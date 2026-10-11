"""配置层：所有环境相关的东西集中在这里。

为什么要拆出来？
  - key 散落在业务代码里 → 一旦要换模型/换平台，要改很多文件
  - 面试点：配置与逻辑分离（12-factor app 的第 3 条）
"""
import os
import sys
from pathlib import Path

# python-dotenv 设为可选依赖：没装也能跑（用下面的极简解析器兜住）
# 这样 clone 下来只要 `pip install requests` 就能启动，少一个新手卡点。
try:
    from dotenv import load_dotenv

    load_dotenv()
    _HAS_DOTENV = True
except ImportError:
    _HAS_DOTENV = False

    def _load_env_fallback() -> None:
        """极简 .env 解析：KEY=VALUE，忽略注释和空行。

        注意：用 encoding="utf-8-sig" 读取 —— Windows 上用 PowerShell / 记事本
        保存的 UTF-8 文件会带 BOM，若按普通 utf-8 读，第一个 key 会变成
        '\\ufeffLLM_API_KEY' 而永远读不到。这个坑很隐蔽。
        """
        env_file = Path(__file__).parent / ".env"
        if not env_file.exists():
            return
        raw = env_file.read_text(encoding="utf-8-sig")
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

    _load_env_fallback()


def _debug(msg: str) -> None:
    print(f"[config] {msg}", file=sys.stderr)


def _get(name: str, default: str = "") -> str:
    """按优先级读配置：环境变量 → Streamlit secrets → 默认值。

    ⚠️ 为什么要多读一个地方（踩过的坑）：
       本地运行时配置来自 .env（被 load_dotenv 写进 os.environ），
       但部署到 **Streamlit Cloud** 时，平台上填的 Secrets 是注入到
       `st.secrets` 里的，**不保证同步成环境变量**。
       结果就是：本地跑得好好的，一上云就报"缺少 LLM_API_KEY"。

       这段兜底让同一个配置层能同时服务 CLI、本地 Streamlit、云端 Streamlit。
       注意用 try/except 包住 —— 不装 streamlit 的环境（比如只跑 CLI）不能因此崩掉。
    """
    value = os.getenv(name)
    if value:
        return value

    try:                                    # 只在导入得到 streamlit 时才有意义
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:                       # noqa: BLE001
        # 没装 streamlit / 不在 Streamlit 环境 / secrets 文件不存在 —— 都属正常
        pass

    return default


class Config:
    """配置集中管理。读取优先级：环境变量 → Streamlit secrets → 默认值。"""

    # 大模型 API
    API_KEY: str = _get("LLM_API_KEY")
    BASE_URL: str = _get("LLM_BASE_URL", "https://api.deepseek.com")
    MODEL: str = _get("LLM_MODEL", "deepseek-flash")

    # 行为参数
    DEBUG: bool = _get("DEBUG", "false").lower() == "true"
    TIMEOUT: int = 60
    MAX_TOOL_ROUNDS: int = 5      # 工具调用最大轮次，防止死循环
    MAX_HISTORY_TURNS: int = 20   # 滑动窗口：只保留最近 N 轮，控制 token 成本

    @classmethod
    def chat_url(cls) -> str:
        return f"{cls.BASE_URL.rstrip('/')}/chat/completions"

    @classmethod
    def validate(cls) -> None:
        if not cls.API_KEY or cls.API_KEY.startswith("sk-在这里"):
            raise RuntimeError(
                "缺少 LLM_API_KEY。请复制 .env.example 为 .env 并填入你的 key。"
            )
        if not _HAS_DOTENV:
            _debug("未安装 python-dotenv，已启用内置 .env 解析（功能等价）")


config = Config()
