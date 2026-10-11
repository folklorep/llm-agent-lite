"""验证 config.py 的优先级逻辑：环境变量 → st.secrets → 默认值。

关键点：本地有 .env 时会污染测试（env 优先，就走不到 secrets 分支），
所以这个脚本用一个子进程 + 清空环境变量来隔离验证。

跑法：python test_config_priority.py
"""
import os
import subprocess
import sys
import tempfile
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))

FAKE_STREAMLIT = textwrap.dedent(
    '''
    import sys, types
    fake = types.ModuleType("streamlit")
    fake.secrets = {
        "LLM_API_KEY": "sk-FROM-SECRETS",
        "LLM_MODEL": "model-FROM-SECRETS",
        "LLM_BASE_URL": "https://base-from-secrets",
    }
    sys.modules["streamlit"] = fake
    '''
)


def run(py_code: str, env: dict) -> str:
    """在干净的子进程里跑一段 python，返回 stdout。"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8", dir=HERE) as f:
        f.write(py_code)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, encoding="utf-8", cwd=HERE, env=env)
        return (r.stdout or "") + (r.stderr or "")
    finally:
        os.unlink(path)


def clean_env() -> dict:
    """复制当前环境，但清掉所有 LLM_* 变量，并让 config 不去读 .env。"""
    env = {k: v for k, v in os.environ.items() if not k.startswith("LLM_")}
    # 用一个不存在的目录当 cwd 不行（要 import config），
    # 所以改为把 .env 临时改名 —— 更直接
    return env


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  <- {detail}" if not ok else ""))
    return ok


def main() -> None:
    print("=" * 64)
    print("  config.py 优先级验证")
    print("=" * 64)

    env_file = os.path.join(HERE, ".env")
    backup = env_file + ".testbak"
    had_env = os.path.exists(env_file)
    if had_env:
        os.rename(env_file, backup)          # 临时藏起 .env
    print(f"\n  本地 .env 存在: {had_env} → 已临时移开，以便隔离测试\n")

    results = []
    env = clean_env()

    try:
        # ---- 用例 1：环境变量有值 → 应该用环境变量 ----
        print("[1] 环境变量优先于 st.secrets")
        code = FAKE_STREAMLIT + textwrap.dedent(
            '''
            import config
            print("RESULT", config.config.API_KEY)
            '''
        )
        e = dict(env, LLM_API_KEY="sk-FROM-ENV")
        out = run(code, e)
        results.append(check("读到环境变量的值（而不是 secrets）",
                             "RESULT sk-FROM-ENV" in out, out.strip()[:120]))

        # ---- 用例 2：环境变量为空 + secrets 有值 → 应该用 secrets ----
        print("\n[2] 没有环境变量时回退到 st.secrets")
        out = run(code, env)     # env 里已清掉 LLM_*
        results.append(check("成功从 st.secrets 读到 key",
                             "RESULT sk-FROM-SECRETS" in out, out.strip()[:120]))

        # ---- 用例 3：secrets 的 MODEL / BASE_URL 也要能读到 ----
        print("\n[3] MODEL / BASE_URL 也能从 secrets 回退")
        code2 = FAKE_STREAMLIT + textwrap.dedent(
            '''
            import config
            print("MODEL", config.config.MODEL)
            print("BASE", config.config.BASE_URL)
            '''
        )
        out = run(code2, env)
        results.append(check("MODEL 来自 secrets",
                             "MODEL model-FROM-SECRETS" in out, out.strip()[:120]))
        results.append(check("BASE_URL 来自 secrets",
                             "BASE https://base-from-secrets" in out, out.strip()[:120]))

        # ---- 用例 4：既没环境变量也没 streamlit → 用默认值，且不能崩 ----
        print("\n[4] 都没有时用默认值（且无 streamlit 也不报错）")
        code3 = textwrap.dedent(
            '''
            import config
            print("KEY", repr(config.config.API_KEY))
            print("MODEL", config.config.MODEL)
            print("BASE", config.config.BASE_URL)
            '''
        )
        out = run(code3, env)
        results.append(check("无 key 时返回空串而不是崩溃",
                             "KEY ''" in out, out.strip()[:120]))
        results.append(check("MODEL 用默认值 deepseek-flash",
                             "MODEL deepseek-flash" in out, out.strip()[:120]))
        results.append(check("BASE_URL 用默认值",
                             "BASE https://api.deepseek.com" in out, out.strip()[:120]))

    finally:
        if had_env:
            os.rename(backup, env_file)      # 恢复 .env
            print("\n  已恢复本地 .env")

    print("\n" + "=" * 64)
    if all(results):
        print(f"  [OK] {len(results)} 项全部通过")
    else:
        print(f"  [!] {results.count(False)} 项失败")
    print("=" * 64)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
