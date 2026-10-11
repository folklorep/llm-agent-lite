"""检查部署所需的依赖状态。跑法：python check_deps.py"""
import importlib
import sys

REQUIRED = {
    "requests": "必需（调 LLM API）",
    "streamlit": "部署必需（网页界面）",
    "dotenv": "可选（config.py 有内置 .env 解析兜底）",
    "fastapi": "可选（HTTP 接口用）",
    "uvicorn": "可选（跑 fastapi 用）",
}

print("=" * 60)
print("  环境检查")
print("=" * 60)
print(f"\n  Python: {sys.version.split()[0]}")
print(f"  解释器: {sys.executable}\n")

missing_required = []
for mod, why in REQUIRED.items():
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", getattr(m, "VERSION", "?"))
        print(f"  [已装] {mod:12} {str(ver):10}  {why}")
    except ImportError:
        print(f"  [缺失] {mod:12} {'':10}  {why}")
        if mod in ("requests", "streamlit"):
            missing_required.append(mod)

print()
if missing_required:
    print("  需要安装：")
    print(f"    python -m pip install {' '.join(missing_required)}")
else:
    print("  [OK] 部署所需的依赖都齐了")

print("=" * 60)
