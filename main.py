"""命令行入口：python main.py

对比你原来的玩法：以前是双击 / cmd 里 `python chat.py`，
现在有了包结构，还能 `python main.py --debug` 看工具调用轨迹 —— 这就叫"工程化"。
"""
import argparse
import sys

from agent import Agent
from config import config


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Agent Lite —— 多工具智能体")
    parser.add_argument("--debug", action="store_true", help="打印工具调用轨迹")
    parser.add_argument("--once", type=str, default=None, help="只问一个问题就退出")
    args = parser.parse_args()

    if args.debug:
        config.DEBUG = True

    try:
        agent = Agent()
    except RuntimeError as e:
        print(f"[配置错误] {e}")
        sys.exit(1)

    # 单次模式（方便写自动化测试 / 录屏演示）
    if args.once:
        try:
            print(agent.chat(args.once))
        except Exception as e:  # noqa: BLE001 —— 入口兜底，只给一行可读错误
            print(f"[请求失败] {type(e).__name__}: {e}")
            sys.exit(1)
        return

    print("=" * 56)
    print("  LLM Agent Lite —— 可用工具：计算器 / 平方根 / 知识库检索 / 时间")
    print("  输入 exit 退出，输入 /trace 查看上次工具调用，输入 /reset 清空对话")
    print("=" * 56)

    while True:
        try:
            user_input = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "退出"}:
            print("再见！")
            break
        if user_input == "/reset":
            agent.reset()
            print("[已清空对话历史]")
            continue
        if user_input == "/trace":
            print("\n".join(agent.trace) if agent.trace else "[本轮没有调用工具]")
            continue

        try:
            answer = agent.chat(user_input)
        except Exception as e:  # noqa: BLE001 —— 入口兜底，不让程序崩
            print(f"[请求失败] {type(e).__name__}: {e}")
            continue

        print(f"\nAI > {answer}")


if __name__ == "__main__":
    main()
