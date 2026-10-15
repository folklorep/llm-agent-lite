"""实验 9 的观察脚本：看清 agent.py 主循环的控制流。

跑法：
    python lab_agent_loop_trace.py

============================================================================
这个脚本在回答什么问题
============================================================================
agent.py 里那个 `for round_no in range(1, MAX_TOOL_ROUNDS + 1)` 循环，
每一步在干什么？有几条路能走出去？

本脚本用一个"会记录被调用情况"的假客户端，把每一轮的
【请求条数 → 模型回了什么 → 程序做了什么决定】都打出来。

============================================================================
三种退出路径（这是本实验的核心）
============================================================================
    路径 1  模型不需要工具，直接回答
            → 循环里碰到 `return answer`，立刻返回

    路径 2  模型要工具，执行完继续循环，下一轮它回答了
            → 也是 `return answer`，只是发生在第 2 轮或更后

    路径 3  循环跑满了 MAX_TOOL_ROUNDS 次，模型还在要工具
            → for 循环自然结束，走"兜底回复"
"""
import json

import config
from agent import Agent
from test_agent import FakeClient


def sep(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


# ============================================================ 演示 1
def demo_path1_direct_answer() -> None:
    """路径 1：模型不需要工具，第 1 轮就回答。"""
    sep("演示 1：路径 1 —— 模型直接回答（第 1 轮就 return）")

    fake = FakeClient([{"content": "你好，我是 AI 助手"}])
    agent = Agent(client=fake)
    answer = agent.chat("你好")

    print(f"\n  问模型的次数: {fake.call_count}")
    print(f"  history 条数: {len(agent.history)}")
    print(f"  trace 内容  : {agent.trace}")
    print(f"  最终回答    : {answer}")
    print(f"""
  ── 这一轮循环内部发生了什么 ──
    进入 for 循环（round_no = 1）
      message = client.chat(...)         → 拿到 {{'content': '你好，我是 AI 助手'}}
      tool_calls = message.get(...)      → None（没有工具调用）
      if not tool_calls: 成立 → return answer   ★ 从这里出去了
""")


# ============================================================ 演示 2
def demo_path2_tool_then_answer() -> None:
    """路径 2：先调工具，下一轮回答。"""
    sep("演示 2：路径 2 —— 先调工具，第 2 轮回答")

    fake = FakeClient([
        {"tool_calls": [{"id": "c1", "name": "calculator",
                         "arguments": json.dumps({"expression": "2+3"})}]},
        {"content": "结果是 5"},
    ])
    agent = Agent(client=fake)
    answer = agent.chat("帮我算 2+3")

    print(f"\n  问模型的次数: {fake.call_count}   ← 两次！")
    print(f"  history 条数: {len(agent.history)}")
    print(f"  trace       : {agent.trace}")
    print(f"  最终回答    : {answer}")
    print(f"""
  ── 循环内部发生了什么 ──
    round_no = 1
      message = client.chat(...)      → 拿到 tool_calls=[calculator]
      tool_calls 非空 → 不走 return
      history.append(message)         → 把"点单"记进对话
      run_tool('calculator', ...)     → 算出 5
      history.append({{role:'tool',...}})  → 把结果记进对话
      ★ 注意：这里【没有 break】→ 循环继续，round_no 变成 2

    round_no = 2
      message = client.chat(...)      → 这次模型看到结果，回 content='结果是 5'
      tool_calls 是 None → return answer   ★ 从这里出去了
""")


# ============================================================ 演示 3
def demo_path3_exhausted() -> None:
    """路径 3：循环跑满，走兜底。"""
    sep("演示 3：路径 3 —— 轮次耗尽（走兜底回复）")

    config.config.MAX_TOOL_ROUNDS = 2      # 调小，快速演示
    # 剧本：每一轮都在点单，永远不回答 → 触发兜底
    fake = FakeClient([
        {"tool_calls": [{"id": "c1", "name": "calculator",
                         "arguments": json.dumps({"expression": "1+1"})}]},
        {"tool_calls": [{"id": "c2", "name": "calculator",
                         "arguments": json.dumps({"expression": "2+2"})}]},
        # 第 3 次如果被调用，就说明循环没停下来（不该发生）
        {"content": "这次不该被用到"},
    ])
    agent = Agent(client=fake)
    answer = agent.chat("一直算下去")

    print(f"\n  MAX_TOOL_ROUNDS = 2")
    print(f"  问模型的次数: {fake.call_count}   ← 正好 2 次（用满了）")
    print(f"  trace       : {agent.trace}")
    print(f"  最终回答    : {answer}")
    print(f"""
  ── 循环内部发生了什么 ──
    round_no = 1 → 点单 → 执行 → 继续
    round_no = 2 → 点单 → 执行 → 继续
    for 循环条件不成立（range(1, 3) 只给 1、2）→ 循环自然结束
      ↓
    执行循环【之后】的代码：拼兜底回复、记进 history、return
      ★ 这是第三条路径 —— 不是 return 出去的，是"掉"出去的

    ⚠️ 假客户端的剧本第 3 条没被用到 —— 说明循环确实在 2 次后停了。
       如果它被用到，假客户端也会抛异常（"剧本耗尽"），双重保险。
""")
    config.config.MAX_TOOL_ROUNDS = 5      # 改回去


# ============================================================ 演示 4
def demo_state_and_reset() -> None:
    """演示：连问两个问题会怎样 + reset() 干什么。"""
    sep("演示 4：连问两个问题，history 会越来越长吗？")

    fake = FakeClient([
        {"content": "第 1 个回答"},
        {"content": "第 2 个回答"},
    ])
    agent = Agent(client=fake)

    agent.chat("第一个问题")
    print(f"\n  问完第 1 个问题后：history = {len(agent.history)} 条，"
          f"trace = {len(agent.trace)} 条")

    agent.chat("第二个问题")
    print(f"  问完第 2 个问题后：history = {len(agent.history)} 条，"
          f"trace = {len(agent.trace)} 条")

    print(f"\n  ── 关键区别 ──")
    print(f"    history ：【累积】的 —— 第 2 个问题能看到第 1 个问题的内容")
    print(f"    trace   ：【每轮重置】—— 每次 chat() 开头都 self.trace = []")
    print(f"""
  为什么 history 要累积？
    → 这就是"多轮对话"的实现方式（实验 2 学过）：把之前的话重发，
      模型才"看起来记得"。

  为什么 trace 要重置？
    → trace 是"这一次问答用了哪些工具"的展示信息。
      如果累积，界面上会把上次的工具调用也显示出来，造成混淆。

  那 reset() 干什么？
    → self.history = [self.history[0]]   ← 只保留 system 那条
      self.trace = []
      也就是"清空对话记忆，但保留人设"。
""")

    agent.reset()
    print(f"  调用 reset() 之后：history = {len(agent.history)} 条 "
          f"（只剩 system）")


def main() -> None:
    sep("实验 9：agent.py 主循环的控制流")
    demo_path1_direct_answer()
    demo_path2_tool_then_answer()
    demo_path3_exhausted()
    demo_state_and_reset()

    sep("总结：这个循环有 3 条退出路径")
    print("""
  ┌─────┬──────────────────────────────┬─────────────────────────┐
  │ 路径 │ 触发条件                      │ 怎么出去的               │
  ├─────┼──────────────────────────────┼─────────────────────────┤
  │  1  │ 模型第 1 轮就不需要工具        │ return answer（第 1 轮） │
  │  2  │ 模型先调工具，后面某轮回答了    │ return answer（第 N 轮） │
  │  3  │ 跑满 MAX_TOOL_ROUNDS 还在点单  │ for 循环结束 → 兜底回复  │
  └─────┴──────────────────────────────┴─────────────────────────┘

  ⚠️ 路径 1 和 2 其实是同一个 return 语句，区别只是发生在第几轮。
     真正独立的出口只有【两个】：return answer / for 跑完。

  ⚠️ 路径 3 必须存在 —— 否则模型反复点单时，用户请求永远不返回。
     这就是实验 1 学的"轮次上限是成本安全阀"在代码里的落点。
""")


if __name__ == "__main__":
    main()
