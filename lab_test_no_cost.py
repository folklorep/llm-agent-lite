"""实验 11 的观察脚本：测试怎么做到"不花一分钱"。

跑法：
    python lab_test_no_cost.py

============================================================================
这个脚本在回答什么问题
============================================================================
113 个测试断言跑完，为什么一次 API 都没调、一分钱都没花？

============================================================================
全部秘密就两件事
============================================================================
    ① 依赖注入（DI）：Agent 的构造函数留了个 client 参数
          生产环境 → 传真的 LLMClient（会发网络请求）
          测试环境 → 传假的 FakeClient（纯内存，不发请求）

    ② 桩对象（Stub）：FakeClient 按"剧本"返回预设的响应
          剧本第 1 条：模型说"我要调 calculator"
          剧本第 2 条：模型说"结果是 5"
          → 完全模拟了真实模型的两次响应，但没有任何网络调用

============================================================================
本脚本做三个对比
============================================================================
    对比 1：真客户端 vs 假客户端 —— 证明"零成本"
    对比 2：FakeClient 的额外能力 —— 它能"记录"发出去的内容
    对比 3：剧本耗尽时报错 —— 把"死循环"变成"测试失败"
"""
import json

import config
from agent import Agent
from test_agent import FakeClient


def sep(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


# ============================================================ 对比 1
def demo_di_vs_no_di() -> None:
    """对比：依赖注入 vs 不使用依赖注入。"""
    sep("对比 1：依赖注入的价值")

    print("""
  ── 看 Agent 的构造函数 ──

      def __init__(self, system_prompt=SYSTEM_PROMPT, client=None):
          self.client = client if client is not None else LLMClient()
                          ↑                              ↑
                  测试时传这个                    生产环境用这个

  ── 如果【没有】这个参数会怎样？ ──

      def __init__(self, system_prompt=SYSTEM_PROMPT):
          self.client = LLMClient()      # ← 写死
""")

    # 实测：假客户端下零网络调用
    print("  ── 实测：用假客户端跑一次完整对话 ──\n")
    fake = FakeClient([
        {"tool_calls": [{"id": "c1", "name": "calculator",
                         "arguments": json.dumps({"expression": "2+3"})}]},
        {"content": "结果是 5"},
    ])
    agent = Agent(client=fake)
    answer = agent.chat("帮我算 2+3")

    print(f"    Agent 用的 client 类型: {type(agent.client).__name__}")
    print(f"    最终回答              : {answer}")
    print(f"    假客户端被调用        : {fake.call_count} 次")
    print(f"    真实的网络请求        : 0 次 ✅")
    print(f"    花费的 token          : 0 ✅")
    print(f"    需要的 API key        : 不需要 ✅")
    print(f"""
  ── 结论 ──

  假客户端和真客户端【接口完全一致】（都有 chat(messages, tools, temperature)），
  所以 Agent 的循环代码【一行都不用改】——
  它根本不知道自己拿到的是真客户端还是假的。

  这就是"面向接口编程"：
      Agent 依赖的不是"LLMClient 这个具体的类"，
      而是"任何有 chat() 方法的对象"。
""")


# ============================================================ 对比 2
def demo_recording_capability() -> None:
    """对比 2：FakeClient 能记录发出去的内容。"""
    sep("对比 2：假客户端的额外能力 —— 记录发出去的内容")

    print("""
  FakeClient 的 chat() 里有一行关键代码：

      self.calls.append([dict(m) for m in messages])   # ← 记录现场

  它把【每次收到的 messages 都存下来】。这让测试能断言一些
  返回值和真调 API 都查不到的东西。
""")

    fake = FakeClient([
        {"tool_calls": [{"id": "c1", "name": "sqrt",
                         "arguments": json.dumps({"x": 144})}]},
        {"content": "12"},
    ])
    agent = Agent(client=fake)
    agent.chat("根号 144 是多少")

    print(f"  ── 实测：这次对话记录到 {len(fake.calls)} 次请求 ──\n")
    for i, msgs in enumerate(fake.calls):
        print(f"    第 {i+1} 次请求共 {len(msgs)} 条：")
        for j, m in enumerate(msgs):
            role = m.get("role", "?")
            if m.get("content"):
                c = str(m["content"]).replace("\n", " ")[:40]
                print(f"      [{j}] {role:<10} {c!r}")
            elif "tool_calls" in m:
                print(f"      [{j}] {role:<10} tool_calls=[{m['tool_calls'][0]['function']['name']}]")
            elif "tool_call_id" in m:
                print(f"      [{j}] {role:<10} tool_call_id={m['tool_call_id']!r}")

    print(f"""
  ── 有了记录，测试能断言什么？ ──

  这些东西【返回值里查不到】，只能通过记录来看：

    ① 第 1 次请求有没有带 tools 参数？（模型是怎么知道有哪些工具的）
    ② payload 的条数有没有超过上限？（成本控制的断言）
    ③ role='tool' 的消息有没有带 tool_call_id？（协议合规性）
    ④ 系统提示词有没有被保留？（reset() 之后应该只剩它）

  举个真实例子（test_agent.py 用例 7 就是这么写的）：

      for msgs in fake.calls:
          assert len(msgs) <= payload_cap    # 每次请求都不能超限

  ⚠️ 这个断言【永远不可能用真 API 写】——
     真 API 不会告诉你"我刚才收到了几条消息"。
     所以"记录能力"是假客户端不可替代的价值，不只是省钱。
""")


# ============================================================ 对比 3
def demo_script_exhausted() -> None:
    """对比 3：剧本耗尽 → 把死循环变成测试失败。"""
    sep("对比 3：剧本耗尽时报错 —— 死循环变成测试失败")

    print("""
  FakeClient 里有一段很聪明的代码：

      if not self.script:
          raise AssertionError("假模型剧本已耗尽，Agent 却还在请求（可能死循环）")

  ── 这不是普通的报错，是一个【测试守护】 ──
""")

    print("\n  ── 实测：给一个「永远点单」的模型，看会发生什么 ──\n")

    config.config.MAX_TOOL_ROUNDS = 3
    # 剧本只给 1 条，但模型会一直被问（如果没有上限的话）
    fake = FakeClient([
        {"tool_calls": [{"id": "c1", "name": "calculator",
                         "arguments": json.dumps({"expression": "1+1"})}]},
    ])
    agent = Agent(client=fake)

    try:
        answer = agent.chat("一直算")
        print(f"    （没触发异常，正常返回）: {answer[:50]}")
    except AssertionError as e:
        print(f"    ✅ 触发了 AssertionError: {e}")

    print(f"""
  ── 为什么要这么设计 ──

  假设 Agent 的循环写错了（比如该停的时候没停）：
    真实情况：会一直调 API，烧钱，而且测试【会一直卡住】直到超时
    有了这个守护：第 2 次请求就【立刻抛 AssertionError】
                  → 测试失败，而且错误信息直接告诉你"可能死循环"

  这是一个"fail fast"的设计：
    把"烧钱 + 卡住"变成"立刻失败 + 明确提示"。

  ⚠️ 注意：这里的意思是"如果模型被多问了一次就应该报错"，
     所以剧本条数 = 你【期望】的模型被调用次数。
     多一条少一条都会暴露问题。
""")
    config.config.MAX_TOOL_ROUNDS = 5


# ============================================================ 对比 4
def demo_assertive_test() -> None:
    """对比 4：看一个真实测试用例长什么样。"""
    sep("对比 4：一个真实测试用例（context 裁剪的那个）")

    print("""
  这是 test_agent.py 用例 7 的核心逻辑（简化版）：
""")
    print("""
      config.config.MAX_HISTORY_TURNS = 3
      fake = FakeClient([{"content": "好的"}] * 20)
      agent = Agent(client=fake)

      for i in range(12):
          agent.chat(f"第{i}轮")

      # 断言 1：history 本身被裁到合理大小
      assert len(agent.history) <= payload_cap + 1

      # 断言 2：⭐ 每次请求的 payload 都不超过上限
      worst = max(len(msgs) for msgs in fake.calls)
      assert worst <= payload_cap, f"最差的一次发出去 {worst} 条"
""")

    # 实测这个断言
    config.config.MAX_HISTORY_TURNS = 3
    limit = config.config.MAX_HISTORY_TURNS * 2
    payload_cap = 1 + limit

    fake = FakeClient([{"content": "好的"}] * 20)
    agent = Agent(client=fake)
    for i in range(12):
        agent.chat(f"第{i}轮")

    worst = max(len(msgs) for msgs in fake.calls)
    print(f"  ── 实测结果 ──")
    print(f"    MAX_HISTORY_TURNS = 3  →  payload 上限 = 1 + 3*2 = {payload_cap}")
    print(f"    实际最差的一次发出去: {worst} 条")
    print(f"    断言 {'通过 ✅' if worst <= payload_cap else '失败 ❌'}")
    print(f"""
  ── 这个断言的价值 ──

  它抓出过一个【真实 bug】：裁剪时机放错，导致每轮多发 2 条。
  这个 bug 的表现是"成本慢慢变高"—— 不报错、不崩溃，
  【只靠肉眼看代码几乎发现不了】。

  但因为假客户端记录了每次 payload 的条数，
  一个 assert 就把它钉死了。

  ⭐ 这就是"可观测的测试"：把抽象的"成本控制"变成
     一行可断言的数字。
""")
    config.config.MAX_HISTORY_TURNS = 20


def main() -> None:
    sep("实验 11：测试怎么做到不花一分钱")
    demo_di_vs_no_di()
    demo_recording_capability()
    demo_script_exhausted()
    demo_assertive_test()

    sep("总结：零成本测试的两个支柱")
    print("""
  ┌──────────────┬────────────────────────────────────────────┐
  │ 支柱          │ 作用                                        │
  ├──────────────┼────────────────────────────────────────────┤
  │ 依赖注入 (DI) │ 让 Agent 不知道自己在跟谁说话                │
  │              │ → 测试时换成假客户端，代码一行不用改          │
  ├──────────────┼────────────────────────────────────────────┤
  │ 桩对象 (Stub) │ 按剧本返回预设响应                          │
  │              │ → 不联网、不花钱、结果可复现                  │
  └──────────────┴────────────────────────────────────────────┘

  ⭐ 一个容易忽略的额外收益：
     假客户端【记录了每次收到的 messages】——
     这让"断言发出去的 payload"成为可能，
     而这种断言【用真 API 永远写不出来】。

  ⭐ 另一个聪明的设计：
     "剧本耗尽就抛异常" —— 把"死循环烧钱"变成"立刻测试失败"。

  面试可以这样讲：
     "我用依赖注入把模型客户端做成可替换的：生产传真实客户端，
      测试传一个按剧本返回的假客户端。所以 113 个断言能在
      不联网、不花钱、不需要 API key 的情况下全部跑完。
      而且假客户端会记录每次发出去的 messages，
      所以我还能断言'payload 条数不超过上限'这种返回值里看不到的东西 ——
      这个断言抓出过一个真实的成本 bug。"
""")


if __name__ == "__main__":
    main()
