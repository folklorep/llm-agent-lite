"""实验 8 的观察脚本：看一次"工具调用"在协议层面到底发生了什么。

不是测试，是**观察工具**。跑法：
    python lab_tool_call_trace.py

============================================================================
这个脚本在回答什么问题
============================================================================
模型说"我要调用 calculator" —— 这句话在网络上传的时候是什么样子？
程序收到之后怎么解析？解析完了又发了什么回去？

一次真实对话（用了工具）其实发生了【两次 API 请求】：
    第 1 次：把用户问题发过去 → 模型回"我要调 calculator"
    第 2 次：把"模型的点单 + 工具的执行结果"一起发过去 → 模型回最终答案

本脚本把这两次请求的**实际内容**和**最终 history**都打出来。
"""
import json

import config
from agent import Agent
from test_agent import FakeClient

config.config.MAX_TOOL_ROUNDS = 5


def sep(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(f"  {title}")
        print("=" * 78)


def show_messages(msgs, label: str) -> None:
    """把一份 messages 列表打得人能看懂。"""
    print(f"\n  【{label}】共 {len(msgs)} 条")
    for i, m in enumerate(msgs):
        role = m.get("role", "?")
        line = f"    [{i}] {role:<9}"
        if m.get("content"):
            c = m["content"].replace("\n", " ")
            line += f" content={c[:46]!r}"
        if "tool_calls" in m:
            tc = m["tool_calls"][0]
            line += f" tool_calls=[{tc['function']['name']}]"
        if "tool_call_id" in m:
            line += f" tool_call_id={m['tool_call_id']!r}"
        print(line)


def main() -> None:
    sep("实验 8：一次工具调用的协议全过程")

    # ---- 用假客户端演一出戏：先在剧本里安排模型的两次回应 ----
    fake = FakeClient(
        [
            # 第 1 轮：模型点单，要算 128*37+15（注意 arguments 是【字符串】）
            {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "calculator",
                        "arguments": json.dumps({"expression": "128*37+15"}),
                    }
                ]
            },
            # 第 2 轮：模型看到工具结果，直接回答
            {"content": "结果是 4751"},
        ]
    )

    agent = Agent(client=fake)
    answer = agent.chat("帮我算 128乘37加15")

    print(f"\n  最终回答：{answer}")

    # ---- 关键：假客户端把每次收到的 messages 都记下来了 ----
    show_messages(fake.calls[0], "第 1 次请求发出去的内容")
    show_messages(fake.calls[1], "第 2 次请求发出去的内容")

    sep("对话结束后 history 里的完整内容")
    for i, m in enumerate(agent.history):
        role = m.get("role", "?")
        print(f"\n  [{i}] role = {role!r}")
        if m.get("content"):
            print(f"       content      = {m['content']!r}")
        if "tool_calls" in m:
            tc = m["tool_calls"][0]
            print("       tool_calls   = [{")
            print(f"           id        : {tc['id']!r}")
            print(f"           name      : {tc['function']['name']!r}")
            print(f"           arguments : {tc['function']['arguments']!r}   ← 字符串!")
            print("       }]")
        if "tool_call_id" in m:
            print(f"       tool_call_id = {m['tool_call_id']!r}   ← 靠它对应上面的 id")

    sep("观察结论")
    print(f"""
  1. 一次"用了工具的对话"，实际发了 {fake.call_count} 次 API 请求
       第 1 次 {len(fake.calls[0])} 条 → 模型回"点单"
       第 2 次 {len(fake.calls[1])} 条 → 模型回最终答案

  2. 模型"点单"时，返回的 message 里：
       content   = None          （没有文字）
       tool_calls = [...]        （要调哪个工具、传什么参数）
       arguments 是【字符串】，程序还要 json.loads 一次

  3. 发回给模型时，必须成对出现：
       assistant 消息（带 tool_calls，含 id）+ tool 消息（带 tool_call_id，值相同）
       —— 这是 OpenAI 兼容协议的硬要求，缺一个 API 就报错

  4. agent.py 里最关键的一行是【没有】的那个 break：
       执行完工具后不 break，回到循环顶部再问模型一次
       —— 这就是"循环"的来源，也是模型能"看到结果再决定下一步"的原因
""")


def demo_multi_tool_calls() -> None:
    """演示：模型一轮点 3 个单时，tool_call_id 是干什么用的。"""
    sep("补充实验：一轮点 3 个单，tool_call_id 怎么配对")

    fake = FakeClient(
        [
            {
                "tool_calls": [
                    {"id": "call_1", "name": "calculator",
                     "arguments": json.dumps({"expression": "144/12"})},
                    {"id": "call_2", "name": "sqrt",
                     "arguments": json.dumps({"x": 144})},
                    {"id": "call_3", "name": "get_current_time",
                     "arguments": "{}"},
                ]
            },
            {"content": "三个都算好了"},
        ]
    )
    agent = Agent(client=fake)
    agent.chat("帮我算三个东西")

    msgs = fake.calls[1]
    print(f"\n  第 2 次请求发了 {len(msgs)} 条，看最后几条：\n")
    for i, m in enumerate(msgs):
        role = m.get("role", "?")
        if role == "assistant" and "tool_calls" in m:
            print(f"    [{i}] {role}")
            for tc in m["tool_calls"]:
                print(f"         点单 id={tc['id']!r}  name={tc['function']['name']!r}")
        elif role == "tool":
            print(f"    [{i}] {role:<6} tool_call_id={m['tool_call_id']!r}  "
                  f"content={m['content']!r}")
        else:
            print(f"    [{i}] {role:<6} content={(m.get('content') or '')[:36]!r}")

    print("""
  ── 结论 ──
  模型一轮点了 3 个单（3 个 id），程序执行后回了 3 条 tool 消息，
  每条靠 tool_call_id 标明"我是哪个点单的结果"。

  如果没有 tool_call_id，模型看到 3 个光秃秃的结果，
  就分不清哪个是 calculator 的、哪个是 sqrt 的。
""")


def demo_error_handling() -> None:
    """演示：工具内部报错时，程序为什么不会崩。"""
    sep("补充实验：工具报错时会发生什么")

    cases = [
        ("正常调用", {"name": "calculator", "arguments": json.dumps({"expression": "1+1"})}),
        ("参数非法（除零）", {"name": "calculator", "arguments": json.dumps({"expression": "1/0"})}),
        ("工具不存在", {"name": "不存在的工具", "arguments": "{}"}),
        ("参数缺失", {"name": "sqrt", "arguments": "{}"}),
        ("参数不是合法 JSON", {"name": "sqrt", "arguments": "这不是json"}),
    ]

    from tools import run_tool

    print()
    for label, call in cases:
        try:
            result = run_tool(call["name"], call["arguments"])
            status = "✅ 正常返回"
            detail = result[:60]
        except Exception as e:  # noqa: BLE001
            status = f"❌ 抛异常 {type(e).__name__}"
            detail = str(e)[:60]
        print(f"  {label:<20} {status}")
        print(f"       {detail}")

    print("""
  ── 结论 ──
  【实测】上面 5 种异常情况【全部】被 run_tool 兜住，
  都以字符串形式（"错误：..."）交回给模型 —— 没有一种让程序崩掉。

  （我原本以为"参数不是合法 JSON"会抛异常，实测发现也被兜住了。
    又一次说明：不要凭印象判断，要实测。）

  为什么这很重要：模型收到"错误：xxx"之后，可以自己决定
  换个参数重试、或者告诉用户做不到。这就是"错误即上下文"。

  对比一下：如果这里直接让异常冒出去，整个请求就 500 了，
  用户只会看到"服务出错"，模型连补救的机会都没有。
""")


if __name__ == "__main__":
    main()
    demo_multi_tool_calls()
    demo_error_handling()
