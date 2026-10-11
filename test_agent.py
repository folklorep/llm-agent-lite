"""Agent 循环测试 —— 用"假模型"代替真实 API。

为什么要这么写（面试点）：
  真实 API 有三个问题：花钱、慢、结果不确定。
  把 LLMClient 换成一个按剧本返回消息的假对象，就能：
    - 精确制造"模型要调工具"和"模型直接回答"两种分支
    - 断言 history 的构造是否符合协议
    - 验证死循环保护、异常兜底是否真的生效
  这就是「依赖注入」带来的可测性。

跑法：python test_agent.py
全部通过会打印 [OK]；任何断言失败会抛出 AssertionError。
"""
import json

import config
from agent import Agent


class FakeClient:
    """假模型：按剧本依次返回预设的 message。

    剧本元素可以是：
      {"content": "..."}                      → 模型直接回答
      {"tool_calls": [{"id":..., "name":..., "arguments":...}]} → 模型点单调工具
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = []          # 记录每次收到的 messages，供断言
        self.call_count = 0

    def chat(self, messages, tools=None, temperature=0.7):
        self.calls.append([dict(m) for m in messages])   # 深拷贝一层，冻结现场
        self.call_count += 1
        if not self.script:
            # 剧本用完了还在问 —— 说明循环没按预期退出
            raise AssertionError("假模型剧本已耗尽，Agent 却还在请求（可能死循环）")
        step = self.script.pop(0)
        if "tool_calls" in step:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.get("id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc.get("arguments", "{}"),
                        },
                    }
                    for i, tc in enumerate(step["tool_calls"])
                ],
            }
        return {"role": "assistant", "content": step["content"]}


def check(label, condition, detail=""):
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f"  <- {detail}" if detail and not condition else ""))
    if not condition:
        raise AssertionError(f"{label} 断言失败 {detail}")


# ---------------------------------------------------------------- 用例 1
def test_direct_answer():
    """模型不需要工具，直接回答。"""
    print("\n[用例 1] 直接回答，不调工具")
    fake = FakeClient([{"content": "你好，我是 AI 助手。"}])
    agent = Agent(client=fake)
    answer = agent.chat("你好")

    check("返回模型内容", answer == "你好，我是 AI 助手。", answer)
    check("只请求了一次", fake.call_count == 1, fake.call_count)
    check("history 长度 = 3（system+user+assistant）", len(agent.history) == 3, len(agent.history))
    check("末条是 assistant", agent.history[-1]["role"] == "assistant")
    check("没有工具轨迹", agent.trace == [], agent.trace)


# ---------------------------------------------------------------- 用例 2
def test_single_tool_call():
    """模型调一次工具 —— 你原来 chat.py 的能力上限。"""
    print("\n[用例 2] 调用一次工具")
    fake = FakeClient([
        {"tool_calls": [{"id": "c1", "name": "calculator",
                         "arguments": json.dumps({"expression": "128*37+15"})}]},
        {"content": "结果是 4751。"},
    ])
    agent = Agent(client=fake)
    answer = agent.chat("帮我算 128乘37加15")

    check("返回最终答案", answer == "结果是 4751。", answer)
    check("请求了两次", fake.call_count == 2, fake.call_count)
    check("产生了 1 条工具轨迹", len(agent.trace) == 1, agent.trace)
    check("工具真的算对了", "4751" in agent.trace[0], agent.trace)

    # 关键：第二次请求里必须带上 assistant 的 tool_calls 和 role=tool 的回传
    second = fake.calls[1]
    roles = [m["role"] for m in second]
    check("第二次请求含 role=tool 消息", "tool" in roles, roles)
    assistant_with_calls = [m for m in second if m.get("tool_calls")]
    check("保留了 assistant 的 tool_calls 原消息", len(assistant_with_calls) == 1)
    tool_msg = [m for m in second if m["role"] == "tool"][0]
    check("tool 消息带 tool_call_id", tool_msg.get("tool_call_id") == "c1", tool_msg)
    check("tool 消息内容是字符串", isinstance(tool_msg["content"], str), type(tool_msg["content"]))


# ---------------------------------------------------------------- 用例 3
def test_multi_round_loop():
    """多工具自动循环 —— 这是本项目的核心卖点。

    模型连续两轮调工具，第三轮才回答。
    你原来的写法（调一次就出答案）在这里会失败。
    """
    print("\n[用例 3] 多轮工具循环（核心能力）")
    fake = FakeClient([
        {"tool_calls": [{"id": "c1", "name": "calculator",
                         "arguments": json.dumps({"expression": "144/12"})}]},
        {"tool_calls": [{"id": "c2", "name": "sqrt",
                         "arguments": json.dumps({"x": 144})}]},
        {"content": "144/12=12，而根号144也是12，两个结果相同。"},
    ])
    agent = Agent(client=fake)
    answer = agent.chat("先算144除以12，再算根号144，比较一下")

    check("请求了三次", fake.call_count == 3, fake.call_count)
    check("产生 2 条工具轨迹", len(agent.trace) == 2, agent.trace)
    check("第一轮是 calculator", "calculator" in agent.trace[0], agent.trace)
    check("第二轮是 sqrt", "sqrt" in agent.trace[1], agent.trace)
    check("两轮结果都在最终上下文里",
          any("12.0" in m.get("content", "") for m in agent.history if m["role"] == "tool"))
    check("返回最终答案", "相同" in answer, answer)


# ---------------------------------------------------------------- 用例 4
def test_parallel_tool_calls():
    """一轮里模型同时点多个工具（并行调用）。"""
    print("\n[用例 4] 一轮内并行调用多个工具")
    fake = FakeClient([
        {"tool_calls": [
            {"id": "c1", "name": "calculator", "arguments": json.dumps({"expression": "2+2"})},
            {"id": "c2", "name": "sqrt", "arguments": json.dumps({"x": 9})},
        ]},
        {"content": "4 和 3。"},
    ])
    agent = Agent(client=fake)
    agent.chat("算 2+2 和 根号9")

    check("产生 2 条轨迹", len(agent.trace) == 2, agent.trace)
    tool_msgs = [m for m in agent.history if m["role"] == "tool"]
    check("回传了 2 条 tool 消息", len(tool_msgs) == 2, len(tool_msgs))
    check("两条 tool_call_id 不同", tool_msgs[0]["tool_call_id"] != tool_msgs[1]["tool_call_id"])


# ---------------------------------------------------------------- 用例 5
def test_round_limit():
    """死循环保护：模型无限想调工具时必须停下。

    这是成本安全阀 —— 没有它，模型抽风会一直烧 token。
    """
    print("\n[用例 5] 死循环保护（轮次上限）")
    limit = config.config.MAX_TOOL_ROUNDS
    fake = FakeClient([
        {"tool_calls": [{"id": f"c{i}", "name": "get_current_time", "arguments": "{}"}]}
        for i in range(limit + 3)          # 故意给足超出上限的剧本
    ])
    agent = Agent(client=fake)
    answer = agent.chat("现在几点")

    check(f"请求次数被限制在 {limit} 次", fake.call_count == limit, fake.call_count)
    check("给出了兜底回复而不是崩溃", "上限" in answer, answer)
    check("兜底回复也进了 history", agent.history[-1]["content"] == answer)


# ---------------------------------------------------------------- 用例 6
def test_tool_error_recovery():
    """工具报错不能让程序崩 —— 错误要回传给模型，让它自己决定怎么办。"""
    print("\n[用例 6] 工具异常兜底")
    fake = FakeClient([
        {"tool_calls": [{"id": "c1", "name": "sqrt",
                         "arguments": json.dumps({"x": -5})}]},
        {"content": "负数没有实数平方根。"},
    ])
    agent = Agent(client=fake)
    answer = agent.chat("根号负五")

    check("程序没崩，拿到答案", answer == "负数没有实数平方根。", answer)
    check("错误信息进了轨迹", "错误" in agent.trace[0], agent.trace)
    err_msg = [m for m in agent.history if m["role"] == "tool"][0]
    check("错误作为字符串回传给模型", err_msg["content"].startswith("错误"))


# ---------------------------------------------------------------- 用例 7
def test_history_trim():
    """上下文裁剪：长对话不能无限膨胀（成本控制）。

    注意这里断言的是**真正发出去的 payload**，不是 session 里躺着的 history。
    两者差 1 条：裁剪发生在"追加本轮 user、还没追加 assistant 回复"的时刻，
    所以一次对话结束后，history 会比 payload 上限多 1 条（最后那条 assistant）。
    这个区别很重要 —— 控制成本看的是 payload。
    """
    print("\n[用例 7] 滑动窗口裁剪 history")
    limit = config.config.MAX_HISTORY_TURNS
    fake = FakeClient([{"content": f"回答{i}"} for i in range(limit * 2 + 5)])
    agent = Agent(client=fake)

    for i in range(limit * 2 + 5):
        agent.chat(f"问题{i}")

    payload_cap = limit * 2 + 1        # system + 最近 N 轮 × 2 条
    check(f"history 被裁到 <= {payload_cap + 1} 条", len(agent.history) <= payload_cap + 1,
          len(agent.history))

    # 逐次回放每次请求的 payload，断言**没有一次**超限
    worst = max(len(c) for c in fake.calls)
    check(f"每次请求的 payload 都不超过 {payload_cap} 条", worst <= payload_cap, worst)

    check("system 消息始终保留在首位", agent.history[0]["role"] == "system")
    check("保留的是最近的对话", agent.history[-1]["content"] == f"回答{limit * 2 + 4}",
          agent.history[-1]["content"])


# ---------------------------------------------------------------- 用例 8
def test_reset():
    print("\n[用例 8] reset 清空对话")
    fake = FakeClient([{"content": "a"}, {"content": "b"}])
    agent = Agent(client=fake)
    agent.chat("第一个问题")
    agent.reset()
    check("history 只剩 system", len(agent.history) == 1, len(agent.history))
    check("trace 被清空", agent.trace == [])


def main():
    print("=" * 60)
    print("  Agent 循环测试（假模型，不消耗 API 额度）")
    print("=" * 60)

    saved = config.config.MAX_HISTORY_TURNS
    config.config.MAX_HISTORY_TURNS = 3      # 调小一点让用例 7 跑得快

    tests = [
        test_direct_answer,
        test_single_tool_call,
        test_multi_round_loop,
        test_parallel_tool_calls,
        test_round_limit,
        test_tool_error_recovery,
        test_history_trim,
        test_reset,
    ]
    for t in tests:
        t()

    config.config.MAX_HISTORY_TURNS = saved
    print("\n" + "=" * 60)
    print(f"  [OK] {len(tests)} 个用例全部通过")
    print("=" * 60)


if __name__ == "__main__":
    main()
