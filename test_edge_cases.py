"""边界场景测试：多轮对话 + 最后一轮工具调用。

这个文件回答两个我在写讲解时突然意识到没验过的问题：

  问题 A：模型在"最后一轮"（轮次上限那轮）才点工具，会发生什么？
          循环退出后 history 末尾是 assistant(tool_calls) + tool 结果，
          **没有对应的 assistant 文本回复**。
          下一轮再 chat 时，这段残缺历史会被发出去 —— 某些 API 实现会报 400。

  问题 B：工具返回非 ASCII（中文）时，回传给模型会不会出问题？

跑法：python test_edge_cases.py
"""
import json

import config
from agent import Agent
from test_agent import FakeClient, check


# ---------------------------------------------------------------- 问题 A
def test_last_round_tool_call():
    """模型在最后一轮点工具 —— 观察 history 的尾部结构。"""
    print("\n[边界 A] 模型在轮次上限那一轮才点工具")
    limit = config.config.MAX_TOOL_ROUNDS

    # 剧本：前 limit-1 轮都点工具，第 limit 轮（最后一轮）也点工具，永远不回答
    fake = FakeClient([
        {"tool_calls": [{"id": f"c{i}", "name": "get_current_time", "arguments": "{}"}]}
        for i in range(limit)
    ])
    agent = Agent(client=fake)
    answer = agent.chat("现在几点")

    check(f"请求次数 = 上限 {limit}", fake.call_count == limit, fake.call_count)
    check("给出了兜底回复", "上限" in answer, answer)

    # 关键观察：history 尾部结构
    tail_roles = [m["role"] for m in agent.history[-3:]]
    print(f"    history 尾部 role 序列: {tail_roles}")

    orphan_tool_calls = [
        m for m in agent.history
        if m["role"] == "assistant" and m.get("tool_calls")
    ]
    tool_msgs = [m for m in agent.history if m["role"] == "tool"]
    print(f"    带 tool_calls 的 assistant 消息: {len(orphan_tool_calls)} 条")
    print(f"    role=tool 的结果消息: {len(tool_msgs)} 条")

    # 如果最后一轮的 tool_calls 之后没有 assistant 文本，这段历史就是"残缺"的
    last_role = agent.history[-1]["role"]
    if last_role != "assistant" or agent.history[-1].get("tool_calls"):
        print("    ⚠️ 发现：history 尾部不是一条纯文本 assistant 回复 —— 存在残缺历史")
        print("       → 下一轮再 chat 时这段会被发出去，部分 API 会因")
        print("         'tool_calls 未全部应答' 报 400")
        return True
    print("    ✅ history 尾部是完整的 assistant 回复")
    return False


def test_next_turn_after_limit():
    """承接上一步：轮次耗尽后再问一句，看发出去的 payload 长什么样。"""
    print("\n[边界 A2] 轮次耗尽后再问一句")
    limit = config.config.MAX_TOOL_ROUNDS
    script = [
        {"tool_calls": [{"id": f"c{i}", "name": "get_current_time", "arguments": "{}"}]}
        for i in range(limit)
    ]
    script.append({"content": "现在是下午三点。"})   # 第二轮对话的正常回复
    fake = FakeClient(script)

    agent = Agent(client=fake)
    agent.chat("现在几点")          # 第一轮：耗尽轮次
    agent.chat("谢谢")             # 第二轮：正常回答

    payload = fake.calls[-1]        # 最后一次真正发出去的 messages
    roles = [m["role"] for m in payload]
    print(f"    第二轮 payload 的 role 序列: {roles}")

    # 检查 protocol 完整性：每个带 tool_calls 的 assistant 后面是否都有对应 tool 应答
    pending = []
    broken = []
    for m in payload:
        if m["role"] == "assistant" and m.get("tool_calls"):
            pending.extend(tc.get("id", "") for tc in m["tool_calls"])
        elif m["role"] == "tool":
            tid = m.get("tool_call_id", "")
            if tid in pending:
                pending.remove(tid)
    if pending:
        broken = pending
        print(f"    ⚠️ 发现 {len(pending)} 个 tool_call 没有对应应答: {pending[:3]}")
        print("       → 这一段历史就是残缺的，真实 API 可能报 400")
    else:
        print("    ✅ 每个 tool_call 都有对应应答")

    return bool(broken)


# ---------------------------------------------------------------- 问题 B
def test_chinese_tool_result():
    """工具返回中文，回传给模型时必须是合法的字符串。"""
    print("\n[边界 B] 工具返回中文内容")
    fake = FakeClient([
        {"tool_calls": [{"id": "c1", "name": "search_knowledge",
                         "arguments": json.dumps({"query": "两数之和"})}]},
        {"content": "两数之和用字典记下标。"},
    ])
    agent = Agent(client=fake)
    agent.chat("两数之和怎么解")

    tool_msg = [m for m in agent.history if m["role"] == "tool"][0]
    check("tool 结果是 str", isinstance(tool_msg["content"], str))
    check("中文内容完整保留", "字典" in tool_msg["content"], tool_msg["content"][:60])
    # 检查能否被 json 序列化（真实请求要走 json= 序列化）
    try:
        json.dumps({"messages": agent.history}, ensure_ascii=False)
        check("整个 history 可 JSON 序列化", True)
    except (TypeError, ValueError) as e:
        check("整个 history 可 JSON 序列化", False, str(e))


def main():
    print("=" * 60)
    print("  边界场景测试")
    print("=" * 60)

    has_broken_history = test_last_round_tool_call()
    broken_ids = test_next_turn_after_limit()
    test_chinese_tool_result()

    print("\n" + "=" * 60)
    if has_broken_history or broken_ids:
        print("  [!] 发现了缺陷 —— 见上文 ⚠️ 标记")
        print("  这两个用例断言的是「观察结果」而非「正确答案」，")
        print("  所以它们会通过；缺陷本身需要改 agent.py 修掉。")
    else:
        print("  [OK] 未发现残缺历史")
    print("=" * 60)


if __name__ == "__main__":
    main()
