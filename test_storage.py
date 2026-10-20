"""存储层测试：守护 SessionStore 的读写契约。

============================================================================
这个测试在守护什么
============================================================================
SessionStore 负责"把对话历史存进 SQLite 再原样读回来"。
它一旦出错，后果是【用户数据丢失或错乱】—— 比功能缺失严重得多。

所以这里重点验证四条：

    ① 原样往返：存进去的 history，读出来必须一模一样
       （尤其 tool_calls 这种嵌套结构 —— 它是 JSON 序列化的重点）
    ② 覆盖写：history 被裁短之后再存，数据库也要跟着变短
    ③ 持久化：关闭连接再打开，数据还在（这是本模块存在的全部意义）
    ④ 隔离：会话之间不能串数据

⚠️ 全程用 ":memory:" 内存库 —— 不碰 data/sessions.db，也不留垃圾文件。

跑法：python test_storage.py
"""
import tempfile
from pathlib import Path

from storage import SessionStore


def check(label: str, ok: bool, detail: object = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  <- {detail}" if not ok else ""))
    if not ok:
        raise AssertionError(f"{label} 断言失败 {detail}")


# 一段真实的 history，含 tool_calls 这种嵌套结构
SAMPLE_HISTORY = [
    {"role": "system", "content": "你是一个严谨的 AI 助手"},
    {"role": "user", "content": "帮我算 2+3"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "calculator", "arguments": '{"expression": "2+3"}'},
        }],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "2+3 = 5"},
    {"role": "assistant", "content": "结果是 5"},
]


def test_roundtrip() -> None:
    print("\n[用例 1] 原样往返：存进去什么，读出来就是什么")
    store = SessionStore(":memory:")
    store.ensure_session("s1")
    n = store.save_messages("s1", SAMPLE_HISTORY)
    check("存入条数正确", n == len(SAMPLE_HISTORY), n)

    back = store.load_messages("s1")
    check("读出条数一致", len(back) == len(SAMPLE_HISTORY), len(back))
    check("整体完全一致（含 tool_calls 嵌套结构）", back == SAMPLE_HISTORY)

    # 单独盯一下最容易出问题的那个字段
    check("assistant 的 tool_calls 被完整保留",
          back[2].get("tool_calls") == SAMPLE_HISTORY[2]["tool_calls"])
    check("tool 消息的 tool_call_id 被保留",
          back[3].get("tool_call_id") == "call_1", back[3])
    store.close()


def test_overwrite_after_trim() -> None:
    print("\n[用例 2] 覆盖写：history 被裁短后，数据库也要变短")
    store = SessionStore(":memory:")
    store.ensure_session("s2")
    store.save_messages("s2", SAMPLE_HISTORY)
    check("先存 5 条", store.count_messages("s2") == 5, store.count_messages("s2"))

    # 模拟 Agent 的 _trim_history 把历史裁短
    trimmed = SAMPLE_HISTORY[:1] + SAMPLE_HISTORY[-2:]
    store.save_messages("s2", trimmed)
    check("覆盖写后变 3 条", store.count_messages("s2") == 3, store.count_messages("s2"))
    check("内容也是裁剪后的版本", store.load_messages("s2") == trimmed)

    # 这个是关键：如果实现成"追加写"，条数会变成 5+3=8
    check("不是追加写（否则会变 8 条）", store.count_messages("s2") != 8)
    store.close()


def test_persistence_across_reopen() -> None:
    print("\n[用例 3] ⭐ 持久化：关闭连接再打开，数据还在")
    tmp = str(Path(tempfile.gettempdir()) / "storage_test.db")
    Path(tmp).unlink(missing_ok=True)

    store = SessionStore(tmp)
    store.ensure_session("s3")
    store.save_messages("s3", SAMPLE_HISTORY)
    store.close()                                    # ← 模拟"程序退出"

    store2 = SessionStore(tmp)
    back = store2.load_messages("s3")
    check("重新打开后数据还在", len(back) == len(SAMPLE_HISTORY), len(back))
    check("内容也一模一样", back == SAMPLE_HISTORY)
    check("会话仍然存在", store2.session_exists("s3"))
    store2.close()

    Path(tmp).unlink(missing_ok=True)


def test_isolation_between_sessions() -> None:
    print("\n[用例 4] 会话隔离：不同会话不能串数据")
    store = SessionStore(":memory:")
    a = [{"role": "user", "content": "我是会话 A"}]
    b = [{"role": "user", "content": "我是会话 B"}]

    store.ensure_session("sA")
    store.ensure_session("sB")
    store.save_messages("sA", a)
    store.save_messages("sB", b)

    check("A 只有自己的消息", store.load_messages("sA") == a)
    check("B 只有自己的消息", store.load_messages("sB") == b)

    # 删掉 A，不能影响 B
    store.delete_session("sA")
    check("删 A 后 A 不存在", not store.session_exists("sA"))
    check("删 A 后 B 不受影响", store.load_messages("sB") == b)
    store.close()


def test_misc() -> None:
    print("\n[用例 5] 其他：排序、计数、会话列表、错误输入")
    store = SessionStore(":memory:")

    # 消息必须按插入顺序读回（靠自增 id 排序）
    store.ensure_session("s5")
    ordered = [{"role": "user", "content": f"第{i}条"} for i in range(10)]
    store.save_messages("s5", ordered)
    check("顺序正确（10 条按顺序读回）", store.load_messages("s5") == ordered)

    # 空历史
    store.ensure_session("s6")
    store.save_messages("s6", [])
    check("存空历史不报错，读出 0 条", store.load_messages("s6") == [])

    # 不存在的会话
    check("读不存在的会话返回空列表", store.load_messages("不存在") == [])
    check("删不存在的会话返回 False", store.delete_session("不存在") is False)

    # 会话列表带消息计数
    sessions = store.list_sessions()
    ids = {s["session_id"]: s["message_count"] for s in sessions}
    check("会话列表包含 s5 且计数为 10", ids.get("s5") == 10, ids)
    store.close()


def main() -> None:
    print("=" * 66)
    print("  存储层测试（SQLite 会话持久化）")
    print("=" * 66)
    test_roundtrip()
    test_overwrite_after_trim()
    test_persistence_across_reopen()
    test_isolation_between_sessions()
    test_misc()
    print("\n" + "=" * 66)
    print("  [OK] 存储层全部通过")
    print("=" * 66)


if __name__ == "__main__":
    main()
