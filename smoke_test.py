"""冒烟测试：不调用真实 API，只验证检索层与工具层能正常工作。

跑法：python smoke_test.py
这也是 P6（pytest）的起点 —— 先有能跑的断言，再谈框架。
"""
import rag
import tools

CASES = [
    ("calculator", '{"expression": "128乘37加15"}'),
    ("calculator", '{"expression": "(3.14*17)^2"}'),
    ("calculator", "not-a-json"),
    ("calculator", '{"expression": "__import__(\'os\').system(\'dir\')"}'),
    ("sqrt", '{"x": 144}'),
    ("sqrt", '{"x": -1}'),
    ("search_knowledge", '{"query": "两数之和"}'),
    ("search_knowledge", '{"query": "红烧肉"}'),
    ("nope", "{}"),
]


def main() -> None:
    print("=== 检索层 ===")
    for q in ["两数之和怎么解", "梯度下降学习率", "红烧肉"]:
        hits = rag.retrieve(q, top_k=2)
        print(f"  {q!r} -> {[(h['tag'], h['score']) for h in hits] or '无命中'}")

    print("\n=== 工具清单 ===")
    print("  " + ", ".join(t["function"]["name"] for t in tools.get_tool_schemas()))

    print("\n=== 工具执行 ===")
    for name, args in CASES:
        print(f"  {name}({args[:40]}) -> {tools.run_tool(name, args)}")

    print("\n[OK] 全部通过（无异常抛出即为通过）")


if __name__ == "__main__":
    main()
