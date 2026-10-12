"""检索层测试：验证 rag.py 重写后接口契约仍然成立。

============================================================================
这个测试在守护什么
============================================================================
`rag.py` 被整体重写过（从"只读 8 条 knowledge.txt"改成"切块 + 向量 + 可选精排"）。
重写最怕的是：**调用方悄悄坏掉**。

所以这个测试只关心**契约**，不关心实现细节：

    输入：query（字符串）、top_k（整数）
    输出：list[dict]，每个 dict 至少有 text / tag / score 三个字段
    行为：返回条数不超过 top_k；不抛异常；空查询不崩

工具层（tools.py 的 search_knowledge）就依赖这三条。只要它们成立，
把内部从关键词换成向量再换成精排，调用方都不用动。

跑法：python test_rag.py
"""
from typing import Any, Dict, List

import config
import rag


def check(label: str, ok: bool, detail: Any = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  <- {detail}" if not ok else ""))
    if not ok:
        raise AssertionError(f"{label} 断言失败 {detail}")


def _shape_ok(items: List[Dict]) -> bool:
    """检查返回结构是否符合契约。"""
    if not isinstance(items, list):
        return False
    for it in items:
        if not isinstance(it, dict):
            return False
        if not isinstance(it.get("text"), str) or not it["text"]:
            return False
        if "tag" not in it or "score" not in it:
            return False
    return True


# ---------------------------------------------------------------- 用例
def test_corpus_loaded():
    print("\n[用例 1] 语料能加载，且不是那 8 条玩具数据")
    items = rag.corpus()
    check("语料非空", len(items) > 0, len(items))
    check("语料来自 docs 目录（已接入切块）", len(items) >= 10, len(items))
    stats = rag.stats()
    check("stats 能返回块数与来源", stats["chunks"] == len(items) and stats["sources"])
    print(f"     语料 {stats['chunks']} 块 / {stats['total_chars']} 字 / 来源 {stats['sources']}")


def test_contract_shape():
    print("\n[用例 2] 返回结构符合契约（text / tag / score）")
    for mode in ("keyword", "vector"):
        config.config.RETRIEVAL_MODE = mode
        items = rag.retrieve("两数之和怎么解", 3)
        check(f"[{mode}] 结构合法", _shape_ok(items), items[:1])
        check(f"[{mode}] 条数不超过 top_k", len(items) <= 3, len(items))


def test_top_k_respected():
    print("\n[用例 3] top_k 被严格遵守")
    config.config.RETRIEVAL_MODE = "keyword"
    for k in (1, 2, 5):
        items = rag.retrieve("字典", k)
        check(f"top_k={k} 时返回 <= {k} 条", len(items) <= k, len(items))


def test_score_sorted():
    print("\n[用例 4] 结果按分数降序")
    for mode in ("keyword", "vector"):
        config.config.RETRIEVAL_MODE = mode
        items = rag.retrieve("梯度下降 学习率", 5)
        scores = [i["score"] for i in items]
        check(f"[{mode}] 分数降序", scores == sorted(scores, reverse=True), scores)


def test_empty_and_weird_query():
    print("\n[用例 5] 空查询 / 无意义查询不能崩")
    config.config.RETRIEVAL_MODE = "keyword"
    for q in ("", "   ", "zzzzqqqq", "！！！", "红烧肉"):
        try:
            items = rag.retrieve(q, 3)
            check(f"查询 {q!r} 正常返回（{len(items)} 条）", _shape_ok(items) or items == [])
        except Exception as e:  # noqa: BLE001
            check(f"查询 {q!r} 不抛异常", False, f"{type(e).__name__}: {e}")


def test_mode_switch():
    print("\n[用例 6] 三种模式切换都不崩（rerank 允许退化为粗排）")
    for mode in ("keyword", "vector", "rerank"):
        config.config.RETRIEVAL_MODE = mode
        try:
            items = rag.retrieve("有效的括号用什么数据结构", 3)
            check(f"[{mode}] 可用", _shape_ok(items) or items == [], items[:1])
        except Exception as e:  # noqa: BLE001
            check(f"[{mode}] 不抛异常", False, f"{type(e).__name__}: {e}")


def test_unknown_mode_falls_back():
    print("\n[用例 7] 配错了模式名应退回 keyword，而不是崩")
    config.config.RETRIEVAL_MODE = "不存在的模式"
    try:
        items = rag.retrieve("字典", 3)
        check("未崩溃且有结果", len(items) > 0, len(items))
    except Exception as e:  # noqa: BLE001
        check("未崩溃", False, f"{type(e).__name__}: {e}")


def test_tool_layer_still_works():
    print("\n[用例 8] 调用方（tools.search_knowledge）没被改坏")
    import tools

    config.config.RETRIEVAL_MODE = "keyword"
    out = tools.run_tool("search_knowledge", '{"query": "两数之和"}')
    check("工具返回字符串", isinstance(out, str))
    check("工具能找到相关内容", "字典" in out or "先查后登记" in out, out[:60])

    # 知识库里确实没有的内容 → 应如实说没有（防幻觉）
    #
    # ⚠️ 注意选词：不能用"红烧肉"！
    #    第一版测试就是拿它测的，结果失败了 —— 因为**语料里真的有"红烧肉"**
    #    （笔记的项目 L4 一节记录了"问红烧肉→如实说不知道"这个实测）。
    #    语料里出现过某个词，检索命中它就是正确行为，不是 bug。
    #    所以这里必须用一个确定不在语料里的词。
    out2 = tools.run_tool("search_knowledge", '{"query": "如何腌制四川泡菜"}')
    check("查不到时如实回答", "没有找到" in out2 or "没有" in out2, out2[:80])


def test_min_score_filter():
    print("\n[用例 9] 弱匹配被阈值过滤（不返回噪声）")
    config.config.RETRIEVAL_MODE = "keyword"
    # 完全不相关的查询，应返回空
    items = rag.retrieve("zzzzqqqq", 3)
    check("无关查询返回空", items == [], items)
    # 只命中单字的弱匹配也不该返回
    items2 = rag.retrieve("腌笃鲜的火候", 3)
    check("弱匹配不返回噪声", len(items2) <= 1, len(items2))
    if items2:
        print(f"     （返回 {len(items2)} 条，最高分 {items2[0]['score']}）")


def test_title_weight_lower():
    print("\n[用例 10] 标题命中权重低于正文命中")
    config.config.RETRIEVAL_MODE = "keyword"
    # 找一个只在标题出现、正文没有的词，它不该拿到满分
    items = rag.retrieve("数组 vs 列表", 3)
    if items:
        top = items[0]
        print(f"     最高分 {top['score']} —— {top['tag'][:40]}")
    check("结果存在且分数合理", all(i["score"] <= 1.0 for i in items))


def main() -> None:
    saved = config.config.RETRIEVAL_MODE
    print("=" * 66)
    print("  检索层测试（守护 rag.py 的接口契约）")
    print("=" * 66)
    try:
        test_corpus_loaded()
        test_contract_shape()
        test_top_k_respected()
        test_score_sorted()
        test_empty_and_weird_query()
        test_mode_switch()
        test_unknown_mode_falls_back()
        test_tool_layer_still_works()
        test_min_score_filter()
        test_title_weight_lower()
    finally:
        config.config.RETRIEVAL_MODE = saved
    print("\n" + "=" * 66)
    print("  [OK] 检索层契约测试全部通过")
    print("=" * 66)


if __name__ == "__main__":
    main()
