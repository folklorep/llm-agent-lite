"""对比两种关键词检索：原始词频打分 vs 长度归一化打分。

============================================================================
为什么做这个对比（这是"诊断"而不是"堆功能"）
============================================================================
跑基线时发现一个可疑现象：语料里最大的那块（1156 字）在 20 题里
有 6 次排第一、13 次进前 3 —— 它只占 35 块的 1/35。

怀疑是**长块偏向**：块越长，累积的词越多，越容易撞上查询词。

本脚本验证这个怀疑，并测试一个标准修法：**按块长度归一化**。

如果归一化后命中率上升，说明"长块偏向"确实是问题 ——
而且这个结论不是猜的，是量出来的。

跑法：
    python compare_retrievers.py
"""
import re
from typing import Dict, List

from eval_retrieval import load_corpus, load_questions, tokenize

# ---------------------------------------------------------------- 检索器 A
def retrieve_raw_tf(query: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
    """基线：命中词数 ÷ 查询词数。

    问题：分母只跟"问题"有关，**跟块长度无关**。
    所以长块天然占便宜 —— 它词多，撞上的机会大。
    """
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return []
    out = []
    for c in chunks:
        d_tokens = set(tokenize(c["text"]))
        hit = len(q_tokens & d_tokens)
        if hit:
            out.append({**c, "score": round(hit / len(q_tokens), 4)})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top_k]


# ---------------------------------------------------------------- 检索器 B
def retrieve_length_norm(query: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
    """修法一：按块长度归一化（类似余弦相似度的思路）。

    得分 = 命中词数 ÷ sqrt(查询词数 × 块词数)

    分母带上了块长度 → 长块不再自动占便宜。
    这跟向量检索里的余弦相似度是同一个思路：
    比较的是"方向"（内容的相似程度），不是"体量"（谁字多）。
    """
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return []
    out = []
    for c in chunks:
        d_tokens = set(tokenize(c["text"]))
        if not d_tokens:
            continue
        hit = len(q_tokens & d_tokens)
        if hit:
            score = hit / ((len(q_tokens) ** 0.5) * (len(d_tokens) ** 0.5))
            out.append({**c, "score": round(score, 4)})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top_k]


# ---------------------------------------------------------------- 检索器 C
def retrieve_idf(query: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
    """修法二：IDF 加权 + 长度归一化。

    思路：越罕见的词越重要。"的""是"这种词到处都是，不该给分；
    而"之字形""失忆"这种词只在少数块里出现，命中就该给高分。
    这是 TF-IDF / BM25 的核心思想。
    """
    import math

    q_tokens = set(tokenize(query))
    if not q_tokens:
        return []

    # 先算每个词在多少块里出现过（文档频率）
    n_docs = len(chunks)
    df: Dict[str, int] = {}
    chunk_tokens = []
    for c in chunks:
        toks = set(tokenize(c["text"]))
        chunk_tokens.append(toks)
        for t in toks:
            df[t] = df.get(t, 0) + 1

    def idf(t: str) -> float:
        # +1 平滑，避免除零；log 让极端罕见词不至于权重爆炸
        return math.log((n_docs + 1) / (df.get(t, 0) + 1)) + 1

    q_weight = {t: idf(t) for t in q_tokens}
    q_norm = sum(w * w for w in q_weight.values()) ** 0.5
    if q_norm == 0:
        return []

    out = []
    for c, d_tokens in zip(chunks, chunk_tokens):
        shared = q_tokens & d_tokens
        if not shared:
            continue
        dot = sum(q_weight[t] * idf(t) for t in shared)
        d_norm = sum(idf(t) ** 2 for t in d_tokens) ** 0.5
        if d_norm == 0:
            continue
        score = dot / (q_norm * d_norm)
        out.append({**c, "score": round(score, 4)})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top_k]


# ---------------------------------------------------------------- 对比
def hit_rate(retriever, chunks, questions, top_k: int):
    """返回 (总命中率, 各分组命中率, 每题的命中情况)"""
    kinds: Dict[str, List[int]] = {}
    detail = []
    for q in questions:
        res = retriever(q["question"], chunks, top_k)
        ok = any(q["keyword"] in r["text"] for r in res)
        detail.append((q["question"], q["kind"], ok))
        kinds.setdefault(q["kind"], []).append(1 if ok else 0)

    total = sum(sum(v) for v in kinds.values())
    n = sum(len(v) for v in kinds.values())
    grouped = {k: (sum(v), len(v)) for k, v in kinds.items()}
    return (total, n, grouped, detail)


def main() -> None:
    chunks = load_corpus()
    questions = load_questions()
    TOP_K = 3

    retrievers = [
        ("A. 词频（基线）", retrieve_raw_tf),
        ("B. 长度归一化", retrieve_length_norm),
        ("C. IDF + 归一化", retrieve_idf),
    ]

    print("=" * 70)
    print("  三种关键词检索策略对比")
    print("=" * 70)
    print(f"\n  语料 {len(chunks)} 块 / 测试集 {len(questions)} 题 / 命中率@{TOP_K}\n")

    results = {}
    for name, fn in retrievers:
        total, n, grouped, detail = hit_rate(fn, chunks, questions, TOP_K)
        results[name] = (total, n, grouped, detail)

    # ---- 汇总表 ----
    print(f"  {'策略':<20} {'总命中率':>10}   {'原词':>8}   {'改写':>8}")
    print("  " + "-" * 54)
    for name, (total, n, grouped, _) in results.items():
        orig = grouped.get("原词", (0, 1))
        rewr = grouped.get("改写", (0, 1))
        print(f"  {name:<20} {total:>3}/{n} = {total / n * 100:>3.0f}%   "
              f"{orig[0]:>2}/{orig[1]} = {orig[0] / orig[1] * 100:>3.0f}%   "
              f"{rewr[0]:>2}/{rewr[1]} = {rewr[0] / rewr[1] * 100:>3.0f}%")

    # ---- 逐题变化 ----
    base = list(results.values())[0][3]
    best = list(results.values())[-1][3]
    print(f"\n  ── 从基线到最佳策略，哪些题改善了 ──")
    improved = [(b[0], b[1]) for b, c in zip(base, best) if not b[2] and c[2]]
    worsened = [(b[0], b[1]) for b, c in zip(base, best) if b[2] and not c[2]]
    for q, kind in improved:
        print(f"    ✅ 变好了 [{kind}] {q}")
    for q, kind in worsened:
        print(f"    ❌ 变差了 [{kind}] {q}")
    if not improved and not worsened:
        print("    （无变化）")

    # ---- 长块偏向在三种策略下的表现 ----
    print(f"\n  ── 长块偏向检查（最大块的排名表现）──")
    biggest = max(chunks, key=lambda c: c["chars"])
    print(f"    最大块：{biggest['chars']} 字 —— {biggest['heading'][:36]}")
    for name, fn in retrievers:
        first = sum(1 for q in questions
                    if fn(q["question"], chunks, TOP_K) and
                    fn(q["question"], chunks, TOP_K)[0]["index"] == biggest["index"])
        print(f"    {name:<20} 当第一名 {first:>2} 次")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
