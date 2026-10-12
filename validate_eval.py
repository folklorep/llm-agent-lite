"""测试集质检：在跑评测之前先检查标注是否可靠。

============================================================================
为什么需要这个（都是踩过的坑）
============================================================================
评测的数字有没有意义，完全取决于标注对不对。我在扩测试集时踩了三类错：

  ① **关键词根本不在语料里**
     v2 初稿里「轮次上限是 5」「备忘录」都不在笔记里 ——
     这两题的"未命中"是标注的错，不是检索器的错。
     如果不查，就会得出"检索效果差"的错误结论。

  ② **关键词太泛，哪个块都算命中**
     「0」出现在 27 个块里、「return」出现在 9 个块里。
     这类关键词等于没标注 —— 检索器返回任何含它的块都算"对"，
     指标会虚高，而且测不出真实差异。

  ③ **关键词在多个块里出现（答案位置不唯一）**
     出现 2 次通常可接受（同一内容被切到相邻块），
     但 3 次以上就要怀疑"这题到底在问哪一块"。

============================================================================
三道检查
============================================================================
    检查 1：关键词是否逐字出现在语料里          → 不在就报错
    检查 2：关键词出现在几个块里              → >2 个块就告警
    检查 3：含关键词的块，在关键词排序里排第几   → 排名太靠后说明召回很难

**检查 3 最有用**：它直接告诉你"这题能不能被召回"，
而不只是"标注对不对"。上一轮正是靠它发现了"答案块排第 20 名"的召回失败。

跑法：
    python validate_eval.py                # 跑三道检查
    python validate_eval.py --verbose      # 打印每题详情
"""
import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List

from eval_retrieval import (QUESTIONS_FILE, load_corpus, load_questions,
                            retrieve_keyword)


def check_keyword_in_corpus(notes_text: str, questions: List[Dict]) -> List[Dict]:
    """检查 1：关键词是否逐字出现在原始语料里。"""
    bad = []
    for q in questions:
        if notes_text.count(q["keyword"]) == 0:
            bad.append(q)
    return bad


def check_keyword_span(chunks: List[Dict], questions: List[Dict],
                       max_ok: int = 2) -> List[tuple]:
    """检查 2：关键词出现在几个块里。"""
    bad = []
    for q in questions:
        n = sum(1 for c in chunks if q["keyword"] in c["text"])
        if n == 0 or n > max_ok:
            bad.append((q, n))
    return bad


def check_recall_rank(chunks: List[Dict], questions: List[Dict]) -> List[tuple]:
    """检查 3：含答案的块在关键词排序里排第几。

    排名越靠后，越依赖"加大召回数"才能被看到。
    """
    rows = []
    for q in questions:
        ranked = retrieve_keyword(q["question"], chunks, 999)
        rank = None
        for i, r in enumerate(ranked, 1):
            if q["keyword"] in r["text"]:
                rank = i
                break
        rows.append((q, rank))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="测试集质检")
    ap.add_argument("--verbose", action="store_true", help="打印每题详情")
    args = ap.parse_args()

    chunks = load_corpus()
    questions = load_questions()
    notes_files = sorted(Path("data/docs").glob("*.md"))
    notes_text = "\n".join(p.read_text(encoding="utf-8-sig") for p in notes_files)

    print("=" * 72)
    print("  测试集质检")
    print("=" * 72)
    print(f"\n  测试集：{QUESTIONS_FILE.name}")
    print(f"  题数：{len(questions)}")
    print(f"  类型分布：{dict(Counter(q['kind'] for q in questions))}")
    print(f"  语料：{len(notes_files)} 个文件 → {len(chunks)} 块")

    problems = 0

    # ---- 检查 1 ----
    print(f"\n  ── 检查 1：关键词是否在语料里 ──")
    bad1 = check_keyword_in_corpus(notes_text, questions)
    if bad1:
        problems += len(bad1)
        print(f"  ❌ {len(bad1)} 题的关键词在语料里找不到（这些题的'未命中'是标注错误）：")
        for q in bad1:
            print(f"       {q['question'][:38]:<40} 关键词={q['keyword']!r}")
    else:
        print(f"  ✅ 全部 {len(questions)} 题的关键词都能在语料里找到")

    # ---- 检查 2 ----
    print(f"\n  ── 检查 2：关键词出现在几个块里 ──")
    bad2 = check_keyword_span(chunks, questions)
    if bad2:
        print(f"  ⚠️ {len(bad2)} 题的关键词分布有问题（>2 块则答案位置不唯一）：")
        for q, n in bad2:
            tag = "不在任何块" if n == 0 else f"出现在 {n} 块"
            print(f"       {q['question'][:38]:<40} {q['keyword']!r} {tag}")
        problems += sum(1 for _, n in bad2 if n > 2)
    else:
        print(f"  ✅ 全部关键词的出现范围都在可接受区间（1-2 块）")

    # ---- 检查 3 ----
    print(f"\n  ── 检查 3：含答案的块在关键词排序里排第几 ──")
    rows = check_recall_rank(chunks, questions)
    ranked_ok = [(q, r) for q, r in rows if r is not None]
    never = [(q, r) for q, r in rows if r is None]

    if never:
        print(f"  ❌ {len(never)} 题的关键词块在关键词检索里从未出现：")
        for q, _ in never:
            print(f"       {q['question'][:44]}")
        problems += len(never)

    if ranked_ok:
        ranks = [r for _, r in ranked_ok]
        print(f"  排名分布：最小 {min(ranks)} / 中位 {sorted(ranks)[len(ranks) // 2]} / 最大 {max(ranks)}")
        bins = {"1-3（易召回）": 0, "4-10（正常）": 0, "11-20（需加大召回）": 0, ">20（很难召回）": 0}
        for r in ranks:
            if r <= 3:
                bins["1-3（易召回）"] += 1
            elif r <= 10:
                bins["4-10（正常）"] += 1
            elif r <= 20:
                bins["11-20（需加大召回）"] += 1
            else:
                bins[">20（很难召回）"] += 1
        for k, v in bins.items():
            bar = "█" * v
            print(f"    {k:<20} {bar} {v}")

        hard = [(q, r) for q, r in ranked_ok if r > 10]
        if hard:
            print(f"\n  ⚠️ 排名 >10 的题（靠加大召回才能救，也最容易成为失败案例）：")
            for q, r in sorted(hard, key=lambda x: -x[1]):
                print(f"       第 {r:>2} 名  [{q['kind']}] {q['question'][:40]}")

    if args.verbose:
        print(f"\n  ── 逐题详情 ──")
        for q, r in rows:
            rk = f"第 {r} 名" if r else "未召回"
            print(f"    [{q['kind']}] {q['question'][:40]:<42} {rk}")

    # ---- 结论 ----
    print("\n" + "=" * 72)
    if problems == 0:
        print("  ✅ 质检通过，可以跑评测")
    else:
        print(f"  ⚠️ 发现 {problems} 处问题，建议先修标注再跑评测")
        print("     （否则评测数字会低估检索效果，或产生歧义）")
    print("=" * 72)


if __name__ == "__main__":
    main()
