"""检索评测脚本：量出"检索准不准"。

============================================================================
为什么这一步是整个 RAG 深化里最有价值的
============================================================================
大部分学生做完 RAG 就停了，说不出"它比原来好多少"。
面试官问"你怎么知道检索得准"，只能答"感觉还行"。

有了评测，你能给出的是**数字**：

    "同样的 20 个问题，关键词检索命中率 55%，向量检索 85%"
    "失败的 3 题都是同义改写的问法"
    "所以向量检索解决的是语义匹配，代价是多一次 embedding 调用"

============================================================================
设计说明（面试可以讲）
============================================================================
1. **先定指标，再改实现**
   先有评测集和基线分数，才谈得上"改进"。否则改了也不知道有没有变好。

2. **答案用"关键词"标注，不用"块编号"**
   块编号会随切块策略变化，关键词不会。这样测试集是稳定的。

3. **检索函数与评测分离**
   `retrieve_keyword()` 只负责"给个 query、返回排好序的块"。
   评测器不关心它是关键词还是向量 —— 换个实现就能对比，这是能测的前提。

跑法：
    python eval_retrieval.py                    # 跑基线（关键词检索）
    python eval_retrieval.py --top-k 5          # 改 k 值
    python eval_retrieval.py --show-miss        # 打印失败案例的详细内容
"""
import argparse
import re
from pathlib import Path
from typing import Callable, Dict, List

from chunker import chunk_file

DOCS_DIR = Path(__file__).parent / "data" / "docs"
QUESTIONS_FILE = Path(__file__).parent / "data" / "eval_questions.txt"


# ============================================================ 语料与测试集
def load_corpus() -> List[Dict]:
    """把 data/docs/ 下所有 markdown 切块，作为检索语料。"""
    chunks: List[Dict] = []
    if not DOCS_DIR.exists():
        return chunks
    for path in sorted(DOCS_DIR.glob("*.md")):
        for c in chunk_file(path):
            c["doc"] = path.name
            chunks.append(c)
    return chunks


def load_questions() -> List[Dict]:
    """读测试集。格式：问题 | 期望命中的关键词 | 类型（可选）

    类型用来分组统计 —— 这是本评测最关键的一点：
      【原词】问题里的词和语料一样，关键词检索也能命中
      【改写】同义改写，关键词检索容易失败
    分组之后才能看出"两个检索器差在哪里"。
    """
    items = []
    if not QUESTIONS_FILE.exists():
        return items
    for line in QUESTIONS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            continue
        items.append(
            {
                "question": parts[0],
                "keyword": parts[1],
                "kind": parts[2] if len(parts) > 2 else "未分类",
            }
        )
    return items


# ============================================================ 分词
_STOP = set("的了是在和与或及等这那你我他它一个什么怎么为什么哪些".replace("", " "))


def tokenize(text: str) -> List[str]:
    """极简分词：英文按单词，中文按字 + 二元组。

    为什么要二元组：中文单字区分度太低（"栈"和"字典"里的"典"都可能命中），
    二元组能保留一点词序信息，在不引入分词库的前提下效果好不少。
    """
    lowered = text.lower()
    words = re.findall(r"[a-z0-9_]+", lowered)
    words = [w for w in words if w not in _STOP and len(w) > 1]

    han = re.findall(r"[\u4e00-\u9fff]", text)
    bigrams = [han[i] + han[i + 1] for i in range(len(han) - 1)]
    return words + han + bigrams


# ============================================================ 检索器
# 所有检索器都是这个签名：给 query，返回排好序的块列表。
# 这样换实现（关键词 → 向量）时，评测器一行都不用改。
Retriever = Callable[[str, List[Dict], int], List[Dict]]


def retrieve_keyword(query: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
    """基线检索器：关键词重合度打分。

    打分方式：查询词有多少比例出现在这一块里。
    属于"字面匹配" —— 用户换个说法就匹配不到了，这正是要暴露的问题。
    """
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return []

    scored = []
    for c in chunks:
        doc_tokens = set(tokenize(c["text"]))
        if not doc_tokens:
            continue
        hit = len(q_tokens & doc_tokens)
        if hit == 0:
            continue
        # 归一化：命中数 ÷ 查询词数（衡量"问题里的词出现了多少"）
        score = hit / len(q_tokens)
        scored.append({**c, "score": round(score, 4)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ============================================================ 评测
def evaluate(retriever: Retriever, chunks: List[Dict], questions: List[Dict],
             top_k: int, name: str, show_miss: bool = False) -> Dict:
    """跑评测，返回统计结果。"""
    hits = []
    for item in questions:
        results = retriever(item["question"], chunks, top_k)
        found = any(item["keyword"] in r["text"] for r in results)
        hits.append(
            {
                "question": item["question"],
                "keyword": item["keyword"],
                "hit": found,
                "top_score": results[0]["score"] if results else 0.0,
                "returned": [r["heading"] for r in results],
            }
        )

    n = len(hits)
    n_hit = sum(1 for h in hits if h["hit"])
    rate = n_hit / n if n else 0.0

    print("=" * 68)
    print(f"  检索评测：{name}")
    print("=" * 68)
    print(f"\n  语料：{len(chunks)} 块")
    print(f"  测试集：{n} 题")
    print(f"  评估口径：命中率@{top_k}（正确答案在前 {top_k} 条里算命中）\n")
    print(f"  ┌─────────────────────────────────────┐")
    print(f"  │  总命中率@{top_k} = {n_hit}/{n} = {rate * 100:5.1f}%        │")
    print(f"  └─────────────────────────────────────┘")

    # ---- 分组统计：原词 vs 改写（这是看出差距的关键）----
    by_kind: Dict[str, List[Dict]] = {}
    for h, q in zip(hits, questions):
        by_kind.setdefault(q["kind"], []).append(h)

    if len(by_kind) > 1:
        print(f"\n  ── 分组命中率 ──")
        for kind, group in by_kind.items():
            gh = sum(1 for g in group if g["hit"])
            bar_len = round(20 * gh / len(group))
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"    {kind:<8} {bar}  {gh}/{len(group)}  ({gh / len(group) * 100:.0f}%)")

    miss = [h for h in hits if not h["hit"]]
    if miss:
        print(f"\n  ── 未命中的 {len(miss)} 题 ──")
        for h in miss:
            print(f"\n  ❌ {h['question']}")
            print(f"     期望关键词：{h['keyword']}")
            print(f"     实际返回：{h['returned'] or '（空）'}")
            if show_miss:
                results = retriever(h["question"], chunks, top_k)
                for r in results:
                    preview = r["text"][:80].replace("\n", " ")
                    print(f"       [score={r['score']}] {preview}...")
    else:
        print(f"\n  🎉 全部命中")

    print("\n" + "=" * 68)
    return {
        "name": name,
        "top_k": top_k,
        "n": n,
        "hit": n_hit,
        "rate": rate,
        "by_kind": {k: (sum(1 for g in v if g["hit"]), len(v)) for k, v in by_kind.items()},
        "detail": hits,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="检索评测")
    ap.add_argument("--top-k", type=int, default=3, help="取前几条算命中（默认 3）")
    ap.add_argument("--show-miss", action="store_true", help="打印失败案例的详细内容")
    args = ap.parse_args()

    chunks = load_corpus()
    questions = load_questions()

    if not chunks:
        print(f"语料为空：请把 markdown 文件放进 {DOCS_DIR}")
        return
    if not questions:
        print(f"测试集为空：请检查 {QUESTIONS_FILE}")
        return

    result = evaluate(retrieve_keyword, chunks, questions, args.top_k,
                      "关键词检索（基线）", args.show_miss)

    print(f"\n  💡 这就是基线。实现向量检索后重跑本脚本，对比命中率变化。")
    print(f"     基线命中率 = {result['rate'] * 100:.1f}%\n")


if __name__ == "__main__":
    main()
