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
from typing import Callable, Dict, List, Optional

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
             top_k: int, name: str) -> Dict:
    """跑评测，**只收集结果不打印**。

    为什么不在这里打印（重要的一次重构）：
      原来每个检索器各自打印自己的报告，输出长这样：
          「关键词检索：未命中的 9 题」+ 40 行详情
          「向量检索：未命中的 6 题」+ 一堆详情
      两个列表挨在一起又不标归属，**极容易看串**
      （实测：用户就把"关键词失败 9 题"看成了"向量失败 9 题"）。

      改成"先收集、后统一汇报"之后，就能画出「每题 × 每种模式」的对照表，
      一眼看清哪题谁命中谁没命中 —— 从根上消除误读。
    """
    hits = []
    for item in questions:
        results = retriever(item["question"], chunks, top_k)
        found = any(item["keyword"] in r["text"] for r in results)
        hits.append(
            {
                "question": item["question"],
                "keyword": item["keyword"],
                "kind": item.get("kind", "未分类"),
                "hit": found,
                "top_score": results[0]["score"] if results else 0.0,
                "returned": [r["heading"] for r in results],
            }
        )

    by_kind: Dict[str, List[Dict]] = {}
    for h, q in zip(hits, questions):
        by_kind.setdefault(q["kind"], []).append(h)

    n = len(hits)
    n_hit = sum(1 for h in hits if h["hit"])
    return {
        "name": name,
        "top_k": top_k,
        "n": n,
        "hit": n_hit,
        "rate": n_hit / n if n else 0.0,
        "by_kind": {k: (sum(1 for g in v if g["hit"]), len(v)) for k, v in by_kind.items()},
        "detail": hits,
    }


def report(results: List[Dict], questions: List[Dict], top_k: int,
           retriever_names: List[str], show_miss: bool = False,
           retrievers: Optional[List] = None, chunks: Optional[List[Dict]] = None) -> None:
    """统一出一份报告：汇总 + 分组 + **每题 × 每种模式的对照表**。"""
    W = 78
    print("=" * W)
    print("  检索评测报告")
    print("=" * W)
    print(f"\n  测试集 {len(questions)} 题 / 评估口径：命中率@{top_k}"
          f"（正确答案出现在返回的前 {top_k} 条里算命中）\n")

    # ---------------- 汇总 ----------------
    print(f"  {'检索器':<30} {'总命中率':>10}  {'原词':>8}  {'改写':>8}")
    print("  " + "-" * 62)
    for r in results:
        g = r["by_kind"]
        o = g.get("原词", (0, 1))
        w = g.get("改写", (0, 1))
        print(f"  {r['name']:<30} {r['hit']:>2}/{r['n']} = {r['rate']*100:>3.0f}%  "
              f"{o[0]:>2}/{o[1]} = {o[0]/o[1]*100:>3.0f}%  {w[0]:>2}/{w[1]} = {w[0]/w[1]*100:>3.0f}%")

    if len(results) > 1:
        delta = (results[-1]["rate"] - results[0]["rate"]) * 100
        print(f"\n  变化：{delta:+.1f} 个百分点")

    # ---------------- ⭐ 每题 × 每种模式 对照表 ----------------
    if len(results) > 1:
        print(f"\n{'=' * W}")
        print("  ⭐ 逐题对照（✓ = 命中，✗ = 未命中）")
        print("=" * W)
        print("  ⚠️ 下面每一行都是【同一道题】在各模式下的表现 —— 不要跨行看串 ⚠️")
        print()

        # 表头（让"哪一种模式"一眼可辨）
        head = f"  {'题目':<40}"
        for i in range(len(results)):
            head += f" {chr(65+i):>3}"
        print(head)
        print("  " + "-" * (40 + 4 * len(results)))
        for i, r in enumerate(results):
            print(f"    {chr(65+i)} = {r['name'][:60]}")
        print()

        # 逐题
        for qi, q in enumerate(questions):
            marks = []
            for r in results:
                d = r["detail"][qi]
                marks.append("  ✓" if d["hit"] else "  ✗")
            line = f"  {q['question'][:38]:<40}" + "".join(f"{m:>4}" for m in marks)
            print(line)

        # 每题的差异小结
        print(f"\n  ── 各模式之间的差异（只列出结果不一致的题）──")
        diff_found = False
        for qi, q in enumerate(questions):
            flags = [r["detail"][qi]["hit"] for r in results]
            if len(set(flags)) > 1:      # 结果不一致
                diff_found = True
                pattern = " ".join(
                    f"{chr(65+i)}:{'✓' if f else '✗'}" for i, f in enumerate(flags)
                )
                win = [chr(65 + i) for i, f in enumerate(flags) if f]
                lose = [chr(65 + i) for i, f in enumerate(flags) if not f]
                print(f"\n  [{q['kind']}] {q['question']}")
                print(f"      期望关键词：{q['keyword']!r}")
                print(f"      {pattern}   → 命中: {','.join(win) or '无'}   "
                      f"未命中: {','.join(lose) or '无'}")
                if show_miss and retrievers and chunks:
                    for i, f in enumerate(flags):
                        if not f:
                            print(f"      {chr(65+i)} 返回的前 3 条：")
                            for r in retrievers[i](q["question"], chunks, top_k):
                                print(f"         [score={r['score']}] {r['heading'][:48]}")
        if not diff_found:
            print("\n    （各模式结果完全一致）")

    # ---------------- 全部失败的题（谁也搞不定）----------------
    if len(results) > 1:
        all_fail = [qi for qi in range(len(questions))
                    if not any(r["detail"][qi]["hit"] for r in results)]
        if all_fail:
            print(f"\n{'=' * W}")
            print(f"  ⚠️ 所有模式都失败的题（{len(all_fail)} 题）—— 这是真正的能力边界")
            print("=" * W)
            for qi in all_fail:
                q = questions[qi]
                print(f"\n  [{q['kind']}] {q['question']}")
                print(f"      期望关键词：{q['keyword']!r}")
                if show_miss and retrievers and chunks:
                    for i, _ in enumerate(results):
                        res = retrievers[i](q["question"], chunks, top_k)
                        print(f"      {chr(65+i)} 返回：{[r['heading'][:30] for r in res]}")
    print("\n" + "=" * W)


def main() -> None:
    ap = argparse.ArgumentParser(description="检索评测")
    ap.add_argument("--top-k", type=int, default=3, help="取前几条算命中（默认 3）")
    ap.add_argument("--show-miss", action="store_true", help="打印失败案例的详细内容")
    ap.add_argument(
        "--retriever",
        choices=["keyword", "vector", "both"],
        default="both",
        help="用哪个检索器评测（默认两个都跑，方便对比）",
    )
    args = ap.parse_args()

    chunks = load_corpus()
    questions = load_questions()

    if not chunks:
        print(f"语料为空：请把 markdown 文件放进 {DOCS_DIR}")
        return
    if not questions:
        print(f"测试集为空：请检查 {QUESTIONS_FILE}")
        return

    results = []
    retrievers = []

    if args.retriever in ("keyword", "both"):
        retrievers.append(retrieve_keyword)
        results.append(
            evaluate(retrieve_keyword, chunks, questions, args.top_k, "关键词检索（基线）")
        )

    if args.retriever in ("vector", "both"):
        # 延迟导入：没装向量库相关依赖时不影响纯关键词评测
        from vector_store import VectorStore, make_vector_retriever

        store = VectorStore()
        store.add(chunks)
        vec = make_vector_retriever(store)
        retrievers.append(vec)
        results.append(
            evaluate(vec, chunks, questions, args.top_k, "向量检索（词袋+TF-IDF+余弦）")
        )

    report(results, questions, args.top_k, [r["name"] for r in results],
           show_miss=args.show_miss, retrievers=retrievers, chunks=chunks)


if __name__ == "__main__":
    main()
