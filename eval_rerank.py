"""LLM 重排序（rerank）：用大模型给候选块精细排序。

============================================================================
为什么需要这一步（接上一轮的实验结论）
============================================================================
上一轮实测：

    关键词检索（三种打分变体）   85%
    向量检索（词袋+TF-IDF+余弦） 85%

**两者一模一样。** 原因是它们本质同源 —— 都在"数词重不重合"。
剩下失败的题属于**词汇完全不重合**：
    用户问"不同单位的数字"      文档写"特征量级差太大"
    用户问"RAG 的三个环节"      文档写"RAG = Retrieval 检索 → ..."

**这类问题只有"懂语义"的东西才能解决。**
而能在本地跑出语义匹配的 embedding 模型，本项目暂时用不了
（DeepSeek API 无 embedding 接口；本地跑要下 2GB+ PyTorch）。

============================================================================
所以改用「两段式检索」：粗排 + 精排
============================================================================
这是工业界真实的 RAG 架构，也是本模块做的事：

    问题
      ↓
    ① 粗排（recall）：用关键词/向量捞出前 N 个候选
       特点：快、免费、召回率高但排序糙
      ↓
    ② 精排（rerank）：把候选 + 问题交给 LLM，让它判断哪个最相关
       特点：准、要花钱、但只处理 N 个候选（成本可控）
      ↓
    返回精排结果

**为什么 LLM 能解决词不重合的问题**：它懂语义。
它知道"不同单位的数字"和"特征量级差太大"说的是同一件事。

============================================================================
成本说明（要如实知道）
============================================================================
每次查询多 1 次 API 调用。以本项目为例：
    20 题 × 1 次 = 20 次调用，按 deepseek-flash 的价格约几分钱。
真实生产里这也是主要成本项，所以候选数 N 不能太大（一般 10-20）。

跑法：
    python eval_rerank.py            # 跑评测，对比粗排 vs 粗排+精排
    python eval_rerank.py --no-cache # 不用缓存，重新请求
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from eval_retrieval import load_corpus, load_questions, retrieve_keyword
from llm_client import LLMClient

CACHE_FILE = Path(__file__).parent / "data" / "rerank_cache.json"
RECALL_N = 10      # 粗排召回多少个候选给 LLM 精排

RERANK_PROMPT = """你是一个检索结果排序助手。

用户的问题：
{question}

以下是 {n} 个候选资料块（编号 + 内容）：

{candidates}

请判断：哪几个块最可能包含回答这个问题的信息？

要求：
1. 只输出编号，按相关度从高到低排列，用逗号分隔。例如：3,7,1
2. 最多输出 3 个编号
3. 只输出编号，不要任何解释、不要其他文字
4. 如果没有任何一个块相关，输出：无

你的回答："""


class _Cache:
    """把 LLM 的排序结果按"问题+候选"的哈希缓存到本地。

    为什么要缓存：
      调 LLM 要花钱、要等。调试评测脚本时会反复跑同样的 20 题，
      缓存之后第二次跑几乎瞬间完成，也不重复花钱。
    """

    def __init__(self, path: Path, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled
        self.data: Dict[str, List[int]] = {}
        if enabled and path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.data = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(question: str, candidates: List[Dict]) -> str:
        raw = question + "||" + "|".join(str(c.get("index")) for c in candidates)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get(self, question: str, candidates: List[Dict]) -> Optional[List[int]]:
        if not self.enabled:
            return None
        k = self._key(question, candidates)
        if k in self.data:
            self.hits += 1
            return self.data[k]
        self.misses += 1
        return None

    def put(self, question: str, candidates: List[Dict], order: List[int]) -> None:
        if not self.enabled:
            return
        self.data[self._key(question, candidates)] = order

    def save(self) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1),
                             encoding="utf-8")


def _parse_order(text: str, valid: List[int]) -> List[int]:
    """从 LLM 的回复里解析出编号顺序。要能容忍各种脏输出。"""
    import re

    if "无" in text and not re.search(r"\d", text):
        return []
    nums = [int(x) for x in re.findall(r"\d+", text)]
    # 只保留合法编号，去重，保持顺序
    seen, out = set(), []
    for n in nums:
        if n in valid and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def make_rerank_retriever(client: LLMClient, cache: _Cache, recall_n: int = RECALL_N,
                          hybrid: bool = False, chunks_ref: Optional[List[Dict]] = None):
    """构造"粗排 + 精排"检索器。

    返回的函数签名和评测框架要求的 Retriever 一致（query, chunks, top_k）。

    hybrid=True 时启用**混合召回**：关键词和向量各召回一批，合并去重后交给精排。
    为什么需要：实测发现关键词粗排会把正确答案排到第 20 名，
    只取前 10 就永远看不到它 —— 这叫"召回失败"，精排再强也救不了。
    """
    # 向量库只在启用混合召回时构建一次（构建有成本）
    _store = None
    if hybrid:
        from vector_store import VectorStore

        _store = VectorStore()
        _store.add(chunks_ref if chunks_ref is not None else [])

    def _recall(query: str, chunks: List[Dict], n: int) -> List[Dict]:
        """粗排：捞候选。"""
        kw = retrieve_keyword(query, chunks, n)
        if not hybrid or _store is None:
            return kw

        # 混合召回：两种方法各取 n 个，按 index 去重合并
        vec = _store.search(query, n)
        merged, seen = [], set()
        # 交替插入，让两种来源的候选都能靠前（避免一种把另一种挤掉）
        for pair in zip(kw + [None] * n, vec + [None] * n):
            for c in pair:
                if c and c.get("index") not in seen:
                    seen.add(c["index"])
                    merged.append(c)
        return merged[: n * 2]

    def retrieve(query: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
        # ---- ① 粗排：捞候选（快、免费）----
        candidates = _recall(query, chunks, recall_n)
        if not candidates:
            return []

        # ---- ② 精排：让 LLM 排序（准、要花钱）----
        cached = cache.get(query, candidates)
        if cached is not None:
            order = cached
        else:
            listing = "\n\n".join(
                f"[{i}] {c['text'][:400]}" for i, c in enumerate(candidates)
            )
            prompt = RERANK_PROMPT.format(
                question=query, n=len(candidates), candidates=listing
            )
            try:
                resp = client.chat([{"role": "user", "content": prompt}])
                raw = resp.get("content") or ""
                order = _parse_order(raw, list(range(len(candidates))))
            except Exception as e:  # noqa: BLE001
                # 精排失败不能让整个检索崩掉 —— 退回粗排结果
                print(f"    [rerank 失败，退回粗排] {type(e).__name__}: {e}")
                order = list(range(len(candidates)))
            cache.put(query, candidates, order)
            time.sleep(0.2)     # 轻微限速，避免触发 API 频率限制

        if not order:
            return []

        picked = [candidates[i] for i in order[:top_k] if i < len(candidates)]
        return picked

    return retrieve


def main() -> None:
    ap = argparse.ArgumentParser(description="粗排 + LLM 精排 对比评测")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--recall-n", type=int, default=RECALL_N,
                    help="粗排召回多少个候选给 LLM 精排")
    ap.add_argument("--no-cache", action="store_true", help="禁用缓存，重新请求")
    ap.add_argument("--show-miss", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="跑一组对照组：不同召回数 / 是否混合召回")
    args = ap.parse_args()

    chunks = load_corpus()
    questions = load_questions()
    if not chunks or not questions:
        print("语料或测试集为空")
        return

    from eval_retrieval import evaluate

    print("=" * 74)
    print("  粗排 + LLM 精排 对比评测")
    print("=" * 74)
    print(f"\n  语料 {len(chunks)} 块 / 测试集 {len(questions)} 题 / 命中率@{args.top_k}\n")

    client = LLMClient()
    cache = _Cache(CACHE_FILE, enabled=not args.no_cache)

    # ---- 对照组 1：纯粗排（不调 LLM）----
    base = evaluate(retrieve_keyword, chunks, questions, args.top_k,
                    "① 纯关键词粗排（基线）", args.show_miss)

    # ---- 实验组 ----
    # 组合：召回数 × 是否混合召回
    configs = []
    if args.sweep:
        configs = [
            (10, False, "② 精排（召回10，纯关键词）"),
            (20, False, "③ 精排（召回20，纯关键词）"),
            (10, True, "④ 精排（召回10，混合召回）"),
            (20, True, "⑤ 精排（召回20，混合召回）"),
        ]
    else:
        configs = [(args.recall_n, False, f"② 精排（召回{args.recall_n}）")]

    results = [base]
    t_all = time.time()
    for recall_n, hybrid, name in configs:
        print(f"\n  ── 正在跑：{name} ──")
        fn = make_rerank_retriever(client, cache, recall_n, hybrid=hybrid,
                                   chunks_ref=chunks)
        r = evaluate(fn, chunks, questions, args.top_k, name, args.show_miss)
        results.append(r)
        cache.save()
    elapsed = time.time() - t_all

    # ---- 汇总对比 ----
    print("\n" + "=" * 74)
    print("  汇总对比")
    print("=" * 74)
    print(f"\n  {'方案':<30} {'总命中率':>9}  {'原词':>7}  {'改写':>7}")
    print("  " + "-" * 62)
    for r in results:
        g = r["by_kind"]
        o = g.get("原词", (0, 1))
        w = g.get("改写", (0, 1))
        print(f"  {r['name']:<30} {r['hit']:>2}/{r['n']} = {r['rate'] * 100:>3.0f}%  "
              f"{o[0]:>2}/{o[1]}  {w[0]:>2}/{w[1]}")

    print(f"\n  相比基线：")
    for r in results[1:]:
        d = (r["rate"] - base["rate"]) * 100
        print(f"    {r['name']:<30} {d:+.0f} 个百分点")

    print(f"\n  总耗时 {elapsed:.0f} 秒（缓存命中 {cache.hits} / 未命中 {cache.misses}）")

    # ---- 逐题变化（对比基线与最后一个方案）----
    best = results[-1]
    up = [(b, a) for b, a in zip(base["detail"], best["detail"]) if not b["hit"] and a["hit"]]
    down = [(b, a) for b, a in zip(base["detail"], best["detail"]) if b["hit"] and not a["hit"]]
    if up:
        print(f"\n  ── 救回来的题（{len(up)} 题）──")
        for b, _ in up:
            print(f"    ✅ [{b.get('kind', '?')}] {b['question']}")
    if down:
        print(f"\n  ── 弄坏的题（{len(down)} 题）──")
        for b, _ in down:
            print(f"    ❌ [{b.get('kind', '?')}] {b['question']}")
    print("\n" + "=" * 74)


if __name__ == "__main__":
    main()
