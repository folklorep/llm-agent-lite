"""检索层：接入切块 + 向量 + 精排的完整链路。

============================================================================
接口契约（这是本模块最重要的设计，没变过）
============================================================================
    retrieve(query, top_k) -> list[{"text": str, "tag"/"heading": str, "score": float}]

**对外接口固定，内部实现从"关键词匹配"换成了"切块 + 向量 + 可选精排"，
但调用方（tools.py）一行都没改。** 这就是当初把这层单独拆出来的价值。

面试点：这叫什么？→ 依赖倒置 / 策略模式。
把「检索」抽象成一个契约，实现可以整体替换。

============================================================================
三级检索模式（config.RETRIEVAL_MODE）
============================================================================
    keyword  只做关键词打分。快、免费、不需要 API。默认。
    vector   词袋向量 + TF-IDF + 余弦相似度。同样是本地计算，免费。
    rerank   粗排（关键词召回 N 个）+ LLM 精排。最准，但每次多一次 API 调用。

**为什么默认用 keyword 而不是最准的 rerank**：
  检索是工具调用链路里的一环，每次工具调用都加一次 LLM 请求会让响应明显变慢，
  而且在"知识库里没有答案"时纯属浪费。
  实测数据（见 README 实验记录）：keyword 82% / vector 88% / rerank 92%。
  **性能与成本的折中应该由使用者决定，所以做成配置项。**

============================================================================
语料来源
============================================================================
优先读 data/docs/*.md（长文档，走切块），
没有时退回 data/knowledge.txt（短条目，兼容早期格式），
再没有就用内置兜底 —— 保证 demo 在任何情况下都能跑起来。
"""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from chunker import chunk_markdown
from config import config

DOCS_DIR = Path(__file__).parent / "data" / "docs"
KNOWLEDGE_FILE = Path(__file__).parent / "data" / "knowledge.txt"

# 最低相关度阈值：低于它的结果直接丢弃。
# 为什么需要：中文二元组分词会产生单字命中（"字典"命中"字"），
# 这种弱匹配如果返回给模型，反而会诱导它基于无关内容瞎编 —— 宁可返回空。
MIN_SCORE = 0.35

# 模块级缓存：语料和向量索引只建一次
_corpus: List[Dict[str, Any]] = []
_store: Optional[Any] = None          # VectorStore 实例，延迟构建


# ============================================================ 语料加载
def _load_from_docs() -> List[Dict[str, Any]]:
    """从 data/docs/*.md 加载：按标题切块。"""
    items: List[Dict[str, Any]] = []
    if not DOCS_DIR.exists():
        return items
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8-sig")
        for c in chunk_markdown(text, source=path.name):
            items.append(
                {
                    "text": c["text"],
                    "tag": c["heading"] or path.stem,
                    "source": path.name,
                }
            )
    return items


def _load_from_knowledge_txt() -> List[Dict[str, Any]]:
    """兼容早期格式：每行 `标签 | 内容`。"""
    items: List[Dict[str, Any]] = []
    if not KNOWLEDGE_FILE.exists():
        return items
    for line in KNOWLEDGE_FILE.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tag, _, text = line.partition("|")
        items.append({"text": text.strip() or tag.strip(), "tag": tag.strip(), "source": "knowledge.txt"})
    return items


def _fallback() -> List[Dict[str, Any]]:
    """内置兜底，保证 demo 永远能跑。"""
    return [
        {"tag": "两数之和", "text": "两数之和用字典记下标，关键是「先查后登记」，防止自己配自己。", "source": "内置"},
        {"tag": "有效的括号", "text": "有效的括号用栈，三道防线：空栈、栈顶不匹配、最后清点栈是否为空。", "source": "内置"},
        {"tag": "梯度下降", "text": "梯度下降核心公式 w = w - lr * grad。学习率太大会发散，太小会收敛慢。", "source": "内置"},
    ]


def corpus() -> List[Dict[str, Any]]:
    """返回检索语料（带缓存）。"""
    global _corpus
    if _corpus:
        return _corpus
    _corpus = _load_from_docs() or _load_from_knowledge_txt() or _fallback()
    return _corpus


# ============================================================ 分词
def _tokenize(text: str) -> set:
    """英文按单词，中文按单字 + 二元组。

    二元组的作用：中文单字区分度太低（"栈"和"字典"里的"典"都可能命中），
    二元组保留了一点词序信息，在不引入分词库的前提下效果好不少。
    """
    lowered = text.lower()
    words = set(re.findall(r"[a-z0-9_]+", lowered))
    han = re.findall(r"[\u4e00-\u9fff]", text)
    words |= {han[i] + han[i + 1] for i in range(len(han) - 1)}
    words |= set(han)
    return words


# ============================================================ 三种检索实现
def _retrieve_keyword(query: str, items: List[Dict], top_k: int) -> List[Dict]:
    """模式一：关键词重合度打分。

    两处设计（都是被实测数据逼出来的）：

    ① **标题和正文分开算权重**（标题 ×0.3）
       原因：实测发现问"红烧肉"时，某块的**标题**里含这几个字，
       于是标题命中被算成了满分 1.0 —— 但它正文里根本没有红烧肉的做法。
       标题只是"这一节叫什么"，不代表正文内容，所以给的权重应该低。

    ② **最低分阈值过滤**
       原因：中文二元组分词会产生单字命中（比如"字典"命中"字"），
       这种弱匹配得分 0.1-0.2，返回给模型纯属噪声 —— 反而容易让它瞎编。
       低于 MIN_SCORE 的直接丢掉，宁可不返回。
    """
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    scored = []
    for item in items:
        # 标题单独算，权重压低 —— 标题是"章节名"，不代表正文内容
        tag_tokens = _tokenize(item["tag"])
        body_tokens = _tokenize(item["text"])

        body_hit = len(q_tokens & body_tokens)
        tag_hit = len(q_tokens & tag_tokens)
        # 正文命中权重 1.0，标题命中权重 0.3，且不重复计入正文
        weighted = body_hit + 0.3 * len(q_tokens & tag_tokens - body_tokens)

        if weighted <= 0:
            continue
        score = weighted / len(q_tokens)
        if score < MIN_SCORE:
            continue          # 弱匹配，丢掉
        scored.append({**item, "score": round(score, 3)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def _get_store():
    """延迟构建向量索引（第一次用到才建，避免拖慢启动）。"""
    global _store
    if _store is None:
        from vector_store import VectorStore

        _store = VectorStore()
        _store.add(corpus())
    return _store


def _retrieve_vector(query: str, items: List[Dict], top_k: int) -> List[Dict]:
    """模式二：词袋向量 + TF-IDF + 余弦相似度。"""
    return _get_store().search(query, top_k)


def _retrieve_rerank(query: str, items: List[Dict], top_k: int) -> List[Dict]:
    """模式三：粗排召回 + LLM 精排。

    注意：精准但每次多一次 API 调用。
    精排失败时自动退回粗排结果 —— 不能让检索环节把整个工具调用搞崩。
    """
    from llm_client import LLMClient

    recall_n = max(config.recall_n(), top_k)
    recall = _retrieve_keyword(query, items, recall_n)
    if len(recall) <= top_k:
        return recall      # 候选本来就少，不用精排

    listing = "\n\n".join(f"[{i}] {c['text'][:400]}" for i, c in enumerate(recall))
    prompt = (
        f"用户的问题：\n{query}\n\n"
        f"以下是 {len(recall)} 个候选资料块（编号 + 内容）：\n\n{listing}\n\n"
        f"请判断哪几个块最可能包含回答这个问题的信息。\n"
        f"只输出编号，按相关度从高到低，用逗号分隔，最多 {top_k} 个。"
        f"不要任何解释。如果都不相关，输出：无"
    )
    try:
        resp = LLMClient().chat([{"role": "user", "content": prompt}])
        raw = resp.get("content") or ""
        order = []
        for n in (int(x) for x in re.findall(r"\d+", raw)):
            if 0 <= n < len(recall) and n not in order:
                order.append(n)
        if order:
            return [recall[i] for i in order[:top_k]]
    except Exception:  # noqa: BLE001 —— 精排是增强，失败不能影响主流程
        pass
    return recall[:top_k]


_MODES = {
    "keyword": _retrieve_keyword,
    "vector": _retrieve_vector,
    "rerank": _retrieve_rerank,
}


# ============================================================ 对外接口
def retrieve(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """检索入口。实现由 config.RETRIEVAL_MODE 决定。

    契约（始终不变）：
        输入：query（问题）、top_k（要几条）
        输出：[{"text": 正文, "tag": 来源标题, "score": 相关度分数}, ...]
    """
    fn = _MODES.get(getattr(config, "RETRIEVAL_MODE", "keyword"), _retrieve_keyword)
    return fn(query, corpus(), top_k)


def stats() -> Dict[str, Any]:
    """给界面/调试用的统计信息。"""
    items = corpus()
    return {
        "mode": getattr(config, "RETRIEVAL_MODE", "keyword"),
        "chunks": len(items),
        "sources": sorted({i.get("source", "?") for i in items}),
        "total_chars": sum(len(i["text"]) for i in items),
    }


if __name__ == "__main__":
    import sys

    print("=" * 68)
    print("  检索层自检")
    print("=" * 68)
    s = stats()
    print(f"\n  模式：{s['mode']}")
    print(f"  语料：{s['chunks']} 块，共 {s['total_chars']} 字")
    print(f"  来源：{s['sources']}")

    q = sys.argv[1] if len(sys.argv) > 1 else "两数之和怎么解"
    print(f"\n  查询：{q}\n")
    for i, r in enumerate(retrieve(q, 3), 1):
        preview = r["text"].replace("\n", " ")[:70]
        print(f"    {i}. [score={r['score']}] {r['tag'][:36]}")
        print(f"       {preview}...")
    print("\n" + "=" * 68)
