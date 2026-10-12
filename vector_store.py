"""向量检索：把每个块转成向量，用余弦相似度找最相关的。

============================================================================
先讲清楚一件重要的事：这里用的不是"神经网络的 embedding"
============================================================================
真正的语义 embedding（比如 text-embedding-3）是一个训练出来的神经网络，
能把"不同单位的数字"和"特征量级差太大"映射到相近的向量 —— 因为它学过语义。

**本项目暂时用不了那种模型**，原因很实在：
  - DeepSeek API 没有 embedding 接口（实测 /embeddings 返回 404）
  - 本地跑 sentence-transformers 要下 PyTorch，2GB+，环境重

**所以这里用的是"词袋向量"（bag-of-words vector）**：
  每个块 = 一个高维向量，每一维对应一个词，值是这个词的 TF-IDF 权重。
  两个块"像不像" = 两个向量的余弦相似度。

============================================================================
那它和关键词检索到底有什么不同？（这是面试必被问的）
============================================================================
    关键词检索：数"命中几个词"，按命中数排序 —— 没有"向量"的概念
    向量检索：  把查询和文档都表示成向量，比"方向是否一致"

三个实质差别：

  ① **长度归一化天然内置**
     余弦相似度除以了两个向量的模长，所以长块不会自动占便宜。
     （上一轮实测：关键词版的长块偏向要单独打补丁，这里不用）

  ② **TF-IDF 加权：越罕见的词越重要**
     "的""是"这种到处都是的词权重接近 0；
     "之字形""失忆"这种只在少数块出现的词权重很高。
     这让"命中了关键术语"的价值远大于"命中了常见字"。

  ③ **可以表达部分匹配的程度**
     关键词版是"命中 3 个词"这种离散计数，向量版得到的是 0~1 的连续相似度，
     能反映"有多像"，排序更细腻。

**但它仍然不是语义匹配。** 词完全不重合时（"不同单位"vs"量级差"），
它和关键词检索一样会失败。这是它的能力边界，要如实说。

============================================================================
架构：为什么写成"向量库"的样子
============================================================================
真实项目里用的是 chromadb / faiss 这类向量数据库。本模块的接口故意做成相似的形状：

    store = VectorStore()
    store.add(chunks)          # 建索引
    store.search(query, k)     # 检索

这样以后要换成真的 embedding + 向量数据库时，**只需要换 store 内部实现，
调用方和评测代码一行都不用改**。这就是"接口与实现分离"。

跑法（自检）：
    python vector_store.py
"""
import math
from typing import Callable, Dict, List, Tuple

from eval_retrieval import load_corpus, load_questions, tokenize

# ============================================================================
# 向量库
# ============================================================================
class VectorStore:
    """一个极简的向量库：词袋 + TF-IDF + 余弦相似度。

    索引结构只有三样东西：
      self._chunks   原始块（保留元数据）
      self._vectors  每块的稀疏向量 {词: 权重}
      self._norms    每块向量的模长（预先算好，检索时不用重复算）
      self._idf      每个词的 IDF 值
    """

    def __init__(self) -> None:
        self._chunks: List[Dict] = []
        self._vectors: List[Dict[str, float]] = []
        self._norms: List[float] = []
        self._idf: Dict[str, float] = {}

    # ---------------------------------------------------------------- 建索引
    def add(self, chunks: List[Dict]) -> None:
        """把所有块转成向量，建好索引。"""
        if not chunks:
            return
        self._chunks = chunks

        # ---- 第 1 步：统计文档频率（DF）—— 有多少块包含这个词 ----
        n_docs = len(chunks)
        df: Dict[str, int] = {}
        tokenized: List[List[str]] = []
        for c in chunks:
            toks = tokenize(c["text"])
            tokenized.append(toks)
            for t in set(toks):
                df[t] = df.get(t, 0) + 1

        # ---- 第 2 步：算 IDF —— 越罕见权重越高 ----
        # 公式：log((N+1)/(df+1)) + 1
        #   +1 平滑：避免分母为 0
        #   最后 +1：保证即使某词在所有块里都出现，权重也不会变成 0
        self._idf = {t: math.log((n_docs + 1) / (d + 1)) + 1 for t, d in df.items()}

        # ---- 第 3 步：把每块转成向量（TF × IDF）----
        self._vectors = []
        self._norms = []
        for toks in tokenized:
            # 词频：这个词在本块里出现了几次
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1

            # 向量：每一维 = 词频 × IDF
            vec = {t: cnt * self._idf.get(t, 1.0) for t, cnt in tf.items()}
            self._vectors.append(vec)
            # 模长（欧几里得范数），预存起来
            self._norms.append(math.sqrt(sum(w * w for w in vec.values())) or 1.0)

    # ---------------------------------------------------------------- 检索
    def _query_vector(self, query: str) -> Tuple[Dict[str, float], float]:
        """把查询也转成同空间的向量。"""
        tf: Dict[str, int] = {}
        for t in tokenize(query):
            tf[t] = tf.get(t, 0) + 1
        vec = {t: cnt * self._idf.get(t, 1.0) for t, cnt in tf.items()}
        norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        return vec, norm

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """按余弦相似度返回最相关的 top_k 块。"""
        if not self._vectors:
            return []

        q_vec, q_norm = self._query_vector(query)

        scored = []
        for i, d_vec in enumerate(self._vectors):
            # 点积：只算两边都有的词（稀疏向量的好处）
            shared = q_vec.keys() & d_vec.keys()
            if not shared:
                continue
            dot = sum(q_vec[t] * d_vec[t] for t in shared)
            # 余弦 = 点积 ÷ (两个模长相乘)
            cos = dot / (q_norm * self._norms[i])
            # 把索引带回去 —— explain() 需要它来取原始向量
            scored.append({**self._chunks[i], "score": round(cos, 4), "_idx": i})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    # ---------------------------------------------------------------- 调试用
    def explain(self, query: str, top_k: int = 1) -> str:
        """解释"为什么这个词排第一" —— 打印共享词及其权重贡献。

        这个方法是给自己排查用的。面试时也很有用：
        被问"你怎么知道它为什么匹配到这一条"，你可以当场打出来看。
        """
        q_vec, q_norm = self._query_vector(query)
        results = self.search(query, top_k)
        lines = [f"查询: {query}", f"查询向量维度: {len(q_vec)}"]
        for r in results:
            idx = r["_idx"]
            d_vec = self._vectors[idx]
            shared = sorted(
                (q_vec[t] * d_vec[t], t) for t in (q_vec.keys() & d_vec.keys())
            )
            top_terms = [f"{t}({v:.3f})" for v, t in shared[-6:]]
            lines.append(f"\n  → {r['heading'][:44]}  score={r['score']}")
            lines.append(f"     贡献最大的共享词: {' '.join(reversed(top_terms))}")
        return "\n".join(lines)


# ============================================================================
# 给评测框架用的检索器（签名和 eval_retrieval.Retriever 一致）
# ============================================================================
def make_vector_retriever(store: VectorStore) -> Callable[[str, List[Dict], int], List[Dict]]:
    """包装成评测器认识的形状。

    注意：评测器传进来的 chunks 参数这里不用 —— 因为向量已经建好索引了。
    这正是"索引"的意义：检索时不必重新处理全部语料。
    """

    def retrieve(query: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
        return store.search(query, top_k)

    return retrieve


# ============================================================================
# 自检
# ============================================================================
def _main() -> None:
    chunks = load_corpus()
    questions = load_questions()
    if not chunks:
        print("语料为空")
        return

    print("=" * 70)
    print("  向量库自检（词袋 + TF-IDF + 余弦相似度）")
    print("=" * 70)

    store = VectorStore()
    store.add(chunks)

    print(f"\n  索引已建：{len(chunks)} 块")
    print(f"  词表大小：{len(store._idf)} 个词")
    print(f"  平均每块向量维度（非零维）：{sum(len(v) for v in store._vectors) // len(store._vectors)}")

    # 看几个 IDF 值，理解"罕见词权重高"
    print(f"\n  ── IDF 权重示例（越高越罕见、越重要）──")
    samples = ["之字形", "失忆", "字典", "栈", "的", "是", "在", "一个"]
    for w in samples:
        if w in store._idf:
            print(f"    {w:<8} IDF = {store._idf[w]:.3f}")
        else:
            print(f"    {w:<8} （不在词表里）")

    # 挑一个失败案例看看它现在能不能找到
    probe = "为什么把不同单位的数字放在一起训练会出问题"
    print(f"\n  ── 探查一个关键词检索失败的案例 ──")
    print(store.explain(probe, top_k=3))

    print("\n" + "=" * 70)
    print("  跑 `python eval_retrieval.py --retriever vector` 看完整评测")
    print("=" * 70)


if __name__ == "__main__":
    _main()
