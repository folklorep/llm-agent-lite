"""检索层：一期关键词匹配 → 二期向量检索（接口不变，实现可换）。

这是本骨架最重要的一处设计：
    retrieve(query, top_k) -> list[{"text": str, "score": float}]

**对外接口固定，内部实现可替换** —— 三期不管怎么升级，调用方（tools.py）一行都不用改。
面试点：这叫什么？→ 依赖倒置 / 策略模式。你把「检索」抽象成了一个契约。

二期升级路线（照着做即可）：
  1. pip install chromadb sentence-transformers
  2. 把知识库文本切块（chunking），embedding 后存进 chroma
  3. 把关键词打分换成余弦相似度，返回同样的结构
"""
import re
from pathlib import Path
from typing import Any, Dict, List

# 知识库文件：每行一条，格式 `标签 | 内容`，用你自己的学习笔记导出即可
KNOWLEDGE_FILE = Path(__file__).parent / "data" / "knowledge.txt"

_cache: List[Dict[str, str]] = []


def _load() -> List[Dict[str, str]]:
    """加载知识库（带缓存，不用每次读盘）。"""
    global _cache
    if _cache:
        return _cache

    items: List[Dict[str, str]] = []
    if KNOWLEDGE_FILE.exists():
        for line in KNOWLEDGE_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tag, _, text = line.partition("|")
            items.append({"tag": tag.strip(), "text": text.strip() or tag.strip()})

    # 知识库为空时给个内置兜底，保证 demo 能跑
    if not items:
        items = [
            {"tag": "两数之和", "text": "两数之和用字典记下标，关键是「先查后登记」，防止自己配自己。"},
            {"tag": "有效的括号", "text": "有效的括号用栈，三道防线：空栈、栈顶不匹配、最后清点栈是否为空。"},
            {"tag": "梯度下降", "text": "梯度下降核心公式 w = w - lr * grad。学习率太大会发散，太小会收敛慢。"},
            {"tag": "特征标准化", "text": "特征量级差太大会导致梯度下降走之字形，解法是标准化 z=(v-mean)/std。"},
        ]

    _cache = items
    return items


def _tokenize(text: str) -> set:
    """极简分词：英文按单词，中文按单字 bigram。够用即可。"""
    lowered = text.lower()
    words = set(re.findall(r"[a-z0-9_]+", lowered))
    han = re.findall(r"[\u4e00-\u9fff]", text)
    words |= {han[i] + han[i + 1] for i in range(len(han) - 1)}
    words |= set(han)
    return words


def retrieve(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """检索入口。一期用关键词重合度打分。"""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scored: List[Dict[str, Any]] = []
    for item in _load():
        doc_tokens = _tokenize(item["tag"] + " " + item["text"])
        overlap = len(query_tokens & doc_tokens)
        if overlap == 0:
            continue
        # 简单归一化：命中的查询词占比
        score = overlap / max(len(query_tokens), 1)
        scored.append({"text": item["text"], "tag": item["tag"], "score": round(score, 3)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
