"""切块模块：把长 Markdown 文档切成适合检索的小块。

============================================================================
为什么需要这个模块（面试时这么讲）
============================================================================
大模型的上下文窗口有限，不可能把整份资料塞进 prompt。所以要：

    长文档 → 切成小块 → 只把最相关的几块塞进 prompt

但"怎么切"是个真问题，切错了后面用再好的向量模型也救不回来。

本模块的切块策略（四条规则，每条都有理由）：

  ① **按 Markdown 标题切，不按固定字数切**
     理由：标题天然就是语义边界。按字数硬切会把一句话、一段代码切成两半。
     实测：把 788 行 / 3.1 万字的笔记按二级标题（###）切，得到 42 块，
     平均 397 字 —— 正好落在"既够完整、又不浪费上下文"的区间。
     而按一级标题（##）切只有 9 块，平均 1891 字，太大。

  ② **代码块永远不拆开**
     理由：一段代码被从中间切断，检索到半截代码等于没检索到。

  ③ **太小的块向前合并**（默认阈值 100 字）
     理由：一个只有 20 字的块（比如一个光秃秃的小标题）单独检索出来毫无意义。

  ④ **每块都带上"它属于哪一节"**
     理由：检索到"三道防线：空栈 / 栈顶不匹配 / 最后清点"这一段时，
     AI 必须知道它说的是"有效的括号"这道题，否则这段信息没用。

============================================================================
用法
============================================================================
    from chunker import chunk_file
    chunks = chunk_file("data/docs/学习笔记.md")
    # chunks = [{"text": ..., "source": ..., "heading": ..., "index": ...}, ...]
"""
import re
from pathlib import Path
from typing import Dict, List

# ------------------------------------------------------------------ 可调参数
MIN_CHUNK_CHARS = 100    # 小于这个字数的块，尝试向前合并
TARGET_MAX_CHARS = 800   # 大于这个字数的块，记录警告（不强行拆，怕破坏语义）

CODE_FENCE = "```"


def _heading_level(line: str) -> int:
    """返回 Markdown 标题的层级；不是标题则返回 0。

    '# 标题'   -> 1
    '## 标题'  -> 2
    '### 标题' -> 3
    '普通文字' -> 0
    """
    m = re.match(r"^(#{1,6})\s+(.*)$", line)
    return len(m.group(1)) if m else 0


def _heading_text(line: str) -> str:
    """取出标题文字（去掉前面的 # 号）。"""
    return re.sub(r"^#{1,6}\s+", "", line).strip()


def chunk_markdown(text: str, source: str = "") -> List[Dict]:
    """把 Markdown 文本切成块。

    核心思路：
      - 逐行扫描，遇到标题就"封口"（把之前攒的内容作为一块输出）
      - 在代码块内部（``` 之间）不认标题 —— 否则代码里的 # 注释会被误判成标题
      - 记录从一级到当前级的标题路径，作为每块的上下文
    """
    lines = text.splitlines()
    chunks: List[Dict] = []

    # 当前正在累积的块
    buf: List[str] = []
    # 标题层级路径，例如 ["三、算法", "1. 两数之和"]
    path: List[str] = []
    # 当前块最贴近的标题（用于给块打标签）
    current_heading = ""

    in_code = False          # 是否处在 ``` 代码块内部

    def flush() -> None:
        """把 buf 里攒的内容封口成一块。"""
        nonlocal buf
        body = "\n".join(buf).strip()
        buf = []
        if not body:
            return

        # 规则 ④：把标题路径拼在正文前面，让块自带上下文
        context = " > ".join(p for p in path if p)
        if context:
            text_with_ctx = f"【{context}】\n{body}"
        else:
            text_with_ctx = body

        chunks.append(
            {
                "text": text_with_ctx,
                "body": body,
                "source": source,
                "heading": current_heading,
                "path": list(path),
                "chars": len(body),
            }
        )

    for line in lines:
        # ---- 代码块状态机：``` 之间的内容不当标题处理 ----
        if line.strip().startswith(CODE_FENCE):
            in_code = not in_code
            buf.append(line)
            continue

        if not in_code:
            level = _heading_level(line)
            if level > 0:
                # 遇到新标题 —— 先把上一块封口
                flush()

                title = _heading_text(line)
                # 维护标题路径：砍掉比当前层级更深的，再补上自己
                path = path[: level - 1]
                while len(path) < level - 1:
                    path.append("")
                path.append(title)
                current_heading = title
                continue

        buf.append(line)

    flush()   # 最后一块

    return _post_process(chunks)


def _post_process(chunks: List[Dict]) -> List[Dict]:
    """规则 ③：把太小的块合并到前一块。"""
    if not chunks:
        return []

    merged: List[Dict] = []
    for c in chunks:
        # 太小的块，且不是第一块 → 并进前一块
        if c["chars"] < MIN_CHUNK_CHARS and merged:
            prev = merged[-1]
            prev["body"] = prev["body"] + "\n\n" + c["body"]
            prev["text"] = prev["text"] + "\n\n" + c["body"]
            prev["chars"] = len(prev["body"])
            continue
        merged.append(c)

    # 重新编号 + 给超大的块打标记（不强行拆，怕破坏语义）
    for i, c in enumerate(merged):
        c["index"] = i
        c["oversize"] = c["chars"] > TARGET_MAX_CHARS

    return merged


def chunk_file(path: str | Path) -> List[Dict]:
    """切一个文件。"""
    p = Path(path)
    return chunk_markdown(p.read_text(encoding="utf-8-sig"), source=p.name)


# ------------------------------------------------------------------ 命令行自检
def _main() -> None:
    """直接运行本文件时，切一份文档并打印统计 —— 用来验证策略效果。

    跑法：
        python chunker.py data/docs/学习笔记.md
    """
    import sys

    if len(sys.argv) < 2:
        print("用法: python chunker.py <markdown文件>")
        print("示例: python chunker.py data/docs/学习笔记.md")
        return

    target = sys.argv[1]
    if not Path(target).exists():
        print(f"文件不存在: {target}")
        return

    chunks = chunk_file(target)
    if not chunks:
        print("没有切出任何块")
        return

    sizes = [c["chars"] for c in chunks]
    print("=" * 66)
    print(f"  切块结果：{Path(target).name}")
    print("=" * 66)
    print(f"\n  总块数        {len(chunks)}")
    print(f"  总字数        {sum(sizes)}")
    print(f"  平均块大小    {sum(sizes) // len(sizes)} 字")
    print(f"  最大块        {max(sizes)} 字")
    print(f"  最小块        {min(sizes)} 字")
    ok = sum(1 for s in sizes if 100 <= s <= 800)
    print(f"  落在 100-800 字的块: {ok} / {len(chunks)}")

    over = [c for c in chunks if c.get("oversize")]
    if over:
        print(f"\n  ⚠️ 超过 {TARGET_MAX_CHARS} 字的块（{len(over)} 个）：")
        for c in over[:5]:
            print(f"     {c['chars']:>5} 字  {c['heading'][:40]}")

    print(f"\n  ── 前 3 块预览 ──")
    for c in chunks[:3]:
        head = c["text"].splitlines()[0][:60]
        print(f"\n  [块 {c['index']}] {head}")
        print(f"    {c['chars']} 字 | 来源 {c['source']}")
    print("\n" + "=" * 66)


if __name__ == "__main__":
    _main()
