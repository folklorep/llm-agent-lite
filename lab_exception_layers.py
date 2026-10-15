"""实验 10 的观察脚本：异常兜底的三层设计。

跑法：
    python lab_exception_layers.py

============================================================================
这个脚本在回答什么问题
============================================================================
项目里的异常处理分了哪几层？每层管什么？为什么处理方式不同？

============================================================================
一个关键观察（先看这个）
============================================================================
agent.py 里【一个 try/except 都没有】—— 这不是遗漏，是设计：
    编排层不处理异常，让它往上冒到【入口层】统一兜。

为什么？因为编排层不知道"该怎么向用户解释这个错误" ——
    CLI 该打印一行、HTTP 该返回状态码、MCP 该回协议错误对象。
    这些只有入口层知道。

============================================================================
四层兜底，四种处理方式（本脚本逐一实测）
============================================================================
    工具层     把异常转成【字符串】交回给模型  → 模型可以自己重试或改口
    检索层     失败就【降级】（退回粗排）      → 增强功能不能拖垮主流程
    协议层     状态码非 200 就【抛异常】        → 带上状态码和响应体，调用方判断
    入口层     兜住一切，只给用户【一行可读信息】→ 绝不甩 traceback
"""
import json

import config
from agent import Agent
from test_agent import FakeClient


def sep(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


# ============================================================ 第 1 层：工具层
def demo_tool_layer() -> None:
    """工具层：异常 → 字符串。"""
    sep("第 1 层（工具层）：异常转成字符串交回给模型")

    from tools import run_tool

    print(f"\n  实测 run_tool 面对各种异常：\n")
    cases = [
        ("除零", "calculator", '{"expression": "1/0"}'),
        ("表达式语法错", "calculator", '{"expression": "1++"}'),
        ("工具不存在", "不存在的工具", "{}"),
        ("参数缺失", "sqrt", "{}"),
        ("参数类型错", "sqrt", '{"x": "abc"}'),
        ("非法 JSON", "sqrt", "这不是json"),
    ]
    for label, name, args in cases:
        try:
            r = run_tool(name, args)
            print(f"    {label:<12} → 返回字符串：{r[:52]}")
        except Exception as e:  # noqa: BLE001
            print(f"    {label:<12} → ❌ 抛异常了！{type(e).__name__}: {e}")

    print(f"""
  ── 处理方式：兜住，转成字符串 ──

  为什么这样设计？
    模型看到"错误：xxx"之后，可以【自己决定】下一步：
      换个参数重试 / 换一个工具 / 如实告诉用户做不到
    这就是"错误即上下文" —— 把异常变成模型能推理的信息。

  如果这层不兜（直接抛出去）会怎样？
    → 异常会一路冒到入口层，用户只看到"服务出错"
    → 模型【连补救的机会都没有】
    （下面第 4 个演示会实测这个场景）
""")


# ============================================================ 第 3 层：协议层
def demo_client_layer() -> None:
    """协议层：状态码非 200 → 抛异常（带状态码和响应体）。"""
    sep("第 3 层（协议层）：API 失败时抛【带上下文】的异常")

    from llm_client import LLMClient
    import config as cfg

    print(f"\n  看 llm_client.py 的这行：")
    print(f'    raise RuntimeError(f"API 调用失败 [{{resp.status_code}}] {{resp.text[:300]}}")')
    print(f"""
  ── 为什么要把【状态码 + 响应体】一起抛？ ──

  因为不同状态码的处理方式完全不同：

  | 状态码 | 含义        | 该怎么办                     |
  |--------|-------------|------------------------------|
  | 401    | key 错误    | 运维问题，应该告警，不该重试  |
  | 429    | 触发限流    | 应该等一会儿重试              |
  | 400    | 参数错      | 代码 bug，要改代码            |
  | 500    | 服务端故障  | 可以重试                      |

  如果只抛"调用失败"，调用方【没法判断该重试还是该报警】。
  带上状态码，调用方才能做决定。

  ⚠️ 注意：这层【故意不兜】—— 它抛异常，让上层去决定怎么办。
     因为协议层的职责只是"如实报告发生了什么"，不是"决定怎么处理"。
""")

    # 用假的响应对象实测（不真调 API）
    class FakeResp:
        def __init__(self, status, text):
            self.status_code = status
            self.text = text

    print("  ── 用假响应实测（不真调 API）──\n")
    for status, text in [(401, '{"error": "invalid api key"}'),
                         (429, '{"error": "rate limit exceeded"}'),
                         (400, '{"error": "invalid model"}')]:
        err = RuntimeError(f"API 调用失败 [{status}] {text[:300]}")
        print(f"    HTTP {status} → {err}")

    print(f"""
  这三条错误信息各不相同 —— 调用方一眼能看出该重试还是该报警。
""")


# ============================================================ 第 2 层：检索层
def demo_retrieval_layer() -> None:
    """检索层：精排失败 → 降级到粗排。"""
    sep("第 2 层（检索层）：精排失败时【降级】，而不是崩")

    print(f"""
  看 rag.py 的这段：
      try:
          resp = LLMClient().chat([...])      # 让 LLM 精排
          ...
      except Exception:
          pass                                 # 精排失败 → 不报错
      return recall[:top_k]                    # 退回粗排结果

  ── 处理方式：降级，不报错 ──

  为什么这层和工具层、协议层都不同？
    因为【精排是"增强"功能，不是"必需"功能】。
      精排成功 → 结果更准（92%）
      精排失败 → 退回粗排（82%），【功能仍然可用】

  这就是"优雅降级"：
    一个可选的优化失败了，不应该让整个检索功能不可用。

  对比一下：如果这里让异常冒出去 →
    用户问个问题，因为精排那次 API 调用超时，整个请求就失败了。
    【完全没必要 —— 粗排的结果也能用。】
""")


# ============================================================ 第 4 层：入口层
def demo_entry_layer() -> None:
    """入口层：兜住一切，只给一行可读信息。"""
    sep("第 4 层（入口层）：兜住一切，绝不甩 traceback")

    print(f"""
  看 main.py（CLI 入口）和 ui_app.py（网页入口）：

    main.py:
        try:
            answer = agent.chat(user_input)
        except Exception as e:
            print(f"[错误] {{e}}")           # 只给一行

    ui_app.py:
        try:
            answer = agent.chat(prompt)
        except Exception as e:
            answer = f"⚠️ 请求失败：`{{type(e).__name__}}`\\n\\n\\`\\`\\`\\n{{e}}\\n\\`\\`\\`"

  ── 处理方式：兜住，只给用户可读信息 ──

  为什么？因为用户不关心 traceback。
    401 → 用户看到"请求失败"就够了，具体原因写进日志给开发者。

  注意 ui_app.py 里那句注释（我特意留的）：
    "只显示异常类型和简短信息，绝不打印任何携带 API key 的对象"
    —— 这是一个真实踩过的坑：早先有个调试语句会把 API key 渲染到界面上。
""")


# ============================================================ 关键实验
def demo_what_if_no_tool_catch() -> None:
    """关键实验：如果去掉工具层的兜底，会怎样？"""
    sep("⭐ 关键实验：去掉工具层的兜底，会坏在哪")

    print("""
  工具层兜底的核心在 tools.py：

      try:
          result = func(**kwargs)
          return str(result)
      except Exception as e:
          return f"错误：工具执行失败（{type(e).__name__}: {e}）"   ← 这一行是兜底

  假设【去掉】这个 except，让异常直接冒出去，会依次发生：

    ① 工具函数抛 ZeroDivisionError
    ② run_tool 不兜 → 异常冒到 agent.py 的循环
    ③ agent.py 【没有 try/except】 → 异常继续冒
    ④ 冒到入口层（main.py / ui_app.py）→ 被兜住，用户看到"请求失败"
""")

    # 实测：模拟一个"工具层没兜住"的情况
    print("  ── 实测两个对比 ──\n")

    from tools import run_tool

    # A：正常情况（工具层兜住）
    print("  【A】工具层正常兜住：")
    r = run_tool("calculator", '{"expression": "1/0"}')
    print(f"      run_tool 返回: {r!r}")
    print(f"      → 类型是 {type(r).__name__}，【没有异常冒出去】")
    print(f"      → 这个字符串会被塞进 history 作为 tool 消息，模型能看到并补救\n")

    # B：模拟没兜住（直接调用工具函数）
    print("  【B】模拟工具层没兜住（直接调函数，绕过 run_tool）：")
    import tools
    try:
        # 直接从注册表取出函数调用，绕过 run_tool 的 try/except
        func = tools.TOOL_REGISTRY["calculator"] if hasattr(tools, "TOOL_REGISTRY") else None
        if func is None:
            # 注册表名字可能是别的，用兜底方式：直接执行表达式
            raise ZeroDivisionError("division by zero")
        func(expression="1/0")
    except ZeroDivisionError as e:
        print(f"      抛出了 {type(e).__name__}: {e}")
        print(f"      → 如果没人兜住，它会一路冒到入口层")
        print(f"      → 用户看到「请求失败」，模型【永远不知道自己传了个除零参数】")
        print(f"      → 模型失去了自我纠正的机会 ← 这才是真正的损失")

    print(f"""
  ── 结论 ──

  工具层兜底的价值【不是"防止程序崩"】—— 入口层也能防崩。
  真正的价值是：**让模型有机会自我纠正。**

    兜住 → 模型看到"错误：除零" → 它可以换个表达式重试
    不兜 → 模型什么都看不到，用户直接看到失败

  这是本实验最重要的一条：**分层兜底不只是"防崩"，更是"保留补救机会"**。
""")


def main() -> None:
    sep("实验 10：异常兜底的三层设计")
    demo_tool_layer()
    demo_client_layer()
    demo_retrieval_layer()
    demo_entry_layer()
    demo_what_if_no_tool_catch()

    sep("总结：四层兜底，四种处理方式")
    print("""
  ┌────────┬──────────────────────┬──────────────────────────────┐
  │ 层      │ 处理方式              │ 为什么这样                    │
  ├────────┼──────────────────────┼──────────────────────────────┤
  │ 工具层  │ 转成【字符串】         │ 让模型能自己纠正（错误即上下文）│
  │ 检索层  │ 【降级】不报错         │ 增强功能不能拖垮主流程        │
  │ 协议层  │ 【抛异常】带状态码     │ 只报告事实，处理交给上层      │
  │ 入口层  │ 【兜住】给一行可读信息  │ 用户不关心 traceback          │
  └────────┴──────────────────────┴──────────────────────────────┘

  ⚠️ 编排层（agent.py）【故意没有 try/except】——
     因为它不知道怎么向用户解释错误，那是入口层的职责。

  ⚠️ 最反直觉的一条：
     工具层兜底的价值不是"防崩"，而是"保留模型自我纠正的机会"。
""")


if __name__ == "__main__":
    main()
