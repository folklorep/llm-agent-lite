"""工具箱：真正"干活"的地方。

设计要点（对比你原来的 chat.py）：
  1. 用「注册表」而不是 if/elif 分支 —— 加工具只改一处，不改分发逻辑
  2. 每个工具有：函数 + JSON Schema 说明书 + 是否可用
  3. 异常在工具层兜住 → 返回错误字符串给模型，而不是让程序崩

面试点：加一个新工具要改几处？
  旧写法：改 2 处（tools 清单 + run_tool 分支），且容易漏
  新写法：加一个 @tool 装饰器函数，1 处。这就是「开闭原则」的实际样子。
"""
import json
import math
from typing import Any, Callable, Dict, List

# 工具注册表：名字 -> {"func": 可调用对象, "schema": JSON Schema}
_REGISTRY: Dict[str, Dict[str, Any]] = {}


def tool(name: str, description: str, parameters: Dict[str, Any]):
    """装饰器：把一个普通函数注册成模型可调用的工具。"""

    def decorator(func: Callable):
        _REGISTRY[name] = {
            "func": func,
            "schema": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            },
        }
        return func

    return decorator


def get_tool_schemas() -> List[Dict[str, Any]]:
    """给模型看的"工具箱说明书"。"""
    return [item["schema"] for item in _REGISTRY.values()]


def run_tool(name: str, arguments: str) -> str:
    """执行模型点单的工具，返回字符串结果（永远是 str，方便塞回 messages）。"""
    if name not in _REGISTRY:
        return f"错误：不存在名为 {name} 的工具"

    try:
        kwargs = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        # 踩坑经验：模型偶尔返回非法 JSON，不能让它把程序带崩
        return f"错误：参数不是合法 JSON：{arguments[:100]}"

    try:
        result = _REGISTRY[name]["func"](**kwargs)
    except TypeError as e:
        return f"错误：参数不匹配（{e}）"
    except Exception as e:  # noqa: BLE001 —— 工具层兜住一切异常，交给模型处理
        return f"错误：工具执行失败（{type(e).__name__}: {e}）"

    return str(result)


# ============================== 具体工具 ==============================

@tool(
    name="calculator",
    description="执行四则运算。当用户需要精确计算时使用。支持 + - * / ** 和括号。",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式，例如 '128*37+15' 或 '(3.14*17)**2'",
            }
        },
        "required": ["expression"],
    },
)
def calculator(expression: str) -> str:
    # 中文符号容错（你自己踩过的坑：用户会用中文说"128乘37加15"）
    for zh, en in (
        ("乘", "*"), ("加", "+"), ("减", "-"), ("除", "/"),
        ("×", "*"), ("÷", "/"), ("^", "**"),
        ("（", "("), ("）", ")"), ("，", ""), (",", ""), (" ", ""),
    ):
        expression = expression.replace(zh, en)
    normalized = expression
    # 安全校验：只允许数字和运算符（面试点：为什么不能直接 eval？→ 任意代码执行）
    allowed = set("0123456789.+-*/() %")
    if not set(normalized) <= allowed:
        return f"错误：表达式含有不支持的字符：{normalized}"
    result = eval(normalized, {"__builtins__": {}}, {})  # noqa: S307 已做白名单
    return f"{expression} = {result}"


@tool(
    name="sqrt",
    description="计算平方根。当用户问'根号X'时使用。",
    parameters={
        "type": "object",
        "properties": {"x": {"type": "number", "description": "被开方的数，必须 >= 0"}},
        "required": ["x"],
    },
)
def sqrt(x: float) -> str:
    if x < 0:
        return "错误：负数没有实数平方根"
    return f"sqrt({x}) = {math.sqrt(x)}"


@tool(
    name="search_knowledge",
    description=(
        "在本地知识库中检索资料。当用户的问题涉及算法笔记、学习资料等私有内容时使用。"
        "如果检索结果为空，说明知识库里没有，不要编造。"
    ),
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "检索关键词或问题"}},
        "required": ["query"],
    },
)
def search_knowledge(query: str) -> str:
    # 一期：关键词匹配（占位）。二期换成向量检索：见 rag.py
    from rag import retrieve

    hits = retrieve(query, top_k=3)
    if not hits:
        return "知识库中没有找到相关内容。"
    return "\n\n".join(f"[{i + 1}] {h['text']}" for i, h in enumerate(hits))


@tool(
    name="get_current_time",
    description="获取当前日期和时间。涉及'今天''现在'的问题时使用。",
    parameters={"type": "object", "properties": {}, "required": []},
)
def get_current_time() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
