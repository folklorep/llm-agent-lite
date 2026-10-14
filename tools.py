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


# ============================================================================
# 下面这个工具是【我自己加的】—— 用来验证"工具注册表"怎么工作
# ============================================================================
#
# 【设计决定】参数用「米」和「千克」，不用「厘米」
# ---------------------------------------------------------------------------
# 为什么这是个需要想清楚的问题：
#   中国人说身高习惯用厘米（"我175"），但 BMI 公式要的是米。
#   所以有两种设计：
#     A. 参数是米（height_m）→ AI 要自己把 175 换算成 1.75
#     B. 参数是厘米（height_cm）→ 函数里自己除以 100
#
#   选 A 的理由：单位写在参数名里（_m / _kg），加上 description 里给了
#   示例 "例如 1.75"，AI 有足够信息自己换算。换算这件事交给 AI 做，
#   因为它是自然语言理解的部分 —— 用户说"175"还是"1.75"它都该懂。
#   如果选 B，参数名叫 height_cm 但用户说"1.75米"时，AI 反而要乘 100，
#   两种错法更难界定。
#
# 【为什么参数要带单位后缀】
#   height_m 这个 _m 后缀是给 AI 看的。参数叫 height 的话，
#   AI 无法判断用户说的 175 该按米还是厘米处理。
# ============================================================================
@tool(
    name="bmi",
    description=(
        "计算 BMI（身体质量指数）并给出健康区间判断。"
        "当用户提供身高和体重、想了解自己胖瘦程度时使用。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "height_m": {
                "type": "number",
                "description": "身高，单位【米】。如果用户说 175，要换算成 1.75",
            },
            "weight_kg": {
                "type": "number",
                "description": "体重，单位【千克】",
            },
        },
        "required": ["height_m", "weight_kg"],
    },
)
def bmi(height_m: float, weight_kg: float) -> str:
    """计算 BMI 并判断健康区间。

    注意两件事（都是照着 sqrt 的写法学的）：

    1. **参数校验放在最前面**
       除以 0 会让 Python 抛异常。虽然 run_tool() 会兜住异常，
       但主动校验能给出更清楚的错误信息（"身高必须大于 0"比"ZeroDivisionError"有用）。

    2. **返回值必须是字符串**
       因为要回传给 AI，它只接受文字。所以最后用 f-string 拼成一句人话，
       而不是返回一个数字（返回数字会导致协议层报错）。
    """
    # ---- 参数校验：不合理就直接返回错误文字，不往下算 ----
    if height_m <= 0 or weight_kg <= 0:
        return "错误：身高和体重必须大于 0"

    # ---- 算 BMI ----
    # height_m ** 2 是 Python 里的"身高平方"（** 表示乘方，不用写成 height_m * height_m）
    value = weight_kg / (height_m ** 2)

    # ---- 判断区间（中国标准）----
    # 注意边界写法：18.5 <= value < 24 这种"链式比较"是 Python 的特性，
    # 等价于 (18.5 <= value) and (value < 24)，但更好读
    if value < 18.5:
        level = "偏瘦"
    elif 18.5 <= value < 24:
        level = "正常"
    elif 24 <= value < 28:
        level = "偏胖"
    else:
        level = "肥胖"

    # ---- 返回字符串 ----
    # {value:.1f} 的含义：把 value 保留 1 位小数（:.1f 是格式说明符）
    # 不加的话会输出 21.224489795918366 这种一串数字，很难看
    return f"BMI = {value:.1f}（{level}）"


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
