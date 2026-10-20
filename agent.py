"""Agent 编排层：本骨架的核心，也是你简历上最值得讲的部分。

你原来 chat.py 的做法：
    问 → 模型点单 → 执行一次 → 把结果塞回去 → 出答案
    （只能调用一次工具，模型第二次想用工具就没辙了）

这里的做法（多工具自动循环）：
    while 轮次 < MAX:
        问模型
        if 模型要调工具:
            挨个执行，把结果塞回 messages，continue   ← 关键：回到循环顶部再问一次
        else:
            输出答案，break
    → 模型可以连续调用多个工具、根据上一步结果决定下一步

面试讲法（这段就是你的"项目亮点"）：
  "单体 Function Calling 只能调一次工具，我把它改成了带轮次上限的
   推理-行动循环（ReAct 风格），模型可以连续调用工具并根据中间结果
   调整策略；同时加了最大轮次保护和工具级异常兜底，防止死循环和崩溃。"
"""
from typing import Any, Callable, Dict, List, Optional

from config import config
from llm_client import LLMClient
from tools import get_tool_schemas, run_tool

SYSTEM_PROMPT = """你是一个严谨的 AI 助手，可以使用工具来完成任务。

规则：
1. 需要精确计算、查资料、查时间时，优先调用工具，不要凭记忆回答。
2. 一次可以调用多个工具；拿到结果后再决定是否需要继续调用。
3. 只输出自然语言回答。不要输出任何工具调用代码、JSON 片段或 <tool_call> 之类的标记文本。
4. 如果工具返回错误或知识库中没有相关资料，如实告知用户，不要编造。
"""


class Agent:
    def __init__(self, system_prompt: str = SYSTEM_PROMPT, client=None,
                 history: Optional[List[Dict[str, Any]]] = None,
                 on_turn_end: Optional[Callable[[List[Dict[str, Any]]], None]] = None) -> None:
        # client 可注入 —— 这是依赖注入（DI）的最小用法：
        #   生产代码传真实客户端；测试时传"假模型"，就能在不花 token、
        #   不需要 API key 的情况下验证整个循环逻辑。
        # 面试点：为什么构造函数里不直接 new 一个 LLMClient？
        #   → 那样 Agent 就和网络绑死了，没法单测。
        self.client = client if client is not None else LLMClient()
        self.tools = get_tool_schemas()

        # history 也可注入 —— 用于"从数据库恢复上次的对话"。
        # 如果不传，就从零开始（只有 system 提示词）。
        if history:
            self.history: List[Dict[str, Any]] = list(history)
        else:
            self.history = [{"role": "system", "content": system_prompt}]

        # on_turn_end 回调：每轮对话结束后调用，参数是当前的 history。
        # 为什么用【回调】而不是让 Agent 直接 import storage？
        #   ① 解耦：Agent 不用知道数据存在哪（SQLite / Redis / 文件都行）
        #   ② 可测：测试时传 None，Agent 就完全不碰存储
        #   这和 client 的注入是同一个思路 —— 依赖由外部提供，而不是内部写死。
        self.on_turn_end = on_turn_end

        self.trace: List[str] = []  # 调用轨迹，便于调试和面试展示

    # ---------- 上下文管理：控制 token 成本 ----------
    def _trim_history(self) -> None:
        """滑动窗口：只保留 system + 最近 N 轮（每轮 2 条消息）。

        面试点：为什么要做？→ 每轮都把完整 history 发出去，
        第 N 轮发送量 ≈ 2N-1 条，长对话成本近似平方增长。
        """
        head, body = self.history[0], self.history[1:]
        limit = config.MAX_HISTORY_TURNS * 2
        if len(body) > limit:
            self.history = [head] + body[-limit:]

    # ---------- 主循环 ----------
    def chat(self, user_input: str) -> str:
        """对外的入口：跑一轮对话，然后持久化。

        为什么要拆成 chat() + _run()：
          循环里有【两个返回点】（正常回答 / 轮次耗尽走兜底）。
          如果直接在 chat() 里写，两个 return 前都要复制一遍"存盘"代码 ——
          容易漏。拆开之后，存盘只在 chat() 里写一次，两条路径都走到。
        """
        answer = self._run(user_input)
        # 每轮结束后把最新 history 交给外部（如果有回调）
        if self.on_turn_end is not None:
            self.on_turn_end(self.history)
        return answer

    def _run(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})
        # 裁剪必须放在"追加用户消息之后"：我们要限制的是**真正发出去的 payload**。
        # 踩过的坑：原来放在 append 之前，于是每轮 payload 都比上限多 2 条
        # （刚追加的 user + 上一轮的 assistant），长对话下误差会累积。
        # 这个 bug 是 test_agent.py 的用例 7 抓出来的。
        self._trim_history()
        self.trace = []

        for round_no in range(1, config.MAX_TOOL_ROUNDS + 1):
            message = self.client.chat(self.history, tools=self.tools)
            tool_calls = message.get("tool_calls")

            if not tool_calls:
                answer = message.get("content") or ""
                self.history.append({"role": "assistant", "content": answer})
                return answer

            # ① 先把模型的"点单"原样记进对话（协议要求，不能省）
            self.history.append(message)

            # ② 挨个执行工具，把结果作为 role=tool 的消息塞回去
            for call in tool_calls:
                name = call["function"]["name"]
                args = call["function"].get("arguments", "")
                result = run_tool(name, args)
                self.trace.append(f"[第{round_no}轮] {name}({args}) -> {result}")

                if config.DEBUG:
                    print(f"  ↳ 调用 {name}({args}) => {result}")

                self.history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": result,
                    }
                )
            # ③ 不加 break —— 回到循环顶部，让模型看着工具结果再决定下一步

        # 兜底：轮次耗尽还没收敛
        fallback = "抱歉，工具调用轮次已达上限，请把问题拆得更具体一些。"
        self.history.append({"role": "assistant", "content": fallback})
        return fallback

    def reset(self) -> None:
        self.history = [self.history[0]]
        self.trace = []
        if self.on_turn_end is not None:
            self.on_turn_end(self.history)
