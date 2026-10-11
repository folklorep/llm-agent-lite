"""大模型客户端：只负责"说话"，不负责"干活"。

职责边界（面试会问）：
  - llm_client：发请求、收响应、解析协议（含 tool_calls）
  - tools：真正干活的函数
  - agent：循环编排（谁先谁后、什么时候停）

对比你原来的 chat.py：一个文件里混了请求、历史、工具分发、主循环四件事。
"""
from typing import Any, Dict, List, Optional

import requests

from config import config


class LLMClient:
    def __init__(self) -> None:
        config.validate()
        self.url = config.chat_url()
        self.headers = {
            "Authorization": f"Bearer {config.API_KEY}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """发一次请求，返回原始 message 对象。

        返回的是 message 而不是 content —— 因为 message 里可能带 tool_calls。
        这是 Function Calling 和你原来 L1 版本最大的区别。
        """
        payload: Dict[str, Any] = {
            "model": config.MODEL,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        resp = requests.post(
            self.url, headers=self.headers, json=payload, timeout=config.TIMEOUT
        )

        if resp.status_code != 200:
            # 面试点：为什么要把状态码和响应体一起抛？
            # 因为 401（key 错）和 429（限流）和 400（参数错）处理方式完全不同
            raise RuntimeError(f"API 调用失败 [{resp.status_code}] {resp.text[:300]}")

        data = resp.json()
        return data["choices"][0]["message"]

    def chat_stream(self, messages: List[Dict[str, Any]]):
        """流式输出：一个字一个字吐。三期接 Web 界面时用。

        面试点：为什么流式体验好？→ 首 token 延迟低，用户不用等全文生成完。
        """
        payload = {
            "model": config.MODEL,
            "messages": messages,
            "stream": True,
        }
        with requests.post(
            self.url, headers=self.headers, json=payload,
            timeout=config.TIMEOUT, stream=True,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                text = line.decode("utf-8").removeprefix("data: ").strip()
                if text == "[DONE]":
                    break
                import json

                try:
                    delta = json.loads(text)["choices"][0]["delta"]
                except (KeyError, IndexError, ValueError):
                    continue
                if "content" in delta and delta["content"]:
                    yield delta["content"]
