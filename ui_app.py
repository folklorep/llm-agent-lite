"""Streamlit 界面：把 Agent 变成一个能演示的网页。

本地跑：
    python -m streamlit run ui_app.py

线上演示：
    https://llm-agent-lite-folklorep.streamlit.app

设计要点（这些是给面试官看的"能演示的东西"）：
  1. **工具调用轨迹默认展开** —— 核心卖点是"模型连续调用多个工具"，
     光看回答看不出来，必须让轨迹可见
  2. **提供示例问题按钮** —— 访客（HR / 面试官）不知道该问什么，给他三个
  3. **防重复提交** —— 用 pending 缓冲 + 处理标记，避免同一次输入被处理多遍
"""
import uuid
from typing import Optional

import streamlit as st

from agent import Agent
from config import config

st.set_page_config(page_title="LLM Agent Lite", page_icon="🤖", layout="wide")

# 示例问题：第三个故意是知识库里没有的，用来展示"防幻觉"
EXAMPLES = [
    "帮我算 128乘37加15",
    "先算 144除以12，再算根号144，比较一下",
    "两数之和的解法是什么？",
    "红烧肉怎么做？",
]


# ------------------------------------------------------------------ 侧边栏
with st.sidebar:
    st.title("🤖 LLM Agent Lite")
    st.caption("多工具智能体 · 多轮对话 · RAG 检索")

    st.subheader("⚙️ 配置")
    st.text_input("模型", value=config.MODEL, key="model_display", disabled=True)
    st.toggle("显示工具调用轨迹", value=True, key="show_trace")

    st.divider()
    if st.button("🗑 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent = None
        st.session_state.pop("pending_prompts", None)
        st.rerun()

    st.divider()
    st.subheader("💡 这个项目的核心能力")
    st.caption(
        "**多工具自动循环**：模型可以先算一步、看结果、再决定下一步。\n\n"
        "第二条示例问题就是演示这个 —— 它会连续调用"
        "「计算器」和「平方根」两个工具。\n\n"
        "第四条问的是知识库里没有的内容，它会**如实说不知道**（防幻觉）。"
    )


# ------------------------------------------------------------------ 初始化
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "pending_prompts" not in st.session_state:
    # 用一个列表当"待处理队列"。Streamlit 每次交互会从头重跑脚本，
    # 队列 + 处理后立即移除，能避免同一次输入被处理多遍。
    st.session_state.pending_prompts = []


def get_agent() -> Optional[Agent]:
    """懒加载 Agent，并把配置错误变成界面提示而不是崩溃。"""
    if st.session_state.agent is None:
        try:
            st.session_state.agent = Agent()
        except RuntimeError as e:
            st.error(f"**配置错误**\n\n{e}")
            st.info(
                "**本地运行**：在项目目录复制 `.env.example` 为 `.env`，填入 `LLM_API_KEY`。\n\n"
                "**Streamlit Cloud**：在 App 右下角 Manage app → Settings → Secrets 里填：\n"
                "```toml\n"
                'LLM_API_KEY = "sk-..."\n'
                'LLM_BASE_URL = "https://api.deepseek.com"\n'
                'LLM_MODEL = "deepseek-flash"\n'
                "```"
            )
            return None
    return st.session_state.agent


def render_trace(trace: list) -> None:
    """渲染工具调用轨迹 —— 这是本项目的核心卖点，默认展开。"""
    if not trace:
        return
    with st.expander(f"🔧 工具调用轨迹（{len(trace)} 次）", expanded=True):
        for line in trace:
            st.code(line, language=None)


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if st.session_state.get("show_trace", True) and msg.get("trace"):
            render_trace(msg["trace"])


# ------------------------------------------------------------------ 主区域
st.title("多工具智能体演示")

agent = get_agent()
if agent is None:
    st.stop()

# 欢迎语
if not st.session_state.messages:
    st.info(
        "👋 这是一个**不依赖任何框架**实现的大模型应用（自行实现协议层、"
        "工具注册与编排循环）。\n\n"
        "**可用工具**：计算器 · 平方根 · 知识库检索 · 当前时间"
    )

# 渲染历史对话
for msg in st.session_state.messages:
    render_message(msg)

# 示例问题按钮（只在对话刚开始时显示，避免刷屏）
if not st.session_state.messages:
    st.write("**试试这些：**")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLES):
        if cols[i % 2].button(q, use_container_width=True, key=f"example_{i}"):
            st.session_state.pending_prompts.append(q)
            st.rerun()

# 输入框
typed = st.chat_input("输入你的问题…")
if typed:
    st.session_state.pending_prompts.append(typed)
    st.rerun()

# ------------------------------------------------------------------ 处理队列
# 一次只处理一个；处理完立刻从队列移除，保证同一条输入不会被处理两次
while st.session_state.pending_prompts:
    prompt = st.session_state.pending_prompts.pop(0)

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("思考中…"):
            try:
                answer = agent.chat(prompt)
                trace = list(agent.trace)
            except Exception as e:  # noqa: BLE001 —— 界面兜底，不把 traceback 甩给用户
                # 只显示异常类型和简短信息，绝不打印任何携带 API key 的对象
                answer = f"⚠️ 请求失败：`{type(e).__name__}`\n\n```\n{e}\n```"
                trace = []

        st.markdown(answer)
        if st.session_state.get("show_trace", True) and trace:
            render_trace(trace)

    st.session_state.messages.append(
        {
            "id": uuid.uuid4().hex[:8],   # 给每条消息一个 id，便于以后做去重/调试
            "role": "assistant",
            "content": answer,
            "trace": trace,
        }
    )
