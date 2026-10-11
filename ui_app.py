"""Streamlit 界面：把 Agent 变成一个能演示的网页。

本地跑：
    python -m pip install streamlit
    python -m streamlit run ui_app.py

为什么先做 Streamlit 而不是 FastAPI：
    面试官/HR 不会 curl 你的接口，但会点一个网页。
    Streamlit 一个文件就能出一个可交互界面，不需要前端代码。

线上演示（可选，见 DEPLOY.md）：
    推到 GitHub 后在 share.streamlit.io 一键部署，免费拿一个公网 URL。
"""
import streamlit as st

from agent import Agent
from config import config
from typing import Optional

st.set_page_config(page_title="LLM Agent Lite", page_icon="🤖", layout="wide")

# ------------------------------------------------------------------ 侧边栏
with st.sidebar:
    st.title("🤖 LLM Agent Lite")
    st.caption("多工具智能体 · 多轮对话 · RAG 检索")

    st.subheader("⚙️ 配置")
    st.text_input("模型", value=config.MODEL, key="model_display", disabled=True)
    debug = st.toggle("显示工具调用轨迹", value=True)
    temperature = st.slider("temperature", 0.0, 1.5, 0.7, 0.1)

    st.divider()
    if st.button("🗑 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent = None
        st.rerun()

    st.divider()
    st.subheader("💡 试试这些")
    st.caption(
        "- 帮我算 128乘37加15\n"
        "- 先算 144除以12，再算根号144，比较一下\n"
        "- 两数之和的解法是什么？\n"
        "- 红烧肉怎么做？（测防幻觉）\n"
        "- 现在几点？"
    )

# ------------------------------------------------------------------ 初始化
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None


def get_agent() -> Optional[Agent]:
    """懒加载 Agent，并把配置错误变成界面提示而不是崩溃。"""
    if st.session_state.agent is None:
        try:
            st.session_state.agent = Agent()
        except RuntimeError as e:
            st.error(f"**配置错误**\n\n{e}")
            st.info(
                "请在项目目录下复制 `.env.example` 为 `.env`，"
                "填入你的 `LLM_API_KEY` 后刷新页面。"
            )
            return None
    return st.session_state.agent


# ------------------------------------------------------------------ 主区域
st.title("多工具智能体演示")

agent = get_agent()

if agent is None:
    st.stop()

# 欢迎语
if not st.session_state.messages:
    st.info(
        "👋 这是一个不依赖框架实现的大模型应用。\n\n"
        "**可用工具**：计算器 · 平方根 · 知识库检索 · 当前时间\n\n"
        "试试问「先算 144除以12，再算根号144，比较一下」——"
        "模型会**连续调用两个工具**再回答，这是它的核心能力。"
    )

# 渲染历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if debug and msg.get("trace"):
            with st.expander(f"🔧 工具调用轨迹（{len(msg['trace'])} 次）"):
                for line in msg["trace"]:
                    st.code(line, language=None)

# 输入
if prompt := st.chat_input("输入你的问题…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("思考中…"):
            try:
                answer = agent.chat(prompt)
                trace = list(agent.trace)
            except Exception as e:  # noqa: BLE001 —— 界面兜底，不把 traceback 甩给用户
                # 只显示异常类型和简短信息，不打印任何对象
                # （早期版本这里有一句 `agent.client` 的调试代码，
                #   会把带 API key 的 client 对象渲染到界面上，已删除）
                answer = f"⚠️ 请求失败：`{type(e).__name__}`\n\n```\n{e}\n```"
                trace = []

        st.markdown(answer)
        if debug and trace:
            with st.expander(f"🔧 工具调用轨迹（{len(trace)} 次）"):
                for line in trace:
                    st.code(line, language=None)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "trace": trace}
    )
