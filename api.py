"""FastAPI 服务化：把 Agent 暴露成 HTTP 接口。

本地跑：
    python -m pip install fastapi uvicorn
    python -m uvicorn api:app --reload --port 8000

测试：
    curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" ^
         -d "{\"message\": \"帮我算 128乘37加15\"}"

接口文档（FastAPI 自动生成，可以直接给面试官看）：
    http://127.0.0.1:8000/docs

设计说明（面试会问）：
    - 会话状态【持久化到 SQLite】（storage.py）——
      进程重启后会话不丢，用户可以接着上次聊。
      （改造前存在内存 dict 里，一重启就全丢。）
    - 内存里仍缓存 Agent 实例（避免每次请求都重新加载），
      但"真相"在数据库里：缓存丢了能从库里重建。
    - 加锁是因为 Agent.history 不是线程安全的：两个请求同时改会串。
      真实生产应该用队列或让每个 session 独占一个实例。
    - 流式接口用 SSE（Server-Sent Events），比 WebSocket 轻，适合单向推送。
"""
import threading
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent import Agent
from storage import SessionStore

app = FastAPI(
    title="LLM Agent Lite API",
    description="多工具智能体服务：多轮对话 + Function Calling 自动循环 + RAG 检索",
    version="0.1.0",
)

# 持久化层：会话历史存在 SQLite 里（进程重启不丢）
#
# 为什么做成"可注入"而不是直接写死 _STORE = SessionStore()？
#   因为写死之后【测试没法隔离】—— 测试里创建的上百个会话会真的写进
#   data/sessions.db，把它污染成垃圾堆（我第一版就是这样，库里积了 110 个会话）。
#   做成可注入后，测试可以换成 db_path=":memory:"，跑完就没了。
#
# 这和要求 client 可注入是同一个思路：依赖由外部提供，不在模块里写死。
_STORE: Optional[SessionStore] = None


def configure_store(db_path: Optional[str] = None) -> SessionStore:
    """设置/替换存储层。传 ":memory:" 就是内存库（测试用）。

    生产不调用这个函数 —— 第一次访问时自动用默认路径创建。
    """
    global _STORE, _SESSIONS
    if _STORE is not None:
        try:
            _STORE.close()
        except Exception:  # noqa: BLE001
            pass
    _STORE = SessionStore(db_path)
    _SESSIONS = {}          # 换存储就清空内存缓存，避免串数据
    return _STORE


def _store() -> SessionStore:
    """取存储层（懒创建）。"""
    global _STORE
    if _STORE is None:
        _STORE = SessionStore()
    return _STORE


# session_id -> Agent（内存缓存，避免每次请求都从库里重建）
# 注意：这里的缓存"可以丢" —— 丢了就从数据库重建，数据本身不丢。
_SESSIONS: Dict[str, Agent] = {}
_LOCK = threading.Lock()
MAX_SESSIONS = 100          # 防止内存缓存无限增长


# ------------------------------------------------------------------ 模型
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="用户输入")
    session_id: str = Field(default="", description="留空则自动创建新会话")


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    tool_calls: List[str] = Field(default_factory=list, description="本轮工具调用轨迹")


# ------------------------------------------------------------------ 工具函数
def _build_agent(session_id: str) -> Agent:
    """根据库里已有的历史，构造一个 Agent。

    ⚠️ 这里必须捕获 Agent() 的构造异常 —— 踩过的坑：
       我最初把 `except RuntimeError` 写在 /chat 路由里，但 Agent() 是在
       **本函数内部**构造的，RuntimeError 在到达路由之前就抛出去了，
       于是"缺 API key 返回 503"这个设计**实际上是失效的，会返回 500**。
       这个 bug 是 test_imports.py 断言异常类型时抓出来的 ——
       只断言"抛异常了"会漏掉它，必须断言**抛的是哪个异常**。

    恢复逻辑：如果库里存过这个会话的消息，就用它们初始化 history；
    否则从零开始（只有 system 提示词）。
    """
    try:
        store = _store()
        saved = store.load_messages(session_id)
        return Agent(
            history=saved if saved else None,
            # 每轮结束把 history 写回库 —— 用回调，Agent 不需要知道存储细节
            on_turn_end=lambda hist: store.save_messages(session_id, hist),
        )
    except RuntimeError as e:
        # 配置错误 → 503（服务暂时不可用），比 500 语义更准：
        # 这不是代码崩了，是服务没配好，重试也没用。
        raise HTTPException(status_code=503, detail=str(e)) from e


def _get_agent(session_id: str) -> tuple[str, Agent]:
    """取或建会话。返回 (session_id, agent)。"""
    with _LOCK:
        if not session_id:
            session_id = uuid.uuid4().hex[:12]
        if session_id not in _SESSIONS:
            agent = _build_agent(session_id)
            # 确保 sessions 表里有这条记录
            _store().ensure_session(session_id)
            if len(_SESSIONS) >= MAX_SESSIONS:
                # 简易淘汰：只从【内存缓存】里踢掉，数据库里的数据不删
                _SESSIONS.pop(next(iter(_SESSIONS)))
            _SESSIONS[session_id] = agent
        return session_id, _SESSIONS[session_id]


# ------------------------------------------------------------------ 路由
@app.get("/health")
def health() -> dict:
    """健康检查。部署到云平台时平台会探这个接口。"""
    return {
        "status": "ok",
        "sessions_cached": len(_SESSIONS),      # 内存缓存数
        "sessions_stored": len(_store().list_sessions()),   # 数据库里的总数
    }



@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """一问一答。模型可能在本轮内部调用多次工具。

    注：缺 API key 的 503 由 _get_agent 内部抛出，这里不需要再兜一次
    （原来这里有一层 `except RuntimeError`，是失效的，已删除 ——
      详见 _get_agent 的注释）。
    """
    session_id, agent = _get_agent(req.session_id)

    with _LOCK:                       # Agent.history 不是线程安全的
        try:
            reply = agent.chat(req.message)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(
                status_code=502, detail=f"上游模型调用失败：{type(e).__name__}: {e}"
            ) from e
        trace = list(agent.trace)

    return ChatResponse(reply=reply, session_id=session_id, tool_calls=trace)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """流式返回（SSE）。前端逐字显示，首字延迟低。

    ⚠️ 已知局限（面试要主动说）：
      真实的多轮对话要维护 history（用户消息 + 助手回复都要落进会话）。
      但流式响应是**逐块吐出**的，拼完整答案需要等流结束；
      而生成器一旦返回就不能再往 history 追加（会打乱消息顺序）。
      所以这里**只做单轮流式展示，不维护多轮上下文**。
      要支持"流式 + 多轮"，正确做法是：边收边拼 → 流结束后
      用拼好的完整回复调用一次 history 落库，而不是在生成器里改。
    """
    session_id, agent = _get_agent(req.session_id)

    payload = agent.history + [{"role": "user", "content": req.message}]

    def event_source():
        yield f"data: [session]{session_id}\n\n"
        try:
            for chunk in agent.client.chat_stream(payload):
                # SSE 的 data 字段不能含裸换行，替换掉
                yield f"data: {chunk.replace(chr(10), ' ')}\n\n"
        except Exception as e:  # noqa: BLE001
            yield f"data: [error]{type(e).__name__}: {e}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.get("/sessions")
def list_sessions() -> dict:
    """列出所有会话（含消息条数）—— 做"历史对话列表"用的就是它。"""
    return {"sessions": _store().list_sessions()}


@app.delete("/session/{session_id}")
def clear_session(session_id: str) -> dict:
    """清空某个会话的对话历史。

    ⚠️ 注意这里查的是【数据库】而不是内存缓存 ——
       因为会话可能已经不在缓存里（被淘汰了，或服务重启过），
       但它仍然在数据库里，应该能清。
    """
    if not _store().session_exists(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    with _LOCK:
        # 清内存里那份（如果还在缓存里）
        agent = _SESSIONS.get(session_id)
        if agent is not None:
            agent.reset()          # 这会把 history 缩到只剩 system，并通过回调写库
        # 再从库里彻底删掉
        _store().delete_session(session_id)
        _SESSIONS.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
