"""导入链路验证：用桩模块顶替未安装的第三方库。

为什么需要这个（面试点）：
    api.py 依赖 fastapi + pydantic，ui_app.py 依赖 streamlit。
    开发/CI 环境不一定装了它们 —— 直接 import 会在第一行就炸，
    于是"语法对不对、导入路径对不对、路由注册对不对"全测不出来。
    把缺失的库塞进 sys.modules，就能在**没装依赖**的情况下验证这些。
    这跟用假模型测 Agent 是同一个思路：**只替换不可控的外部依赖，其余走真实代码。**

局限（必须说清）：
    这只验证「导入 + 装饰器注册 + 会话管理逻辑」，
    **不能替代真实运行**。路由真的能响应请求、Streamlit 真的能渲染，
    仍然需要装上依赖后实测。

跑法：python test_imports.py
"""
import sys
import types
from typing import Any


# ============================================================ 桩模块
def _stub(name: str, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


class _SessionState(dict):
    """假的 st.session_state。

    必须是真的 dict 子类，因为 ui_app.py 会做
    `if "messages" not in st.session_state` 这种成员判断 ——
    用普通假对象会报 `argument of type '_Ctx' is not iterable`。
    这个报错本身就说明：桩要模拟**被用到的协议**，不能只模拟"能被访问"。
    """


class _Columns(list):
    """假的 st.columns() 返回值。

    真实 Streamlit 的 st.columns(n) 返回一个**可下标的列表**，元素是列对象。
    用普通 _Ctx 会在 `cols[0].button(...)` 处报
    "'_Ctx' object is not subscriptable" ——
    这个报错再次说明：桩要模拟**被用到的协议**（这里是"可下标"），
    不能只模拟"属性能被访问"。
    """

    def __getattr__(self, item):
        return _Ctx()


class _Ctx:
    """既能被调用、也能用在 with 语句、还能链式取属性的万能假对象。"""

    def __call__(self, *a, **k):
        # st.columns(n) 要返回可下标的列列表
        if len(a) == 1 and isinstance(a[0], int) and not k:
            return _Columns([_Ctx() for _ in range(a[0])])
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, item):
        if item == "session_state":
            return _Ctx.session_state
        return self


class _StreamlitModule(types.ModuleType):
    """Streamlit 的 API 太杂，用 catch-all 代替逐个打桩。"""

    def __getattr__(self, item):
        if item == "session_state":
            return _Ctx.session_state
        return _Ctx()


_Ctx.session_state = _SessionState()


def install_stubs() -> None:
    # ---------------- fastapi ----------------
    class HTTPException(Exception):
        def __init__(self, status_code=500, detail=""):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

    class FastAPI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.routes = []

        def _register(self, method, path):
            def decorator(func):
                self.routes.append((method, path, func.__name__))
                return func
            return decorator

        def get(self, path, **kw):
            return self._register("GET", path)

        def post(self, path, **kw):
            return self._register("POST", path)

        def delete(self, path, **kw):
            return self._register("DELETE", path)

    class StreamingResponse:
        def __init__(self, content, media_type=None):
            self.content = content
            self.media_type = media_type

    _stub("fastapi", FastAPI=FastAPI, HTTPException=HTTPException)
    _stub("fastapi.responses", StreamingResponse=StreamingResponse)

    # ---------------- pydantic ----------------
    class FieldInfo:
        def __init__(self, default=None, **kwargs):
            self.default = default
            self.kwargs = kwargs

    def Field(default=None, **kwargs):  # noqa: N802 —— 模仿 pydantic 命名
        return FieldInfo(default, **kwargs)

    class BaseModel:
        """简化版 pydantic，必须模拟真实语义，否则测的是桩不是被测代码：
          ① 类属性上的普通值 = 字段默认值
          ② 类属性是 Field(...) 时取它的 default
          ③ default_factory 每次调用一次（不能共享同一个可变对象）
        """

        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            fields = {}
            for klass in reversed(cls.__mro__):
                for key, val in vars(klass).items():
                    if key.startswith("_") or callable(val):
                        continue
                    fields[key] = val
            cls.model_fields = fields

        def __init__(self, **data):
            for name, decl in type(self).model_fields.items():
                if name in data:
                    setattr(self, name, data[name])
                elif isinstance(decl, FieldInfo):
                    factory = decl.kwargs.get("default_factory")
                    setattr(self, name, factory() if factory else decl.default)
                else:
                    setattr(self, name, decl)
            for k, v in data.items():
                if k not in type(self).model_fields:
                    setattr(self, k, v)

    _stub("pydantic", BaseModel=BaseModel, Field=Field)
    _stub("uvicorn", run=lambda *a, **k: None)
    sys.modules["streamlit"] = _StreamlitModule("streamlit")


# ============================================================ 断言
def check(label: str, condition: bool, detail: Any = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}"
          + (f"  <- {detail}" if not condition else ""))
    if not condition:
        raise AssertionError(f"{label} 断言失败 {detail}")


# ============================================================ 测试
def test_routes(api) -> None:
    print("\n[1] api.py 导入与路由注册")
    check("app 对象已创建", hasattr(api, "app"))
    check("标题正确", api.app.kwargs.get("title") == "LLM Agent Lite API")

    routes = {(m, p) for m, p, _ in api.app.routes}
    for method, path in [
        ("GET", "/health"),
        ("POST", "/chat"),
        ("POST", "/chat/stream"),
        ("DELETE", "/session/{session_id}"),
    ]:
        check(f"路由 {method} {path} 已注册", (method, path) in routes, sorted(routes))

    check("/health 可直接调用", api.health()["status"] == "ok")


def test_models(api) -> None:
    print("\n[2] 请求/响应模型默认值")
    check("ChatRequest.session_id 默认为空串",
          api.ChatRequest(message="hi").session_id == "")
    check("ChatRequest.message 被正确赋值",
          api.ChatRequest(message="hi").message == "hi")
    check("ChatResponse.tool_calls 默认空列表",
          api.ChatResponse(reply="x", session_id="s").tool_calls == [])
    # 两次构造必须是各自独立的列表（default_factory 语义）
    r1 = api.ChatResponse(reply="a", session_id="s1")
    r2 = api.ChatResponse(reply="b", session_id="s2")
    r1.tool_calls.append("x")
    check("tool_calls 不是共享的可变对象", r2.tool_calls == [], r2.tool_calls)


def test_missing_key_returns_503(api) -> None:
    """缺 API key 时应该返回 HTTP 503，而不是把裸 RuntimeError 抛给框架。

    ⚠️ 踩坑记录：这里必须改**类属性**而不是实例属性。
       Config 的字段全是类属性，validate() 又是 @classmethod，
       它读的是 cls.API_KEY。写 `config.config.API_KEY = "..."` 只会创建
       一个同名**实例属性**把类属性遮住 —— 实例读得到，类方法读不到。
       这个坑我踩过一次，所以在这里留注释。
    """
    print("\n[3] 缺少 API key 时的错误处理")
    import config

    real_agent = api.Agent
    saved_class_key = config.Config.API_KEY
    config.Config.API_KEY = ""          # 改类属性，validate() 才看得到
    try:
        api._get_agent("no_key_session")
        check("缺 key 时抛 HTTPException", False, "没有抛异常")
    except api.HTTPException as e:
        check("抛的是 HTTPException(503)", e.status_code == 503, e.status_code)
    except Exception as e:  # noqa: BLE001
        import fastapi as _f
        print(f"     [诊断] 实际异常类型 = {type(e).__mro__[:3]}")
        print(f"     [诊断] api.HTTPException = {api.HTTPException}")
        print(f"     [诊断] sys.modules['fastapi'].HTTPException = {_f.HTTPException}")
        check("缺 key 时抛 HTTPException", False, f"抛的是 {type(e).__name__}")
    finally:
        config.Config.API_KEY = saved_class_key
        api.Agent = real_agent


def test_sessions(api) -> None:
    """会话管理逻辑 —— 用假 Agent 隔离掉网络与配置依赖。"""
    print("\n[4] 会话管理")
    created: list = []

    class FakeAgent:
        def __init__(self):
            self.history = []
            created.append(self)

        def reset(self):
            self.history = []

    import config
    saved_class_key = config.Config.API_KEY
    config.Config.API_KEY = "sk-fake"        # 让真 Agent 的校验能过
    real_agent = api.Agent
    api.Agent = FakeAgent                    # 注入假 Agent，不碰网络
    try:
        sid1, a1 = api._get_agent("")
        check("空 session_id 自动生成 12 位", len(sid1) == 12, sid1)
        check("已创建 1 个 Agent", len(created) == 1, len(created))

        sid2, a2 = api._get_agent(sid1)
        check("同一 session_id 复用同一 Agent", a2 is a1 and sid2 == sid1)
        check("没有重复创建", len(created) == 1, len(created))

        sid3, a3 = api._get_agent("")
        check("不同 session_id 得到不同 Agent", a3 is not a1)
        check("新 session_id 与旧的不同的", sid3 != sid1)

        # 容量上限与淘汰
        api._SESSIONS.clear()
        created.clear()
        for i in range(api.MAX_SESSIONS + 5):
            api._get_agent(f"flood{i}")
        check(f"会话数被限制在 {api.MAX_SESSIONS}",
              len(api._SESSIONS) == api.MAX_SESSIONS, len(api._SESSIONS))

        # 淘汰后保留的是最新的那批
        check("保留了最新的会话", f"flood{api.MAX_SESSIONS + 4}" in api._SESSIONS)

        api._SESSIONS.clear()
    finally:
        api.Agent = real_agent
        config.Config.API_KEY = saved_class_key


def test_ui_app() -> None:
    print("\n[5] ui_app.py 导入（Streamlit 桩）")
    import ui_app

    check("模块已导入", ui_app is not None)
    check("没有把 api 的会话表带进来", not hasattr(ui_app, "_SESSIONS"))


def main() -> None:
    print("=" * 62)
    print("  导入链路验证（桩模块顶替 fastapi / pydantic / streamlit）")
    print("=" * 62)

    install_stubs()
    import api

    test_routes(api)
    test_models(api)
    test_missing_key_returns_503(api)
    test_sessions(api)
    test_ui_app()

    print("\n" + "=" * 62)
    print("  [OK] 导入链路、路由注册、模型默认值、会话管理全部通过")
    print("  [!] 这不等于运行时已验证 —— 路由真正响应请求、")
    print("      Streamlit 真正渲染，仍需装依赖后实测。")
    print("=" * 62)


if __name__ == "__main__":
    main()
