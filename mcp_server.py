"""MCP 服务器：把 tools.py 里的工具按 MCP 协议暴露出去。

============================================================================
这个文件在做什么（一句话）
============================================================================
把项目里的 4 个工具，按 **MCP（Model Context Protocol）** 标准暴露成一个服务器。
效果：任何支持 MCP 的客户端（Claude Desktop / Cursor / 其他 Agent）
      都能直接调用这些工具，**不需要改客户端的任何代码**。

============================================================================
为什么要写它（而不是直接用 Function Calling）
============================================================================
Function Calling 是"每家一套写法"：
    OpenAI 一套格式、Anthropic 一套、通义/智谱/豆包又各自不同。
后果是：你写一次工具，换个平台就要重写定义；别人写好的工具你也用不了。

**MCP 就是把这件事标准化** —— 类比：Function Calling 是各家自己的充电口，
MCP 是 USB-C。服务器按协议实现一次，所有客户端都能接。

============================================================================
协议要实现的三个方法（MCP 的核心）
============================================================================
    initialize   握手：交换协议版本、声明能力
    tools/list   客户端问"你有哪些工具"
    tools/call   客户端说"帮我调用某个工具"

另外还有一个通知（不需要回复）：
    notifications/initialized   客户端告知"我初始化完了"

底层是 **JSON-RPC 2.0**，消息格式长这样：
    请求  {"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}
    响应  {"jsonrpc":"2.0","id":1,"result":{...}}
    错误  {"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"..."}}

============================================================================
⚠️ 一个实际会踩的坑：stdio 的帧格式有两种
============================================================================
MCP 的 stdio 传输在不同版本里用过不同的分帧方式：

  ① **每行一个 JSON**（当前规范，主流客户端）
         {"jsonrpc":"2.0",...}\n
         {"jsonrpc":"2.0",...}\n

  ② **Content-Length 头**（早期规范，仿 LSP）
         Content-Length: 123\r\n\r\n{"jsonrpc":"2.0",...}

本实现**两种都支持**：读到 "Content-Length" 开头就走 ②，否则按 ① 逐行解析。
不这么做的话，遇到用旧格式的客户端会直接卡死 —— 而且报错信息很难懂。

============================================================================
怎么用
============================================================================
自检（不需要客户端）：
    python mcp_server.py --selftest

手动喂一条请求试试：
    echo {"jsonrpc":"2.0","id":1,"method":"tools/list"} | python mcp_server.py

接到 Claude Desktop：
    见 README 的「接入 MCP 客户端」一节，或跑 python mcp_server.py --config
"""
import json
import sys
from typing import Any, Dict, List, Optional

# 协议版本：客户端会带自己的版本过来，不一致时我们回自己能支持的最高版本
PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "llm-agent-lite"
SERVER_VERSION = "0.1.0"

# JSON-RPC 标准错误码
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


# ============================================================ 工具适配层
def _list_tools() -> List[Dict[str, Any]]:
    """把项目里的工具转成 MCP 的 tools/list 结果。

    注意字段名的差异（这是接协议时最容易错的地方）：
        项目内部（OpenAI 格式）：{"type":"function","function":{"name":..,"parameters":..}}
        MCP 格式：              {"name":..,"description":..,"inputSchema":..}

    也就是说：**去掉外面那层 function 包装，把 parameters 改名成 inputSchema。**
    """
    from tools import get_tool_schemas

    out = []
    for s in get_tool_schemas():
        fn = s.get("function", {})
        out.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "inputSchema": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return out


def _call_tool(name: str, arguments: Any) -> Dict[str, Any]:
    """执行工具，把结果包成 MCP 的 tools/call 结果格式。

    MCP 要求返回 content 数组，元素形如 {"type":"text","text":"..."}。
    项目里的 run_tool 已经保证返回字符串，这里只做包装。
    """
    from tools import run_tool

    # arguments 应该是对象；有的客户端会传字符串形式的 JSON
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}

    try:
        text = run_tool(name, json.dumps(arguments, ensure_ascii=False))
    except Exception as e:  # noqa: BLE001 —— 工具异常要作为结果回传，而不是让服务器崩
        text = f"错误：工具执行失败（{type(e).__name__}: {e}）"

    return {"content": [{"type": "text", "text": text}], "isError": False}


# ============================================================ 协议分发
def handle(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """处理一条 JSON-RPC 请求，返回响应（通知类请求返回 None）。"""
    method = request.get("method")
    req_id = request.get("id")
    params = request.get("params") or {}

    # ---- 通知：不需要回复 ----
    if method == "notifications/initialized" or (req_id is None and method and method.startswith("notifications/")):
        return None

    # ---- initialize：握手 ----
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        }

    # ---- tools/list：返回工具清单 ----
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": _list_tools()}}

    # ---- tools/call：执行工具 ----
    if method == "tools/call":
        name = params.get("name")
        if not name:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": INVALID_PARAMS, "message": "缺少参数 name"},
            }
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": _call_tool(name, params.get("arguments")),
        }

    # ---- 其他方法：按协议返回"方法不存在" ----
    # ping 是常见的健康检查，顺手支持一下
    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": METHOD_NOT_FOUND, "message": f"不支持的方法：{method}"},
    }


# ============================================================ stdio 传输
def _emit(obj: Dict[str, Any], framing: str = "line") -> None:
    """把响应写出去。framing 要和收到的格式保持一致。"""
    payload = json.dumps(obj, ensure_ascii=False)
    if framing == "content-length":
        data = payload.encode("utf-8")
        sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii"))
        sys.stdout.buffer.write(data)
    else:
        sys.stdout.write(payload + "\n")
    sys.stdout.flush()


def _read_messages():
    """从 stdin 连续读消息，同时支持两种帧格式。"""
    stream = sys.stdin.buffer
    while True:
        first = stream.readline()
        if not first:
            return                      # EOF，客户端关掉了连接
        line = first.decode("utf-8", errors="replace").strip()
        if not line:
            continue

        # ---- 格式 ②：Content-Length 头 ----
        if line.lower().startswith("content-length:"):
            try:
                length = int(line.split(":", 1)[1].strip())
            except ValueError:
                continue
            # 跳过紧随其后的空行（可能有多余的 \r\n）
            while True:
                sep = stream.readline()
                if not sep or sep.strip() == b"":
                    break
            body = stream.read(length)
            try:
                yield json.loads(body.decode("utf-8")), "content-length"
            except json.JSONDecodeError:
                yield None, "content-length"
            continue

        # ---- 格式 ①：每行一个 JSON ----
        try:
            yield json.loads(line), "line"
        except json.JSONDecodeError:
            yield None, "line"


def serve() -> None:
    """主循环：读请求 → 分发 → 写响应。"""
    # 注意：日志必须写到 stderr！stdout 是协议通道，混进任何非 JSON 内容都会破坏协议
    print(f"[mcp] {SERVER_NAME} 已启动，等待客户端连接…", file=sys.stderr)
    for request, framing in _read_messages():
        if request is None:
            _emit(
                {"jsonrpc": "2.0", "id": None,
                 "error": {"code": PARSE_ERROR, "message": "JSON 解析失败"}},
                framing,
            )
            continue
        response = handle(request)
        if response is not None:
            _emit(response, framing)
    print("[mcp] 客户端断开，退出", file=sys.stderr)


# ============================================================ 自检与辅助
def _selftest() -> None:
    """不需要客户端就能验证协议实现是否正确。"""
    print("=" * 70)
    print("  MCP 服务器自检")
    print("=" * 70)

    checks = []

    def check(label, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  <- {detail}" if not ok else ""))
        checks.append(ok)
        return ok

    # ---- initialize ----
    print("\n[1] initialize 握手")
    resp = handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                   "params": {"protocolVersion": PROTOCOL_VERSION,
                              "clientInfo": {"name": "test", "version": "1"}}})
    check("返回 jsonrpc 2.0", resp.get("jsonrpc") == "2.0")
    check("回显请求 id", resp.get("id") == 1)
    check("带 protocolVersion", "protocolVersion" in resp["result"])
    check("声明 tools 能力", "tools" in resp["result"]["capabilities"])
    print(f"      serverInfo = {resp['result']['serverInfo']}")

    # ---- tools/list ----
    print("\n[2] tools/list 工具清单")
    resp = handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools_list = resp["result"]["tools"]
    check("返回了工具数组", isinstance(tools_list, list) and len(tools_list) > 0, len(tools_list))
    names = [t["name"] for t in tools_list]
    check("字段名符合 MCP（inputSchema 而非 parameters）",
          all("inputSchema" in t for t in tools_list))
    check("不残留 OpenAI 的 function 包装",
          all("function" not in t for t in tools_list))
    print(f"      工具：{names}")

    # ---- tools/call ----
    print("\n[3] tools/call 执行工具")
    resp = handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                   "params": {"name": "calculator",
                              "arguments": {"expression": "128*37+15"}}})
    content = resp["result"]["content"]
    check("返回 content 数组", isinstance(content, list) and content, content)
    check("元素是 text 类型", content[0]["type"] == "text")
    check("算出正确结果", "4751" in content[0]["text"], content[0]["text"])

    # ---- 错误处理 ----
    print("\n[4] 错误处理")
    resp = handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                   "params": {"arguments": {}}})
    check("缺 name 时返回 error", "error" in resp, resp)
    check("错误码是 INVALID_PARAMS", resp["error"]["code"] == INVALID_PARAMS)

    resp = handle({"jsonrpc": "2.0", "id": 5, "method": "totally/unknown"})
    check("未知方法返回 METHOD_NOT_FOUND",
          resp["error"]["code"] == METHOD_NOT_FOUND, resp.get("error"))

    # ---- 工具内部报错不能崩服务器 ----
    resp = handle({"jsonrpc": "2.0", "id": 6, "method": "tools/call",
                   "params": {"name": "sqrt", "arguments": {"x": -1}}})
    check("工具报错时服务器不崩", "result" in resp, resp)
    check("错误内容作为结果回传", "错误" in resp["result"]["content"][0]["text"],
          resp["result"]["content"][0]["text"])

    # ---- 通知不回复 ----
    print("\n[5] 通知类消息不产生响应")
    resp = handle({"jsonrpc": "2.0", "method": "notifications/initialized"})
    check("notifications 返回 None", resp is None, resp)

    print("\n" + "=" * 70)
    if all(checks):
        print(f"  [OK] {len(checks)} 项全部通过")
    else:
        print(f"  [!] {sum(1 for c in checks if not c)} 项失败")
    print("=" * 70)


def _print_config() -> None:
    """打印接入 Claude Desktop 用的配置片段。"""
    from pathlib import Path

    server_path = Path(__file__).resolve()
    python_path = sys.executable
    cfg = {
        "mcpServers": {
            "llm-agent-lite": {
                "command": python_path,
                "args": [str(server_path)],
            }
        }
    }
    print("把下面这段合并进 Claude Desktop 的配置文件：")
    print()
    print(json.dumps(cfg, indent=2, ensure_ascii=False))
    print()
    print("配置文件位置（Windows）：")
    print(r"  %APPDATA%\Claude\claude_desktop_config.json")
    print()
    print("改完要完全退出 Claude Desktop 再重开（不是关窗口，要退出进程）。")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    elif "--config" in sys.argv:
        _print_config()
    else:
        serve()
