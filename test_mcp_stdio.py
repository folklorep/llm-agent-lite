"""MCP 服务器的 stdio 通道测试：把服务器当子进程跑，走真实的输入输出。

============================================================================
和 mcp_server.py --selftest 的区别
============================================================================
    --selftest   直接调 handle() 函数，测的是**协议分发逻辑**
    本文件       把服务器当**独立进程**启动，喂 stdin、读 stdout，
                 测的是**传输层**：能不能正确读消息、写响应、两种帧格式是否都支持

两者都要有。历史上协议类实现的 bug 大多出在传输层：
  - 输出里混进了日志 → 客户端 JSON 解析失败（所以日志必须走 stderr）
  - 帧格式判断错了 → 客户端卡死等响应
  - 没有 flush → 响应被缓冲住发不出去

跑法：python test_mcp_stdio.py
"""
import json
import subprocess
import sys
from pathlib import Path

SERVER = Path(__file__).parent / "mcp_server.py"


def check(label: str, ok: bool, detail: object = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  <- {detail}" if not ok else ""))
    return ok


def run_lines(messages) -> list:
    """按"每行一个 JSON"格式喂消息，返回解析出的响应列表。"""
    stdin = "\n".join(json.dumps(m, ensure_ascii=False) for m in messages) + "\n"
    proc = subprocess.run(
        [sys.executable, str(SERVER)],
        input=stdin.encode("utf-8"),
        capture_output=True,
        cwd=str(SERVER.parent),
        timeout=60,
    )
    out = []
    for line in proc.stdout.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({"_raw": line})
    return out, proc


def run_content_length(messages) -> list:
    """按 Content-Length 头格式喂消息。"""
    parts = []
    for m in messages:
        body = json.dumps(m, ensure_ascii=False).encode("utf-8")
        parts.append(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
    stdin = b"".join(parts)

    proc = subprocess.run(
        [sys.executable, str(SERVER)],
        input=stdin,
        capture_output=True,
        cwd=str(SERVER.parent),
        timeout=60,
    )
    # 解析 Content-Length 响应
    raw = proc.stdout
    out = []
    while raw:
        idx = raw.find(b"\r\n\r\n")
        if idx < 0:
            break
        header = raw[:idx].decode("ascii", errors="replace")
        if not header.lower().startswith("content-length:"):
            break
        length = int(header.split(":", 1)[1].strip())
        body = raw[idx + 4: idx + 4 + length]
        try:
            out.append(json.loads(body.decode("utf-8")))
        except json.JSONDecodeError:
            out.append({"_raw": body.decode("utf-8", errors="replace")})
        raw = raw[idx + 4 + length:]
    return out, proc


def main() -> None:
    print("=" * 72)
    print("  MCP 服务器 stdio 通道测试")
    print("=" * 72)

    results = []

    # ---------------------------------------------------------- 行格式
    print("\n[1] 帧格式①：每行一个 JSON")
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "calculator", "arguments": {"expression": "128*37+15"}}},
    ]
    resp, proc = run_lines(messages)

    results.append(check("进程正常退出", proc.returncode == 0, proc.returncode))
    results.append(check("收到 3 条响应（通知不产生响应）", len(resp) == 3, len(resp)))
    results.append(check("没有混入非 JSON 的输出（日志走了 stderr）",
                         all("_raw" not in r for r in resp),
                         [r for r in resp if "_raw" in r]))
    if len(resp) >= 3:
        results.append(check("id 顺序正确", [r.get("id") for r in resp] == [1, 2, 3],
                             [r.get("id") for r in resp]))
        results.append(check("tools/list 有 4 个工具",
                             len(resp[1]["result"]["tools"]) == 4,
                             len(resp[1]["result"]["tools"])))
        results.append(check("calculator 算出 4751",
                             "4751" in resp[2]["result"]["content"][0]["text"],
                             resp[2]["result"]["content"][0]["text"]))
    # stderr 应该有启动日志
    err = proc.stderr.decode("utf-8", errors="replace")
    results.append(check("启动日志写在 stderr", "[mcp]" in err, err[:80]))

    # ---------------------------------------------------------- Content-Length 格式
    print("\n[2] 帧格式②：Content-Length 头")
    resp2, proc2 = run_content_length([
        {"jsonrpc": "2.0", "id": 11, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 12, "method": "tools/call",
         "params": {"name": "sqrt", "arguments": {"x": 144}}},
    ])
    results.append(check("进程正常退出", proc2.returncode == 0, proc2.returncode))
    results.append(check("收到 2 条响应", len(resp2) == 2, len(resp2)))
    results.append(check("响应也是 Content-Length 格式", all("_raw" not in r for r in resp2),
                         [r for r in resp2 if "_raw" in r]))
    if len(resp2) >= 2:
        results.append(check("sqrt(144) = 12",
                             "12" in resp2[1]["result"]["content"][0]["text"],
                             resp2[1]["result"]["content"][0]["text"]))

    # ---------------------------------------------------------- 脏输入
    print("\n[3] 脏输入不能让服务器卡死")
    stdin = 'not json at all\n{"jsonrpc":"2.0","id":21,"method":"tools/list"}\n'
    proc3 = subprocess.run([sys.executable, str(SERVER)],
                           input=stdin.encode("utf-8"), capture_output=True,
                           cwd=str(SERVER.parent), timeout=60)
    lines3 = [l for l in proc3.stdout.decode("utf-8", "replace").splitlines() if l.strip()]
    results.append(check("进程没卡死", proc3.returncode == 0, proc3.returncode))
    # 注意：json.dumps 输出的是 "code": -32700（冒号后有空格），
    # 断言时不要写成 "-32700"，那会匹配失败 —— 我第一次就踩了这个
    results.append(check("对坏消息回了 PARSE_ERROR（-32700）",
                         any("-32700" in l for l in lines3), lines3))
    results.append(check("坏消息之后的正常请求仍被处理",
                         any('"id": 21' in l for l in lines3), lines3))

    # ---------------------------------------------------------- 空输入
    print("\n[4] 空输入（客户端直接断开）")
    proc4 = subprocess.run([sys.executable, str(SERVER)], input=b"",
                           capture_output=True, cwd=str(SERVER.parent), timeout=30)
    results.append(check("立即退出、不挂起", proc4.returncode == 0, proc4.returncode))
    results.append(check("没有输出垃圾到 stdout", proc4.stdout.strip() == b"",
                         proc4.stdout[:60]))

    print("\n" + "=" * 72)
    if all(results):
        print(f"  [OK] {len(results)} 项全部通过")
    else:
        print(f"  [!] {results.count(False)} 项失败")
    print("=" * 72)


if __name__ == "__main__":
    main()
