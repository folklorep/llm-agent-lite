"""会话持久化层：把对话历史存进 SQLite。

============================================================================
它解决什么问题
============================================================================
改造前（api.py）：
    _SESSIONS: Dict[str, Agent] = {}     # 纯内存，进程重启就全丢

改造后（本模块）：
    历史写进 SQLite 文件，进程重启后能读回来 —— 用户可以接着上次聊。

============================================================================
表设计（两个决定，都是经过实测验证的）
============================================================================
决定 ①：一间表存【每条消息】，不是存整段对话
    理由：消息是天然独立的记录（有自己的 id、时间、角色）。
          拆成行之后，分页 / 统计 / 过滤 / 搜索都能直接用 SQL 搞定；
          存整段 JSON 的话，这些操作都得先把整块捞出来解析。

决定 ②：复杂结构（tool_calls）用 JSON 字符串存
    理由：json.dumps / json.loads 能【原样往返】——
          不用写"怎么拆成列、怎么再拼回来"这种麻烦代码。

最终是"混血"设计：
    role / content 拆成独立列   → 为了能查询、能过滤
    payload 存整条 message JSON → 为了原样还原 tool_calls

    ┌──────────────┬─────────┬─────────┬─────────┬──────────┐
    │ id (自增)     │ session │ role    │ content │ payload  │
    ├──────────────┼─────────┼─────────┼─────────┼──────────┤
    │ 1            │ abc123  │ system  │ 你是…    │ {整条JSON} │
    │ 2            │ abc123  │ user    │ 我叫小明 │ {整条JSON} │
    │ 3            │ abc123  │ assistant│ 你好…   │ {整条JSON} │
    └──────────────┴─────────┴─────────┴─────────┴──────────┘

    为什么 payload 要存【整条】而不是只存 tool_calls？
      → 因为读回来时可以直接 json.loads 还原，
        不依赖"这个字段该放哪一列"的映射逻辑。少一处能出错的地方。
"""
import json
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional

# 数据库文件位置：默认放项目下的 data/sessions.db
# 测试时可以传入 ":memory:" 让它完全在内存里跑（不落盘，跑完就没）
DEFAULT_DB = Path(__file__).parent / "data" / "sessions.db"

# 建表 SQL
SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at  TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT,
    payload     TEXT NOT NULL,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- 索引：按 session_id 查消息是最常用的操作，建索引能显著加速
CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, id);
"""


class SessionStore:
    """会话存储。所有数据库操作都走这个类。

    为什么单独封一个类（而不是在 api.py 里直接写 SQL）：
      ① 职责分离 —— api.py 管 HTTP，本模块管存储
      ② 可替换 —— 将来要换成 Redis / MySQL，只改这个类，
         api.py 的代码一行不用动（和检索层的思路一样）
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        """db_path 传 ":memory:" 就是内存库（测试用），传路径就是文件库。"""
        self.db_path = str(db_path or DEFAULT_DB)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False：允许在不同线程使用同一个连接。
        # ⚠️ 这需要自己保证并发安全 —— 所以我们外面还要加锁（见 _lock）
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # ---------------------------------------------------------------- 会话
    def ensure_session(self, session_id: str) -> None:
        """确保会话存在（不存在就建一条）。"""
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id) VALUES (?)",
                (session_id,),
            )
            self._conn.commit()

    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return row is not None

    def list_sessions(self) -> List[Dict]:
        """列出所有会话（含消息条数）—— 将来做"历史列表"用的就是这个。"""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT s.session_id, s.created_at, s.updated_at,
                       COUNT(m.id) AS msg_count
                FROM sessions s
                LEFT JOIN messages m ON m.session_id = s.session_id
                GROUP BY s.session_id
                ORDER BY s.updated_at DESC
                """
            ).fetchall()
        return [
            {"session_id": r[0], "created_at": r[1],
             "updated_at": r[2], "message_count": r[3]}
            for r in rows
        ]

    # ---------------------------------------------------------------- 消息
    def save_messages(self, session_id: str, messages: List[Dict]) -> int:
        """把一整段 history 写进去（覆盖写：先删旧的，再插新的）。

        为什么是"覆盖写"而不是"追加写"：
          因为 Agent 的 history 会被 _trim_history() 裁剪 ——
          它可能变短！如果只追加，数据库里的记录会和内存不一致。
          所以每次都当成"全量快照"重写，保证数据库 = 内存里的最新状态。

        ⚠️ 覆盖写的代价：每轮都重写全部消息。
           消息多的时候会慢。更优的做法是"增量写"（只写新增的），
           但那样要处理裁剪导致的数据不一致 —— 复杂度高。
           这是本实现的已知局限，在面试里应该主动说出来。
        """
        with self._lock:
            try:
                self._conn.execute("BEGIN")
                # 1) 先删掉这个会话的旧消息
                self._conn.execute(
                    "DELETE FROM messages WHERE session_id = ?", (session_id,)
                )
                # 2) 逐条插入新消息
                for m in messages:
                    self._conn.execute(
                        "INSERT INTO messages (session_id, role, content, payload)"
                        " VALUES (?, ?, ?, ?)",
                        (
                            session_id,
                            m.get("role", "user"),
                            m.get("content"),
                            # 整条 message 序列化成 JSON —— 这样 tool_calls
                            # 这种嵌套结构也能原样存下来
                            json.dumps(m, ensure_ascii=False),
                        ),
                    )
                # 3) 更新会话的最后活跃时间
                self._conn.execute(
                    "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP"
                    " WHERE session_id = ?",
                    (session_id,),
                )
                self._conn.commit()
                return len(messages)
            except Exception:
                self._conn.rollback()   # 出错回滚，不留半截数据
                raise

    def load_messages(self, session_id: str) -> List[Dict]:
        """按顺序读回整段 history。

        只读 payload 列就够了 —— 因为它是"整条 message 的 JSON"，
        json.loads 之后就是原始结构（含 tool_calls）。
        这就是决定② 的好处：读写对称，不用做字段映射。
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def count_messages(self, session_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
            ).fetchone()
        return int(row[0])

    # ---------------------------------------------------------------- 维护
    def delete_session(self, session_id: str) -> bool:
        """删掉一个会话及其全部消息。返回是否真的删到了东西。"""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
            self._conn.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ============================================================================
# 自检：不需要启动服务就能验证存储层
# ============================================================================
if __name__ == "__main__":
    import tempfile

    print("=" * 74)
    print("  SessionStore 自检（用临时文件，不碰项目的 data/sessions.db）")
    print("=" * 74)

    tmp = str(Path(tempfile.gettempdir()) / "store_selftest.db")
    Path(tmp).unlink(missing_ok=True)

    store = SessionStore(tmp)
    sid = "sess_demo"

    # ---- 1. 会话 ----
    store.ensure_session(sid)
    print(f"\n[1] 建会话 {sid!r}: 存在 = {store.session_exists(sid)}")

    # ---- 2. 存一段真实的 history（含 tool_calls）----
    history = [
        {"role": "system", "content": "你是一个严谨的 AI 助手"},
        {"role": "user", "content": "帮我算 2+3"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "calculator",
                             "arguments": '{"expression": "2+3"}'},
            }],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "2+3 = 5"},
        {"role": "assistant", "content": "结果是 5"},
    ]
    n = store.save_messages(sid, history)
    print(f"[2] 存了 {n} 条消息")

    # ---- 3. 读回并验证原样还原 ----
    back = store.load_messages(sid)
    print(f"[3] 读回 {len(back)} 条，逐条比对：")
    all_same = True
    for i, (a, b) in enumerate(zip(history, back)):
        same = a == b
        all_same = all_same and same
        print(f"      [{i}] {a['role']:<9} 一致={same}")
    print(f"   ⭐ 整体原样还原 = {all_same}")

    # ---- 4. 覆盖写（模拟 history 被裁剪后变短）----
    print(f"\n[4] 模拟裁剪：history 从 5 条变 3 条，再存一次")
    trimmed = history[:1] + history[-2:]
    store.save_messages(sid, trimmed)
    print(f"      数据库里现在 {store.count_messages(sid)} 条（应该是 3）")

    # ---- 5. 会话列表 ----
    print(f"\n[5] 会话列表: {store.list_sessions()}")

    # ---- 6. ⭐ 关掉再打开，数据还在吗 ----
    store.close()
    store2 = SessionStore(tmp)
    back2 = store2.load_messages(sid)
    print(f"\n[6] 关闭后重新打开：读出 {len(back2)} 条")
    print(f"      第一条: {back2[0]['role']} / {back2[0]['content'][:20]}")
    print(f"   ⭐ 持久化生效 = {len(back2) == len(trimmed)}")

    # ---- 7. 删除 ----
    print(f"\n[7] 删会话: {store2.delete_session(sid)}")
    print(f"      删后还存在吗: {store2.session_exists(sid)}")
    store2.close()

    Path(tmp).unlink(missing_ok=True)
    print("\n" + "=" * 74)
    print("  自检完成（临时库已删除）")
    print("=" * 74)
