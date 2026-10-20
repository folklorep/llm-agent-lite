"""实验 13 第 1 步：SQLite 入门（不碰项目，先玩明白）。

跑法：
    python lab_sqlite_basics.py

============================================================================
先搞清一件事：SQLite 是什么，和其他数据库什么区别
============================================================================
    SQLite      一个【文件】就是一个数据库。Python 内置，不用装、不用起服务。
    MySQL       要单独安装、启动服务、配账号密码，通过网络连接。适合多机共享。
    Redis       内存数据库，极快，但默认不落盘。常用来做缓存/会话（所以 api.py
                的注释里说"真实生产应该放 Redis"）。

    本实验用 SQLite 的理由：
      ① Python 内置，不需要装任何东西（避免又卡在环境上）
      ② 数据存文件，重启后还在 —— 正好解决"会话重启就丢"的问题
      ③ SQL 语法和 MySQL 基本一样，学到的能迁移

============================================================================
本脚本演示五件事
============================================================================
    ① 建表（CREATE TABLE）—— 定义"数据长什么样"
    ② 插入（INSERT）—— 写数据
    ③ 查询（SELECT）—— 读数据
    ④ 更新/删除（UPDATE / DELETE）
    ⑤ ⭐ 关闭再打开，数据还在吗？（这就是"持久化"）
"""
import os
import sqlite3
import tempfile

DB_PATH = os.path.join(tempfile.gettempdir(), "lab_sqlite_demo.db")


def sep(t: str) -> None:
    print("\n" + "=" * 74)
    print(f"  {t}")
    print("=" * 74)


def main() -> None:
    # 每次从干净的库开始
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    sep("① 建表：定义「数据长什么样」")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS demo (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT    NOT NULL,
            score    INTEGER,
            created  TEXT    DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    print(f"""
  执行的 SQL：
      CREATE TABLE demo (
          id      INTEGER PRIMARY KEY AUTOINCREMENT,   -- 自增主键
          name    TEXT NOT NULL,                       -- 文本，不能为空
          score   INTEGER,                             -- 整数，可空
          created TEXT DEFAULT CURRENT_TIMESTAMP       -- 默认填当前时间
      )

  ── 逐项解释 ──
      INTEGER PRIMARY KEY AUTOINCREMENT
          → 每一行自动有个编号（1, 2, 3...），用来唯一标识这行
      TEXT / INTEGER
          → 这一列存什么类型
      NOT NULL
          → 这一列不能空着
      DEFAULT CURRENT_TIMESTAMP
          → 不填的话自动填当前时间

  库文件：{DB_PATH}
""")

    # ---------------------------------------------------------------- ②
    sep("② 插入：写数据")

    # ⚠️ 看这两行的区别 —— 这是 SQL 注入的由来
    print("""
  ── 两种写法，一定要用第二种 ──

  ❌ 字符串拼接（危险）：
      cur.execute(f"INSERT INTO demo (name, score) VALUES ('{name}', {score})")
      如果 name 里含有单引号，SQL 就被破坏了 —— 这叫 SQL 注入

  ✅ 参数化（安全）：
      cur.execute("INSERT INTO demo (name, score) VALUES (?, ?)", (name, score))
      用 ? 占位，数据库自己处理转义
""")

    rows = [("小明", 88), ("小红", 92), ("小刚", 77)]
    for name, score in rows:
        cur.execute("INSERT INTO demo (name, score) VALUES (?, ?)", (name, score))
    conn.commit()
    print(f"  已插入 {len(rows)} 行：{rows}")

    # ---------------------------------------------------------------- ③
    sep("③ 查询：读数据")

    print("  [查询全部]  SELECT * FROM demo")
    for row in cur.execute("SELECT id, name, score, created FROM demo"):
        print(f"      id={row[0]}  name={row[1]!r}  score={row[2]}  created={row[3]}")

    print("\n  [带条件]  SELECT * FROM demo WHERE score > 80")
    for row in cur.execute("SELECT name, score FROM demo WHERE score > 80"):
        print(f"      {row[0]} -> {row[1]}")

    print("\n  [排序]    SELECT * FROM demo ORDER BY score DESC")
    for row in cur.execute("SELECT name, score FROM demo ORDER BY score DESC"):
        print(f"      {row[0]} -> {row[1]}")

    print("\n  [聚合]    SELECT COUNT(*), AVG(score) FROM demo")
    row = cur.execute("SELECT COUNT(*), AVG(score) FROM demo").fetchone()
    print(f"      共 {row[0]} 行，平均分 {row[1]:.1f}")

    # ---------------------------------------------------------------- ④
    sep("④ 更新 / 删除")

    cur.execute("UPDATE demo SET score = ? WHERE name = ?", (95, "小刚"))
    conn.commit()
    row = cur.execute("SELECT score FROM demo WHERE name = '小刚'").fetchone()
    print(f"  UPDATE 之后，小刚的分数 = {row[0]}")

    cur.execute("DELETE FROM demo WHERE name = ?", ("小红",))
    conn.commit()
    n = cur.execute("SELECT COUNT(*) FROM demo").fetchone()[0]
    print(f"  DELETE 之后，还剩下 {n} 行")

    # ---------------------------------------------------------------- ⑤
    sep("⑤ ⭐ 关闭连接，再重新打开 —— 数据还在吗？")

    conn.close()
    print("  已关闭连接（模拟「程序退出」）")

    conn2 = sqlite3.connect(DB_PATH)
    cur2 = conn2.cursor()
    n2 = cur2.execute("SELECT COUNT(*) FROM demo").fetchone()[0]
    print(f"\n  重新打开后，表里还有 {n2} 行：")
    for row in cur2.execute("SELECT name, score FROM demo"):
        print(f"      {row[0]} -> {row[1]}")

    print(f"""
  ── 这就是"持久化" ──

  数据写进了文件（{os.path.basename(DB_PATH)}），
  所以程序重启、电脑关机、连接关闭 —— 数据都还在。

  对比你项目现在用的内存字典：
      _SESSIONS = {{}}          ← 程序一退出，里面全没了

  这就是我们要改造的点。
""")
    conn2.close()

    # ---------------------------------------------------------------- 收尾
    sep("小结：SQLite 的四个核心操作")
    print("""
      CREATE TABLE   建表（定义结构）
      INSERT         插入（写）
      SELECT         查询（读）
      UPDATE/DELETE  修改/删除

  以及一条铁律：
      ⚠️ 永远用参数化查询（? 占位），不要拼字符串 —— 否则会有 SQL 注入

  文件位置（可以自己去看看）：
      {DB_PATH}
""".replace("{DB_PATH}", DB_PATH))

    os.remove(DB_PATH)
    print("  （演示用的库文件已删除）")


if __name__ == "__main__":
    main()
