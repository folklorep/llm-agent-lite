"""实验 12 补充：亲手复现"线程不安全"。

跑法：
    python lab_thread_safety.py

============================================================================
这个脚本在演示什么
============================================================================
Agent.history 是共享数据。多个请求（线程）同时改它，会出什么问题？

脚本做两次实验：
    实验 A：不加锁 —— 观察 history 条数错乱
    实验 B：加锁   —— 观察恢复正常

⚠️ 注意：线程问题【不是每次都能复现】（取决于调度时机），
   所以这里跑多轮统计，用"出错次数"来说明，而不是拿单次结果下结论。
"""
import threading
import time

import config


class MiniAgent:
    """一个极简的 Agent，只保留"读长度 → 追加 → 按长度裁剪"三步。

    为什么这么写能稳定复现竞态：
      trim 的裁剪量依赖它【读到】的长度。先读长度、再追加、再按旧长度裁剪 ——
      如果两个线程都读到旧长度，它们各自以为列表还是那么大，
      于是一次裁剪只砍掉一条，谁也没把列表压到上限内。
      多轮累积下来，列表会一路涨上去。
    """

    def __init__(self):
        self.history = [{"role": "system", "content": "sys"}]
        self.limit = 2
        # 只在【裁剪完成之后】采样 —— 那才是"不变量应该成立"的时刻。
        # （如果在 append 之后采样，本来就会多出刚追加的那一条，测不准）
        self.max_after_trim = 0
        self.trim_count = 0

    def chat(self, user_input):
        seen_len = len(self.history)
        self.history.append({"role": "user", "content": user_input})
        time.sleep(0.0002)                       # ← 窗口：别人可以插进来
        head, body = self.history[0], self.history[1:]
        if seen_len - 1 > self.limit:
            self.history = [head] + body[-(seen_len - 1):]
        # 裁剪完成后采样
        self.trim_count += 1
        self.max_after_trim = max(self.max_after_trim, len(self.history))


def run_once(use_lock: bool) -> int:
    """跑一轮并发，返回 history 出现过的最大条数。

    用"最大条数"而不是"违规次数"来对比，原因：
      在锁外数违规，加锁组也会因为"检查时被别人插进来"而误报。
      而 max_seen 是 Agent 内部记录的，不依赖外部检查时机。
    """
    agent = MiniAgent()
    lock = threading.Lock()

    def worker(n):
        for i in range(20):
            if use_lock:
                with lock:
                    agent.chat(f"user{n}-{i}")
            else:
                agent.chat(f"user{n}-{i}")

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return agent.max_after_trim


def main() -> None:
    print("=" * 76)
    print("  实验 12 补充：线程不安全是怎么发生的")
    print("=" * 76)
    print("""
  设定：
    8 个线程同时往同一个 Agent 里写，每个线程写 20 次
    Agent 的 limit = 2
    采样点：只在【裁剪完成之后】记录 history 条数（那才是不变量该成立时）

  ⚠️ 正常上限是 limit + 2 = 4，不是 limit + 1 ——
     因为裁剪条件是"追加【前】的长度 > limit"，
     所以会先涨到 4 才触发裁剪（我第一版算错了这个上限）。
     单线程实测：2 → 3 → 4 → 4 → 4……稳定在 4。

  预期：不加锁时最大条数会远超 4；加锁时应该始终 <= 4
""")

    print(f"  {'轮次':>4}  {'不加锁：history 最大条数':>24}  {'加锁：最大条数':>16}")
    print("  " + "-" * 52)
    no_lock_vals, with_lock_vals = [], []
    for r in range(5):
        no_lock = run_once(use_lock=False)
        with_lock = run_once(use_lock=True)
        no_lock_vals.append(no_lock)
        with_lock_vals.append(with_lock)
        print(f"  {r+1:>4}  {no_lock:>24}  {with_lock:>16}")

    print("  " + "-" * 52)
    print(f"  {'平均':>4}  {sum(no_lock_vals)/len(no_lock_vals):>24.1f}  "
          f"{sum(with_lock_vals)/len(with_lock_vals):>16.1f}")

    print(f"""
  ── 结论 ──

    正常上限是 4 条。
    加锁  ：最大 {max(with_lock_vals)} 条 —— 稳定守住上限 ✅
    不加锁：最大 {max(no_lock_vals)} 条 —— 数据已经乱了 ❌

  ── 为什么会涨？ ──

  "读长度 → 追加 → 按【旧长度】裁剪"之间有个窗口。没锁时两个线程会这样交错：

      线程A: 读到长度=2，append → [sys, a, A]
      线程B: 读到长度=3，append → [sys, a, A, B]     ← 插进 A 的窗口
      线程A: 按旧长度 2 裁 → [sys, a, A]              ← 只砍掉 B
      线程B: 按旧长度 3 裁 → [sys, a, A, B]           ← 又按过时的长度算

  两个线程都按【过时的长度】裁剪，裁剪量不足，列表就一路涨上去。

  ── 为什么加锁能解决 ──

  锁保证"读长度 + append + trim"这一串【原子执行】：
  同一时刻只有一个线程在改，不会读到过时的长度。

  ── 面试怎么讲 ──

  "HTTP 服务并发处理请求，而 Agent.history 是共享的可变状态。
   我实测过不加锁的后果：8 个线程并发写 20 轮，
   正常上限 4 条的 history 会涨到 {max(no_lock_vals)} 条 —— 数据完全乱了。
   所以我用一个锁把'取会话 + 执行 chat'护住。
   ⚠️ 但这是单进程方案：多实例部署时锁不跨进程，
   正确做法是每个会话独占实例，或把状态放 Redis。我在代码注释里写了这个局限。"

  ── 三个诚实的补充 ──

  ① 我第一版脚本跑出来"不加锁 0 次违规" —— 因为线程 bug 本身【偶发】，
     复现需要刻意设计时序（读长度→追加→按旧长度裁剪）。
     这也是为什么线程 bug 在真实项目里特别难查。

  ② 我第二版把采样点放在 append 之后，结果加锁组也"超限"——
     因为那个时刻本来就会多出刚追加的一条。采样点必须选对。

  ③ 我第三版把上限算成了 limit + 1，实际是 limit + 2。
     三次修正都是靠【实测数据对不上】发现的 —— 不是靠读代码想出来的。

  ⭐ 所以：线程问题不能靠"跑一遍没问题"来证明不存在，
     要靠【代码审查 + 压力测试 + 明确的不变量 + 正确的采样点】。
""")


if __name__ == "__main__":
    main()
