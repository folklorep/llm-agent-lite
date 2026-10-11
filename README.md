# LLM Agent Lite

一个基于大模型 API 的多工具智能体（Agent）应用：支持**多轮对话**、**角色扮演**、
**向量/关键词混合知识库检索（RAG）**、**多工具自动循环调用（Function Calling / ReAct 风格）**。

> 这是本人从零实现的大模型应用项目：不依赖 LangChain 等框架，自行实现协议层、
> 工具注册与编排循环，以便完整掌握大模型应用的底层机制。

---

## ✨ 特性

| 能力 | 说明 |
|---|---|
| 多轮对话 | 自行维护 `history`，支持滑动窗口裁剪控制 token 成本 |
| 系统人设 | `system` prompt 定义角色与行为边界 |
| 工具调用 | Function Calling，模型返回结构化 `tool_calls`，由本地代码真实执行 |
| **多工具自动循环** | 模型可连续调用多个工具，并根据中间结果决定下一步（带轮次上限保护）|
| **RAG 检索** | 知识库检索 + 防幻觉约束；检索层接口固定，可替换为向量检索 |
| 异常兜底 | API 错误、非法 JSON 参数、工具内部异常均被捕获并回传给模型 |
| 调用轨迹 | `--debug` 或 `/trace` 可查看每轮工具调用与返回结果 |

## 🏗 架构

```
main.py          CLI 入口（参数解析、交互循环、命令）
api.py           HTTP 服务（FastAPI：/chat、/chat/stream SSE、会话管理）
ui_app.py        网页界面（Streamlit）
  └── agent.py   编排层：推理-行动循环、上下文裁剪、异常兜底   ← 核心
        ├── llm_client.py  协议层：HTTP 请求、tool_calls 解析、流式输出
        ├── tools.py       工具层：注册表 + Schema + 执行分发
        └── rag.py         检索层：知识库加载、打分排序（可替换为向量检索）
              └── data/knowledge.txt  知识库（每行：标签 | 内容）
config.py       配置层：环境变量集中管理
```

**分层原则**：协议层只管通信，工具层只管干活，编排层只管决策。
换模型平台只改 `config.py`，加工具只加一个装饰器函数。

**三个入口共用同一个 `Agent`**：CLI、HTTP、网页只是三种"壳"，核心逻辑只有一份。

## 🚀 快速开始

```bash
# 1. 安装依赖
python -m pip install -r requirements.txt

# 2. 配置密钥
copy .env.example .env      # Windows
# 然后编辑 .env，填入你的 LLM_API_KEY

# 3. 运行
python main.py              # 交互模式
python main.py --debug      # 显示工具调用轨迹
python main.py --once "帮我算 128乘37加15"   # 单次提问
```

## 💬 效果示例

```
你 > 帮我算 128乘37加15
  ↳ 调用 calculator({"expression": "128*37+15"}) => 128*37+15 = 4751
AI > 128 × 37 + 15 = 4751

你 > 什么是两数之和的解法？
  ↳ 调用 search_knowledge({"query": "两数之和"}) => 用字典记录「数 → 下标」...
AI > 两数之和的解法是：用字典记录「数 → 下标」，边遍历边查 target - nums[i]
     是否出现过。关键顺序是先查后登记，防止自己配自己。

你 > 红烧肉怎么做？
  ↳ 调用 search_knowledge({"query": "红烧肉"}) => 知识库中没有找到相关内容。
AI > 抱歉，我的知识库中没有红烧肉的做法，无法回答。   ← 防幻觉生效
```

## 🧠 三个值得讲的设计决策

1. **为什么不用 LangChain？**
   框架把协议层和编排层都封装了，出问题难以定位。自己实现一遍后，
   能清楚说出 `tool_calls` 的消息协议、`role=tool` 回传的格式要求、
   以及为什么必须原样保留 assistant 的 tool_calls 消息。

2. **为什么要有轮次上限？**
   模型可能陷入"反复调用同一工具"的循环，无上限会持续烧 token。
   `MAX_TOOL_ROUNDS` 是成本与能力的折中。

3. **检索层为什么要抽成函数接口？**
   一期用关键词打分就能跑通全链路，二期换向量检索时调用方一行不改。
   先跑通链路再优化单点，比一开始就上向量库更容易交付。

## 🧪 测试

```bash
python test_agent.py        # Agent 主循环 8 个用例（34 个断言）
python test_edge_cases.py   # 边界场景：最后一轮调工具、中文返回值
python test_imports.py      # 导入链路 + 路由 + 会话管理（桩模块，无需装 web 依赖）
python smoke_test.py        # 检索层与工具层冒烟
```

| 文件 | 覆盖 | 依赖 |
|---|---|---|
| `test_agent.py` | 多轮循环、并行工具调用、轮次上限、异常兜底、上下文裁剪、reset | 无（假模型）|
| `test_edge_cases.py` | 轮次耗尽后的 history 完整性、中文工具结果可序列化 | 无 |
| `test_imports.py` | api 路由注册、模型默认值、503 错误处理、会话管理与淘汰 | 无（桩模块）|
| `smoke_test.py` | 检索打分、工具执行与异常分支 | 无 |

**全部测试不消耗 API 额度、不需要 API key。**
做法是用**依赖注入**把 `LLMClient` 换成假模型（`FakeClient`），
用 `sys.modules` 打桩顶替 fastapi / pydantic / streamlit。

> ⚠️ `test_imports.py` 只验证「导入 + 注册 + 逻辑」，**不等于运行时已验证**。
> 路由真正响应请求、Streamlit 真正渲染，仍需装依赖后实测。

### 测试抓到的两个真实缺陷

| # | 缺陷 | 怎么发现的 |
|---|---|---|
| 1 | **上下文裁剪时机错误**：`_trim_history()` 在追加用户消息**之前**执行，导致每次发出去的 payload 比上限多 2 条，长对话下持续漂移、token 成本失控 | `test_agent.py` 用例 7 断言"history 长度 ≤ 上限"，实测 8 ≠ 7 |
| 2 | **503 错误处理是失效的**：`except RuntimeError` 写在 `/chat` 路由里，但 `Agent()` 是在 `_get_agent()` **内部**构造的，异常在到达路由前就抛出去了 → 缺 key 实际返回 500 而非 503 | `test_imports.py` 断言**异常类型**时抓到（只断言"抛异常了"会漏掉）|

> **教训**：断言要断言**具体的东西**。
> 第 1 个 bug 是断言了精确数值才暴露；第 2 个 bug 如果只写 `assertRaises(Exception)` 就永远发现不了。

## 🛣 路线图

- [x] P1 工程化拆分（config / llm_client / tools / rag / agent / cli）
- [x] P2 多工具自动循环 + 轮次上限 + 异常兜底
- [x] P6-部分 单测（4 个测试文件，60+ 断言）
- [x] P4-部分 FastAPI 服务化（`api.py`：/chat、/chat/stream SSE、会话管理）
- [x] P5-部分 Streamlit 界面（`ui_app.py`）
- [ ] P3 向量检索（chromadb + embedding）替换关键词打分
- [ ] P4-完成 装上 fastapi 后实测路由；流式支持多轮上下文
- [ ] P5-完成 部署上线（Streamlit Cloud / HF Spaces），拿到公网 URL
- [ ] P6-完成 迁移到 pytest + GitHub Actions

## ⚠️ 踩坑记录

| 问题 | 现象 | 解决 |
|---|---|---|
| 模型名过期 | 旧模型名仍可调用但行为退化 | 始终以官方文档当前模型名为准，配置化 |
| prompt 锁死工具 | "严格根据资料不要编造"导致模型不敢调工具 | prompt 需明确"介绍工具 + 说明使用时机" |
| 模型"演"工具调用 | 不返回 JSON 协议，而是输出文本标签 | prompt 明确禁止输出标记文本；代码只认标准 `tool_calls` |
| 多 Python 版本 | 装了库但运行时报 `ModuleNotFoundError` | 统一用 `python -m pip install`，保证装进当前解释器 |
| **UTF-8 BOM** | `.env` 第一个变量永远读不到 | 用 `encoding="utf-8-sig"` 读取（PowerShell / 记事本保存的 UTF-8 带 BOM）|
| **类属性被实例属性遮蔽** | 测试里 `config.API_KEY = "x"` 设了值，`validate()` 仍报缺 key | `validate()` 是 `@classmethod`，读的是 `cls.API_KEY`；实例赋值只是遮住类属性。改类属性才对 |
| **except 写错层级** | 缺 key 时返回 500 而不是设计的 503 | 异常在**被调用函数内部**抛出时，兜底必须写在该函数里，写在调用方是失效的 |

## 🚀 启动 Web 服务 / 界面（需要额外依赖）

```bash
# HTTP 接口（自带交互式文档）
python -m pip install fastapi uvicorn
python -m uvicorn api:app --reload --port 8000
# 接口文档：http://127.0.0.1:8000/docs

# 网页界面（最适合演示）
python -m pip install streamlit
python -m streamlit run ui_app.py
```

## 📄 License

MIT
