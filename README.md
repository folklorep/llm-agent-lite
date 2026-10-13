# LLM Agent Lite

一个基于大模型 API 的多工具智能体（Agent）应用：支持**多轮对话**、**角色扮演**、
**可评测的检索增强（RAG）**、**多工具自动循环调用（Function Calling）**，
并把工具按 **MCP 协议**暴露成标准服务器（任何 MCP 客户端都能直接调用）。

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
| **可评测的 RAG** | 三种检索模式（82% / 88% / 92%），有 50 题测试集与量化实验记录 ↓ |
| **MCP 服务器** | 工具按 MCP 协议暴露，任何 MCP 客户端（Claude Desktop / Cursor）可直接调用 |
| 异常兜底 | API 错误、非法 JSON 参数、工具内部异常均被捕获并回传给模型 |
| 调用轨迹 | `--debug` 或 `/trace` 可查看每轮工具调用与返回结果 |

## 🏗 架构

```
main.py          CLI 入口（参数解析、交互循环、命令）
api.py           HTTP 服务（FastAPI：/chat、/chat/stream SSE、会话管理）
ui_app.py        网页界面（Streamlit，可切换检索模式对比效果）
mcp_server.py    MCP 服务器（手写 JSON-RPC 2.0，把工具按协议暴露给外部客户端）
  └── agent.py   编排层：推理-行动循环、上下文裁剪、异常兜底   ← 核心
        ├── llm_client.py  协议层：HTTP 请求、tool_calls 解析、流式输出
        ├── tools.py       工具层：注册表 + Schema + 执行分发
        └── rag.py         检索层：三种模式（keyword / vector / rerank）
              ├── chunker.py       切块：按 Markdown 标题切，带上下文
              └── vector_store.py  向量库：词袋 + TF-IDF + 余弦相似度
                    └── data/docs/*.md  语料（切块后 35 块 / 1.9 万字）
config.py       配置层：环境变量集中管理
```

**分层原则**：协议层只管通信，工具层只管干活，编排层只管决策。
换模型平台只改 `config.py`，加工具只加一个装饰器函数。

**四个入口共用同一个 `Agent`**：CLI、HTTP、网页、MCP 只是四种"壳"，核心逻辑只有一份。

**检索层的接口契约（`retrieve(query, top_k)`）从"关键词匹配"到"向量 + LLM 精排"
换了两轮实现，调用方（`tools.py`）一行都没改** —— 这是当初把检索单独拆层的回报。

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

## 🔬 RAG 检索优化：一次完整的实验记录

**每一步都有量化数据**，包括被证伪的假设和负面结果。

### 实验设计

- **语料**：`data/docs/学习笔记.md`（3.1 万字），按标题切成 **35 块**，平均 496 字
- **测试集**：`data/eval_questions.txt`，**50 题**（17 题原词问 + 33 题同义改写问）
- **质检**：`validate_eval.py` 三道检查（关键词是否在语料里 / 是否太泛 / 答案块排第几）
- **指标**：命中率@3（正确答案出现在返回的前 3 条里）

> **为什么要分"原词/改写"两组**：原词题（问法和语料用词一样）关键词检索也能命中；
> 改写题才是关键词检索的弱点。**不分组的评测证明不了任何事。**

### 最终结果（50 题）

| 方案 | 总命中率 | 原词 | 改写 |
|---|---|---|---|
| ① 纯关键词粗排（基线）| 41/50 = **82%** | 16/17 | 25/33 |
| ② 精排（召回 10，纯关键词）| 45/50 = **90%** | 17/17 | 28/33 |
| ③ **精排（召回 20，纯关键词）** | **46/50 = 92%** ⭐ | 17/17 | 29/33 |
| ④ 精排（召回 10，混合召回）| 44/50 = 88% | 17/17 | 27/33 |
| ⑤ 精排（召回 20，混合召回）| 45/50 = 90% | 16/17 | 29/33 |

**最佳：③ 召回 20 + 纯关键词 + LLM 精排 → 92%（基线 82%，+10 分）**

### 五个被数据证实或推翻的判断

**① "测试集 20 题就够了" —— 被证伪（这一条最重要）**

20 题时，关键词检索和向量检索**都是 85%，完全一样**，我据此得出
"词袋向量与关键词本质同源"的结论。**扩到 50 题后结论变了**：

| | 20 题 | 50 题 |
|---|---|---|
| 关键词 | 85% | 82% |
| 向量（词袋+TF-IDF+余弦）| 85%（无差别）| **88%（+6 分）** |

**样本太小，真实差异会被噪声吃掉。** 这是本次实验最大的方法论教训。

**② "加真实向量检索就能解决语义匹配" —— 部分成立但受限**

词袋向量 + TF-IDF + 余弦相似度做到 88%（+6 分），主要收益来自改写题。
但**它仍不是语义匹配** —— 词完全不重合时一样失败。
真正的语义 embedding 本项目暂时用不了：DeepSeek API 无 embedding 接口
（实测 `/embeddings` 返回 404），本地跑 sentence-transformers 要下 2GB+ PyTorch。

**③ "瓶颈在打分公式" —— 被证伪**

试了三种关键词打分：原始词频、按长度归一化、IDF 加权 + 归一化。
在 20 题集上**三种都是 85%**，纹丝不动。说明瓶颈不在公式，在"字面匹配"这个范式本身。

**④ "长块会污染检索" —— 被证实**

诊断发现：语料里最大的块（1156 字）在 20 题里**有 6 次排第一**（它只占 1/35）。
按长度归一化后降到 **1 次**。典型的"长文档偏向"。

**⑤ ⚠️ "混合召回会更好" —— 被证伪（反直觉的负面结果）**

我原以为"关键词 + 向量"两种来源合并会比单一方法好。**实测反而更差**：

```
③ 纯关键词 + 召回 20   → 92%   ← 最好
⑤ 混合召回 + 召回 20   → 90%   ← 差 2 分
④ 混合召回 + 召回 10   → 88%   ← 比纯关键词同配置还差
```

推测原因：混合召回把候选池从 20 扩大到 40，**精排要处理的噪声变多**，
或者是我那个"交替插入"的合并策略不合理。

### 核心结论：决定成败的是召回深度

对比 ②③：召回 10 → 90%，召回 20 → 92%。
逐题查证后发现（`validate_eval.py` 的检查 3）：

```
50 题里 41 题的答案块位于关键词排序前 3 名
        6 题位于 4-10 名
        3 题位于 11-20 名   ← 召回 10 时看不到它们
        0 题位于 20 名之后
```

> **这就是 RAG 里最经典的坑：召回失败（recall failure）。**
> 候选池里没有正确答案时，精排再强也救不回来 —— 它只能从给定候选里挑。

### 可复现的命令

```bash
python validate_eval.py                   # 测试集质检（跑评测前先跑这个）
python chunker.py data/docs/学习笔记.md   # 切块结果与统计
python eval_retrieval.py                  # 关键词 vs 向量 对比
python compare_retrievers.py              # 三种关键词打分策略对比
python eval_rerank.py --sweep             # 召回深度 × 混合召回 对照组
```

> ⚠️ **诚实的局限**：语料只有 35 块 / 3.1 万字，仍然偏小；
> 92% 这个数字未必能外推到真实文档规模。50 题也还不算多
> （工业界的检索评测通常几百到几千条 query）。

## 🔌 接入 MCP 客户端

`mcp_server.py` 把项目里的工具按 **MCP（Model Context Protocol）** 标准暴露成服务器。
接好之后，**任何支持 MCP 的客户端都能直接调用这些工具**，不需要改客户端代码。

### 为什么写它（和 Function Calling 的区别）

Function Calling 是"每家一套写法"：OpenAI 一套、Anthropic 一套、通义/智谱又各自不同。
**MCP 把这件事标准化** —— 类比：Function Calling 是各家自己的充电口，MCP 是 USB-C。

### 实现了什么

| 方法 | 作用 |
|---|---|
| `initialize` | 握手：交换协议版本、声明能力 |
| `tools/list` | 客户端问"你有哪些工具" |
| `tools/call` | 客户端说"帮我调用某个工具" |
| `notifications/initialized` | 通知（按协议不产生响应）|

**底层是手写的 JSON-RPC 2.0，没有依赖官方 SDK。**

### ⚠️ 一个实际踩到的坑：stdio 有两种帧格式

| 格式 | 样子 | 出现在 |
|---|---|---|
| ① 每行一个 JSON | `{"jsonrpc":...}\n` | 当前规范，主流客户端 |
| ② Content-Length 头 | `Content-Length: 123\r\n\r\n{...}` | 早期规范（仿 LSP）|

**本实现两种都支持**（读到 `Content-Length` 就走 ②，否则逐行解析）。
不这么做的话，遇到用旧格式的客户端会直接卡死，而且报错信息很难懂。

另一个必须守的规矩：**日志只能写 stderr**。stdout 是协议通道，
混进任何非 JSON 内容都会让客户端解析失败。

### 试一下

```bash
python mcp_server.py --selftest    # 协议分发自检（16 项）
python test_mcp_stdio.py           # stdio 传输测试（16 项，含两种帧格式和脏输入）
python mcp_server.py --config      # 打印接入 Claude Desktop 的配置片段

# 手动喂一条请求
echo {"jsonrpc":"2.0","id":1,"method":"tools/list"} | python mcp_server.py
```

### 接入 Claude Desktop

跑 `python mcp_server.py --config` 会打印现成的配置 JSON，
合并进 `%APPDATA%\Claude\claude_desktop_config.json`，然后**完全退出再重开** Claude Desktop。

## 🛣 路线图

- [x] P1 工程化拆分（config / llm_client / tools / rag / agent / cli）
- [x] P2 多工具自动循环 + 轮次上限 + 异常兜底
- [x] P4-部分 FastAPI 服务化（`api.py`：/chat、/chat/stream SSE、会话管理）
- [x] P5 部署上线（Streamlit Cloud），拿到公网 URL
- [x] **P3 检索优化**：切块 + 评测框架 + 粗排精排两段式检索（见上节实验记录）
- [x] **P7 MCP 服务器**：手写 JSON-RPC 2.0，工具可被外部 MCP 客户端调用
- [x] P6-部分 测试（7 个文件，127 断言）
- [ ] P3-续 用真实 embedding 模型替换词袋向量（需要可用的 embedding API）
- [ ] P4-完成 装上 fastapi 后实测路由；流式支持多轮上下文
- [ ] P8 自主检索循环（检索 → 模型判断资料是否够 → 不够就改写查询再检索）
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
