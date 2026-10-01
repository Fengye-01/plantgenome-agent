# PlantGenome Agent - Failure Cases（边界与失败用例）

> 5 个边界/失败用例，展示系统的边界意识和优雅降级能力。
> Failure case 不是缺点，反而体现系统的鲁棒性和错误处理能力——面试官喜欢看到你知道系统的局限。

---

## 用例总览

| # | 场景 | 输入 | 预期行为 | 体现能力 |
|---|------|------|---------|---------|
| 1 | 知识库无答案 | "今天天气怎么样？" | 明确说明无法回答，不编造 | 防幻觉 + 边界意识 |
| 2 | 非法 FASTA | "帮我统计：hello world" | 提示"未检测到有效的 FASTA 序列" | 输入校验 + 友好提示 |
| 3 | 过短 DNA 序列 | "扫描 CpG 岛：ATCG" | 提示序列长度不足（<200bp） | 参数校验 + 边界处理 |
| 4 | 意图模糊 | "帮我分析一下" | router 分类到最接近的类型，或给出通用回答 | 意图分类鲁棒性 |
| 5 | 工具执行异常 | （构造异常输入） | 显示错误信息，不崩溃，给出建议 | 异常处理 + 优雅降级 |

---

## 用例详情

### Failure Case 1：知识库无答案

- **场景**：用户问一个知识库完全不相关的问题
- **用户输入**：`今天天气怎么样？`
- **预期 Agent 链路**：`router → tool(search_pdf_knowledge) → answer`
- **预期意图**：`literature_search`（router 可能分类为文献检索）
- **预期行为**：
  - RAG 检索返回低相似度结果或无结果
  - answer_node 基于检索结果回答，明确说明"根据现有文献资料，无法回答这个问题"
  - 不编造天气信息
  - 可能建议用户换一个与生信相关的问题
- **为什么重要**：
  - 体现防幻觉能力：系统知道自己不知道什么
  - RAG 的核心价值就是"基于证据回答，没有证据就说不知道"
  - 面试时可以讲："我在 Prompt 里明确要求模型，如果检索结果没有相关信息，必须说明无法回答，不能编造。"

---

### Failure Case 2：非法 FASTA 输入

- **场景**：用户在聊天框输入非 FASTA 格式的内容，要求统计
- **用户输入**：`帮我统计一下这个序列：hello world`
- **预期 Agent 链路**：`router → tool(parse_fasta_stats) → answer`
- **预期意图**：`fasta_analysis`
- **预期工具**：`parse_fasta_stats`
- **预期行为**：
  - parse_fasta_stats 返回 `{"error": "未检测到有效的 FASTA 序列，请检查输入格式（序列 ID 应以 > 开头）"}`
  - answer_node 基于工具结果回答，说明输入格式不正确
  - 给出正确的 FASTA 格式示例
  - 不崩溃，不抛出异常
- **为什么重要**：
  - 体现输入校验能力：工具函数能识别非法输入
  - 友好的错误提示：告诉用户正确的格式是什么
  - 异常不传播：工具失败不会导致整个 Agent 崩溃

---

### Failure Case 3：过短 DNA 序列

- **场景**：用户输入很短的 DNA 序列要求扫描 CpG 岛
- **用户输入**：（侧边栏粘贴 `ATCG`，点击"扫描 CpG 岛"按钮）
- **预期 Agent 链路**：（前端直接调用 scan_cpg_islands）
- **预期意图**：`cpg_scan`
- **预期工具**：`scan_cpg_islands`
- **预期行为**：
  - scan_cpg_islands 检测到序列长度（4bp）小于窗口大小（200bp）
  - 返回 `[{"warning": "序列长度(4bp)小于窗口大小(200bp)，无法进行滑动窗口扫描", ...}]`
  - 前端显示警告信息，不崩溃
  - 说明 CpG 岛扫描需要至少 200bp 的序列
- **为什么重要**：
  - 体现参数校验：算法有明确的输入要求（窗口大小 200bp）
  - 边界处理：过短序列不会导致数组越界或除零错误
  - 友好提示：告诉用户为什么不能扫描，需要多长的序列

---

### Failure Case 4：意图模糊

- **场景**：用户输入非常模糊的问题，没有明确意图
- **用户输入**：`帮我分析一下`
- **预期 Agent 链路**：`router → tool(...) → answer` 或 `router → answer`
- **预期行为**：
  - router_node 尝试分类到最接近的意图（可能是 literature_search 或 pipeline_suggest）
  - 如果分类为 literature_search，调用 search_pdf_knowledge，检索结果可能不相关
  - answer_node 基于结果回答，可能询问用户澄清具体需求
  - 不会崩溃，不会抛出异常
- **为什么重要**：
  - 体现意图分类的鲁棒性：即使输入模糊，router 也能给出一个最接近的分类
  - 优雅降级：不会因为输入模糊而崩溃
  - 可以改进的方向：V2 可以增加"澄清问题"节点，当意图置信度低时主动询问用户

---

### Failure Case 5：工具执行异常

- **场景**：构造异常输入，触发工具执行异常
- **用户输入**：（构造各种异常输入，如空字符串、特殊字符、超大输入等）
- **预期行为**：
  - tool_node 用 try-except 包裹工具调用
  - 工具执行失败时，tool_result = `{"error": "错误信息"}`
  - execution_log 记录 status="failed" 和错误信息
  - answer_node 基于错误结果回答，说明失败原因并给出建议
  - 整个 Agent 不崩溃，正常返回
- **为什么重要**：
  - 体现异常处理能力：tool_node 的 try-except 确保工具失败不会传播
  - 错误可追溯：execution_log 记录了哪个工具失败、失败原因、耗时
  - 优雅降级：即使工具失败，Agent 也能给出有意义的回答（说明失败原因 + 建议）

---

## 系统局限性说明（面试可讲）

在 README 和面试中，可以主动说明系统的局限性：

| 局限性 | 说明 | V2 改进方向 |
|--------|------|------------|
| 单工具调用 | 当前一次只调用一个工具，不支持多工具链式调用 | 增加 ReAct 循环，支持多轮工具调用 |
| 无多轮记忆 | 每次对话独立，不保留上下文 | 增加 Memory 模块，支持多轮对话 |
| 7B 模型限制 | 本地 7B 模型在复杂推理和 JSON 输出上不稳定 | 换用更大模型（14B/72B）或 API 模型 |
| 文本型 PDF | 当前只处理文本型 PDF，不支持扫描件/OCR | 增加 OCR 支持（PaddleOCR/Tesseract） |
| 简单评估 | 当前只用 HitRate@3，没有 RAGAS 专业评估 | 引入 RAGAS 框架，评估 Faithfulness/Context Precision |
| 无 MCP | 工具注册是硬编码的，不支持动态加载 | 引入 MCP 协议，支持外部工具动态接入 |

主动说明局限性反而体现你对系统的深入理解和工程思维。

---

## 检索评估失败用例（三模式复现，2026-10）

> 基于隔离环境重新规范入库 12 篇 PDF、18 条 answerable query 的三模式评估
> （结果见 `docs/retrieval_eval_3modes_*.json`）。18 条中 17 条命中，1 条三模式均失败。

### Case 18：中文 query 的词法召回失效 + Dense 语义偏移

- **Query**：`什么是正向选择和纯化选择？怎么检测？`
- **Gold 相关文档**：PAML（Yang 2007）。

**各通道真实表现：**

| 通道/模式 | 结果 |
|-----------|------|
| Dense 通道 | 正确 chunk `doc1_chunk5` 在 Top-20 中排第 11，distance=0.506（略超阈值），Top-5 全部为 Ashikawa（CpG） |
| BM25 通道 | **完全为空**：query 为纯中文，tokenizer 正则 `[a-z0-9]+` 丢弃所有中文字符，无有效查询词 |
| dense_only 最终 | rank=None（miss） |
| candidate_rrf 最终 | rank=None（miss） |
| global_rrf 最终 | rank=None（miss） |

**根因（两层）：**

1. **召回/词法层**：BM25 tokenizer 不支持中文分词，纯中文 query 无法匹配英文正文，Sparse 通道对这类 query 完全失效——这是 global_rrf 未能补回的直接原因。
2. **语义层**：Dense 将中文 query 误导向 CpG 主题，正确 PAML chunk 排名靠后（11）且距离略超门控阈值。

**诚实的面试表述：**

> 当前版本 global_rrf 在含英文术语/软件名的 query 上稳定提升排序，但对纯中文 query，受限于英文词法 tokenizer，BM25 无法生效；这是当前实现的明确限制。如果继续优化，我会优先：① 引入支持中文的分词（如 jieba / 字符 n-gram）让 Sparse 通道覆盖中文；② 用 HyDE/MQE 对 query 做改写改善 Dense 语义偏移；③ 再评估多语言 Cross-Encoder Reranker。

**为什么不优先 Reranker：** 当前唯一失败同时包含中文词法失效（Reranker 前的召回问题）和语义偏移，单纯 Reranker 无法补回未被有效召回的证据，应先解决中文召回与 query 改写。

---

*PlantGenome Agent Failure Cases · D13*
