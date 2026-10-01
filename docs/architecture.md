# PlantGenome Agent - 架构图

> 3 张核心架构图，用 Mermaid 格式编写，可直接在 GitHub / Typora / VS Code 中渲染。
> 如果需要 PNG 图片，可以用 https://mermaid.live 导出。

---

## 图 1：系统架构图（System Architecture）

展示用户 → 前端 → 后端 → Agent → 数据层的完整链路。

```mermaid
graph TB
    User([👤 用户]) --> Streamlit[🖥️ Streamlit UI<br/>聊天界面 + 工具面板 + 执行过程可视化]

    Streamlit --> FastAPI[⚡ FastAPI API<br/>POST /chat + GET /health]

    FastAPI --> Agent[🤖 LangGraph Agent<br/>状态机工作流]

    subgraph Agent_Nodes ["Agent 内部节点"]
        Router[🧭 Router Node<br/>意图分类 + 工具选择]
        Tool[🔧 Tool Node<br/>工具执行 + 异常处理]
        Answer[💬 Answer Node<br/>结果整合 + 回答生成]
    end

    Agent --> Router
    Router -->|需要工具| Tool
    Router -->|直接回答| Answer
    Tool --> Answer

    subgraph Tools ["4 个生信工具"]
        T1[📄 search_pdf_knowledge<br/>RAG 文献检索]
        T2[🧬 parse_fasta_stats<br/>FASTA 序列统计]
        T3[🔬 scan_cpg_islands<br/>CpG 岛扫描]
        T4[📋 suggest_pipeline<br/>分析流程推荐]
    end

    Tool --> T1
    Tool --> T2
    Tool --> T3
    Tool --> T4

    subgraph DataLayer ["数据层"]
        Chroma[(💾 Chroma 向量库<br/>BGE-m3 embedding<br/>287 chunks)]
        PDFStore[📁 PDF 文件存储<br/>10 篇生信文献]
    end

    T1 --> Chroma
    Chroma --> PDFStore

    subgraph LLM ["LLM 层"]
        LLMAPI[☁️ 硅基流动 API<br/>Qwen2.5-7B-Instruct]
    end

    Router --> LLMAPI
    Answer --> LLMAPI
    T4 --> LLMAPI

    style Agent fill:#e1f5fe,stroke:#01579b
    style Chroma fill:#fff3e0,stroke:#e65100
    style LLMAPI fill:#f3e5f5,stroke:#4a148c
```

---

## 图 2：PDF RAG 流程图（PDF RAG Pipeline）

展示从 PDF 上传到回答生成的完整 RAG 流水线，对应 Hello Agents 第8章。

```mermaid
graph LR
    subgraph Offline ["离线：知识库构建"]
        PDF[📄 PDF 文献上传] --> Parse[🔍 PyMuPDF 解析<br/>文本 + Markdown 双模式]
        Parse --> Clean[🧹 TextCleaner 清洗<br/>页眉页脚 + 重复行 + 噪声]
        Clean --> Chunk[✂️ Markdown 智能分块<br/>标题层次感知 + Token 精确控制]
        Chunk --> Embed[🔢 BGE-m3 向量化<br/>中英文混合 embedding]
        Embed --> Store[(💾 Chroma 持久化<br/>余弦相似度索引)]
    end

    subgraph Online ["在线：问答生成"]
        Query[❓ 用户提问] --> Retrieve[🔍 检索 Top-K<br/>向量相似度匹配]
        Store --> Retrieve
        Retrieve --> Context[📝 ContextBuilder<br/>GSSC 流水线 + Prompt 组装]
        Context --> LLM[🤖 LLM 生成回答<br/>Qwen2.5-7B-Instruct]
        LLM --> Answer[💬 最终回答<br/>+ sources 来源引用]
    end

    style Offline fill:#e8f5e9,stroke:#2e7d32
    style Online fill:#e3f2fd,stroke:#1565c0
    style Store fill:#fff3e0,stroke:#e65100
```

**RAG 流水线关键步骤说明：**

| 步骤 | 模块 | 对应教程 | 说明 |
|------|------|---------|------|
| 1. 文档载入 | PDFParser | 第8章 8.3.1 | PyMuPDF 页级提取，支持纯文本和 Markdown 双模式 |
| 2. 文本清洗 | TextCleaner | 第8章 8.3.2 | 两阶段清洗，9 个噪声正则，去除页眉页脚和重复行 |
| 3. 智能分块 | Chunker | 第8章 8.3.3 | Markdown 标题层次感知 + 段落语义保持 + BGE-m3 Token 精确控制 + 智能 overlap |
| 4. 向量化 | VectorStore | 第8章 8.3.4 | BGE-m3 embedding，Chroma 持久化，余弦相似度 |
| 5. 检索 | Retriever | 第8章 8.3.5 | Top-K 检索，MQE/HyDE 预留接口，sources 组装 |
| 6. 上下文工程 | ContextBuilder | 第9章 | GSSC 流水线（Gather → Select → Sort → Compose），Token 预算控制 |
| 7. 生成 | RAGGenerator | 第8章 8.3.6 | LLM 生成回答，强制基于参考资料，防幻觉 |

---

## 图 3：Agent 状态机图（Agent State Machine）

展示 LangGraph Agent 的状态流转，对应 Hello Agents 第6章。

```mermaid
stateDiagram-v2
    [*] --> Router: 用户输入 query

    state "🧭 Router Node" as Router
    state "🔧 Tool Node" as Tool
    state "💬 Answer Node" as Answer

    Router --> Tool: 需要调用工具<br/>(literature_search / fasta_analysis / cpg_scan / pipeline_suggest)
    Router --> Answer: 直接回答<br/>(direct_answer)

    Tool --> Answer: 工具执行完成<br/>返回 tool_result

    Answer --> [*]: 生成最终回答<br/>+ sources + execution_log

    note right of Router
        输入: query, messages
        输出: intent, tool_name, tool_input
        关键: LLM 意图分类 + JSON 解析
              多级容错（直接解析/正则修复/关键词提取）
    end note

    note right of Tool
        输入: tool_name, tool_input
        输出: tool_result, sources
        关键: ToolExecutor 统一调度
              try-except 异常处理
              兜底参数提取（正则）
              execution_log 记录
    end note

    note right of Answer
        输入: query, tool_result, sources, intent
        输出: final_answer
        关键: 按意图格式化工具结果
              8 条回答规则
              强制基于参考资料
    end note
```

**AgentState 状态字段：**

| 字段 | 类型 | 说明 |
|------|------|------|
| query | str | 用户原始问题 |
| messages | list | 对话历史（MVP 单轮） |
| execution_log | list | 执行日志（每个节点的耗时、状态、错误） |
| intent | str | 分类意图（5 类） |
| tool_name | str | 选中的工具名 |
| tool_input | dict | 工具输入参数 |
| tool_result | any | 工具执行结果 |
| sources | list | 文献来源引用 |
| final_answer | str | 最终回答 |

**5 类意图：**

| 意图 | 工具 | 说明 |
|------|------|------|
| literature_search | search_pdf_knowledge | 文献/概念/方法问题 |
| fasta_analysis | parse_fasta_stats | FASTA 序列统计 |
| cpg_scan | scan_cpg_islands | CpG 岛扫描 |
| pipeline_suggest | suggest_pipeline | 分析流程推荐 |
| direct_answer | （无） | 直接回答，不调用工具 |

---

## 图 4：检索模式与用户级全库 BM25（Retrieval Modes）

系统支持三种可配置、可回滚的检索模式，由环境变量 `RETRIEVAL_MODE` 控制：

| 模式 | Dense 召回 | BM25 召回范围 | 说明 |
|------|-----------|--------------|------|
| `dense_only` | Chroma Top-K | 不使用 | 纯向量检索，最简单 |
| `candidate_rrf`（默认） | Chroma Top-K×5 | **仅在 Dense 候选内** | BM25 只能重排候选，无法补回 Dense 漏召回 |
| `global_rrf` | Chroma 全库 Top-N | **用户全库独立 Top-N** | Dense / Sparse 各自独立召回后 RRF 融合 |

非法配置值保守回退 `candidate_rrf`；运行期异常最终兜底 `dense_only`，检索接口不直接崩溃。

```mermaid
flowchart TD
    Q[用户 Query + user_id] --> D[Chroma Dense 独立召回 Top-N<br/>filter user_id]
    Q --> S[用户级 BM25 快照<br/>独立召回 Sparse Top-N]
    D --> R[RRF 融合<br/>按共同 chunk_id 去重]
    S --> R
    R --> G[分通道证据门控]
    G --> K[Final Top-K + Sources]
```

**用户级 BM25 快照（`app/rag/bm25_snapshot.py`）**

- 快照内容：`user_id`、`corpus_version`、`built_at`、`BM25Index`、chunk_id → 正文/元数据映射。
- 数据源：从 SQL 经 `documents JOIN document_chunks` 加载，只含该用户、`status='completed'`、正文非空、`chroma_id` 非空的 chunk；不加载其他用户数据，也不从 Dense 候选构建。
- 进程内懒加载，用户级锁（SingleFlight）防并发重复构建，构建成功后原子替换；构建失败保留旧快照并标记 `stale`。

**跨进程失效（Redis 版本键）**

API 与 ARQ Worker 是不同进程，不能共享 Python 内存。版本键 `rag:bm25:version:{user_id}`：

- API 使用快照前比对 Redis 版本，版本变化则重建；
- Worker 在 SQL + Chroma 均入库成功后递增版本；删除成功后也递增；失败路径不误递增；
- Redis 不可用时降级为进程内 TTL（默认 300s）判断，并记录警告，不阻断检索。

**当前边界与限制**

- 融合键为 `chroma_id`，规则 `doc{document_id}_chunk{index}`，在当前文档代次内可用；重新切分后身份不保证稳定，稳定 UUID 属后续演进。
- BM25 tokenizer 为英文/数字词法匹配，**不支持中文分词**：纯中文 query 的词法召回会失效（见 failure_cases.md case 18）。
- 快照为进程内状态，当前方案适用于单机多进程；多实例水平扩展需演进为共享索引或集中式词法服务。
- 历史 `data/chroma` 早期向量（SQL 无对应记录、ID 含 UUID）未纳入本次升级验证；三模式评估基于隔离环境中重新规范入库的语料。

---

## 如何导出为 PNG

1. 打开 https://mermaid.live
2. 把上面的 Mermaid 代码粘贴进去
3. 点击 "Actions" → "Export PNG"
4. 保存到 `docs/images/` 目录

---

*PlantGenome Agent Architecture · D15*
