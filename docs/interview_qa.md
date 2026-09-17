# PlantGenome Agent 面试问答清单

> 基于项目真实实现整理，所有答案可直接背诵，被追问时能展开讲细节。

---

## 一、项目整体介绍（STAR 法则，1-2 分钟版）

### Q：介绍一下这个项目？

**S（情境）**：植物比较基因组研究中，研究人员需要阅读大量文献、分析序列、选择生信工具，但这些操作分散在不同软件中，缺乏统一入口。

**T（任务）**：我独立设计并实现了一个基于 LLM 的植物基因组研究 Agent，用自然语言对话完成文献检索、序列分析、工具调用等任务。

**A（行动）**：
- 后端用 FastAPI + LangGraph 构建循环状态机，支持 7 类意图自动路由和 6 个生信工具调用
- RAG 系统用 BGE-m3 向量检索 + BM25 关键词检索 + RRF 融合的混合检索方案
- 集成 PubMed 和 NCBI E-utilities，Agent 可以自动搜索文献和基因信息
- 前端 Streamlit 交互，JWT 认证，异步任务队列处理 PDF 入库

**R（结果）**：工具调用准确率 95%（19/20 测试用例），支持多工具链式调用，响应时间约 19 秒。

---

## 二、RAG 相关（最高频追问）

### Q1：你的 RAG 是怎么做的？整个流程？

PDF 上传后走流水线：
1. **解析**：PyMuPDF 双模式提取——纯文本模式 + Markdown 模式（保留标题层次）
2. **清洗**：两阶段清洗，9 个正则去除页眉页脚、参考文献等噪声
3. **分块**：Markdown 感知分块，按标题层次切分，用 BGE-m3 tokenizer 精确控制每块 600 token，重叠 50 token
4. **向量化**：BGE-m3 embedding 模型，存入 Chroma 向量库
5. **检索**：用户提问时，向量召回候选 + BM25 重排 + RRF 融合，返回 Top-3
6. **生成**：把检索到的上下文 + 用户问题拼入 prompt，LLM 生成回答，标注来源

### Q2：为什么用混合检索？纯向量检索不够吗？

纯向量检索的问题：
- 对**精确关键词**（如基因名 "AT1G01010"、软件名 "PAML"）匹配不好，语义相近但关键词不同会漏检
- BM25 基于词频，对精确术语匹配强，但不理解语义

混合检索优势：互补——向量理解语义，BM25 抓精确词。我用 **RRF（Reciprocal Rank Fusion）** 融合两路排序，公式是 `score = Σ 1/(k + rank)`，k=60，不需要调权重，比加权求和更鲁棒。

### Q3：BM25 的索引是怎么建的？为什么不建全局索引？

我没有建全局 BM25 索引，而是**在向量召回的候选集上动态构建**：
1. 向量检索先召回 15 个候选 chunk
2. 在这 15 个 chunk 上临时构建 BM25 索引
3. 对 query 做 BM25 打分，得到第二路排序
4. RRF 融合两路排序，取 Top-3

**为什么这么设计**：
- 数据量不大（几百 chunk），在小候选集上建索引开销可忽略
- 避免维护全局 BM25 索引的同步问题（新增文档时要重建索引）
- 向量检索已经做了第一轮粗筛，BM25 只需要在候选中精排

### Q4：为什么选 BGE-m3？embedding 模型怎么选的？

- **中英文混合**：BGE-m3 支持中英文，项目文献有中文也有英文
- **生物领域**：BGE-m3 对生物专有名词（基因名、物种名）表示较好
- **多粒度**：支持 dense、sparse、multi-vector 三种检索方式，我用了 dense 向量

### Q5：分块策略怎么设计的？为什么是 600 token？

- **Markdown 感知**：按标题层次切分，不会把一个小标题下的内容切成两半
- **token 精确控制**：用 BGE-m3 的 tokenizer 计算，不是按字符数估算，确保每块不超过 embedding 模型上限
- **600 token**：太小会丢上下文，太大引入噪声。600 是在 287 个 chunk 上试过的合理值
- **50 token 重叠**：防止跨块语义断裂

### Q6：怎么抑制幻觉？

- **来源引用**：每个回答句子后标注 [1] [2]，对应具体 PDF 片段
- **prompt 约束**：明确要求"不要编造检索结果中没有的信息"
- **来源传递**：answer_node 从 tool_result 中提取 sources，不依赖 intent 分类
- **三维度来源匹配**：评估时检查文件名、摘要片段、章节路径，不只是文件名

---

## 三、LangGraph / Agent 架构（核心亮点）

### Q1：为什么用 LangGraph？它和普通 Chain 有什么区别？

普通 LangChain Chain 是线性的（输入→处理→输出），但 Agent 需要**根据中间结果决定下一步做什么**：
- 第一轮调 search_pubmed 搜文献
- 发现信息不够，第二轮调 search_pdf_knowledge 检索已入库文献
- 都拿到后生成回答

LangGraph 用**有向图 + 条件边**实现这种循环控制，比 Chain 灵活。我的图结构：
```
router → [条件边] → tool → router（循环）
                   → answer → END
```

### Q2：多工具链式调用怎么实现的？怎么防止死循环？

- **循环边**：tool 执行完不直接到 answer，而是回到 router
- **迭代计数**：state 里有 `iteration` 字段，每次 router 自增，达到 MAX_ITERATIONS=3 强制结束
- **防重复调用**：router prompt 里传入"已调用工具历史"，如果某个工具已经调过，router 必须选 direct_answer 结束
- **条件边**：`should_call_tool` 判断——direct_answer / tool_name 为 None / 达到迭代上限，三种情况跳到 answer

### Q3：State 怎么设计的？为什么这么设计？

```python
class AgentState(TypedDict):
    query: str                    # 用户原始问题
    intent: str                   # 分类结果
    tool_name: str                # 选中的工具
    tool_input: dict              # 工具参数
    tool_result: any               # 最近一次工具结果
    tool_history: list            # 所有工具调用历史
    iteration: int                 # 迭代计数
    final_answer: str             # 最终回答
    sources: list                  # 来源引用
    execution_log: list           # 执行日志
```

设计原则：
- **最小传递**：每个节点只返回需要更新的字段，LangGraph 自动 merge
- **tool_history 与 tool_result 分离**：tool_result 保留最近一次供 answer_node 用，tool_history 供 router 多轮判断
- **execution_log 记录每步**：方便调试和前端展示执行过程

### Q4：Router 怎么做意图分类？准确率多少？

- **LLM 分类**：prompt 里列出 7 类意图 + 每类的 tool_name 和参数格式 + 示例，让 LLM 输出 JSON
- **低温**：temperature=0.1，确保分类稳定
- **兜底**：JSON 解析失败时用关键词匹配兜底
- **准确率**：自建 20 条测试集，工具调用准确率 95%（19/20）

### Q5：7 类意图怎么设计的？为什么这么分？

1. literature_search：检索已入库文献（RAG）
2. literature_discovery：PubMed 全网搜索新文献
3. gene_query：NCBI 基因信息查询
4. fasta_analysis：FASTA 序列统计
5. cpg_scan：CpG 岛识别
6. pipeline_suggest：分析流程推荐
7. direct_answer：闲聊直接回答

**设计思路**：按"是否需要外部工具"分——前 6 类各对应一个工具，第 7 类不调工具直接 LLM 回答。literature_search 和 literature_discovery 的区分是为了区分"查已有知识库"和"搜全网"。

---

## 四、Tool Calling 细节

### Q1：工具怎么注册和调度的？

用一个 ToolExecutor 类：
- `registerTool(name, description, func)` 注册工具
- 每个工具有 description，告诉 router 什么时候用
- 执行时从 state 取 tool_name 和 tool_input，用 `**tool_input` 关键字参数调用
- try-except 包裹，失败时返回 error 字典

### Q2：6 个工具分别做什么？

| 工具 | 功能 | 输入 |
|------|------|------|
| search_pdf_knowledge | RAG 混合检索 | query |
| search_pubmed | PubMed 搜索文献元数据 | keyword |
| query_ncbi_gene | NCBI 基因信息查询 | gene_name |
| parse_fasta_stats | FASTA 序列统计 | fasta_text |
| scan_cpg_islands | CpG 岛识别扫描 | sequence |
| suggest_pipeline | 生信分析流程推荐 | research_goal |

### Q3：为什么 PubMed 下载用异步任务，搜索用同步？

- **PubMed 搜索**（search_pubmed_tool）：只调 E-utilities API 获取元数据，2-3 秒完成，同步调用没问题
- **PubMed 下载入库**（pubmed_fetcher）：要下载 PDF、解析、分块、向量化、入库，可能 1-2 分钟，必须异步
- 异步用 ARQ + Redis 队列，前端轮询任务状态

### Q4：NCBI API 怎么处理的？遇到什么问题？

- E-utilities 的 esearch 搜 PMID，efetch 获取 XML 元数据
- **踩坑**：NCBI PMC 直接下载 PDF 返回 403，改用 Europe PMC API 获取 PDF 直链
- fpdf2 默认字体不支持中文，摘要 PDF 全部用英文生成

---

## 五、工程与数据库

### Q1：数据库为什么从 PostgreSQL 切到 SQLite？

- 本地开发环境 PostgreSQL 认证配置复杂（scram-sha-256），没有管理员权限重启服务
- 项目数据量小（几百 chunk、几十个文档），SQLite 完全够用
- **关键改动**：模型中把 `JSONB`（PostgreSQL 特定）改成 SQLAlchemy 通用 `JSON` 类型，这样 SQLite 和 PostgreSQL 都兼容
- 生产环境切回 PostgreSQL 只需改 DATABASE_URL

### Q2：JWT 认证怎么做的？

- 注册：bcrypt 哈希密码（rounds=12）存数据库
- 登录：OAuth2 密码流，验证用户名密码后签发 JWT（HS256，24 小时过期）
- 后续请求：从 Authorization header 解析 Bearer token，验证后注入 user_id
- 用户隔离：向量库按 user_id 分 collection，不同用户数据隔离

### Q3：异步任务怎么设计的？

- 用 ARQ（基于 Redis 的异步任务队列）
- PDF 上传后立即返回 task_id，后台 worker 处理
- worker 执行：解析 PDF → 清洗 → 分块 → 向量化 → 存 Chroma + PostgreSQL
- 前端轮询 `/api/tasks/{task_id}` 获取进度

### Q4：uvicorn reload 在 Windows 上有什么问题？

- `reload=True` 启动 reloader + worker 两个进程，Windows 上 worker 偶尔卡住不响应请求
- 改为 `reload=False` 解决，开发时手动重启或用 Streamlit 热重载

---

## 六、评估体系

### Q1：怎么评估 Agent 效果？

自建 20 条测试集，4 个维度：
- **工具调用准确率**：router 选的工具是否正确（95%）
- **HitRate@3**：RAG 检索结果是否包含期望来源（三维度匹配：文件名 + 摘要片段 + 章节路径）
- **关键词命中率**：回答是否包含期望关键词（68%）
- **响应时间**：完整链路平均 19 秒

### Q2：HitRate@3 初版只有 40%，怎么分析的？

三个原因：
1. **sources 传递 Bug**：answer_node 只在 literature_search 意图时提取 sources，router 分类偏差时来源丢失 → 修复为只要 tool_result 有 sources 键就提取
2. **评估脚本太严格**：只匹配文件名，但 PDF 文件名是论文标题不含主题关键词 → 改为文件名+snippet+heading_path 三维度
3. **回答重复字符**：LLM 生成时有"并，并，，并"重复 → prompt 增加"严禁重复"指令

---

## 七、可能被挑刺的问题（提前准备）

### Q：这个项目和普通 RAG demo 有什么区别？为什么用 LangGraph？

区别在三点：
1. **多工具链式调用**：不是单轮 RAG，而是 Agent 可以根据中间结果连续调多个工具
2. **混合检索**：不是纯向量检索，而是 BM25 + 向量 + RRF，有工程优化
3. **完整工程闭环**：有认证、异步任务、评估脚本、数据库，不是 notebook demo

### Q：BGE-m3 2.3GB，部署成本怎么考虑？

- 本地开发用 HF 镜像下载，缓存到本地
- 生产环境可以用 API 调用 embedding 服务（如硅基流动的 embedding API），不需要本地部署模型
- Chroma 向量库持久化到磁盘，重启不需要重新 embedding

### Q：如果数据量到百万级 chunk，架构怎么改？

当前架构在百万级会有瓶颈：
- Chroma 换成 Milvus/Qdrant 等分布式向量库
- BM25 从动态构建改成全局索引（Elasticsearch/OpenSearch）
- 异步任务队列加 workers 水平扩展
- 向量库按用户分片

### Q：为什么不用 LangChain Agent，要自己用 LangGraph 写？

- LangChain 的 Agent 封装太重，可控性差
- LangGraph 的图结构清晰，条件边和循环显式定义，便于调试
- State 完全自定义，知道每一步传了什么数据
- 不需要 ReAct prompt 的复杂推理，规则路由 + LLM 分类更稳定

---

## 八、简历措辞建议

简历上可以这么写：

> **PlantGenome Agent** | Python / FastAPI / LangGraph / Chroma / LLM
> - 设计并实现基于 LangGraph 循环状态机的研究 Agent，支持 7 类意图自动路由、6 个工具调用、多工具链式推理（最多 3 轮迭代）
> - 实现 BM25 + 向量检索 + RRF 融合的混合检索方案，在向量召回候选集上动态构建 BM25 索引，兼顾语义理解与精确关键词匹配
> - 集成 PubMed/NCBI E-utilities，支持文献搜索下载入库、基因信息查询，异步任务队列处理 PDF 解析流水线
> - 自建 20 条测试集评估体系，工具调用准确率 95%，实现三维度来源匹配评估
> - Streamlit 交互前端 + JWT 认证 + 用户数据隔离，完整工程闭环
