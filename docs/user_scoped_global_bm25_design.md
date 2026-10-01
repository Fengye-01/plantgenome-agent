# 用户级全库 BM25 + Dense + RRF 检索架构设计

> 状态：待确认，尚未实施  
> 选型：PostgreSQL 规范 Chunk 事实源 + 用户级进程内 BM25 不可变快照 + PostgreSQL 持久化索引代次 + Redis 失效通知 + 独立 Dense/Sparse 召回 + RRF  
> 本文只描述设计；业务代码、数据库、Chroma 数据和 Git 分支均未因本文发生变更。

## 1. 背景与目标

当前在线检索先由 Chroma 完成稠密召回，再在候选上临时建立 BM25 并通过 RRF 重排。它能改善候选排序，但 BM25 无法补回 Dense 漏召回的 Chunk，也没有完整覆盖上传、重试、删除、重建和跨进程缓存失效。

目标：

1. PostgreSQL 中可追踪、可恢复的 Chunk 是唯一规范事实源。
2. 用户边界内分别执行 Dense Top-N 与 Sparse Top-N。
3. PostgreSQL、Chroma、BM25 使用同一稳定 `chunk_id`。
4. RRF 只融合排名，不直接相加异构原始分数。
5. PostgreSQL 保存权威语料代次，Redis 只做加速与通知。
6. 每个 API 进程维护用户级不可变 BM25 快照，不按请求全量重建。
7. 补齐上传、删除、重建、重试、补偿、隔离和可观测性。
8. 在同一语料、用户、Gold Set 和 Top-K 下比较三种检索模式。

本阶段不实现 Cross-Encoder Reranker、分布式检索服务、OpenSearch/Elasticsearch，也不推测修复旧向量归属。

## 2. 当前架构

### 2.1 在线调用链

~~~text
JWT 认证 user_id
  -> AgentState.user_id
  -> Router
  -> search_pdf_knowledge
  -> Chroma where={user_id}，召回 top_k * 5
  -> 对 Dense 候选临时构建 BM25
  -> 候选集内 RRF
  -> Dense distance 证据门控
  -> Answer 或拒答
~~~

主要入口：`app/tools/search_pdf_knowledge.py`、`app/rag/vector_store.py`、`app/rag/bm25_store.py`、`app/agents/nodes.py`。

### 2.2 当前入库链路

~~~text
Document/Task -> ARQ Worker -> 解析/清洗/分块
-> 按列表位置生成 doc{document_id}_chunk{i}
-> Chroma add
-> 删除旧 SQL Chunk
-> 写入新 SQL Chunk
-> Document=completed
~~~

### 2.3 当前删除链路

删除接口先忽略文件删除异常，再调用当前不存在的 `VectorStore.delete_by_ids` 并吞掉异常，最后硬删除 SQL Document/Chunk。结果可能是 SQL 已删除但 Chroma 仍残留。

## 3. 审计发现

### 3.1 检索实现

- 在线 `hybrid_search` 仅在 Dense 候选中计算 BM25，不能补回 Dense miss。
- 在线 BM25 每请求重建，RRF `k=60` 硬编码，并可能返回零分文档。
- `scripts/stage5_hybrid.py` 是全集合 Dense/BM25 独立召回实验，但没有用户过滤、服务生命周期和一致性处理。
- 历史 `MRR=0.900` 来自 18 条可回答问题的离线原型，不能作为当前在线指标。
- 当前没有 Cross-Encoder Reranker。

### 3.2 ID 与生命周期

- Chunker 生成 UUID，但 Worker 丢弃它并使用列表位置 ID。
- SQL 使用自增 `DocumentChunk.id`，另有 nullable `chroma_id`，没有统一稳定 ID。
- Chroma 使用 `add`，重试可能重复 ID；Chroma 成功而 SQL 失败会产生孤儿向量。
- 重分块数量减少可能留下旧向量。
- Worker 未重新验证任务 `user_id` 与 `Document.user_id`。

### 3.3 用户隔离漏洞

直接文献检索支持 `user_id` 过滤，但 `suggest_pipeline` 内部调用检索时没有传认证用户 ID。它必须作为独立安全修复先完成，且客户端/LLM 提供的 user_id 必须被丢弃。

### 3.4 真实数据质量

只读检查得到：

- Chroma 共 398 条，365 条正文非空，33 条为空。
- 287 条缺少 `document_id`，并存在重复文本。
- 本地 SQL 没有与 398 条向量对应的完整 Document/Chunk 事实记录。
- 365 条非空 Chunk 中 347 条明显英文主导。
- 当前 100 条 Agent 评估问题均含中文，75 条同时含英文术语。

因此不能按文本相似度猜测旧向量归属，也不能直接把不可追踪向量写入新 SQL 事实表。

### 3.5 评估边界

| 结果 | 范围 | 定位 |
|---|---|---|
| 18 条可回答问题，MRR 0.900 | 离线全库 Hybrid | 历史基线，不代表在线 |
| 100 条，工具选择 96% | Agent Router/Tool | 不是检索指标 |
| 历史 20 条，95% | 旧端到端评估 | 历史结果 |
| 历史 15 条，HitRate@3 93.3% | 旧语料检索 | 不与新结果混算 |
| 新线上同构评估 | 尚未执行 | 实施后单独报告 |

## 4. 目标架构

~~~text
                       PostgreSQL
       Document / DocumentChunk / UserRetrievalIndexState
                 corpus_generation（权威）
                         |
             +-----------+-----------+
             |                       |
       Chroma Dense              API 进程内
       user_id 过滤          BM25IndexSnapshot
             |                 built_generation
             +-----------+-----------+
                         |
              按稳定 chunk_id 做 RRF
                         |
                  证据门控 / 拒答

Redis：缓存权威代次、通知失效、保护共享写操作；
       不保存权威版本，不共享 Python BM25 对象。
~~~

查询时先取得权威 `corpus_generation`，比较本地 `built_generation`。过期时进入本进程 user 级 SingleFlight，从 SQL 加载有效 Chunk，构建新快照并在代次复核后原子替换。随后 Dense 与 Sparse 独立召回，按 chunk_id 融合，再经 SQL 校验和证据门控。

## 5. 为什么选择方案 A

方案 A 最符合当前规模与代码基础：

- 当前只有数百 Chunk，用户级重建成本可控。
- PostgreSQL 已是业务数据目标事实源，便于恢复、审计和隔离。
- 不增加第三个持久化检索数据源。
- Redis 故障时可从 PostgreSQL 恢复。
- 不可变快照适合读多写少，查询无需全局读锁。
- 可保留 Dense-only 和候选集 Hybrid 用于灰度与回滚。

暂不选择本地持久化 BM25：它会增加文件锁、格式迁移、损坏恢复和第三份数据一致性。暂不选择 OpenSearch/Elasticsearch：部署、映射、权限和同步成本与当前规模不匹配。用户语料达到十万级 Chunk 或需要跨实例共享稀疏索引时再评估。

## 6. PostgreSQL、Redis 与进程内快照职责

### PostgreSQL

- 保存 Document/Chunk 规范事实、稳定 chunk_id、有效代次和软删除状态。
- 保存用户权威 `corpus_generation`。
- 在同一事务完成文档有效代次切换和 generation 递增。
- Redis 丢失后提供恢复依据。

### Redis

- 缓存 PostgreSQL generation，发布失效通知。
- 为上传、删除、重建提供文档/用户级共享写锁。
- Redis 值只能由 PostgreSQL权威值刷新，不能反向覆盖 PostgreSQL。
- 不保存、不共享进程内 BM25 对象。

建议键：

~~~text
rag:corpus-generation:{user_id}
rag:index-invalidated:{user_id}
rag:document-write-lock:{document_id}
rag:user-index-write-lock:{user_id}
~~~

### API 进程内快照

- 每个进程独立维护；保存自己的 `built_generation`。
- 快照不可变，构建完成后原子替换引用。
- 构建失败保留旧快照，但必须标记 stale/degraded。
- Redis 分布式锁不能让其他进程直接获得该内存对象。

## 7. 数据模型迁移

迁移必须使用 Alembic；当前仓库虽声明依赖，但未发现规范迁移目录，不能继续仅依赖 `create_all`。

### 7.1 `UserRetrievalIndexState`

建议表名 `user_retrieval_index_states`：

| 字段 | 约束/职责 |
|---|---|
| `user_id` | FK users.id，主键 |
| `corpus_generation` | BigInteger，非空，默认 0；权威语料代次 |
| `bm25_status` | empty/ready/rebuilding/failed；运维状态，不代表每个进程缓存 |
| `last_indexed_generation` | 最近一次完成构建验证的代次；不能推断所有进程已更新 |
| `last_error` | 最近错误摘要 |
| `updated_at` | 带时区更新时间 |

generation 使用原子 SQL 自增，并与 Document READY/DELETED 切换处于同一事务。

### 7.2 `Document`

建议新增：

- `content_sha256`：同用户上传去重和审计。
- `active_generation`：当前查询可见代次。
- `pending_generation`：正在构建代次。
- `failure_step`、`retry_count`、`updated_at`、`deleted_at`。

目标状态：

~~~text
PENDING -> INDEXING -> READY
                    -> PARTIAL_FAILED
READY -> INDEXING（重建）
READY -> DELETING -> DELETED
                 -> PARTIAL_FAILED
~~~

旧小写状态通过迁移兼容转换。

### 7.3 `DocumentChunk`

新增：

- `chunk_id`：UUID/String(36)，非空、唯一、不可变。
- `document_generation`：非空。
- `content_sha256`：完整性和重复诊断。
- `deleted_at`：软删除标记。

约束：

~~~text
UNIQUE(chunk_id)
UNIQUE(document_id, document_generation, chunk_index)
~~~

`chroma_id` 在 expand/contract 迁移期保留并回填，最终废弃。新链路始终满足：

~~~text
PostgreSQL DocumentChunk.chunk_id
== Chroma collection id
== BM25 snapshot chunk_id
~~~

随机 UUID 必须先持久化到 SQL；ARQ 重试复用同一文档代次已有 UUID。

## 8. 上传、删除与重建状态机

### 8.1 首次上传

~~~text
1. JWT 确定 user_id，计算 PDF content_sha256。
2. 创建 Document=PENDING 和幂等 Task。
3. Worker 获取 document 写锁，重新查询并校验归属。
4. 设置 INDEXING，分配 pending_generation。
5. 解析、清洗、分块。
6. SQL 写入该代 Chunk 并生成 UUID；提交。
7. 从 SQL 读取同代 Chunk，按持久化 chunk_id Chroma upsert。
8. 验证 SQL/Chroma 数量、ID、user/document/generation metadata。
9. 同一 SQL 事务切换 active_generation、设 READY、递增 corpus_generation。
10. 提交后刷新 Redis 并发布失效通知。
~~~

SQL Chunk 提交后 Worker 即使崩溃，重试仍复用相同 ID。

### 8.2 重新解析

保留旧 active_generation 对外服务；新 pending_generation 使用新 UUID。Chroma upsert 并验证后，SQL 原子切换 active_generation、递增用户 generation，最后异步清理旧代。清理失败记录补偿任务，不回退已验证的新代。

### 8.3 删除

~~~text
READY -> DELETING
-> 按 SQL chunk_id 删除并验证 Chroma
-> SQL 软删除 Chunk，Document=DELETED
-> 同一事务递增 corpus_generation
-> 提交后刷新 Redis
-> 最后尽力删除本地文件
~~~

Chroma/SQL 失败进入 `PARTIAL_FAILED`，保留 Chunk ID 并允许幂等重试；接口不能假装 204 成功。检索端以 SQL READY + active_generation 为准，丢弃 DELETING、DELETED、孤儿或错误用户 Dense hit。

### 8.4 向量生命周期能力

`VectorStore` 计划新增：

- `upsert_documents`
- `delete_by_ids`
- `get_by_ids`
- `delete_by_document_generation`

一致性检查器报告 SQL 当前代缺向量、Chroma 孤儿、metadata 不一致、空正文和正文哈希不一致。默认只读，修复需显式确认。

## 9. 并发模型

每个 API 进程维护：

~~~text
snapshots[user_id] -> BM25IndexSnapshot
singleflight[user_id] -> asyncio.Lock/Future
~~~

快照至少保存：

~~~text
user_id
built_generation
chunk_ids
tokenized_corpus
documents/metadata
bm25_index
created_at
tokenizer_version
bm25_config_version
~~~

规则：

- 同一进程同一用户只有一个构建者；其他请求等待或 Dense-only 降级。
- 进入锁后再次比较版本。
- 构建完成前再次检查权威 generation；变化则丢弃过期结果。
- 完整成功后才原子替换；失败保留旧对象。
- 不同用户可并行构建。

当前 Agent 工具链是同步调用。推荐将检索服务与 Tool 链路逐步异步化，使用 LangGraph/FastAPI 异步路径；过渡方案是同语义的 `threading.Condition/Lock`。禁止在运行中的事件循环里调用 `asyncio.run`。此项需要实施前确认。

上传、删除、重建使用 Redis 写锁，但 generation 正确性最终由 PostgreSQL 原子更新/行锁保证。

## 10. 一致性与补偿

PostgreSQL 与 Chroma 只能做到可验证、可补偿的最终一致，不宣称强事务。

| 故障点 | 状态与补偿 |
|---|---|
| SQL Chunk 提交前失败 | 回滚 SQL，无 Chroma 新数据 |
| SQL 已提交，Chroma 失败 | INDEXING/PARTIAL_FAILED；复用 chunk_id 重试 upsert |
| Chroma 成功，READY 切换失败 | 新代不可见；重试切换或删除新代向量 |
| READY 成功，Redis 失败 | PostgreSQL 已权威生效；查询回源，后台补发通知 |
| 删除 Chroma 失败 | PARTIAL_FAILED，保留 Chunk ID 并重试 |
| 快照构建失败 | 旧快照标 stale 或 Dense-only，记录 degraded |

## 11. 用户隔离

- `search_pdf_knowledge` 和 `suggest_pipeline` 内部检索必须要求认证 user_id。
- Router/LLM/客户端提交的 user_id 一律删除，由代码注入。
- SQL BM25 语料按 `Document.user_id`、READY、active_generation 过滤。
- Chroma 查询强制 `where={\"user_id\": current_user.id}`。
- 融合前用 SQL 二次校验 user、状态和代次。
- 缓存键、SingleFlight 和 generation 按 user_id 隔离。
- 用户 A 的写入或失效不能影响用户 B 快照。

该漏洞先独立修复并提交，不与 BM25 架构混合。

## 12. 旧数据处理

先设置人工确认门：

1. 备份 `data/chroma`、关系数据库和上传目录。
2. 生成只读质量报告及可重建/不可追踪/需人工判断清单。
3. 查找原始 PDF，以文件哈希建立可靠映射。
4. 报告确认前不删除、移动或改写现有 398 条向量。

找到原始 PDF 的数据按新链路重建。无法确认来源的数据导出并移入 legacy collection，不进入在线检索。禁止按文本相似度猜 document_id。

## 13. 统一分词策略

### 13.1 语料事实

英文论文正文占主导，但用户问题以中文为主并混有英文软件名、基因名、物种名和 accession。典型实体包括 `mTERF1`、`XP_072092248.1`、`Arachis hypogaea`、`dN/dS`、`L-INS-i`。

### 13.2 第一阶段规范

新增共享 tokenizer，在线 BM25、离线评估、索引重建和单测只能调用同一实现：

1. 使用 Unicode `casefold()` 归一化大小写。
2. 优先保留英文/数字及内部 `. _ - /` 的完整 token。
3. 同时生成必要子词，例如保留 `xp_072092248.1` 并生成 `xp`、`072092248`、`1`。
4. `dN/dS`、`L-INS-i` 同时保留完整规范 token 和子词。
5. 物种双名保留为两个词；不一刀切删除短 token。
6. 第一阶段不引入未经评估的中文分词依赖。中文连续文本采用明确、可测试的 Han 字符串/字符 bigram 实验，与英文实体基线比较后决定是否启用。
7. 停用词默认关闭，只有同构评估证明收益后才加入版本化列表。
8. 空文本产生空 token，不进入快照。

定义 `TOKENIZER_VERSION`。快照身份为：

~~~text
(user_id, corpus_generation, tokenizer_version, bm25_config_version)
~~~

Tokenizer 变化需要重建，但不应伪造语料 corpus_generation 变化。

## 14. 三种检索模式

配置建议：

~~~text
RETRIEVAL_MODE=dense_only|candidate_bm25_rrf|global_bm25_rrf
DENSE_TOP_N=20
SPARSE_TOP_N=20
RRF_K=60
FINAL_TOP_K=3
BM25_REBUILD_WAIT_MS=...
~~~

- `dense_only`：Chroma 用户过滤 + SQL 有效性校验，用于基线和回滚。
- `candidate_bm25_rrf`：保留候选内 BM25 基线，统一 tokenizer/ID，但仍不能补 Dense miss。
- `global_bm25_rrf`：Dense 与用户快照独立 Top-N，Sparse 过滤 `score <= 0`，按 chunk_id RRF，SQL 校验后返回 Final Top-K。

结果记录 `retrieval_sources`、Dense rank/distance、Sparse rank/score、RRF score、文档/页码/章节和代次。原始分数不直接相加。

当前门控只看 Dense distance。目标门控按通道区分：Sparse 需正分、最低术语匹配、SQL 有效和非空正文；双通道记录共同证据；无有效证据确定性拒答。

降级：

- BM25 缺失/超时：Dense-only + degraded + 调度重建。
- Redis 不可用：回源 PostgreSQL。
- Chroma 不可用：仅在 Sparse 门控完成后允许 Sparse-only，否则明确失败。
- SQL 不一致：丢弃 hit 并记录 consistency_error。
- 两通道故障：返回检索不可用，不伪装成知识库无答案。

## 15. 测试矩阵

### 用户隔离

- A 的直接检索和 pipeline 检索都不能返回 B Chunk。
- 伪造 user_id 被认证态覆盖。
- A 的 generation 变化不影响 B 快照。

### 模型与生命周期

- chunk_id 非空、唯一、不可变，且与 Chroma ID 一致。
- ARQ 重试复用 UUID；upsert 不增加记录数。
- Worker 拒绝 user_id 不匹配。
- 新代失败时旧代仍可检索。
- READY/DELETED 只递增一次 generation。
- delete_by_ids 真实删除；失败进入 PARTIAL_FAILED。
- 重分块更少时旧代向量被清理。

### 快照与并发

- 同进程同用户并发只构建一次，不同用户可并行。
- generation 一致不加载全量 SQL。
- 版本变化后原子重建。
- 构建中版本变化不发布过期快照。
- 构建失败保留旧快照并降级。
- Redis 清空/不可用时从 PostgreSQL 恢复。
- Redis 旧值不能覆盖 PostgreSQL 新值。

### Tokenizer、融合与故障

- 在线、离线、重建复用同一 tokenizer。
- 覆盖上述生物学实体、大小写和空文本。
- Sparse 零分过滤。
- Dense/Sparse 按 chunk_id 去重，RRF k 可配置，来源标注正确。
- 构造 Dense miss 被 Sparse 补回。
- Redis、Chroma、SQL、BM25 各故障有确定行为。
- 只读一致性脚本能检出孤儿、缺失和 metadata 错误。

## 16. 评估方案

在同一规范语料、测试用户、Gold Set、Top-K、相关性判定和 tokenizer 下比较：

~~~text
Dense-only
Candidate-BM25 + RRF
Global-BM25 + RRF
~~~

不增加虚假 Reranker 对照。报告：

- 总数、可回答/不可回答数。
- HitRate@3、Recall@K、MRR、无结果率。
- p50/p95 延迟。
- 首次构建、缓存命中和版本变化后重建时间。
- Dense miss 被 Sparse 补回案例。
- 新方案退化案例和故障降级次数。

新报告必须带语料 manifest 和配置，不覆盖 `docs/hybrid_baseline.json`。

## 17. 回滚方案

1. 新代码合入后默认保持 `candidate_bm25_rrf`，评估后再切全库模式。
2. 全库模式异常时切 `dense_only`，无需删除新数据。
3. 数据库采用 expand/contract：新增字段、回填验证、最后加非空约束。
4. 迁移期保留 `chroma_id` 和旧 collection。
5. 旧数据先备份；清理必须二次确认。
6. Redis 故障不回滚 PostgreSQL generation。
7. 新快照只在构建与代次复核成功后替换。

## 18. 分阶段 Commit 计划

现有“工具参数校验 + 证据不足拒答”必须先单独保存，不混入以下提交。

1. `fix: enforce authenticated user isolation in pipeline retrieval`
2. `feat: add stable chunk identity and retrieval index state`
3. `fix: make document vector lifecycle idempotent and recoverable`
4. `docs: add legacy retrieval data quality report`
5. `feat: add user-scoped bm25 snapshot manager`
6. `feat: add independent dense sparse retrieval and rrf fusion`
7. `test: cover retrieval isolation lifecycle and fallback`
8. `eval: compare user-scoped retrieval modes`
9. `docs: document retrieval architecture and verified metrics`

每阶段单独运行相关测试、全量测试和格式检查。旧数据清理不与业务代码提交混合。

## 19. 计划修改文件与测试

本轮均未修改，仅列计划。

预计修改：

- `app/models/document.py`
- `app/models/__init__.py`
- `app/core/config.py`
- `app/api/documents.py`
- `app/worker/tasks.py`
- `app/rag/vector_store.py`
- `app/rag/bm25_store.py`
- `app/tools/search_pdf_knowledge.py`
- `app/tools/pipeline_suggest.py`
- `app/agents/tool_schemas.py`
- `.env.example`
- `docker-compose.yml`（只在配置需注入容器时）

预计新增：

- `app/models/retrieval_index_state.py`
- `app/rag/tokenizer.py`
- `app/rag/bm25_snapshot.py`
- `app/rag/retrieval_service.py`
- Alembic 配置与迁移文件
- 只读一致性审计/重建脚本
- 同构评估脚本与新报告

计划测试：

- `tests/test_pipeline_user_isolation.py`
- `tests/test_chunk_identity.py`
- `tests/test_vector_lifecycle.py`
- `tests/test_retrieval_index_state.py`
- `tests/test_bm25_tokenizer.py`
- `tests/test_bm25_snapshot.py`
- `tests/test_retrieval_fusion.py`
- `tests/test_retrieval_fallback.py`
- `tests/test_retrieval_consistency.py`

单测使用临时 SQL、Fake Chroma/Fake Redis；另保留 PostgreSQL/Redis/Chroma 集成测试验证真实事务与 metadata 过滤。

## 20. 已知局限

- 多 API 进程会重复占用 BM25 内存和构建时间。
- PostgreSQL 与 Chroma 只能最终一致。
- 本地 Chroma/SQLite 的并发写能力有限。
- 第一阶段不承诺高质量中文分词，策略由评估决定。
- BM25 score 与 Dense distance 不可横向比较。
- 当前没有 Reranker。
- 旧 398 条向量需要重建或隔离。
- Alembic 基础设施尚需规范初始化。

## 21. 当前实现与目标实现对照

| 维度 | 当前 | 目标 |
|---|---|---|
| Sparse 范围 | Dense 候选 | 用户全库独立召回 |
| BM25 构建 | 每请求 | generation 控制的不可变快照 |
| 版本事实源 | 无 | PostgreSQL corpus_generation |
| Redis | ARQ | 代次缓存、通知、写锁 |
| Chunk ID | 列表位置/nullable chroma_id | SQL 先持久化 UUID |
| Chroma 写入 | add | upsert + 验证 |
| 删除 | 吞异常，可能残留 | 状态机 + 补偿 |
| 用户隔离 | pipeline 路径缺失 | 认证态强制注入 + SQL 校验 |
| 融合 | 候选集 RRF | 独立召回按 chunk_id RRF |
| 门控 | Dense distance | 通道感知门控 |
| 评估 | 多套历史结果 | 同构三模式比较 |
| Reranker | 未实现 | 本阶段不实现、不宣称 |

## 22. 面试口径

当前可安全陈述：

> 当前在线系统先由 Chroma 按用户过滤进行稠密召回，再在候选集内计算 BM25 并用 RRF 重排，因此 BM25 不能补回 Dense 漏召回的 Chunk。项目另有离线全库 Hybrid 原型，在 18 条可回答问题上得到 MRR 0.900，但该结果不能代表在线用户级检索。下一阶段计划以 PostgreSQL 规范 Chunk 为事实源，为每个用户维护带持久代次和进程内 SingleFlight 的 BM25 快照，实现 Dense 与 Sparse 独立召回，同时补齐上传、删除、重建和故障降级。

完成实现和新评估前禁止声称“已升级完成”“在线 MRR 0.900”“已实现 Reranker”“三存储强一致”或“Redis 跨进程共享 BM25 索引”。

## 23. 实施前待确认决策

1. **调用模型**：是否将检索服务与 Agent Tool 链路异步化以使用原生 `asyncio` SingleFlight；若面试前风险过高，是否先用同语义同步进程锁过渡。
2. **状态命名**：是否接受 `PENDING/INDEXING/READY/PARTIAL_FAILED/DELETING/DELETED` 并迁移旧状态。
3. **旧数据策略**：是否确认先备份和生成只读报告，二次确认后才重建或隔离。
4. **迁移基础设施**：是否先初始化 Alembic，再进行 expand/contract 迁移。
5. **默认模式**：是否保持 `candidate_bm25_rrf`，同构评估后再切 `global_bm25_rrf`。
6. **中文 tokenizer**：是否确认第一阶段不引入新中文分词依赖，先建立共享可版本化基线。
7. **删除语义**：是否允许删除改为可观察的异步状态机，而非立即返回 204。

