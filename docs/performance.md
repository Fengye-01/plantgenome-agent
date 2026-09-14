# PlantGenome Agent - 性能基线

> 记录 10 个 Demo Case 的端到端耗时，作为性能基线。
> 测试环境：Windows 11 + Python 3.12 + 硅基流动 API（Qwen2.5-7B-Instruct）+ BGE-m3 本地嵌入。

---

## 测试环境

| 项目 | 配置 |
|------|------|
| 操作系统 | Windows 11 |
| Python | 3.12.14 |
| LLM | 硅基流动 API - Qwen/Qwen2.5-7B-Instruct |
| Embedding | BAAI/bge-m3（本地运行，CPU） |
| 向量数据库 | Chroma（本地持久化） |
| Agent 框架 | LangGraph 1.2.11 |
| 知识库 | 10 个生信 PDF，287 个 chunks |

---

## 性能基线（10 个 Demo Case）

| # | 场景 | 意图 | Router耗时 | Tool耗时 | Answer耗时 | 总耗时 | 状态 |
|---|------|------|-----------|---------|-----------|--------|------|
| 1 | PAML omega 解释 | literature_search | ~2s | ~1s | ~8s | ~11s | ✅ |
| 2 | OrthoFinder 输入格式 | literature_search | ~2s | ~1s | ~7s | ~10s | ✅ |
| 3 | MAFFT 算法选择 | literature_search | ~2s | ~1s | ~9s | ~12s | ✅ |
| 4 | mTERF 家族功能 | literature_search | ~2s | ~1s | ~10s | ~13s | ✅ |
| 5 | FASTA 序列统计 | fasta_analysis | - | ~0.1s | - | ~0.1s | ✅ |
| 6 | CpG 岛扫描（阳性） | cpg_scan | - | ~0.05s | - | ~0.05s | ✅ |
| 7 | CpG 岛扫描（阴性） | cpg_scan | - | ~0.05s | - | ~0.05s | ✅ |
| 8 | mTERF 进化流程推荐 | pipeline_suggest | ~2s | ~15s | ~8s | ~25s | ✅ |
| 9 | 选择压力分析流程 | pipeline_suggest | ~2s | ~18s | ~9s | ~29s | ✅ |
| 10 | 直接问候 | direct_answer | ~2s | - | ~3s | ~5s | ✅ |

> 注：FASTA/CpG 用例走侧边栏专用工具（前端直接调用纯计算工具，不经过 LLM），所以没有 Router/Answer 耗时。

---

## 性能分析

### 1. 各阶段耗时占比（以 literature_search 为例）

```
总耗时 ~11s
├── Router 节点: ~2s (18%)    — LLM 意图分类
├── Tool 节点: ~1s (9%)        — RAG 检索（BGE-m3 嵌入 + Chroma 查询）
└── Answer 节点: ~8s (73%)    — LLM 生成最终回答
```

### 2. 瓶颈分析

| 瓶颈 | 原因 | 优化方向 |
|------|------|---------|
| LLM 推理慢 | 7B 模型通过 API 调用，网络延迟 + 推理时间 | 换用更大模型的 API 或本地部署更快的模型 |
| suggest_pipeline 慢 | 工具内部需要 RAG 检索 + LLM 生成流程（两次 LLM 调用） | 缓存常用流程，减少重复生成 |
| BGE-m3 嵌入慢 | 本地 CPU 运行，首次查询需要加载模型 | 模型已缓存，后续查询快；可换用 GPU |

### 3. 纯计算工具性能

| 工具 | 输入规模 | 耗时 | 复杂度 |
|------|---------|------|--------|
| parse_fasta_stats | 5 条序列，~5000bp | ~0.1s | O(n) |
| scan_cpg_islands | 1000bp 序列 | ~0.05s | O(n) |
| scan_cpg_islands | 10000bp 序列 | ~0.1s | O(n) |

纯计算工具性能优秀，大序列也能毫秒级处理，不会卡死。

---

## 优化记录

### D12 性能优化：大序列不经过 LLM

**问题**：大 FASTA/DNA 序列作为 query 传给 LLM（router 和 answer 节点），导致 LLM 处理大输入超时、页面卡死。

**解决方案**：侧边栏的 FASTA/CpG 按钮直接调用纯计算工具，完全不经过 LLM。

**优化效果**：
- 优化前：大序列走完整 Agent 链路，可能 >60s 或卡死
- 优化后：大序列直接调用纯计算工具，<0.1s 完成

---

## 面试可讲的性能数据

> "本地 7B 模型端到端平均响应时间约 12 秒（文献检索类），主要瓶颈在 LLM 推理（占 73%）。纯计算工具（FASTA 统计、CpG 岛扫描）走前端直接调用，毫秒级响应，大序列也不会卡死。如果换用 API 模型或更大模型，LLM 推理时间可以缩短到 3-5 秒。"

> "针对大序列输入导致页面卡死的问题，我做了性能优化：识别出 FASTA 统计和 CpG 岛扫描是纯计算工具，不需要 LLM 理解意图，所以在前端直接调用这些工具，完全绕过 LLM，把响应时间从可能的 >60s 降到 <0.1s。"

---

*PlantGenome Agent Performance Baseline · D13*
