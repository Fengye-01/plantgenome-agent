# PlantGenome Agent - 第3周包装素材清单

> D14 准备：列出第3周需要的所有包装材料，逐项打勾完成。
> 第3周目标：Streamlit 完善 + Demo + 评估 + Docker + 简历 + 投递。

---

## 一、README 需要

### 1. 项目简介
- [ ] 一句话介绍（标题下方）
- [ ] 详细描述（3-5 句，说明解决什么问题、核心功能、技术亮点）
- [ ] 项目定位（面向什么用户、什么场景）

### 2. 架构图
- [ ] 系统架构图（用户 → Streamlit → FastAPI → LangGraph Agent → RAG/Tools）
- [ ] PDF RAG 流程图（PDF → 解析 → 清洗 → 分块 → 嵌入 → Chroma → 检索 → LLM）
- [ ] Agent 状态机图（router → [条件边] → tool/answer → END）
- [ ] 用 Mermaid 或 draw.io 绘制，放在 README 顶部

### 3. 技术栈列表
- [ ] 后端：Python, FastAPI, Uvicorn
- [ ] Agent：LangGraph, LangChain Core
- [ ] RAG：Chroma, BGE-m3, PyMuPDF
- [ ] 前端：Streamlit
- [ ] 工具：FASTA 统计, CpG 岛扫描, 流程推荐
- [ ] 部署：Docker（V2）

### 4. 快速启动步骤
- [ ] 环境要求（Python 3.10+, conda）
- [ ] 安装依赖（pip install -r requirements.txt）
- [ ] 配置 .env（LLM API Key, 模型选择）
- [ ] 准备知识库（放入 data/pdfs/，运行入库脚本）
- [ ] 启动后端（python -m app.api.main）
- [ ] 启动前端（streamlit run frontend/app.py）
- [ ] 验证（打开 http://localhost:8502）

### 5. Demo 截图/GIF
- [ ] 主界面截图
- [ ] 文献问答截图（带 sources）
- [ ] FASTA 统计截图（带表格）
- [ ] CpG 岛扫描截图
- [ ] 流程推荐截图（带步骤卡片）
- [ ] Agent 执行过程面板截图
- [ ] 1 分钟演示 GIF（可选，放在 README 顶部）

### 6. 目录结构说明
- [ ] 项目目录树（app/, frontend/, tests/, docs/, data/, scripts/）
- [ ] 每个目录的作用说明
- [ ] 核心文件说明

### 7. 评估结果
- [ ] 检索评估：HitRate@1 = 93.3%, HitRate@3 = 93.3%（15 题测试集）
- [ ] RAG 端到端测试：4/5 = 80% 通过
- [ ] Router 分类准确率：9/10 = 90%
- [ ] Agent 端到端测试：5/5 = 100%
- [ ] 性能基线：文献检索类平均 ~12s，纯计算工具 <0.1s

### 8. Future Work（V2/V3 规划）
- [ ] V2：MQE/HyDE 高级检索策略
- [ ] V2：混合检索（BM25 + 向量）+ Reranker
- [ ] V2：RAGAS 自动化评估
- [ ] V2：NCBI API 工具调用
- [ ] V2：多轮对话记忆
- [ ] V3：MCP 协议支持
- [ ] V3：OCR 支持扫描件 PDF
- [ ] V3：Docker 容器化部署

---

## 二、简历需要

### 1. 项目名称 + 时间
- [ ] 项目名称：PlantGenome Agent｜植物基因组研究流程智能助手
- [ ] 时间：2026.09 - 至今

### 2. 项目简介（2-3 句）
- [ ] 面向植物比较基因组与基因家族分析场景，设计并实现基于 PDF 文献知识库的 RAG + Tool Calling 智能体系统
- [ ] 支持上传论文和软件说明书 PDF，自动完成文本解析、语义分块、向量化入库和来源追踪
- [ ] 通过 LangGraph Agent 根据用户意图调用文献检索、FASTA 统计、CpG 岛识别和分析流程推荐等工具

### 3. 技术栈
- [ ] Python、FastAPI、LangGraph、Chroma、PyMuPDF、BGE-m3、Streamlit、Redis（V2）、Docker（V2）

### 4. 4 条负责内容（带量化指标）
- [ ] ① 实现 PDF 文献解析与知识库构建流程，支持 Markdown 智能分块（标题层次感知 + Token 精确控制），完成 10 篇生信文献、287 个 chunks 的向量化入库
- [ ] ② 基于 Chroma + BGE-m3 构建向量检索模块，支持 top-k 召回与来源引用，在自建 15 题测试集上 HitRate@3 达到 93.3%
- [ ] ③ 基于 LangGraph 构建 Router-Tool-Answer 三阶段 Agent 流程，实现 4 类意图分类（准确率 90%）与 4 个生信工具自动调用，5 类端到端测试 100% 通过
- [ ] ④ 使用 Streamlit 构建演示前端，支持 PDF 上传、聊天问答、工具结果结构化展示和 Agent 执行过程可视化；针对大序列输入做性能优化，纯计算工具响应时间 <0.1s

### 5. 评估数据
- [ ] HitRate@3 = 93.3%（15 题检索测试集）
- [ ] Router 分类准确率 = 90%（10 题）
- [ ] Agent 端到端通过率 = 100%（5 类）
- [ ] 纯计算工具响应 <0.1s，文献检索平均 ~12s

---

## 三、面试 Q&A 需要

### 1. 项目难点
- [ ] PDF 解析难点：双栏排版、表格、公式 → 当前方案：PyMuPDF + Markdown 转换，MVP 只处理文本型 PDF
- [ ] 分块策略难点：固定字符数会切断语义 → 方案：Markdown 智能分块（标题层次感知 + 段落语义保持 + Token 精确控制 + 智能 overlap）
- [ ] Agent 路由难点：7B 模型输出 JSON 不稳定 → 方案：多级容错解析（直接解析 + 正则修复 + 关键词提取 + 强制映射）
- [ ] 大序列性能难点：大序列走 LLM 导致卡死 → 方案：纯计算工具前端直接调用，绕过 LLM

### 2. 技术选型原因
- [ ] 为什么用 PyMuPDF？：性能好，支持文本和 Markdown 双模式提取，活跃维护
- [ ] 为什么用 Chroma 而不是 Milvus？：轻量级，Python 原生，适合 MVP 和本地开发；Milvus 适合企业级大规模场景
- [ ] 为什么用 BGE-m3？：支持多语言（中英文混合），对生物专有名词支持好，开源可本地部署
- [ ] 为什么用 LangGraph？：状态机 Agent 框架，大厂认可，支持复杂工作流和条件边，比 LangChain Agent 更可控
- [ ] 为什么用 Streamlit？：Python 原生，适合 AI 应用快速搭建和演示，学习成本低

### 3. RAG 优化手段
- [ ] 分块策略：Markdown 智能分块 vs 固定字符数
- [ ] Token 精确控制：用 BGE-m3 tokenizer 计算 token 数，而不是字符数
- [ ] Prompt 优化：三轮 A/B 对比，确定 max_tokens=2048, temperature=0.3, 300-800 字
- [ ] 来源引用：回答标注 sources，降低幻觉
- [ ] V2 规划：混合检索（BM25+向量）、Reranker、MQE/HyDE

### 4. 工具调用设计
- [ ] 为什么选这 4 个工具？：覆盖文献检索（RAG）、序列统计（纯计算）、CpG 分析（算法）、流程推荐（LLM+RAG），展示不同类型的工具调用
- [ ] 工具注册机制：用 ToolExecutor 统一注册，工具描述详细（会被拼接到 Agent Prompt）
- [ ] 异常处理：tool_node 用 try-except 包裹，失败不崩溃，execution_log 记录
- [ ] 参数提取：router 提取 + tool_node 兜底提取（正则），双重保障

### 5. 评估方法
- [ ] 检索评估：HitRate@3，自建 15 题测试集（人工标注标准答案和参考文档）
- [ ] Router 评估：分类准确率，10 题测试集
- [ ] Agent 端到端评估：5 类意图各 1 题，检查是否正确调用工具并生成回答
- [ ] 为什么不用 RAGAS？：MVP 阶段用简单指标，RAGAS 需要 LLM 作为评判器，成本高；V2 引入
- [ ] HitRate@3 的局限：只衡量检索是否命中，不衡量回答质量；V2 补充 Faithfulness 和 Answer Relevancy

### 6. 局限性和改进方向
- [ ] 单工具调用：当前一次只调用一个工具 → V2 增加 ReAct 循环，支持多轮工具调用
- [ ] 无多轮记忆：每次对话独立 → V2 增加 Memory 模块
- [ ] 7B 模型限制：复杂推理和 JSON 输出不稳定 → 换用更大模型或 API 模型
- [ ] 文本型 PDF：不支持扫描件/OCR → V3 增加 OCR
- [ ] 简单评估：只有 HitRate@3 → V2 引入 RAGAS
- [ ] 无 MCP：工具硬编码 → V3 引入 MCP 协议

---

## 四、其他包装材料

### 1. GitHub 仓库
- [ ] 仓库名称：plantgenome-agent
- [ ] 仓库描述：面向植物比较基因组研究的 PDF-RAG + Tool Calling 智能体
- [ ] Topics：rag, agent, langgraph, genomics, bioinformatics, llm, streamlit
- [ ] .gitignore（排除 data/chroma/, .env, __pycache__, .streamlit/）
- [ ] LICENSE（MIT）
- [ ] requirements.txt

### 2. 演示视频/GIF
- [ ] 1 分钟演示视频（可选，上传到 B站/YouTube）
- [ ] 演示 GIF（放在 README 顶部，展示核心功能）

### 3. 博客文章（可选）
- [ ] 技术博客：从零构建一个植物基因组 RAG+Agent 智能体
- [ ] 发布到知乎/CSDN/个人博客

---

## 完成进度追踪

| 类别 | 已完成 | 总计 | 进度 |
|------|--------|------|------|
| README | 0 | 8 | 0% |
| 简历 | 0 | 5 | 0% |
| 面试 Q&A | 0 | 6 | 0% |
| 其他 | 0 | 3 | 0% |
| **总计** | **0** | **22** | **0%** |

---

*PlantGenome Agent Packing Checklist · D14*
