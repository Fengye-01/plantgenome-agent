# PlantGenome Agent - Demo Cases（演示用例）

> 10 个稳定可演示的用例，覆盖 4 类意图 + 边界场景。
> 每个用例都经过实际运行验证，可用于面试演示和 README 展示。

---

## 用例总览

| # | 场景 | 意图 | 工具 | 演示亮点 |
|---|------|------|------|---------|
| 1 | PAML omega 解释 | literature_search | search_pdf_knowledge | RAG 检索 + 来源引用 |
| 2 | OrthoFinder 输入格式 | literature_search | search_pdf_knowledge | 软件文档检索 |
| 3 | MAFFT 算法选择 | literature_search | search_pdf_knowledge | 工具比较查询 |
| 4 | mTERF 家族功能 | literature_search | search_pdf_knowledge | 基因家族文献检索 |
| 5 | FASTA 序列统计 | fasta_analysis | parse_fasta_stats | 纯计算工具 + 表格展示 |
| 6 | CpG 岛扫描（阳性） | cpg_scan | scan_cpg_islands | 滑动窗口算法 + 结果表格 |
| 7 | CpG 岛扫描（阴性） | cpg_scan | scan_cpg_islands | 边界场景：无 CpG 岛 |
| 8 | mTERF 进化流程推荐 | pipeline_suggest | suggest_pipeline | 流程规划 + 步骤卡片 |
| 9 | 选择压力分析流程 | pipeline_suggest | suggest_pipeline | 专业分析流程 |
| 10 | 直接问候 | direct_answer | （无工具） | 条件边：不调用工具直接回答 |

---

## 用例详情

### 用例 1：PAML omega 解释

- **场景**：用户询问 PAML 软件中 omega 值的生物学意义
- **用户输入**：`PAML 中的 omega (dN/dS) 值怎么解释？`
- **预期 Agent 链路**：`router → tool(search_pdf_knowledge) → answer`
- **预期意图**：`literature_search`
- **预期工具**：`search_pdf_knowledge`
- **预期输出**：
  - 回答基于 PAML 论文和文档
  - 解释 omega = dN/dS 的含义
  - 说明 omega > 1（正选择）、omega = 1（中性）、omega < 1（纯化选择）
  - 标注来源引用 [1] [2]
- **演示要点**：展示 RAG 检索 + 来源引用，回答专业准确

---

### 用例 2：OrthoFinder 输入格式

- **场景**：用户询问 OrthoFinder 软件的输入要求
- **用户输入**：`OrthoFinder 的输入需要什么格式？`
- **预期 Agent 链路**：`router → tool(search_pdf_knowledge) → answer`
- **预期意图**：`literature_search`
- **预期工具**：`search_pdf_knowledge`
- **预期输出**：
  - 基于 OrthoFinder 论文回答
  - 说明输入是蛋白序列 FASTA 文件，每个物种一个文件
  - 说明文件命名规范（物种名.fasta）
  - 标注来源引用
- **演示要点**：展示软件文档检索能力

---

### 用例 3：MAFFT 算法选择

- **场景**：用户询问 MAFFT 不同比对算法的区别
- **用户输入**：`MAFFT 的 L-INS-i 和 G-INS-i 有什么区别？`
- **预期 Agent 链路**：`router → tool(search_pdf_knowledge) → answer`
- **预期意图**：`literature_search`
- **预期工具**：`search_pdf_knowledge`
- **预期输出**：
  - 基于 MAFFT 论文回答
  - 说明 L-INS-i 适合局部保守的序列，G-INS-i 适合全局保守
  - 说明适用场景和计算复杂度
  - 标注来源引用
- **演示要点**：展示工具比较查询，回答有区分度

---

### 用例 4：mTERF 家族功能

- **场景**：用户询问 mTERF 基因家族在植物中的功能
- **用户输入**：`mTERF 基因家族在植物中的功能是什么？`
- **预期 Agent 链路**：`router → tool(search_pdf_knowledge) → answer`
- **预期意图**：`literature_search`
- **预期工具**：`search_pdf_knowledge`
- **预期输出**：
  - 基于 mTERF 相关综述回答
  - 说明 mTERF 蛋白参与线粒体/叶绿体基因表达调控
  - 说明在植物发育和逆境响应中的作用
  - 标注来源引用
- **演示要点**：展示基因家族文献检索，结合用户研究背景

---

### 用例 5：FASTA 序列统计

- **场景**：用户粘贴 FASTA 序列，要求统计
- **用户输入**：（侧边栏粘贴 5 条序列的 FASTA，点击"分析 FASTA"按钮）
- **预期 Agent 链路**：（前端直接调用纯计算工具，不经过 LLM）
- **预期意图**：`fasta_analysis`
- **预期工具**：`parse_fasta_stats`
- **预期输出**：
  - 序列数量、总长度、平均长度、最短/最长
  - GC 含量
  - 长度分布柱状图
  - 序列详情表格（ID、长度、GC含量）
- **演示要点**：
  - 大序列也能快速处理（纯计算，毫秒级）
  - 结构化展示（metric + chart + dataframe）
  - 执行过程面板显示"前端直接调用"

---

### 用例 6：CpG 岛扫描（阳性）

- **场景**：用户粘贴高 GC 高 CpG 密度的 DNA 序列
- **用户输入**：（侧边栏粘贴 `CGCG` 重复 100 次的序列，点击"扫描 CpG 岛"按钮）
- **预期 Agent 链路**：（前端直接调用纯计算工具，不经过 LLM）
- **预期意图**：`cpg_scan`
- **预期工具**：`scan_cpg_islands`
- **预期输出**：
  - 检测到 1 个 CpG 岛
  - CpG 岛表格（start, end, length, gc_content, obs_exp_ratio）
  - GC 含量对比柱状图
- **演示要点**：
  - 滑动窗口算法（window_size=200, step=100）
  - Gardiner-Garden & Frommer 标准（GC>50%, Obs/Exp>0.6）
  - 阳性结果展示

---

### 用例 7：CpG 岛扫描（阴性）

- **场景**：用户粘贴低 GC 的 DNA 序列（AT 丰富）
- **用户输入**：（侧边栏粘贴 `ATAT` 重复 100 次的序列，点击"扫描 CpG 岛"按钮）
- **预期 Agent 链路**：（前端直接调用纯计算工具，不经过 LLM）
- **预期意图**：`cpg_scan`
- **预期工具**：`scan_cpg_islands`
- **预期输出**：
  - 显示"未检测到符合标准的 CpG 岛"
  - 显示扫描参数（window_size, step, gc_threshold, oe_threshold）
  - 不崩溃，友好提示
- **演示要点**：
  - 边界场景处理
  - 阴性结果友好展示
  - 体现系统的鲁棒性

---

### 用例 8：mTERF 进化流程推荐

- **场景**：用户询问 mTERF 家族进化研究的分析流程
- **用户输入**：`研究 mTERF 家族进化需要什么分析流程？`
- **预期 Agent 链路**：`router → tool(suggest_pipeline) → answer`
- **预期意图**：`pipeline_suggest`
- **预期工具**：`suggest_pipeline`
- **预期输出**：
  - 流程名称：mTERF 基因家族进化分析流程
  - 步骤卡片（每步显示：编号、名称、工具、输入、输出）
    1. 序列收集（BLAST/OrthoFinder）
    2. 多序列比对（MAFFT）
    3. 系统发育树构建（IQ-TREE/MEGA）
    4. 选择压力分析（PAML/codeml）
  - 注意事项
  - 文献来源
- **演示要点**：
  - 展示 Agent 自动调用 suggest_pipeline 工具
  - 步骤卡片展示，层次清晰
  - 结合 RAG 检索文献，流程有依据

---

### 用例 9：选择压力分析流程

- **场景**：用户询问基因家族正选择压力分析的方法
- **用户输入**：`怎么做基因家族的正选择压力分析？`
- **预期 Agent 链路**：`router → tool(suggest_pipeline) → answer`
- **预期意图**：`pipeline_suggest`
- **预期工具**：`suggest_pipeline`
- **预期输出**：
  - 流程名称：基因家族正选择压力分析流程
  - 步骤卡片：
    1. 同源基因鉴定（OrthoFinder）
    2. 多序列比对（MAFFT + trimAl 修剪）
    3. 系统发育树构建（IQ-TREE）
    4. 选择压力分析（PAML codeml，分支模型/位点模型/分支位点模型）
    5. 结果可视化（ggtree）
  - 注意事项：模型选择、多重检验校正
- **演示要点**：
  - 专业分析流程，体现生信背景
  - 步骤详细，工具推荐准确
  - 可作为面试时展示专业深度的用例

---

### 用例 10：直接问候

- **场景**：用户简单问候
- **用户输入**：`你好`
- **预期 Agent 链路**：`router → answer`（条件边：direct_answer 不调用工具）
- **预期意图**：`direct_answer`
- **预期工具**：`None`（不调用工具）
- **预期输出**：
  - 简洁的问候回复
  - 询问用户需要什么帮助
  - 执行过程面板只有 router + answer 两个节点（没有 tool）
- **演示要点**：
  - 展示条件边逻辑：direct_answer 直接到 answer，不调用工具
  - 执行链路短，响应快
  - 体现 Agent 的意图路由能力

---

## 演示建议顺序

面试演示时建议按以下顺序展示，层层递进：

1. **用例 10（直接问候）**：展示 Agent 启动和条件边
2. **用例 1（PAML omega）**：展示 RAG 检索 + 来源引用
3. **用例 5（FASTA 统计）**：展示纯计算工具 + 结构化展示
4. **用例 6（CpG 岛扫描）**：展示滑动窗口算法
5. **用例 8（流程推荐）**：展示 Agent 自动调工具 + 步骤卡片
6. **用例 7（阴性 CpG）**：展示边界处理和鲁棒性

全程展开"Agent 执行过程"面板，让面试官看到完整的 router→tool→answer 链路。

---

*PlantGenome Agent Demo Cases · D13*
