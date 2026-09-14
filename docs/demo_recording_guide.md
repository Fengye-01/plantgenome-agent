# PlantGenome Agent - 演示 GIF 录制指南

> D16 任务：录制 1 分钟演示 GIF，放在 README 顶部。
> 由于 GIF 需要手动录屏，本文档提供详细的录制脚本、工具推荐和优化建议。

---

## 一、推荐工具

| 工具 | 平台 | 优点 | 下载 |
|------|------|------|------|
| **ScreenToGif** | Windows | 免费开源，功能强大，支持编辑/压缩/导出 | https://www.screentogif.com |
| **LICEcap** | Windows/Mac | 轻量，简单直接 | https://www.cockos.com/licecap/ |
| **OBS Studio** | 全平台 | 专业录屏，可先录 MP4 再转 GIF | https://obsproject.com |
| **在线工具** | 浏览器 | Ezgif.com 支持视频转 GIF、压缩、裁剪 | https://ezgif.com |

**推荐 ScreenToGif**：功能最全，支持录制后编辑（删除帧、调整大小、添加文字），导出时可以优化压缩。

---

## 二、演示脚本（60 秒）

按照以下顺序录制，控制在 60 秒以内。每个步骤的时间是建议值，可以根据实际情况调整。

### 步骤 1：打开页面（0-5s）

- 打开浏览器，访问 http://localhost:8502
- 展示 PlantGenome Agent 主界面
- 可以稍微滚动一下，展示侧边栏的 PDF 上传区和序列分析工具区

**录制要点**：页面加载完成后停留 2-3 秒，让观众看清界面布局。

### 步骤 2：文献问答（5-20s）

- 在聊天输入框输入：`PAML 中的 omega (dN/dS) 值怎么解释？`
- 按回车发送
- 等待回答生成（展示 LLM 生成过程）
- 回答完成后，展开"📚 来源引用"，展示 sources
- 展开"🔍 Agent 执行过程"，展示 router→tool→answer 链路

**录制要点**：
- 这是核心演示，一定要展示 sources 和执行过程面板
- 如果回答生成太慢，可以加速 GIF 或提前准备好回答

### 步骤 3：FASTA 统计（20-30s）

- 滚动到侧边栏的"FASTA 统计"区域
- 粘贴一段 FASTA 序列（3-5 条，每条 100-200bp）
- 点击"📊 分析 FASTA"按钮
- 展示统计结果（metric 卡片 + 长度分布柱状图 + 序列详情表格）

**示例 FASTA 序列**：
```
>seq1
ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG
>seq2
GGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCC
>seq3
ATATATATATATATATATATATATATATATATATATATAT
>seq4
AGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT
>seq5
CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG
```

**录制要点**：展示纯计算工具的速度（几乎瞬间完成），和文献问答形成对比。

### 步骤 4：CpG 岛扫描（30-40s）

- 滚动到侧边栏的"CpG 岛扫描"区域
- 粘贴一段高 GC 高 CpG 密度的序列（`CGCG` 重复 50-100 次）
- 点击"🔬 扫描 CpG 岛"按钮
- 展示 CpG 岛结果表格和 GC 含量对比图

**示例 DNA 序列**：
```
CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG
```

**录制要点**：展示滑动窗口算法的结果，CpG 岛表格里有 start/end/length/gc_content/obs_exp_ratio。

### 步骤 5：流程推荐（40-52s）

- 回到聊天输入框
- 输入：`研究 mTERF 基因家族进化需要什么分析流程？`
- 按回车发送
- 等待回答生成
- 展示流程推荐的步骤卡片（每步显示编号、名称、工具、输入、输出）

**录制要点**：展示 Agent 自动调用 suggest_pipeline 工具，步骤卡片层次清晰。

### 步骤 6：执行过程面板（52-60s）

- 展开"🔍 Agent 执行过程"面板
- 展示完整的 router→tool→answer 链路
- 每个节点的信息（意图、工具名、耗时）
- 底部的总节点数和总耗时

**录制要点**：这是 Agent 项目的核心亮点，一定要让观众看清执行过程。

---

## 三、GIF 优化建议

### 1. 帧率

- 建议 **8-10 fps**（不需要太高，10fps 足够流畅）
- 太高的帧率会导致文件过大，GitHub 显示限制是 10MB

### 2. 尺寸

- 建议宽度 **1000-1200px**
- 高度按比例缩放，保持浏览器窗口的宽高比
- ScreenToGif 录制时可以选择区域，只录制浏览器内容区域

### 3. 压缩

- ScreenToGif 导出时选择"优化"，可以减少颜色数和帧
- 如果还是太大，用 https://ezgif.com/optimize 在线压缩
- 目标：**<10MB**（GitHub README 能正常显示）

### 4. 备选方案

如果 GIF 实在太大（>10MB），可以：
- 拆成 2-3 个小 GIF，分别展示不同功能
- 转成 MP4 视频，上传到 B站/YouTube，README 里放视频链接
- 用静态截图 + 文字说明代替 GIF

---

## 四、截图清单

除了 GIF，还需要截取以下静态截图，放在 README 和 docs 里：

| 截图 | 文件名 | 说明 |
|------|--------|------|
| 主界面 | `main_interface.png` | 完整页面，展示侧边栏 + 聊天区 |
| 文献问答 | `demo1_literature_search.png` | 回答 + sources + 执行过程 |
| FASTA 统计 | `demo2_fasta_stats.png` | metric 卡片 + 柱状图 + 表格 |
| CpG 岛扫描 | `demo3_cpg_island.png` | CpG 岛表格 + GC 对比图 |
| 流程推荐 | `demo4_pipeline.png` | 步骤卡片 |
| 执行过程面板 | `demo5_execution_log.png` | router→tool→answer 链路 |
| 知识库无答案 | `failure1_no_answer.png` | 展示"无法回答"，不编造 |

截图保存到 `docs/screenshots/` 目录。

---

## 五、README 插入 GIF

GIF 录制完成后，保存为 `docs/images/demo.gif`，然后在 README 顶部（项目简介下方）插入：

```markdown
## 🎬 演示

![PlantGenome Agent Demo](docs/images/demo.gif)
```

如果用静态截图代替，可以放 3-5 张关键截图：

```markdown
## 📸 截图

| 文献问答 | FASTA 统计 | Agent 执行过程 |
|---------|-----------|--------------|
| ![](docs/screenshots/demo1_literature_search.png) | ![](docs/screenshots/demo2_fasta_stats.png) | ![](docs/screenshots/demo5_execution_log.png) |
```

---

## 六、录制前检查清单

录制前确保以下事项：

- [ ] 后端服务已启动（http://localhost:8000）
- [ ] 前端服务已启动（http://localhost:8502）
- [ ] 知识库已入库（287 chunks）
- [ ] LLM API 可用（硅基流动额度充足）
- [ ] 浏览器窗口大小合适（建议 1280x800）
- [ ] 关闭无关的浏览器标签页和通知
- [ ] 准备好要粘贴的 FASTA 和 DNA 序列（复制到剪贴板）
- [ ] ScreenToGif 已安装并打开

---

*PlantGenome Agent Demo Recording Guide · D16*
