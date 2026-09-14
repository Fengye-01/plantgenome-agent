"""
工具注册表：用 ToolExecutor 注册 4 个生信工具

参考 Hello Agents 第 4 章 tools.py 的 ToolExecutor 类：
- registerTool(name, description, func)：注册工具
- getTool(name)：获取工具函数
- getAvailableTools()：返回格式化的工具列表字符串（会被拼接到 Agent Prompt 中）

工具名用下划线命名，和 INTENT_TO_TOOL 映射表保持一致。
"""
from __future__ import annotations

from app.core.tools import ToolExecutor
from app.tools.search_pdf_knowledge import search_pdf_knowledge
from app.tools.fasta_stats import parse_fasta_stats
from app.tools.cpg_island import scan_cpg_islands
from app.tools.pipeline_suggest import suggest_pipeline


# 初始化工具执行器单例
tool_executor = ToolExecutor()


# ═══════════════════════════════════════════════════════════════
# 注册 4 个生信工具
# ═══════════════════════════════════════════════════════════════

# 工具 1：PDF 文献检索
tool_executor.registerTool(
    name="search_pdf_knowledge",
    description="文献知识库检索工具。当用户询问生信工具用法、基因家族概念、分析方法、软件参数等文献相关问题时使用。输入参数：query（检索关键词字符串），可选 top_k（返回数量，默认3）。",
    func=search_pdf_knowledge,
)

# 工具 2：FASTA 序列统计
tool_executor.registerTool(
    name="parse_fasta_stats",
    description="FASTA 序列统计工具。当用户提供或粘贴 FASTA 序列，要求统计序列数量、长度、GC含量等信息时使用。输入参数：fasta_text（FASTA格式文本字符串）。",
    func=parse_fasta_stats,
)

# 工具 3：CpG 岛扫描
tool_executor.registerTool(
    name="scan_cpg_islands",
    description="CpG 岛扫描工具。当用户提供 DNA 序列并要求识别或扫描 CpG 岛时使用。输入参数：sequence（DNA序列字符串），可选 window_size（窗口大小，默认200）、step（步长，默认100）。",
    func=scan_cpg_islands,
)

# 工具 4：分析流程推荐
tool_executor.registerTool(
    name="suggest_pipeline",
    description="分析流程推荐工具。当用户询问研究方案、分析流程、实验设计、需要什么工具时使用。输入参数：research_goal（研究目标描述字符串）。",
    func=suggest_pipeline,
)


# ═══════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════

def get_tool_names() -> list:
    """获取所有已注册的工具名列表。"""
    return list(tool_executor.tools.keys())


def get_tool_descriptions() -> str:
    """获取格式化的工具描述字符串（可拼接到 Prompt 中）。"""
    return tool_executor.getAvailableTools()


# ═══════════════════════════════════════════════════════════════
# 测试入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    print("=" * 70)
    print("工具注册表测试")
    print("=" * 70)

    print(f"\n已注册工具数量: {len(get_tool_names())}")
    print(f"工具名列表: {get_tool_names()}")

    print(f"\n工具描述（getAvailableTools 输出）:")
    print("-" * 70)
    print(get_tool_descriptions())
    print("-" * 70)

    # 测试 getTool
    print("\n测试 getTool:")
    for name in get_tool_names():
        tool_func = tool_executor.getTool(name)
        print(f"  {name}: {'✅ 已找到' if tool_func else '❌ 未找到'}")

    # 测试未知工具
    unknown = tool_executor.getTool("nonexistent_tool")
    print(f"  nonexistent_tool: {'✅ 正确返回 None' if unknown is None else '❌ 异常'}")

    print("\n" + "=" * 70)
    print("所有测试完成！")
    print("=" * 70)
