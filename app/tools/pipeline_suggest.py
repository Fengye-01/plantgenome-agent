"""
工具 4：suggest_pipeline（研究流程推荐工具）

基于知识库 RAG + LLM 生成步骤化的分析流程。
这是 4 个工具中唯一依赖 LLM 的工具，其他 3 个都是纯计算。

工作流程：
1. 用 search_pdf_knowledge 检索相关文献
2. 构建 Prompt（研究目标 + 检索到的文献资料）
3. 调用 LLM 生成步骤化分析流程（JSON 格式）
4. 多级容错 JSON 解析
5. 返回流程 + sources

参考 Hello Agents 第 4 章工具调用 + 第 8 章 RAG + 第 9 章上下文工程。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from app.core.llm import HelloAgentsLLM
from app.tools.search_pdf_knowledge import search_pdf_knowledge


# ═══════════════════════════════════════════════════════════════
# Prompt 模板
# ═══════════════════════════════════════════════════════════════

PIPELINE_PROMPT = """你是植物比较基因组学分析专家。根据用户的研究目标和检索到的文献资料，推荐一个步骤化的分析流程。

【要求】
1. 输出 3-5 个步骤，流程顺序：数据准备 → 比对 → 分析 → 可视化
2. 基于文献资料，不要编造不存在的工具
3. 严格按以下格式输出，不要输出任何其他文字、解释或 markdown

【输出格式】
第一行：流程名称
然后每个步骤一行，格式：步骤编号. 步骤名称 | 推荐工具 | 输入 | 输出
最后一行：注意事项：xxx

【示例】
基因家族进化与选择压力分析流程
1. 序列收集 | BLAST/OrthoFinder | 目标物种蛋白序列 | 同源基因家族列表
2. 多序列比对 | MAFFT | 基因家族序列 | 比对结果
3. 选择压力分析 | PAML/codeml | 比对结果+系统发育树 | dN/dS比值
注意事项：注意选择合适的进化模型，建议先做模型选择

【研究目标】
{goal}

【文献资料】
{context}

【你的输出】
"""


# ═══════════════════════════════════════════════════════════════
# 纯文本解析（对 7B 小模型更友好，避免复杂 JSON 输出不稳定）
# ═══════════════════════════════════════════════════════════════

def parse_pipeline_output(response: str) -> Dict[str, Any]:
    """
    解析 LLM 输出的纯文本格式分析流程。

    预期格式：
        流程名称
        1. 步骤名称 | 推荐工具 | 输入 | 输出
        2. 步骤名称 | 推荐工具 | 输入 | 输出
        注意事项：xxx

    解析策略：
    1. 去除 markdown 代码块和多余空行
    2. 第一行作为流程名称
    3. 匹配 "数字. 名称 | 工具 | 输入 | 输出" 格式的行作为步骤
    4. 匹配 "注意事项：" 开头的行作为 notes
    5. 兜底：如果没有匹配到步骤，返回原始输出作为 notes

    Args:
        response: LLM 原始输出

    Returns:
        解析后的流程 dict
    """
    # 预处理：去除 markdown 代码块标记
    cleaned = re.sub(r'^```(?:json|text)?\s*', '', response.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE)

    # 按行分割，去除空行，截断过长行（可能是模型重复循环）
    lines = []
    for line in cleaned.split('\n'):
        line = line.strip()
        if line:
            # 截断超过 300 字符的行（可能是重复循环）
            if len(line) > 300:
                line = line[:300] + "...（截断）"
            lines.append(line)

    if not lines:
        return {
            "pipeline_name": "分析流程建议",
            "steps": [],
            "notes": "LLM 输出为空。",
        }

    # 第一行作为流程名称（如果不是步骤行的话）
    pipeline_name = lines[0]
    start_idx = 1

    # 如果第一行看起来像步骤行（以数字.开头），则流程名称用默认值
    if re.match(r'^\d+\.', lines[0]):
        pipeline_name = "分析流程建议"
        start_idx = 0

    # 解析步骤
    steps = []
    notes = ""

    for line in lines[start_idx:]:
        # 匹配步骤行："数字. 名称 | 工具 | 输入 | 输出"（支持 | 和 / 作为分隔符）
        step_match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if step_match:
            step_num = int(step_match.group(1))
            content = step_match.group(2)

            # 优先用 | 分割，如果 | 数量不够，再用 / 分割
            if content.count('|') >= 2:
                parts = [p.strip() for p in content.split('|')]
            elif content.count('/') >= 2:
                parts = [p.strip() for p in content.split('/')]
            else:
                # 都没有，整行作为名称，其他字段为空
                parts = [content.strip(), "", "", ""]

            # 确保有 4 个部分（名称、工具、输入、输出）
            while len(parts) < 4:
                parts.append("")

            # 清理 markdown 加粗标记（**xxx**）
            parts = [re.sub(r'\*\*', '', p).strip() for p in parts]

            steps.append({
                "step": step_num,
                "name": parts[0],
                "tool": parts[1],
                "input": parts[2],
                "output": parts[3],
            })
            continue

        # 匹配注意事项行
        notes_match = re.match(r'^注意事项[：:]\s*(.+)$', line)
        if notes_match:
            notes = notes_match.group(1)
            continue

        # 其他行追加到 notes
        if notes:
            notes += " " + line
        else:
            notes = line

    # 如果没有解析到步骤，返回原始输出作为 notes
    if not steps:
        return {
            "pipeline_name": pipeline_name,
            "steps": [],
            "notes": "未能解析到步骤。原始输出：" + cleaned[:500],
        }

    return {
        "pipeline_name": pipeline_name,
        "steps": steps,
        "notes": notes if notes else "（无额外注意事项）",
    }


# ═══════════════════════════════════════════════════════════════
# 主工具函数
# ═══════════════════════════════════════════════════════════════

def suggest_pipeline(research_goal: str, top_k: int = 3) -> Dict[str, Any]:
    """
    基于知识库 RAG + LLM 生成步骤化的分析流程。

    这是 Agent 的工具之一，用于回答"怎么做XX分析""需要什么流程""用什么工具"等问题。

    Args:
        research_goal: 研究目标描述
        top_k: RAG 检索的文档数量，默认 3

    Returns:
        dict，包含：
        - pipeline_name: 流程名称
        - steps: 步骤列表，每步含 step、name、tool、input、output
        - notes: 注意事项和建议
        - sources: 文献来源列表
        - has_rag_result: RAG 是否检索到相关文献
        - raw_response: LLM 原始输出（调试用）

    失败情况：
        - RAG 检索失败
        - LLM 调用失败
        - JSON 解析失败（有兜底）
    """
    # 第一步：检索相关文献
    rag_result = search_pdf_knowledge(research_goal, top_k=top_k)
    context = rag_result.get("context", "")
    sources = rag_result.get("sources", [])
    has_rag_result = rag_result.get("has_result", False)

    # 如果没有检索到文献，在 context 中说明
    if not has_rag_result or not context:
        context = "（知识库中未检索到与该研究目标直接相关的文献，以下流程基于通用生信分析经验给出）"

    # 第二步：构建 Prompt
    prompt = PIPELINE_PROMPT.format(goal=research_goal, context=context)

    # 第三步：调用 LLM（temperature=0.1 减少重复循环，max_tokens=2000 避免生成过长）
    llm = HelloAgentsLLM()
    try:
        response = llm.invoke(prompt, temperature=0.1, max_tokens=2000)
    except Exception as e:
        return {
            "pipeline_name": "分析流程建议（LLM 调用失败）",
            "steps": [],
            "notes": f"LLM 调用失败：{str(e)}",
            "sources": sources,
            "has_rag_result": has_rag_result,
            "error": str(e),
        }

    # 第四步：解析 JSON（多级容错）
    pipeline = parse_pipeline_output(response)

    # 第五步：补充 sources 和元信息
    pipeline["sources"] = sources
    pipeline["has_rag_result"] = has_rag_result
    pipeline["raw_response"] = response  # 调试用，前端可以不展示

    return pipeline


# ═══════════════════════════════════════════════════════════════
# 测试入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # 项目根路径导入
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    print("=" * 70)
    print("工具测试：suggest_pipeline")
    print("=" * 70)

    test_goals = [
        "分析 mTERF 基因家族在植物中的进化与选择压力",
        "研究 CpG 岛甲基化与基因表达的关系",
    ]

    for i, goal in enumerate(test_goals, 1):
        print(f"\n{'─' * 70}")
        print(f"测试 {i}: {goal}")
        print(f"{'─' * 70}")

        result = suggest_pipeline(goal, top_k=3)

        print(f"\n流程名称: {result.get('pipeline_name')}")
        print(f"RAG 检索到文献: {result.get('has_rag_result')}")
        print(f"步骤数量: {len(result.get('steps', []))}")

        print(f"\n步骤详情:")
        for step in result.get("steps", []):
            print(f"  步骤 {step.get('step')}: {step.get('name')}")
            print(f"    工具: {step.get('tool')}")
            print(f"    输入: {step.get('input')}")
            print(f"    输出: {step.get('output')}")

        print(f"\n注意事项: {result.get('notes', '')[:300]}")

        print(f"\nSources:")
        for s in result.get("sources", []):
            print(f"  [{s['index']}] {s['filename']} (p.{s['page_num']})")

    print("\n" + "=" * 70)
    print("所有测试完成！")
    print("=" * 70)
