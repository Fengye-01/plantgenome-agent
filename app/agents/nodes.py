"""
Agent 节点实现（D8 任务 2）

目前实现 router_node（路由节点），后续 D9 加 tool_node 和 answer_node。

参考 Hello Agents 第 6 章 LangGraph 基础 + 第 4 章 ReAct/Agent 基础：
- router_node 是 Agent 的"大脑"，负责理解用户意图并选择工具
- 用 LLM 做意图分类 + 工具参数提取
- 输出 JSON 格式，便于后续节点解析
"""
from __future__ import annotations

import json
import re
import time
from typing import Dict, Any

from app.core.llm import HelloAgentsLLM
from app.agents.state import (
    AgentState,
    VALID_INTENTS,
    INTENT_TO_TOOL,
    INTENT_DIRECT_ANSWER,
)


# ═══════════════════════════════════════════════════════════════
# Router Prompt（路由分类提示词）
# ═══════════════════════════════════════════════════════════════

ROUTER_PROMPT = """你是一个专业的生物信息学任务路由分类器。

根据用户的问题，判断属于以下哪类任务，并提取对应的工具参数。

【任务类型】
1. literature_search - 文献检索、方法查询、概念解释、生信工具用法、软件参数说明（检索用户已上传入库的文献）
2. literature_discovery - PubMed 文献搜索、查找最新研究、了解某领域有哪些论文、搜索相关文献（搜索 PubMed 全网，返回标题/作者/摘要）
3. gene_query - NCBI 基因信息查询、基因功能查询、Locus Tag 查询、基因位置查询（用户输入基因名或Locus Tag要求查询信息）
4. fasta_analysis - FASTA 序列统计分析（用户提供或粘贴了 FASTA 序列，要求统计）
5. cpg_scan - CpG 岛识别扫描（用户提供 DNA 序列并要求找 CpG 岛）
6. pipeline_suggest - 研究流程/分析方案推荐（用户问"怎么做XX分析""需要什么流程""用什么工具"）
7. direct_answer - 仅限纯闲聊、问候、感谢、道歉、或与生物信息学完全无关的简单问题

【重要规则 - 必须严格遵守】
- 涉及基因、序列、文献、生信工具、分析方法、专业概念的问题，必须调用工具，绝对不要 direct_answer
- 即使你认为自己知道答案，只要问题涉及专业知识，也必须调用检索或分析工具
- "XX是什么""XX怎么用""XX有什么功能""怎么做XX分析"这类问题，一律调用对应工具
- direct_answer 只用于"你好""谢谢""再见""你是谁"这类纯对话

【literature_search vs literature_discovery 的区别】
- literature_search：用户问的是具体概念/方法/工具用法，答案应该在已上传的文献中找
- literature_discovery：用户明确要求"搜索文献""找最新研究""有哪些相关论文"，需要去 PubMed 全网搜索
- 不确定时优先用 literature_search

【多工具链式调用说明】
如果之前已经调用过工具（见下方"已调用工具历史"），请判断：
- 如果已有工具结果已经足够回答用户问题，选择 direct_answer 结束
- 如果还需要调用其他工具补充信息，选择对应的工具类型继续
- 不要重复调用同一个工具

【已调用工具历史】
{tool_history_str}

【输出要求】
严格输出 JSON 格式，不要输出任何其他文字、解释或 markdown 代码块。
JSON 中每个字段之间必须用逗号分隔，括号必须配对。

JSON 格式：
{{"intent": "任务类型", "tool_name": "工具名或null", "tool_input": {{参数对象}}}}

【工具名映射】
- literature_search → search_pdf_knowledge
- literature_discovery → search_pubmed
- gene_query → query_ncbi_gene
- fasta_analysis → parse_fasta_stats
- cpg_scan → scan_cpg_islands
- pipeline_suggest → suggest_pipeline
- direct_answer → null

【tool_input 参数说明】
- search_pdf_knowledge: {{"query": "检索关键词"}}
- search_pubmed: {{"keyword": "搜索关键字"}}
- query_ncbi_gene: {{"gene_name": "基因名或Locus Tag"}}
- parse_fasta_stats: {{"fasta_text": "FASTA序列内容"}}
- scan_cpg_islands: {{"sequence": "DNA序列"}}
- suggest_pipeline: {{"research_goal": "研究目标描述"}}
- direct_answer: {{}}

【示例】
示例1：
用户问题：PAML codeml 怎么设置分支模型？
输出：{{"intent": "literature_search", "tool_name": "search_pdf_knowledge", "tool_input": {{"query": "PAML codeml 分支模型设置"}}}}

示例2：
用户问题：帮我搜索一下植物 mTERF 基因家族的最新研究文献
输出：{{"intent": "literature_discovery", "tool_name": "search_pubmed", "tool_input": {{"keyword": "plant mTERF gene family"}}}}

示例3：
用户问题：研究 mTERF 家族进化需要什么流程？
输出：{{"intent": "pipeline_suggest", "tool_name": "suggest_pipeline", "tool_input": {{"research_goal": "mTERF 家族进化研究"}}}}

示例4：
用户问题：你好
输出：{{"intent": "direct_answer", "tool_name": null, "tool_input": {{}}}}

【用户问题】
{query}

【你的输出（仅JSON）】
"""


# ═══════════════════════════════════════════════════════════════
# 工具函数：解析 LLM 输出的 JSON
# ═══════════════════════════════════════════════════════════════

def _fix_json_string(json_str: str) -> str:
    """
    修复 LLM 输出中常见的 JSON 语法错误。

    常见问题：
    1. 字段之间缺少逗号（如 "a": 1 "b": 2）
    2. 缺少结尾括号
    3. 多余的逗号（如 "a": 1, }）
    4. 单引号代替双引号

    Args:
        json_str: 可能有语法错误的 JSON 字符串

    Returns:
        修复后的 JSON 字符串
    """
    fixed = json_str.strip()

    # 1. 修复缺少结尾括号：统计 { 和 } 的数量，如果不匹配则补全
    open_count = fixed.count('{')
    close_count = fixed.count('}')
    if open_count > close_count:
        fixed += '}' * (open_count - close_count)

    # 2. 修复字段之间缺少逗号：在 " 后面紧跟 " 的地方加逗号
    #    例如："tool_name": "xxx" "tool_input": {...}
    fixed = re.sub(r'"\s+"', '", "', fixed)

    # 3. 修复 } 后面紧跟 " 的地方加逗号
    fixed = re.sub(r'}\s+"', '}, "', fixed)

    # 4. 修复多余的逗号（如 {, "a": 1} 或 {"a": 1,}）
    fixed = re.sub(r'\{\s*,', '{', fixed)
    fixed = re.sub(r',\s*}', '}', fixed)
    fixed = re.sub(r',\s*,', ',', fixed)

    return fixed


def _extract_intent_from_text(text: str) -> str:
    """
    当 JSON 解析完全失败时，从文本中尝试提取 intent。

    策略：
    1. 搜索常见的 intent 关键词
    2. 如果找到，返回对应的 intent
    3. 否则返回 direct_answer

    Args:
        text: LLM 输出文本

    Returns:
        提取到的 intent
    """
    text_lower = text.lower()

    # 按优先级匹配
    if any(kw in text_lower for kw in ['literature_discovery', 'pubmed', '搜索文献', '查找文献', '最新研究', '相关论文', '有哪些文献']):
        return "literature_discovery"
    if any(kw in text_lower for kw in ['gene_query', '基因查询', '基因信息', '基因功能', 'locus tag', '基因位置', '查一下基因']):
        return "gene_query"
    if any(kw in text_lower for kw in ['literature_search', '文献', '检索', '方法查询', '概念解释']):
        return "literature_search"
    if any(kw in text_lower for kw in ['fasta_analysis', 'fasta', '序列统计']):
        return "fasta_analysis"
    if any(kw in text_lower for kw in ['cpg_scan', 'cpg', '甲基化岛']):
        return "cpg_scan"
    if any(kw in text_lower for kw in ['pipeline_suggest', '流程', '方案', '怎么分析', '需要什么']):
        return "pipeline_suggest"

    return INTENT_DIRECT_ANSWER


def parse_router_output(response: str) -> Dict[str, Any]:
    """
    解析 router_node 的 LLM 输出，提取 JSON。

    LLM 输出可能不稳定（尤其是 7B 小模型）：
    - 带 markdown 代码块包裹
    - 有多余的解释文字
    - JSON 格式不完整（缺少逗号、缺少括号等）
    - tool_name 值不对

    处理策略（多级容错）：
    1. 去除 markdown 代码块
    2. 提取第一个 {...} 块
    3. 尝试直接 json.loads
    4. 失败则尝试 _fix_json_string 修复后再解析
    5. 仍失败则从文本中提取 intent（_extract_intent_from_text）
    6. tool_name 始终从 INTENT_TO_TOOL 映射表获取，不信任 LLM 输出的 tool_name

    Args:
        response: LLM 的原始输出文本

    Returns:
        解析后的字典，包含 intent、tool_name、tool_input
    """
    # 去除 markdown 代码块标记
    cleaned = response.strip()
    cleaned = re.sub(r'^```json\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^```\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)

    # 提取第一个 {...} 块（支持嵌套）
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if not match:
        # 没有找到 JSON，尝试从文本提取 intent
        intent = _extract_intent_from_text(cleaned)
        return {
            "intent": intent,
            "tool_name": INTENT_TO_TOOL.get(intent),
            "tool_input": {},
        }

    json_str = match.group()

    # 尝试 1：直接解析
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        # 尝试 2：修复 JSON 后解析
        try:
            fixed = _fix_json_string(json_str)
            parsed = json.loads(fixed)
        except json.JSONDecodeError:
            # 尝试 3：从文本提取 intent
            intent = _extract_intent_from_text(cleaned)
            return {
                "intent": intent,
                "tool_name": INTENT_TO_TOOL.get(intent),
                "tool_input": {},
            }

    # 解析成功，提取字段
    intent = parsed.get("intent", INTENT_DIRECT_ANSWER)
    tool_input = parsed.get("tool_input", {})

    # 校验 intent 是否合法
    if intent not in VALID_INTENTS:
        intent = _extract_intent_from_text(cleaned)

    # tool_name 始终从映射表获取，不信任 LLM 输出的 tool_name
    # （因为 LLM 经常输出错误的 tool_name，如 "pdf_knowledge" 而不是 "search_pdf_knowledge"）
    tool_name = INTENT_TO_TOOL.get(intent)

    return {
        "intent": intent,
        "tool_name": tool_name,
        "tool_input": tool_input if isinstance(tool_input, dict) else {},
    }


# ═══════════════════════════════════════════════════════════════
# router_node 实现
# ═══════════════════════════════════════════════════════════════

def router_node(state: AgentState) -> AgentState:
    """
    路由节点：用 LLM 做意图分类 + 工具参数提取。

    这是 Agent 工作流的第一个节点（也是多工具链式调用的循环节点），负责：
    1. 理解用户问题的意图
    2. 选择合适的工具
    3. 提取工具需要的输入参数
    4. 多轮循环时，基于已调用工具历史决定是否继续

    Args:
        state: 当前 Agent 状态（必须包含 query）

    Returns:
        部分更新的 AgentState（intent, tool_name, tool_input, iteration, execution_log）
        LangGraph 会自动把这些字段 merge 到原状态中
    """
    query = state["query"]
    iteration = state.get("iteration", 0) + 1  # 每次 router 调用，迭代 +1

    # 构建工具历史字符串（供多轮 router 判断是否需要继续调用）
    tool_history = state.get("tool_history", [])
    if tool_history:
        history_parts = []
        for i, h in enumerate(tool_history, 1):
            tool = h.get("tool", "?")
            status = h.get("status", "?")
            history_parts.append(f"{i}. {tool} (状态: {status})")
        tool_history_str = "\n".join(history_parts)
    else:
        tool_history_str = "（无，首次调用）"

    # 初始化 LLM（每次调用都创建新实例，避免状态污染）
    llm = HelloAgentsLLM()

    # 构建 prompt
    prompt = ROUTER_PROMPT.format(query=query, tool_history_str=tool_history_str)

    # 调用 LLM
    try:
        response = llm.invoke(prompt, temperature=0.1)  # 低温，确保分类稳定
    except Exception as e:
        # LLM 调用失败，默认 direct_answer
        response = '{"intent": "direct_answer", "tool_name": null, "tool_input": {}}'

    # 解析 LLM 输出
    parsed = parse_router_output(response)
    intent = parsed["intent"]
    tool_name = parsed["tool_name"]
    tool_input = parsed["tool_input"]

    # 防止重复调用同一个工具（多工具链式调用的保护机制）
    # 按"工具名+参数"去重，允许同一工具不同参数多次调用（如先查PAML再查MAFFT）
    if tool_history and tool_name:
        current_call = f"{tool_name}:{json.dumps(tool_input, sort_keys=True, ensure_ascii=False)}"
        called_calls = [
            f"{h.get('tool')}:{json.dumps(h.get('tool_input', {}), sort_keys=True, ensure_ascii=False)}"
            for h in tool_history
        ]
        if current_call in called_calls and intent != INTENT_DIRECT_ANSWER:
            # 完全相同的调用（工具名+参数都一样），强制 direct_answer 结束循环
            intent = INTENT_DIRECT_ANSWER
            tool_name = None
            tool_input = {}

    # 记录执行日志
    log_entry = {
        "node": "router",
        "intent": intent,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "iteration": iteration,
        "raw_response": response[:500],  # 只保存前 500 字符，避免日志过大
    }
    state["execution_log"].append(log_entry)

    # 返回部分更新的状态
    return {
        "intent": intent,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "iteration": iteration,
        "execution_log": state["execution_log"],
    }


# ═══════════════════════════════════════════════════════════════
# tool_node：工具调用节点（D10 任务 2）
# ═══════════════════════════════════════════════════════════════

def _extract_fasta_from_query(query: str) -> str:
    """
    从用户问题中兜底提取 FASTA 序列。

    当 router 没有正确提取 fasta_text 参数时，从 query 中提取。
    FASTA 格式：以 > 开头的 ID 行，后续是序列内容。

    提取策略：
    1. 找到第一个 > 的位置
    2. 从该位置开始取到字符串结束
    3. 清理掉可能的多余内容（如结尾的客套话）

    Args:
        query: 用户原始问题

    Returns:
        提取到的 FASTA 文本，没找到返回空字符串
    """
    # 找到第一个 > 的位置
    gt_pos = query.find('>')
    if gt_pos == -1:
        return ""

    # 从 > 开始取到字符串结束
    fasta_text = query[gt_pos:].strip()

    # 简单验证：是否包含 > 开头的行
    lines = fasta_text.split('\n')
    if not lines or not lines[0].startswith('>'):
        return ""

    return fasta_text


def _extract_dna_sequence_from_query(query: str) -> str:
    """
    从用户问题中兜底提取 DNA 序列。

    当 router 没有正确提取 sequence 参数时，从 query 中提取。
    匹配连续的 ATCGN 字符（长度 >= 20）。

    Args:
        query: 用户原始问题

    Returns:
        提取到的 DNA 序列，没找到返回空字符串
    """
    # 匹配连续的 DNA 字符（长度 >= 20）
    seq_matches = re.findall(r'[ATCGNatcgn]{20,}', query)
    if seq_matches:
        # 返回最长的匹配
        return max(seq_matches, key=len).upper()
    return ""


def tool_node(state: AgentState) -> AgentState:
    """
    工具调用节点：执行 router 选中的工具，处理参数和异常，记录调用日志。

    工作流程：
    1. 从 state 中获取 tool_name 和 tool_input
    2. 通过 tool_executor.getTool() 获取工具函数
    3. 兜底参数提取：fasta/cpg 工具如果参数缺失，从 query 中提取
    4. 执行工具（try-except，确保工具失败不会导致整个 Agent 崩溃）
    5. 记录 execution_log（工具名、参数、耗时、状态）
    6. 返回更新后的 state

    参考 Hello Agents 第 4 章工具调用 + 第 6 章 LangGraph 节点。

    Args:
        state: Agent 状态（含 tool_name、tool_input、query、execution_log）

    Returns:
        部分更新的 AgentState（tool_result、execution_log）
    """
    tool_name = state.get("tool_name")
    tool_input = state.get("tool_input", {}) or {}
    query = state.get("query", "")

    # 情况 1：没有指定工具（direct_answer 意图不应该走到这个节点）
    if not tool_name:
        error_msg = "未指定工具"
        state["tool_result"] = {"error": error_msg}
        state["execution_log"].append({
            "node": "tool",
            "tool": None,
            "error": error_msg,
            "latency": 0,
            "status": "failed",
        })
        return state

    # 情况 2：通过 tool_executor 获取工具函数
    from app.agents.tool_registry import tool_executor
    tool_func = tool_executor.getTool(tool_name)

    if not tool_func:
        error_msg = f"未知工具: {tool_name}"
        state["tool_result"] = {"error": error_msg}
        state["execution_log"].append({
            "node": "tool",
            "tool": tool_name,
            "error": error_msg,
            "latency": 0,
            "status": "failed",
        })
        return state

    # 情况 3：兜底参数提取
    # FASTA 统计工具：优先用从 query 中兜底提取的完整序列（router 提取的经常不准确）
    if tool_name == "parse_fasta_stats":
        fasta_text = _extract_fasta_from_query(query)
        if fasta_text:
            # 从 query 中提取到了完整的 FASTA，优先使用
            tool_input["fasta_text"] = fasta_text
        elif "fasta_text" not in tool_input:
            # 没提取到且 tool_input 里也没有，保持原样（工具会返回错误）
            pass

    # CpG 岛扫描工具：如果 sequence 缺失，从 query 中提取
    if tool_name == "scan_cpg_islands" and "sequence" not in tool_input:
        sequence = _extract_dna_sequence_from_query(query)
        if sequence:
            tool_input["sequence"] = sequence

    # 用户身份注入（数据隔离）：
    # user_id 来自 state（源头是 JWT 认证），由代码强制注入检索工具，
    # 不使用 LLM 在 tool_input 中输出的任何 user_id（防止 prompt 注入越权）。
    user_id = state.get("user_id")
    if tool_name == "search_pdf_knowledge" and user_id is not None:
        tool_input["user_id"] = user_id

    # 情况 4：执行工具（try-except）
    start_time = time.time()
    try:
        # 用关键字参数调用工具函数（我们的工具函数接收关键字参数，不是单个字符串）
        result = tool_func(**tool_input)
        latency = round(time.time() - start_time, 3)

        state["tool_result"] = result
        state["execution_log"].append({
            "node": "tool",
            "tool": tool_name,
            "input_keys": list(tool_input.keys()),
            "latency": latency,
            "status": "success",
        })
        # 累积到 tool_history（供多工具链式调用时 router 判断）
        state.setdefault("tool_history", []).append({
            "tool": tool_name,
            "tool_input": tool_input,
            "status": "success",
            "latency": latency,
        })
    except Exception as e:
        latency = round(time.time() - start_time, 3)
        error_msg = str(e)

        state["tool_result"] = {"error": error_msg}
        state["execution_log"].append({
            "node": "tool",
            "tool": tool_name,
            "input_keys": list(tool_input.keys()),
            "error": error_msg,
            "latency": latency,
            "status": "failed",
        })
        # 累积到 tool_history
        state.setdefault("tool_history", []).append({
            "tool": tool_name,
            "tool_input": tool_input,
            "status": "failed",
            "error": error_msg,
            "latency": latency,
        })

    # 返回部分更新的状态
    # tool_history 累积所有已调用工具的结果（供多工具链式调用时 router 判断）
    # tool_result 保留最近一次的结果（供 answer_node 使用）
    return {
        "tool_result": state["tool_result"],
        "tool_history": state.get("tool_history", []),
        "execution_log": state["execution_log"],
    }


# ═══════════════════════════════════════════════════════════════
# answer_node：回答生成节点（D11 任务 1）
# ═══════════════════════════════════════════════════════════════

ANSWER_PROMPT = """你是植物基因组学研究助手。根据工具执行结果和用户问题，生成最终回答。

【要求】
1. 如果有文献检索结果，回答必须基于检索内容，在相关句子后标注来源 [1] [2]
2. 如果有工具执行结果（FASTA统计/CpG扫描/流程推荐），把结果整理成清晰的格式，数字和统计结果要准确呈现
3. 如果工具执行失败，说明失败原因并给出建议
4. 如果是直接回答（没有调用工具），基于你的知识简洁回答
5. 回答专业、简洁、有结构，300-800字
6. 不要编造检索结果中没有的信息
7. 【严禁重复】绝对不要重复同一个词、短语或句子。不要出现"并，并，，并"这类重复。如果发现自己在重复，立即停止并换一种表述。
8. 【语言流畅】使用通顺的中文，专业术语保留英文原文。不要堆砌关键词，不要逐字翻译。

【用户问题】
{query}

【意图】
{intent}

【工具执行结果】
{tool_result_str}

【文献来源】
{sources_str}

【你的回答】
"""


def _format_tool_result(tool_result: Any, intent: str) -> str:
    """
    格式化工具执行结果，转为适合 LLM 阅读的文本。

    处理不同类型的工具结果：
    - RAG 检索：提取 context 和 sources 摘要
    - FASTA 统计：提取关键统计指标
    - CpG 岛扫描：提取 CpG 岛列表
    - 流程推荐：提取流程名称和步骤
    - 错误：显示错误信息

    Args:
        tool_result: 工具执行结果（dict 或 list）
        intent: 意图类型，用于选择格式化方式

    Returns:
        格式化后的文本字符串
    """
    if tool_result is None:
        return "（无工具执行结果，直接回答）"

    # 错误结果
    if isinstance(tool_result, dict) and "error" in tool_result:
        return f"工具执行失败：{tool_result['error']}"

    # RAG 检索结果
    if intent == "literature_search" and isinstance(tool_result, dict):
        context = tool_result.get("context", "")
        count = tool_result.get("count", 0)
        return f"检索到 {count} 个相关文献片段：\n{context[:2000]}"

    # FASTA 统计结果
    if intent == "fasta_analysis" and isinstance(tool_result, dict):
        lines = [
            f"序列数量: {tool_result.get('num_sequences', 'N/A')}",
            f"总长度: {tool_result.get('total_length', 'N/A')} bp",
            f"平均长度: {tool_result.get('avg_length', 'N/A')} bp",
            f"最短/最长: {tool_result.get('min_length', 'N/A')} / {tool_result.get('max_length', 'N/A')} bp",
            f"GC含量: {tool_result.get('gc_content', 'N/A')}",
            f"长度分布: {tool_result.get('length_distribution', {})}",
        ]
        return "\n".join(lines)

    # CpG 岛扫描结果
    if intent == "cpg_scan" and isinstance(tool_result, list):
        if not tool_result:
            return "未检测到 CpG 岛"
        if isinstance(tool_result[0], dict) and "warning" in tool_result[0]:
            return tool_result[0]["warning"]
        if isinstance(tool_result[0], dict) and "result" in tool_result[0]:
            return tool_result[0]["result"]

        lines = [f"检测到 {len(tool_result)} 个 CpG 岛："]
        for i, island in enumerate(tool_result[:5], 1):  # 最多显示 5 个
            if isinstance(island, dict):
                lines.append(
                    f"  岛 {i}: 位置 {island.get('start', '?')}-{island.get('end', '?')} "
                    f"({island.get('length', '?')}bp), "
                    f"GC={island.get('gc_content', '?')}, "
                    f"Obs/Exp={island.get('obs_exp_ratio', '?')}"
                )
        if len(tool_result) > 5:
            lines.append(f"  ... 还有 {len(tool_result) - 5} 个 CpG 岛")
        return "\n".join(lines)

    # 流程推荐结果
    if intent == "pipeline_suggest" and isinstance(tool_result, dict):
        pipeline_name = tool_result.get("pipeline_name", "未命名流程")
        steps = tool_result.get("steps", [])
        notes = tool_result.get("notes", "")

        lines = [f"推荐流程：{pipeline_name}", ""]
        for step in steps[:10]:  # 最多显示 10 步
            if isinstance(step, dict):
                lines.append(
                    f"  步骤 {step.get('step', '?')}: {step.get('name', '?')}\n"
                    f"    工具: {step.get('tool', '?')}\n"
                    f"    输入: {step.get('input', '?')}\n"
                    f"    输出: {step.get('output', '?')}"
                )
        if notes:
            lines.append(f"\n注意事项: {notes}")
        return "\n".join(lines)

    # PubMed 文献搜索结果
    if intent == "literature_discovery" and isinstance(tool_result, dict):
        articles = tool_result.get("articles", [])
        if not articles:
            return tool_result.get("message", "未搜索到相关文献")

        lines = [f"PubMed 搜索到 {len(articles)} 篇相关文献：", ""]
        for i, art in enumerate(articles[:5], 1):  # 最多显示 5 篇
            lines.append(f"[{i}] {art.get('title', '?')}")
            lines.append(f"    作者: {art.get('authors', '?')[:80]}")
            lines.append(f"    期刊: {art.get('journal', '?')} ({art.get('year', '?')})")
            lines.append(f"    PMID: {art.get('pmid', '?')} | 全文: {'有' if art.get('has_full_text') else '无'}")
            abstract = art.get('abstract', '')
            if abstract:
                lines.append(f"    摘要: {abstract[:200]}...")
            lines.append("")
        lines.append("提示：如需下载全文并入库 RAG，请在左侧 PubMed 面板输入关键字。")
        return "\n".join(lines)

    # NCBI 基因查询结果
    if intent == "gene_query" and isinstance(tool_result, dict):
        genes = tool_result.get("genes", [])
        if not genes:
            return tool_result.get("message", "未查询到相关基因信息")

        lines = [f"NCBI 查询到 {len(genes)} 个相关基因：", ""]
        for i, g in enumerate(genes[:3], 1):  # 最多显示 3 个
            lines.append(f"[{i}] {g.get('name', '?')} (Gene ID: {g.get('gene_id', '?')})")
            lines.append(f"    物种: {g.get('organism', '?')}")
            if g.get("description"):
                lines.append(f"    描述: {g['description'][:100]}")
            if g.get("chromosome_location"):
                lines.append(f"    染色体位置: {g['chromosome_location']}")
            if g.get("synonyms"):
                lines.append(f"    别名: {', '.join(g['synonyms'][:5])}")
            if g.get("summary"):
                lines.append(f"    功能摘要: {g['summary'][:150]}...")
            lines.append("")
        return "\n".join(lines)

    # 默认：直接转字符串，截断
    result_str = str(tool_result)
    if len(result_str) > 2000:
        result_str = result_str[:2000] + "...（截断）"
    return result_str


def _format_sources(sources: list) -> str:
    """
    格式化文献来源列表。

    Args:
        sources: 来源列表，每个来源是 dict，含 filename、page_num、snippet 等

    Returns:
        格式化后的来源文本
    """
    if not sources:
        return "（无文献来源）"

    lines = []
    for i, s in enumerate(sources[:5], 1):  # 最多显示 5 个来源
        if isinstance(s, dict):
            filename = s.get("filename", "未知文件")
            page_num = s.get("page_num", "?")
            snippet = s.get("snippet", "")[:100]
            lines.append(f"[{i}] {filename} (p.{page_num}): {snippet}")
    if len(sources) > 5:
        lines.append(f"... 还有 {len(sources) - 5} 个来源")
    return "\n".join(lines)


def answer_node(state: AgentState) -> AgentState:
    """
    回答生成节点：基于工具执行结果和用户问题，生成最终回答。

    工作流程：
    1. 从 state 中获取 query、intent、tool_result、sources
    2. 格式化工具结果（根据 intent 选择不同的格式化方式）
    3. 格式化文献来源
    4. 构建 ANSWER_PROMPT
    5. 调用 LLM 生成最终回答
    6. 记录 execution_log
    7. 返回更新后的 state

    参考 Hello Agents 第 9 章上下文工程（回答生成 Prompt 设计）。

    Args:
        state: Agent 状态（含 query、intent、tool_result、sources、execution_log）

    Returns:
        部分更新的 AgentState（final_answer、execution_log）
    """
    query = state.get("query", "")
    intent = state.get("intent", "direct_answer")
    tool_result = state.get("tool_result")
    sources = state.get("sources", []) or []

    # 从 tool_result 中提取 sources（只要工具返回了 sources 就提取，不限制 intent）
    # 这样即使 router 分类有偏差，sources 也能正确传递到最终回答
    if isinstance(tool_result, dict) and "sources" in tool_result:
        extracted = tool_result.get("sources", [])
        if extracted:
            sources = extracted

    # 格式化工具结果和来源
    tool_result_str = _format_tool_result(tool_result, intent)
    sources_str = _format_sources(sources)

    # 构建 Prompt
    prompt = ANSWER_PROMPT.format(
        query=query,
        intent=intent,
        tool_result_str=tool_result_str[:3000],  # 限制长度，避免超出上下文
        sources_str=sources_str,
    )

    # 调用 LLM 生成回答
    llm = HelloAgentsLLM()
    start_time = time.time()
    try:
        answer = llm.invoke(prompt, temperature=0.3, max_tokens=2048)
        latency = round(time.time() - start_time, 3)
        status = "success"
        error = None
    except Exception as e:
        latency = round(time.time() - start_time, 3)
        answer = f"回答生成失败：{str(e)}"
        status = "failed"
        error = str(e)

    # 记录执行日志
    state["execution_log"].append({
        "node": "answer",
        "intent": intent,
        "answer_length": len(answer),
        "latency": latency,
        "status": status,
        "error": error,
    })

    # 返回部分更新的状态
    return {
        "final_answer": answer,
        "sources": sources,
        "execution_log": state["execution_log"],
    }
