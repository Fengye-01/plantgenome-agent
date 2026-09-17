"""
AgentState 定义（D8 任务 1）

定义 PlantGenome Agent 的状态数据结构，在 LangGraph 节点之间传递。

参考 Hello Agents 第 6 章 LangGraph 基础：
- State 用 TypedDict 定义
- 每个节点返回部分字段，LangGraph 自动 merge
- State 在整个工作流中传递，记录所有中间结果
"""
from __future__ import annotations

from typing import TypedDict, Optional, Any, List, Dict


# ═══════════════════════════════════════════════════════════════
# 意图分类常量（5 类）
# ═══════════════════════════════════════════════════════════════

INTENT_LITERATURE_SEARCH = "literature_search"   # 文献/方法/概念问题（检索已有知识库）
INTENT_LITERATURE_DISCOVERY = "literature_discovery"  # PubMed 文献搜索（发现新文献）
INTENT_GENE_QUERY = "gene_query"                   # NCBI 基因信息查询
INTENT_FASTA_ANALYSIS = "fasta_analysis"         # FASTA 序列统计
INTENT_CPG_SCAN = "cpg_scan"                     # CpG 岛识别
INTENT_PIPELINE_SUGGEST = "pipeline_suggest"     # 研究流程规划
INTENT_DIRECT_ANSWER = "direct_answer"           # 闲聊/直接可答

# 所有合法意图列表
VALID_INTENTS = [
    INTENT_LITERATURE_SEARCH,
    INTENT_LITERATURE_DISCOVERY,
    INTENT_GENE_QUERY,
    INTENT_FASTA_ANALYSIS,
    INTENT_CPG_SCAN,
    INTENT_PIPELINE_SUGGEST,
    INTENT_DIRECT_ANSWER,
]

# 意图 → 工具名映射
INTENT_TO_TOOL = {
    INTENT_LITERATURE_SEARCH: "search_pdf_knowledge",
    INTENT_LITERATURE_DISCOVERY: "search_pubmed",
    INTENT_GENE_QUERY: "query_ncbi_gene",
    INTENT_FASTA_ANALYSIS: "parse_fasta_stats",
    INTENT_CPG_SCAN: "scan_cpg_islands",
    INTENT_PIPELINE_SUGGEST: "suggest_pipeline",
    INTENT_DIRECT_ANSWER: None,  # 不调工具，直接回答
}


# ═══════════════════════════════════════════════════════════════
# AgentState 定义
# ═══════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """
    PlantGenome Agent 的状态数据结构。

    在 LangGraph 工作流中，State 在节点之间传递，记录所有中间结果。
    每个节点只返回需要更新的字段，LangGraph 会自动 merge 到原状态中。

    字段说明：
    - query: 用户原始问题
    - user_id: 当前登录用户 ID（来自 JWT 认证，不经过 LLM，用于向量库用户隔离）
    - messages: 对话历史（用于多轮对话）
    - intent: 路由分类结果（6 类意图之一）
    - tool_name: 选中的工具名（router_node 决定）
    - tool_input: 工具输入参数（router_node 提取）
    - tool_result: 工具执行结果（tool_node 产出，最近一次）
    - tool_history: 工具调用历史（多工具链式调用时，记录所有已调用的工具和结果）
    - iteration: 当前迭代次数（多工具链式调用的循环计数，达到上限后强制结束）
    - retrieved_context: RAG 检索结果（search_pdf_knowledge 工具产出）
    - final_answer: 最终回答（answer_node 产出）
    - sources: 来源引用（answer_node 产出）
    - execution_log: 执行日志（记录每个节点的执行情况，用于调试和展示）
    """
    query: str                                    # 用户原始问题
    user_id: Optional[int]                         # 当前用户 ID（认证态注入，用于数据隔离）
    messages: List[Dict[str, str]]                # 对话历史
    intent: Optional[str]                          # 路由分类结果
    tool_name: Optional[str]                       # 选中的工具名
    tool_input: Optional[Dict[str, Any]]           # 工具输入参数
    tool_result: Optional[Any]                     # 工具执行结果（最近一次）
    tool_history: List[Dict[str, Any]]             # 工具调用历史（多工具链式调用）
    iteration: int                                 # 当前迭代次数
    retrieved_context: Optional[List[Dict]]        # RAG 检索结果
    final_answer: Optional[str]                    # 最终回答
    sources: Optional[List[str]]                   # 来源引用
    execution_log: List[Dict[str, Any]]            # 执行日志


# ═══════════════════════════════════════════════════════════════
# 初始状态工厂函数
# ═══════════════════════════════════════════════════════════════

def create_initial_state(
    query: str,
    messages: Optional[List[Dict]] = None,
    user_id: Optional[int] = None,
) -> AgentState:
    """
    创建初始状态。

    Args:
        query: 用户问题
        messages: 对话历史（可选，默认空列表）
        user_id: 当前登录用户 ID（来自 JWT 认证，用于向量库用户隔离；
                 独立脚本/测试场景可为 None，表示不过滤）

    Returns:
        初始化的 AgentState
    """
    return {
        "query": query,
        "user_id": user_id,
        "messages": messages if messages is not None else [],
        "intent": None,
        "tool_name": None,
        "tool_input": None,
        "tool_result": None,
        "tool_history": [],
        "iteration": 0,
        "retrieved_context": None,
        "final_answer": None,
        "sources": None,
        "execution_log": [],
    }
