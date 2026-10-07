"""
LangGraph 工作流构建（D11 任务 2：完整图 + 条件边）

完整的 Agent 工作流（支持多工具链式调用）：
    START → router → [条件边] → tool → router（循环，可继续调用其他工具）
                            ↓
                          answer → END

循环控制：
- 最大工具调用次数 MAX_TOOL_CALLS = 3（防止死循环）
- 当 router 返回 direct_answer 或没有选中工具时，跳出循环到 answer
- 达到最大迭代次数时，强制到 answer

参考 Hello Agents 第 6 章 LangGraph：
- StateGraph: 状态图
- add_node: 添加节点
- add_conditional_edges: 条件边（根据状态决定下一个节点）
- add_edge / set_entry_point: 设置边和入口
- compile: 编译图
- invoke / stream: 运行图
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from langgraph.graph import END, StateGraph

from app.agents.nodes import answer_node, router_node, tool_node
from app.agents.state import INTENT_DIRECT_ANSWER, AgentState, create_initial_state

# 多工具链式调用的真实执行上限；直接按 tool_history 计数，避免 Router 轮次偏差。
MAX_TOOL_CALLS = 3


# ═══════════════════════════════════════════════════════════════
# 条件边函数
# ═══════════════════════════════════════════════════════════════


def should_call_tool(state: AgentState) -> str:
    """
    条件边：router 执行后，判断是否需要调用工具，还是直接回答。

    判断逻辑（多工具链式调用版本）：
    - intent == direct_answer → 直接回答（不需要工具）
    - tool_name 为 None → 直接回答（没有选中工具）
    - tool_validation_error 非空 → 直接回答，不执行工具
    - 已执行工具数达到 MAX_TOOL_CALLS → 强制结束（防止死循环）
    - 其他情况 → 调用工具（调用完后会回到 router，可继续调用其他工具）

    Args:
        state: Agent 状态（含 intent、tool_name、iteration）

    Returns:
        "tool" 或 "answer"，对应条件边的 key
    """
    intent = state.get("intent")
    tool_name = state.get("tool_name")
    # Router 已判定参数不合法，不允许进入 Tool 节点。
    if state.get("tool_validation_error") or state.get("router_error"):
        return "answer"

    # 达到最大工具调用次数，强制结束（防止死循环）
    if len(state.get("tool_history", [])) >= MAX_TOOL_CALLS:
        return "answer"

    # direct_answer 意图或没有选中工具 → 直接回答
    if intent == INTENT_DIRECT_ANSWER or tool_name is None:
        return "answer"

    # 其他情况 → 调用工具
    return "tool"


# ═══════════════════════════════════════════════════════════════
# 图构建函数
# ═══════════════════════════════════════════════════════════════


def build_graph():
    """
    构建 PlantGenome Agent 的完整 LangGraph 工作流（支持多工具链式调用）。

    图结构：
        START → router → [条件边]
                            ├─ tool → router（循环，可继续调用其他工具）
                            └─ answer → END

    条件边逻辑：
    - intent == direct_answer 或 tool_name == None → 直接到 answer
    - 已执行工具数达到 MAX_TOOL_CALLS → 强制到 answer
    - 其他情况 → 到 tool（调用完后回到 router）

    Returns:
        编译后的 LangGraph 应用
    """
    # 创建 StateGraph，指定 State 类型
    workflow = StateGraph(AgentState)

    # 添加 3 个节点
    workflow.add_node("router", router_node)
    workflow.add_node("tool", tool_node)
    workflow.add_node("answer", answer_node)

    # 设置入口点（图的开始节点）
    workflow.set_entry_point("router")

    # 条件边：router → tool 或 answer
    # should_call_tool 返回 "tool" 或 "answer"，和 mapping 的 key 对应
    workflow.add_conditional_edges(
        "router",
        should_call_tool,
        {
            "tool": "tool",
            "answer": "answer",
        },
    )

    # 循环边：tool → router（工具执行完后回到路由节点，可继续调用其他工具）
    # 这是多工具链式调用的核心：tool 执行完不直接到 answer，而是回到 router
    workflow.add_edge("tool", "router")

    # 普通边：answer → END（回答生成后结束）
    workflow.add_edge("answer", END)

    # 编译图
    app = workflow.compile()

    return app


@lru_cache(maxsize=1)
def get_compiled_graph():
    """Compile the stateless graph once per process."""
    return build_graph()


# ═══════════════════════════════════════════════════════════════
# 便捷调用函数
# ═══════════════════════════════════════════════════════════════


def run_agent(
    query: str,
    messages: Optional[list] = None,
    stream: bool = False,
    user_id: Optional[int] = None,
):
    """
    便捷函数：运行完整 Agent，传入用户问题，返回最终状态。

    Args:
        query: 用户问题
        messages: 对话历史（可选）
        stream: 是否使用流式模式（可以看到每个节点的输出）
        user_id: 当前登录用户 ID（来自 JWT 认证，透传给检索工具做用户隔离；
                 None 时不过滤，用于独立脚本/测试）

    Returns:
        最终的 AgentState（含 final_answer、sources、execution_log 等）
    """
    app = get_compiled_graph()

    # 创建初始状态（携带 user_id，沿 LangGraph 状态传递，不经过 LLM）
    initial_state = create_initial_state(query, messages, user_id=user_id)

    if stream:
        # 流式模式：逐个产出每个节点的输出
        print("=== Agent 执行过程 ===")
        final_state = initial_state
        for snapshot in app.stream(initial_state, stream_mode="values"):
            final_state = snapshot
            print(
                f"\n📍 intent={snapshot.get('intent')} "
                f"tool={snapshot.get('tool_name')} iteration={snapshot.get('iteration')}"
            )
        print("\n=== 执行完成 ===")
        return final_state
    else:
        # 同步模式：直接返回最终状态
        return app.invoke(initial_state)


# ═══════════════════════════════════════════════════════════════
# 图可视化
# ═══════════════════════════════════════════════════════════════


def print_graph_structure():
    """打印图结构（文字描述，不需要额外依赖）。"""
    print("=" * 60)
    print("PlantGenome Agent 工作流图结构（支持多工具链式调用）")
    print("=" * 60)
    print("""
    ┌─────────┐
    │  START  │
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │ router  │  ← 意图分类 + 工具选择（可循环多次）
    └────┬────┘
         │
    ┌────┴────┐  ← 条件边 (should_call_tool)
    │         │
    ▼         ▼
┌───────┐ ┌────────┐
│ tool  │ │ answer │  ← direct_answer / 达到最大迭代次数
└───┬───┘ └───┬────┘
    │         │
    └────┬────┘
         │ 循环边：tool → router（可继续调用其他工具）
         ▼
    ┌─────────┐
    │   END   │
    └─────────┘
    """)
    print("节点说明:")
    print("  router: 用 LLM 做意图分类，选择工具和参数（多轮循环时基于工具历史判断）")
    print("  tool:   执行选中的工具（RAG检索/FASTA统计/CpG扫描/流程推荐/PubMed搜索）")
    print("  answer: 基于工具结果生成最终回答")
    print()
    print("循环说明:")
    print("  tool 执行完后回到 router，可继续调用其他工具（多工具链式调用）")
    print(f"  最大工具调用次数: {MAX_TOOL_CALLS}（防止死循环）")
    print("  当 router 返回 direct_answer 或达到工具调用上限时，跳到 answer 结束")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════
# 测试入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # 项目根路径导入
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    # 打印图结构
    print_graph_structure()

    # 测试问题
    test_queries = [
        "你好",  # direct_answer
        "PAML omega 值是什么意思？",  # literature_search
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─' * 60}")
        print(f"测试 {i}: {query}")
        print(f"{'─' * 60}")

        result = run_agent(query, stream=True)

        print("\n📋 最终状态:")
        print(f"   query: {result['query']}")
        print(f"   intent: {result['intent']}")
        print(f"   tool_name: {result['tool_name']}")
        print(f"   final_answer 长度: {len(result.get('final_answer', ''))}")
        print(f"   sources 数量: {len(result.get('sources', []))}")
        print(f"   execution_log 条数: {len(result['execution_log'])}")

        print("\n📝 执行日志:")
        for j, log in enumerate(result["execution_log"], 1):
            print(
                f"   {j}. node={log.get('node')}, status={log.get('status')}, "
                f"latency={log.get('latency')}s"
            )

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
