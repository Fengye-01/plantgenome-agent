"""
LangGraph 工作流构建（D11 任务 2：完整图 + 条件边）

完整的 Agent 工作流：
    START → router → [条件边] → tool → answer → END
                            ↓
                          answer（direct_answer 直接回答，不调用工具）

参考 Hello Agents 第 6 章 LangGraph：
- StateGraph: 状态图
- add_node: 添加节点
- add_conditional_edges: 条件边（根据状态决定下一个节点）
- add_edge / set_entry_point: 设置边和入口
- compile: 编译图
- invoke / stream: 运行图
"""
from __future__ import annotations

from typing import Optional

from langgraph.graph import StateGraph, END

from app.agents.state import AgentState, create_initial_state, INTENT_DIRECT_ANSWER
from app.agents.nodes import router_node, tool_node, answer_node


# ═══════════════════════════════════════════════════════════════
# 条件边函数
# ═══════════════════════════════════════════════════════════════

def should_call_tool(state: AgentState) -> str:
    """
    条件边：router 执行后，判断是否需要调用工具，还是直接回答。

    判断逻辑：
    - intent == direct_answer → 直接回答（不需要工具）
    - tool_name 为 None → 直接回答（没有选中工具）
    - 其他情况 → 调用工具

    Args:
        state: Agent 状态（含 intent、tool_name）

    Returns:
        "tool" 或 "answer"，对应条件边的 key
    """
    intent = state.get("intent")
    tool_name = state.get("tool_name")

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
    构建 PlantGenome Agent 的完整 LangGraph 工作流。

    图结构：
        START → router → [条件边]
                            ├─ tool → answer → END
                            └─ answer → END

    条件边逻辑：
    - intent == direct_answer 或 tool_name == None → 直接到 answer
    - 其他情况 → 到 tool（调用工具后再到 answer）

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

    # 普通边：tool → answer（工具执行完后生成回答）
    workflow.add_edge("tool", "answer")

    # 普通边：answer → END（回答生成后结束）
    workflow.add_edge("answer", END)

    # 编译图
    app = workflow.compile()

    return app


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
    # 构建图
    app = build_graph()

    # 创建初始状态（携带 user_id，沿 LangGraph 状态传递，不经过 LLM）
    initial_state = create_initial_state(query, messages, user_id=user_id)

    if stream:
        # 流式模式：逐个产出每个节点的输出
        print("=== Agent 执行过程 ===")
        for output in app.stream(initial_state):
            for node_name, node_output in output.items():
                print(f"\n📍 节点: {node_name}")
                if node_name == "router":
                    print(f"   intent: {node_output.get('intent')}")
                    print(f"   tool_name: {node_output.get('tool_name')}")
                    print(f"   tool_input: {node_output.get('tool_input')}")
                elif node_name == "tool":
                    tool_result = node_output.get("tool_result", {})
                    if isinstance(tool_result, dict):
                        print(f"   tool_result keys: {list(tool_result.keys())}")
                        if "error" in tool_result:
                            print(f"   error: {tool_result['error']}")
                    elif isinstance(tool_result, list):
                        print(f"   tool_result 是列表，长度: {len(tool_result)}")
                elif node_name == "answer":
                    answer = node_output.get("final_answer", "")
                    print(f"   answer 长度: {len(answer)}")
                    print(f"   answer 前 200 字: {answer[:200]}")
        print("\n=== 执行完成 ===")
        # 流式模式需要再 invoke 一次获取最终状态
        return app.invoke(initial_state)
    else:
        # 同步模式：直接返回最终状态
        return app.invoke(initial_state)


# ═══════════════════════════════════════════════════════════════
# 图可视化
# ═══════════════════════════════════════════════════════════════

def print_graph_structure():
    """打印图结构（文字描述，不需要额外依赖）。"""
    print("=" * 60)
    print("PlantGenome Agent 工作流图结构")
    print("=" * 60)
    print("""
    ┌─────────┐
    │  START  │
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │ router  │  ← 意图分类 + 工具选择
    └────┬────┘
         │
    ┌────┴────┐  ← 条件边 (should_call_tool)
    │         │
    ▼         ▼
┌───────┐ ┌────────┐
│ tool  │ │ answer │  ← direct_answer 直接回答
└───┬───┘ └───┬────┘
    │         │
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │   END   │
    └─────────┘
    """)
    print("节点说明:")
    print("  router: 用 LLM 做意图分类，选择工具和参数")
    print("  tool:   执行选中的工具（RAG检索/FASTA统计/CpG扫描/流程推荐）")
    print("  answer: 基于工具结果生成最终回答")
    print()
    print("条件边说明:")
    print("  intent == direct_answer 或 tool_name == None → 直接到 answer")
    print("  其他情况 → 到 tool（调用工具后再到 answer）")
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

        print(f"\n📋 最终状态:")
        print(f"   query: {result['query']}")
        print(f"   intent: {result['intent']}")
        print(f"   tool_name: {result['tool_name']}")
        print(f"   final_answer 长度: {len(result.get('final_answer', ''))}")
        print(f"   sources 数量: {len(result.get('sources', []))}")
        print(f"   execution_log 条数: {len(result['execution_log'])}")

        print(f"\n📝 执行日志:")
        for j, log in enumerate(result["execution_log"], 1):
            print(f"   {j}. node={log.get('node')}, status={log.get('status')}, "
                  f"latency={log.get('latency')}s")

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
