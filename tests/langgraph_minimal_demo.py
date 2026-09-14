"""
LangGraph 最小示例（D7 任务 4 - 预习）

2 个节点 + 1 条边，不调用 LLM，纯逻辑，帮助理解 LangGraph 基本机制。

流程：
    START → process_query → generate_answer → END

运行：
    cd C:/Users/YeFeng/Desktop/自救计划/plantgenome-agent
    python tests/langgraph_minimal_demo.py
"""
from __future__ import annotations

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# ═══════════════════════════════════════════════════════════════
# 1. 定义 State（状态）
# ═══════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """
    Agent 状态：在节点之间传递的数据结构。

    每个节点接收 State，处理后返回更新后的 State（部分更新，LangGraph 会自动合并）。
    """
    query: str              # 用户原始查询
    processed_query: str    # 处理后的查询（节点 1 的输出）
    answer: str             # 最终回答（节点 2 的输出）
    step: str               # 当前步骤（用于调试）


# ═══════════════════════════════════════════════════════════════
# 2. 定义 Node（节点）
# ═══════════════════════════════════════════════════════════════

def process_query_node(state: AgentState) -> AgentState:
    """
    节点 1：处理用户查询。

    模拟"理解用户需求"——把查询转成大写，加上前缀。
    实际项目中这里可以是 LLM 调用（查询重写、意图识别等）。

    Args:
        state: 当前状态（包含 query）

    Returns:
        部分更新的状态（processed_query 和 step）
        LangGraph 会自动把返回的字段合并到原状态中
    """
    print(f"  [节点 1 - process_query] 收到查询: {state['query']}")

    # 模拟处理：转成大写，加上前缀
    original_query = state["query"]
    processed = f"[已处理] {original_query.upper()}"

    print(f"  [节点 1 - process_query] 处理结果: {processed}")

    # 返回部分更新的状态（LangGraph 会自动合并）
    return {
        "processed_query": processed,
        "step": "query_processed",
    }


def generate_answer_node(state: AgentState) -> AgentState:
    """
    节点 2：生成最终回答。

    模拟"基于处理后的查询生成回答"。
    实际项目中这里可以是 LLM 调用（RAG 生成、工具调用结果总结等）。

    Args:
        state: 当前状态（包含 processed_query）

    Returns:
        部分更新的状态（answer 和 step）
    """
    print(f"  [节点 2 - generate_answer] 收到处理后的查询: {state['processed_query']}")

    # 模拟生成回答
    processed = state["processed_query"]
    answer = f"这是针对 '{processed}' 的回答。答案是 42。"

    print(f"  [节点 2 - generate_answer] 生成回答: {answer}")

    return {
        "answer": answer,
        "step": "answer_generated",
    }


# ═══════════════════════════════════════════════════════════════
# 3. 构建 Graph（图）
# ═══════════════════════════════════════════════════════════════

def build_graph():
    """
    构建 LangGraph 工作流。

    步骤：
    1. 创建 StateGraph（传入 State 类型）
    2. 添加节点（add_node）
    3. 添加边（add_edge），定义数据流方向
    4. 编译（compile），得到可执行的 app
    """
    # 1. 创建 StateGraph
    workflow = StateGraph(AgentState)

    # 2. 添加节点（节点名, 节点函数）
    workflow.add_node("process_query", process_query_node)
    workflow.add_node("generate_answer", generate_answer_node)

    # 3. 添加边（定义数据流方向）
    #    START 是特殊节点，表示图的入口
    #    END 是特殊节点，表示图的出口
    workflow.add_edge(START, "process_query")
    workflow.add_edge("process_query", "generate_answer")
    workflow.add_edge("generate_answer", END)

    # 4. 编译图，得到可执行的 app
    app = workflow.compile()

    return app


# ═══════════════════════════════════════════════════════════════
# 4. 运行图
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("LangGraph 最小示例 - 2 节点工作流")
    print("=" * 70)

    # 构建图
    app = build_graph()
    print("\n✅ 图构建完成！")
    print("   节点: process_query → generate_answer")
    print("   边: START → process_query → generate_answer → END")

    # 初始状态
    initial_state: AgentState = {
        "query": "什么是 LangGraph？",
        "processed_query": "",
        "answer": "",
        "step": "start",
    }

    print(f"\n📝 初始状态:")
    print(f"   query: {initial_state['query']}")
    print(f"   step: {initial_state['step']}")

    # 运行图（stream 模式，可以看到每个节点的输出）
    print("\n🚀 开始运行图...\n")

    final_state = None
    for output in app.stream(initial_state):
        # output 是一个字典，key 是节点名，value 是该节点返回的部分状态
        for node_name, node_output in output.items():
            print(f"\n  📍 节点 '{node_name}' 执行完成")
            print(f"     返回: {node_output}")

    # 获取最终状态（app.stream 最后一个 output 就是最终状态）
    # 也可以用 app.invoke() 直接获取最终状态
    final_state = app.invoke(initial_state)

    print("\n" + "=" * 70)
    print("✅ 图运行完成！")
    print("=" * 70)
    print(f"\n📋 最终状态:")
    print(f"   query: {final_state['query']}")
    print(f"   processed_query: {final_state['processed_query']}")
    print(f"   answer: {final_state['answer']}")
    print(f"   step: {final_state['step']}")

    print(f"\n💡 最终回答: {final_state['answer']}")


if __name__ == "__main__":
    main()
