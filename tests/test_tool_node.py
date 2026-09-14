"""
D10 任务 3：tool_node 端到端测试

测试 4 个工具通过 tool_node 调用，以及异常处理。
"""
import sys
from pathlib import Path

# 项目根路径导入
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.state import create_initial_state
from app.agents.nodes import tool_node


def run_test(test_name: str, state_updates: dict, expected_check: str):
    """运行单个测试，打印结果。"""
    print(f"\n{'─' * 70}")
    print(f"测试: {test_name}")
    print(f"{'─' * 70}")

    # 创建初始状态
    state = create_initial_state(state_updates.get("query", "test"))
    # 合并测试指定的状态更新
    for key, value in state_updates.items():
        state[key] = value

    # 执行 tool_node
    result = tool_node(state)

    # 打印结果
    tool_result = result.get("tool_result", {})
    execution_log = result.get("execution_log", [])

    print(f"tool_result 类型: {type(tool_result).__name__}")
    if isinstance(tool_result, dict):
        print(f"tool_result keys: {list(tool_result.keys())}")
        if "error" in tool_result:
            print(f"error: {tool_result['error']}")
    print(f"execution_log 条数: {len(execution_log)}")
    if execution_log:
        last_log = execution_log[-1]
        print(f"最后一条日志: node={last_log.get('node')}, tool={last_log.get('tool')}, "
              f"status={last_log.get('status')}, latency={last_log.get('latency')}s")

    # 检查预期
    print(f"\n预期检查: {expected_check}")
    return result


def main():
    print("=" * 70)
    print("tool_node 端到端测试")
    print("=" * 70)

    results = []

    # 测试 1: RAG 检索工具
    print("\n" + "=" * 70)
    print("测试 1: RAG 检索工具 (search_pdf_knowledge)")
    print("=" * 70)
    r1 = run_test(
        "RAG 检索",
        {
            "query": "PAML omega 值是什么意思？",
            "tool_name": "search_pdf_knowledge",
            "tool_input": {"query": "PAML omega", "top_k": 2},
        },
        "tool_result 包含 context 和 sources，sources 数量为 2",
    )
    has_context = "context" in r1.get("tool_result", {})
    has_sources = "sources" in r1.get("tool_result", {})
    source_count = len(r1.get("tool_result", {}).get("sources", []))
    passed1 = has_context and has_sources and source_count == 2
    print(f"\n结果: {'✅ 通过' if passed1 else '❌ 失败'} (context={has_context}, sources={has_sources}, count={source_count})")
    results.append(("RAG 检索", passed1))

    # 测试 2: FASTA 统计工具
    print("\n" + "=" * 70)
    print("测试 2: FASTA 统计工具 (parse_fasta_stats)")
    print("=" * 70)
    fasta = ">seq1\nATCGATCGATCG\n>seq2\nGGCCGGCCGGCC\n>seq3\nATATATATATAT"
    r2 = run_test(
        "FASTA 统计",
        {
            "query": f"统计这个FASTA：{fasta}",
            "tool_name": "parse_fasta_stats",
            "tool_input": {"fasta_text": fasta},
        },
        "tool_result 包含 num_sequences=3, gc_content, length_distribution",
    )
    num_seq = r2.get("tool_result", {}).get("num_sequences", 0)
    has_gc = "gc_content" in r2.get("tool_result", {})
    has_dist = "length_distribution" in r2.get("tool_result", {})
    passed2 = num_seq == 3 and has_gc and has_dist
    print(f"\n结果: {'✅ 通过' if passed2 else '❌ 失败'} (num_sequences={num_seq}, gc_content={has_gc}, length_distribution={has_dist})")
    results.append(("FASTA 统计", passed2))

    # 测试 3: CpG 岛扫描工具
    print("\n" + "=" * 70)
    print("测试 3: CpG 岛扫描工具 (scan_cpg_islands)")
    print("=" * 70)
    # 构造高 GC 高 CpG 序列（CGCG 重复 100 次 = 400bp）
    cpg_seq = "CGCG" * 100
    r3 = run_test(
        "CpG 岛扫描",
        {
            "query": f"扫描CpG岛：{cpg_seq}",
            "tool_name": "scan_cpg_islands",
            "tool_input": {"sequence": cpg_seq},
        },
        "tool_result 是列表，包含至少 1 个 CpG 岛（含 start, end, length, gc_content）",
    )
    tool_result3 = r3.get("tool_result", [])
    is_list = isinstance(tool_result3, list)
    has_island = len(tool_result3) > 0 and isinstance(tool_result3[0], dict) and "start" in tool_result3[0]
    passed3 = is_list and has_island
    if is_list and has_island:
        island = tool_result3[0]
        print(f"检测到 CpG 岛: start={island['start']}, end={island['end']}, length={island['length']}, gc={island['gc_content']}")
    print(f"\n结果: {'✅ 通过' if passed3 else '❌ 失败'} (is_list={is_list}, has_island={has_island})")
    results.append(("CpG 岛扫描", passed3))

    # 测试 4: 流程推荐工具
    print("\n" + "=" * 70)
    print("测试 4: 流程推荐工具 (suggest_pipeline)")
    print("=" * 70)
    r4 = run_test(
        "流程推荐",
        {
            "query": "mTERF家族进化分析流程",
            "tool_name": "suggest_pipeline",
            "tool_input": {"research_goal": "mTERF家族进化分析"},
        },
        "tool_result 包含 pipeline_name 和 steps",
    )
    has_pipeline_name = "pipeline_name" in r4.get("tool_result", {})
    has_steps = "steps" in r4.get("tool_result", {})
    passed4 = has_pipeline_name and has_steps
    if has_pipeline_name:
        print(f"流程名称: {r4['tool_result']['pipeline_name']}")
    if has_steps:
        print(f"步骤数量: {len(r4['tool_result']['steps'])}")
    print(f"\n结果: {'✅ 通过' if passed4 else '❌ 失败'} (pipeline_name={has_pipeline_name}, steps={has_steps})")
    results.append(("流程推荐", passed4))

    # 测试 5: 异常处理 - 未知工具
    print("\n" + "=" * 70)
    print("测试 5: 异常处理 - 未知工具")
    print("=" * 70)
    r5 = run_test(
        "未知工具",
        {
            "query": "test",
            "tool_name": "NonexistentTool",
            "tool_input": {},
        },
        "tool_result 包含 error，execution_log status=failed",
    )
    has_error = "error" in r5.get("tool_result", {})
    last_log = r5.get("execution_log", [{}])[-1] if r5.get("execution_log") else {}
    status_failed = last_log.get("status") == "failed"
    passed5 = has_error and status_failed
    print(f"error: {r5.get('tool_result', {}).get('error', 'N/A')}")
    print(f"\n结果: {'✅ 通过' if passed5 else '❌ 失败'} (error={has_error}, status={last_log.get('status')})")
    results.append(("未知工具异常", passed5))

    # 测试 6: 异常处理 - 未指定工具
    print("\n" + "=" * 70)
    print("测试 6: 异常处理 - 未指定工具")
    print("=" * 70)
    r6 = run_test(
        "未指定工具",
        {
            "query": "test",
            "tool_name": None,
            "tool_input": {},
        },
        "tool_result 包含 error，execution_log status=failed",
    )
    has_error6 = "error" in r6.get("tool_result", {})
    last_log6 = r6.get("execution_log", [{}])[-1] if r6.get("execution_log") else {}
    status_failed6 = last_log6.get("status") == "failed"
    passed6 = has_error6 and status_failed6
    print(f"error: {r6.get('tool_result', {}).get('error', 'N/A')}")
    print(f"\n结果: {'✅ 通过' if passed6 else '❌ 失败'} (error={has_error6}, status={last_log6.get('status')})")
    results.append(("未指定工具异常", passed6))

    # 测试汇总
    print("\n" + "=" * 70)
    print("测试汇总")
    print("=" * 70)
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    print(f"总测试数: {total_count}")
    print(f"通过: {passed_count}/{total_count} ({passed_count/total_count*100:.1f}%)")
    print(f"\n详细结果:")
    for name, passed in results:
        print(f"  {'✅' if passed else '❌'} {name}")

    if passed_count == total_count:
        print("\n🎉 所有测试通过！")
    else:
        print(f"\n⚠️  有 {total_count - passed_count} 个测试失败，需要检查。")

    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
