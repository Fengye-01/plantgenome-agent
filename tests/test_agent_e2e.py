"""
D11 任务 3：完整 Agent 端到端测试

测试 5 类问题通过完整 LangGraph 工作流（router → tool/answer → END）：
1. 文献检索（literature_search）
2. FASTA 统计（fasta_analysis）
3. CpG 岛扫描（cpg_scan）
4. 流程推荐（pipeline_suggest）
5. 直接回答（direct_answer）
"""
import sys
import time
from pathlib import Path

# 项目根路径导入
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.graph import build_graph
from app.agents.state import create_initial_state


def run_agent(query: str) -> dict:
    """运行完整 Agent，返回最终状态。"""
    agent = build_graph()
    initial_state = create_initial_state(query)
    return agent.invoke(initial_state)


def print_result(test_name: str, result: dict):
    """打印测试结果。"""
    print(f"\n{'─' * 70}")
    print(f"测试: {test_name}")
    print(f"{'─' * 70}")
    print(f"  query: {result.get('query', '')[:80]}")
    print(f"  intent: {result.get('intent')}")
    print(f"  tool_name: {result.get('tool_name')}")
    print(f"  final_answer 长度: {len(result.get('final_answer', ''))}")
    print(f"  sources 数量: {len(result.get('sources', []))}")
    print(f"  execution_log 条数: {len(result.get('execution_log', []))}")

    print(f"\n  执行链路:")
    for i, log in enumerate(result.get("execution_log", []), 1):
        node = log.get("node", "?")
        status = log.get("status", "?")
        latency = log.get("latency", "?")
        print(f"    {i}. {node} (status={status}, latency={latency}s)")

    answer = result.get("final_answer", "")
    if answer:
        print(f"\n  回答（前 300 字）:")
        print(f"    {answer[:300]}")


def main():
    print("=" * 70)
    print("PlantGenome Agent 端到端测试（5 类问题）")
    print("=" * 70)

    results = []
    total_start = time.time()

    # ═══════════════════════════════════════════════════════════
    # 测试 1: 文献检索（literature_search）
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("测试 1/5: 文献检索（literature_search）")
    print("=" * 70)
    start = time.time()
    r1 = run_agent("PAML omega 值是什么意思？")
    latency1 = round(time.time() - start, 1)

    intent_ok = r1.get("intent") == "literature_search"
    tool_ok = r1.get("tool_name") == "search_pdf_knowledge"
    answer_ok = r1.get("final_answer") is not None and len(r1["final_answer"]) > 0
    sources_ok = len(r1.get("sources", [])) > 0
    passed1 = intent_ok and tool_ok and answer_ok and sources_ok

    print_result("文献检索", r1)
    print(f"\n  检查: intent={intent_ok}, tool={tool_ok}, answer={answer_ok}, sources={sources_ok}")
    print(f"  耗时: {latency1}s")
    print(f"  结果: {'✅ 通过' if passed1 else '❌ 失败'}")
    results.append(("文献检索", passed1, latency1))

    # ═══════════════════════════════════════════════════════════
    # 测试 2: FASTA 统计（fasta_analysis）
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("测试 2/5: FASTA 统计（fasta_analysis）")
    print("=" * 70)
    fasta = ">gene1\nATCGATCGATCGATCG\n>gene2\nGGCCGGCCGGCCGGCC\n>gene3\nATATATATATATATAT"
    start = time.time()
    r2 = run_agent(f"帮我统计这个FASTA序列：\n{fasta}")
    latency2 = round(time.time() - start, 1)

    intent_ok = r2.get("intent") == "fasta_analysis"
    tool_result = r2.get("tool_result", {})
    num_seq_ok = isinstance(tool_result, dict) and tool_result.get("num_sequences") == 3
    answer_ok = r2.get("final_answer") is not None and len(r2["final_answer"]) > 0
    passed2 = intent_ok and num_seq_ok and answer_ok

    print_result("FASTA 统计", r2)
    if isinstance(tool_result, dict):
        print(f"\n  工具结果: num_sequences={tool_result.get('num_sequences')}, "
              f"gc_content={tool_result.get('gc_content')}")
    print(f"\n  检查: intent={intent_ok}, num_seq={num_seq_ok}, answer={answer_ok}")
    print(f"  耗时: {latency2}s")
    print(f"  结果: {'✅ 通过' if passed2 else '❌ 失败'}")
    results.append(("FASTA 统计", passed2, latency2))

    # ═══════════════════════════════════════════════════════════
    # 测试 3: CpG 岛扫描（cpg_scan）
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("测试 3/5: CpG 岛扫描（cpg_scan）")
    print("=" * 70)
    cpg_seq = "CGCG" * 100  # 400bp 高 GC 高 CpG 序列
    start = time.time()
    r3 = run_agent(f"这段序列有没有CpG岛？{cpg_seq}")
    latency3 = round(time.time() - start, 1)

    intent_ok = r3.get("intent") == "cpg_scan"
    tool_result = r3.get("tool_result", [])
    has_island = isinstance(tool_result, list) and len(tool_result) > 0 and isinstance(tool_result[0], dict) and "start" in tool_result[0]
    answer_ok = r3.get("final_answer") is not None and len(r3["final_answer"]) > 0
    passed3 = intent_ok and has_island and answer_ok

    print_result("CpG 岛扫描", r3)
    if has_island:
        island = tool_result[0]
        print(f"\n  工具结果: 检测到 CpG 岛 start={island['start']}, end={island['end']}, "
              f"length={island['length']}, gc={island['gc_content']}")
    print(f"\n  检查: intent={intent_ok}, has_island={has_island}, answer={answer_ok}")
    print(f"  耗时: {latency3}s")
    print(f"  结果: {'✅ 通过' if passed3 else '❌ 失败'}")
    results.append(("CpG 岛扫描", passed3, latency3))

    # ═══════════════════════════════════════════════════════════
    # 测试 4: 流程推荐（pipeline_suggest）
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("测试 4/5: 流程推荐（pipeline_suggest）")
    print("=" * 70)
    start = time.time()
    r4 = run_agent("研究 mTERF 基因家族进化需要什么分析流程？")
    latency4 = round(time.time() - start, 1)

    intent_ok = r4.get("intent") == "pipeline_suggest"
    tool_result = r4.get("tool_result", {})
    has_steps = isinstance(tool_result, dict) and "steps" in tool_result
    answer_ok = r4.get("final_answer") is not None and len(r4["final_answer"]) > 0
    passed4 = intent_ok and has_steps and answer_ok

    print_result("流程推荐", r4)
    if has_steps:
        print(f"\n  工具结果: pipeline_name={tool_result.get('pipeline_name')}, "
              f"steps={len(tool_result.get('steps', []))}")
    print(f"\n  检查: intent={intent_ok}, has_steps={has_steps}, answer={answer_ok}")
    print(f"  耗时: {latency4}s")
    print(f"  结果: {'✅ 通过' if passed4 else '❌ 失败'}")
    results.append(("流程推荐", passed4, latency4))

    # ═══════════════════════════════════════════════════════════
    # 测试 5: 直接回答（direct_answer）
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("测试 5/5: 直接回答（direct_answer）")
    print("=" * 70)
    start = time.time()
    r5 = run_agent("你好")
    latency5 = round(time.time() - start, 1)

    intent_ok = r5.get("intent") == "direct_answer"
    tool_none = r5.get("tool_name") is None
    answer_ok = r5.get("final_answer") is not None and len(r5["final_answer"]) > 0
    # direct_answer 应该只有 router + answer 两个节点，没有 tool
    log_nodes = [log.get("node") for log in r5.get("execution_log", [])]
    no_tool = "tool" not in log_nodes
    passed5 = intent_ok and tool_none and answer_ok and no_tool

    print_result("直接回答", r5)
    print(f"\n  检查: intent={intent_ok}, tool_none={tool_none}, answer={answer_ok}, no_tool={no_tool}")
    print(f"  耗时: {latency5}s")
    print(f"  结果: {'✅ 通过' if passed5 else '❌ 失败'}")
    results.append(("直接回答", passed5, latency5))

    # ═══════════════════════════════════════════════════════════
    # 测试汇总
    # ═══════════════════════════════════════════════════════════
    total_latency = round(time.time() - total_start, 1)
    passed_count = sum(1 for _, passed, _ in results if passed)
    total_count = len(results)

    print("\n" + "=" * 70)
    print("测试汇总")
    print("=" * 70)
    print(f"总测试数: {total_count}")
    print(f"通过: {passed_count}/{total_count} ({passed_count/total_count*100:.1f}%)")
    print(f"总耗时: {total_latency}s")
    print(f"\n详细结果:")
    for name, passed, latency in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {name} ({latency}s)")

    if passed_count == total_count:
        print("\n🎉 所有测试通过！完整 Agent 工作流跑通！")
    else:
        print(f"\n⚠️  有 {total_count - passed_count} 个测试失败，需要检查。")

    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
