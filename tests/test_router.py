"""
router_node 测试（D8 任务 3）

10 个测试问题，验证 router 分类准确率。
目标：10 个中至少 8 个分类正确。

运行：
    cd C:/Users/YeFeng/Desktop/自救计划/plantgenome-agent
    $env:HF_ENDPOINT = "https://hf-mirror.com"
    python tests/test_router.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 项目根路径导入
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.nodes import router_node
from app.agents.state import create_initial_state


# ═══════════════════════════════════════════════════════════════
# 测试用例
# ═══════════════════════════════════════════════════════════════

TEST_CASES = [
    {
        "id": 1,
        "query": "PAML codeml 怎么设置分支模型？",
        "expected_intent": "literature_search",
        "description": "生信工具用法查询",
    },
    {
        "id": 2,
        "query": ">seq1\nATCGATCG\n>seq2\nATCGATCGATCG\n帮我统计一下这个 FASTA",
        "expected_intent": "fasta_analysis",
        "description": "FASTA 序列统计",
    },
    {
        "id": 3,
        "query": "这段序列里有没有 CpG 岛？ATCGATCGATCGATCGATCGATCG",
        "expected_intent": "cpg_scan",
        "description": "CpG 岛识别",
    },
    {
        "id": 4,
        "query": "研究 mTERF 家族进化需要什么流程？",
        "expected_intent": "pipeline_suggest",
        "description": "研究流程规划",
    },
    {
        "id": 5,
        "query": "你好",
        "expected_intent": "direct_answer",
        "description": "闲聊问候",
    },
    {
        "id": 6,
        "query": "OrthoFinder 结果怎么解读？",
        "expected_intent": "literature_search",
        "description": "软件结果解读",
    },
    {
        "id": 7,
        "query": "帮我分析这个基因家族的选择压力，需要用什么工具？",
        "expected_intent": "pipeline_suggest",
        "description": "分析方案推荐",
    },
    {
        "id": 8,
        "query": "MAFFT 和 ClustalW 哪个比对效果好？",
        "expected_intent": "literature_search",
        "description": "工具比较查询",
    },
    {
        "id": 9,
        "query": ">gene1\nATCGATCGATCGATCGATCGATCGATCGATCG\n扫描 CpG 岛",
        "expected_intent": "cpg_scan",
        "description": "DNA 序列 CpG 扫描",
    },
    {
        "id": 10,
        "query": "谢谢，你的回答很有帮助",
        "expected_intent": "direct_answer",
        "description": "感谢闲聊",
    },
]


# ═══════════════════════════════════════════════════════════════
# 运行测试
# ═══════════════════════════════════════════════════════════════

def run_tests():
    print("=" * 80)
    print("router_node 意图分类测试")
    print("=" * 80)

    results = []
    correct_count = 0

    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n--- 测试 {i}/{len(TEST_CASES)} ---")
        print(f"问题: {case['query'][:80]}{'...' if len(case['query']) > 80 else ''}")
        print(f"描述: {case['description']}")
        print(f"期望 intent: {case['expected_intent']}")

        # 创建初始状态
        state = create_initial_state(case["query"])

        # 调用 router_node
        try:
            updated_state = router_node(state)
            actual_intent = updated_state["intent"]
            actual_tool = updated_state["tool_name"]
            actual_input = updated_state["tool_input"]

            # 判断是否正确
            is_correct = actual_intent == case["expected_intent"]
            if is_correct:
                correct_count += 1

            print(f"实际 intent: {actual_intent} {'✅' if is_correct else '❌'}")
            print(f"实际 tool: {actual_tool}")
            print(f"实际 tool_input: {actual_input}")

            results.append({
                "id": case["id"],
                "query": case["query"],
                "expected": case["expected_intent"],
                "actual": actual_intent,
                "tool": actual_tool,
                "tool_input": actual_input,
                "correct": is_correct,
            })

        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
            results.append({
                "id": case["id"],
                "query": case["query"],
                "expected": case["expected_intent"],
                "actual": "ERROR",
                "correct": False,
                "error": str(e),
            })

    # ═══════════════════════════════════════════════════════════
    # 汇总报告
    # ═══════════════════════════════════════════════════════════

    total = len(TEST_CASES)
    print("\n" + "=" * 80)
    print("测试汇总报告")
    print("=" * 80)
    print(f"总测试用例数: {total}")
    print(f"分类正确: {correct_count}/{total} ({correct_count/total*100:.1f}%)")
    print(f"目标: ≥8/10 (80%)")
    print(f"结果: {'✅ 达标' if correct_count >= 8 else '❌ 未达标，需要优化 ROUTER_PROMPT'}")

    print("\n详细结果:")
    print(f"{'#':<3} {'期望':<20} {'实际':<20} {'结果':<6} {'描述'}")
    print("-" * 80)
    for r in results:
        status = "✅" if r["correct"] else "❌"
        print(f"{r['id']:<3} {r['expected']:<20} {r['actual']:<20} {status:<6} {TEST_CASES[r['id']-1]['description']}")

    # 错误用例分析
    failed = [r for r in results if not r["correct"]]
    if failed:
        print("\n❌ 分类错误用例分析:")
        for r in failed:
            print(f"\n  [{r['id']}] 问题: {r['query'][:80]}")
            print(f"       期望: {r['expected']}")
            print(f"       实际: {r['actual']}")
            print(f"       工具: {r.get('tool')}")
            print(f"       建议: 优化 ROUTER_PROMPT 中该类意图的描述和示例")
    else:
        print("\n🎉 所有用例全部分类正确！")

    print("\n" + "=" * 80)
    return results


if __name__ == "__main__":
    results = run_tests()
