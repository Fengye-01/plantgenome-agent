"""
D13 稳定性测试：验证 3 类意图的端到端稳定性
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.graph import run_agent


def run_case(name, query):
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"输入: {query[:80]}")
    print(f"{'='*60}")

    try:
        r = run_agent(query)
        intent = r.get("intent")
        tool_name = r.get("tool_name")
        answer = r.get("final_answer", "")
        sources = r.get("sources", [])
        execution_log = r.get("execution_log", [])
        tool_result = r.get("tool_result")

        print(f"  intent: {intent}")
        print(f"  tool_name: {tool_name}")
        print(f"  answer_len: {len(answer)}")
        print(f"  sources: {len(sources)}")
        print(f"  log_nodes: {[log.get('node') for log in execution_log]}")

        if isinstance(tool_result, dict) and "steps" in tool_result:
            print(f"  steps: {len(tool_result['steps'])}")

        # 检查是否有错误
        has_error = False
        for log in execution_log:
            if log.get("status") == "failed":
                has_error = True
                print(f"  ⚠️  节点失败: {log.get('node')} - {log.get('error', 'unknown')}")

        if not has_error and answer:
            print(f"  ✅ 测试通过")
            return True
        else:
            print(f"  ❌ 测试失败")
            return False

    except Exception as e:
        print(f"  ❌ 异常: {str(e)}")
        return False


def main():
    print("=" * 60)
    print("PlantGenome Agent 稳定性测试")
    print("=" * 60)

    results = []

    # 测试 1: direct_answer
    results.append(("direct_answer", run_case("直接问候", "你好")))

    # 测试 2: literature_search
    results.append(("literature_search", run_case("PAML omega", "PAML omega 值是什么意思？")))

    # 测试 3: pipeline_suggest
    results.append(("pipeline_suggest", run_case("流程推荐", "研究 mTERF 家族进化需要什么流程？")))

    # 汇总
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    passed = sum(1 for _, p in results if p)
    total = len(results)
    print(f"通过: {passed}/{total}")
    for name, p in results:
        print(f"  {'✅' if p else '❌'} {name}")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
