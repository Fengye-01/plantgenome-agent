r"""
PlantGenome Agent - 评估脚本

对应 Hello Agents 第12章（性能评估）。
自动运行 20 条测试，计算 HitRate@3、工具调用准确率、回答关键词命中率。

用法：
    cd C:/Users/YeFeng/Desktop/自救计划/plantgenome-agent
    $env:HF_ENDPOINT = "https://hf-mirror.com"
    python scripts/run_evaluation.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.eval_dataset import EVAL_DATASET, get_dataset_summary
from app.agents.graph import run_agent


def evaluate_item(item: dict) -> dict:
    """评估单条测试用例。"""
    question = item["question"]
    category = item["category"]
    expected_tool = item["expected_tool"]
    expected_keywords = item["expected_keywords"]
    expected_source = item.get("expected_source")

    print(f"  [{item['id']:2d}] {question[:60]}...", end=" ", flush=True)

    start_time = time.time()
    try:
        result = run_agent(question)
        elapsed = time.time() - start_time

        intent = result.get("intent", "")
        tool_name = result.get("tool_name")
        final_answer = result.get("final_answer", "")
        sources = result.get("sources", [])
        execution_log = result.get("execution_log", [])

        # 1. 工具调用准确率
        tool_correct = (tool_name == expected_tool)

        # 2. HitRate@3（仅 literature_search 类）
        hit_at_3 = None
        if category == "literature_search" and expected_source:
            source_files = []
            for s in sources:
                if isinstance(s, dict):
                    fname = s.get("filename", "").lower()
                    source_files.append(fname)
                elif isinstance(s, str):
                    source_files.append(s.lower())
            # 检查期望来源关键词是否出现在任一来源文件名中
            hit_at_3 = any(expected_source.lower() in sf for sf in source_files)

        # 3. 回答关键词命中率
        answer_lower = final_answer.lower()
        keyword_hits = [kw for kw in expected_keywords if kw.lower() in answer_lower]
        keyword_hit_rate = len(keyword_hits) / len(expected_keywords) if expected_keywords else 0

        status = "✅" if tool_correct else "❌"
        print(f"{status} ({elapsed:.1f}s) intent={intent}, tool={tool_name}")

        return {
            "id": item["id"],
            "question": question,
            "category": category,
            "difficulty": item["difficulty"],
            "intent": intent,
            "tool_name": tool_name,
            "expected_tool": expected_tool,
            "tool_correct": tool_correct,
            "hit_at_3": hit_at_3,
            "expected_source": expected_source,
            "keyword_hits": keyword_hits,
            "keyword_hit_rate": keyword_hit_rate,
            "sources_count": len(sources),
            "answer_length": len(final_answer),
            "answer_preview": final_answer[:200],
            "elapsed_seconds": round(elapsed, 1),
            "execution_log_nodes": [log.get("node") for log in execution_log],
            "error": None,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ 异常: {str(e)[:80]}")
        return {
            "id": item["id"],
            "question": question,
            "category": category,
            "difficulty": item["difficulty"],
            "intent": "",
            "tool_name": None,
            "expected_tool": expected_tool,
            "tool_correct": False,
            "hit_at_3": None,
            "expected_source": expected_source,
            "keyword_hits": [],
            "keyword_hit_rate": 0,
            "sources_count": 0,
            "answer_length": 0,
            "answer_preview": "",
            "elapsed_seconds": round(elapsed, 1),
            "execution_log_nodes": [],
            "error": str(e),
        }


def main():
    print("=" * 70)
    print("PlantGenome Agent - 系统评估")
    print("=" * 70)

    summary = get_dataset_summary()
    print(f"\n测试集: {summary['total']} 条")
    print(f"分布: {summary['by_category']}")
    print(f"难度: {summary['by_difficulty']}")
    print(f"\n开始评估...\n")

    results = []
    for item in EVAL_DATASET:
        result = evaluate_item(item)
        results.append(result)

    # ═══════════════════════════════════════════════════════════
    # 统计汇总
    # ═══════════════════════════════════════════════════════════
    total = len(results)
    tool_correct_count = sum(1 for r in results if r["tool_correct"])
    tool_accuracy = tool_correct_count / total if total else 0

    # HitRate@3（仅 literature_search）
    lit_results = [r for r in results if r["category"] == "literature_search" and r["hit_at_3"] is not None]
    hit_count = sum(1 for r in lit_results if r["hit_at_3"])
    hit_rate_at_3 = hit_count / len(lit_results) if lit_results else 0

    # 关键词命中率
    avg_keyword_rate = sum(r["keyword_hit_rate"] for r in results) / total if total else 0

    # 平均耗时
    avg_elapsed = sum(r["elapsed_seconds"] for r in results) / total if total else 0

    # 分类统计
    category_stats = {}
    for cat in ["literature_search", "fasta_analysis", "cpg_scan", "pipeline_suggest"]:
        cat_results = [r for r in results if r["category"] == cat]
        if cat_results:
            cat_tool_acc = sum(1 for r in cat_results if r["tool_correct"]) / len(cat_results)
            cat_kw_rate = sum(r["keyword_hit_rate"] for r in cat_results) / len(cat_results)
            cat_avg_time = sum(r["elapsed_seconds"] for r in cat_results) / len(cat_results)
            category_stats[cat] = {
                "count": len(cat_results),
                "tool_accuracy": cat_tool_acc,
                "keyword_rate": cat_kw_rate,
                "avg_time": cat_avg_time,
            }

    # ═══════════════════════════════════════════════════════════
    # 输出汇总
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("评估结果汇总")
    print("=" * 70)
    print(f"  总测试数:          {total}")
    print(f"  RAG HitRate@3:     {hit_rate_at_3:.1%} ({hit_count}/{len(lit_results)}) [仅文献检索类]")
    print(f"  工具调用准确率:     {tool_accuracy:.1%} ({tool_correct_count}/{total})")
    print(f"  回答关键词命中率:   {avg_keyword_rate:.1%}")
    print(f"  平均响应时间:       {avg_elapsed:.1f}s")

    print(f"\n  分类统计:")
    for cat, stats in category_stats.items():
        print(f"    {cat:25s}: {stats['count']} 条, 工具准确率 {stats['tool_accuracy']:.0%}, "
              f"关键词命中 {stats['keyword_rate']:.0%}, 平均 {stats['avg_time']:.1f}s")

    # 错误分析
    print(f"\n  工具调用错误的用例:")
    wrong_tools = [r for r in results if not r["tool_correct"]]
    if wrong_tools:
        for r in wrong_tools:
            print(f"    [{r['id']:2d}] 期望: {r['expected_tool']}, 实际: {r['tool_name']}, "
                  f"intent: {r['intent']}")
    else:
        print("    （无）")

    # HitRate 未命中
    print(f"\n  HitRate@3 未命中的用例:")
    missed = [r for r in lit_results if not r["hit_at_3"]]
    if missed:
        for r in missed:
            print(f"    [{r['id']:2d}] 期望来源: {r['expected_source']}, "
                  f"sources数: {r['sources_count']}")
    else:
        print("    （无）")

    # ═══════════════════════════════════════════════════════════
    # 保存结果
    # ═══════════════════════════════════════════════════════════
    output = {
        "summary": {
            "total": total,
            "hit_rate_at_3": round(hit_rate_at_3, 4),
            "hit_count": hit_count,
            "lit_count": len(lit_results),
            "tool_accuracy": round(tool_accuracy, 4),
            "tool_correct_count": tool_correct_count,
            "avg_keyword_rate": round(avg_keyword_rate, 4),
            "avg_elapsed_seconds": round(avg_elapsed, 1),
            "category_stats": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                                     for kk, vv in v.items()}
                                for k, v in category_stats.items()},
        },
        "details": results,
    }

    output_path = PROJECT_ROOT / "docs" / "evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n评估结果已保存到: {output_path}")
    print("=" * 70)

    return output


if __name__ == "__main__":
    main()
