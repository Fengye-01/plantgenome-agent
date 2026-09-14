"""
检索效果评估脚本：构建测试集，评估 RAG 检索质量。

对应 Hello Agents 第 8 章 RAG 的「Evaluation（评估）」阶段。
用法：
    $env:HF_ENDPOINT = "https://hf-mirror.com"  # 第一次需要，模型缓存后不需要
    python tests/retrieval_evaluation.py

评估指标：
    - HitRate@1：top-1 结果来自正确 PDF 的比例
    - HitRate@3：top-3 中至少有一个来自正确 PDF 的比例
    - KeywordHitRate：top-1 结果包含期望关键词的比例
    - AvgDistance：top-1 结果的平均余弦距离（越小越好）
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from typing import List, Dict

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.rag.vector_store import VectorStore


# ═══════════════════════════════════════════════════════════════
# 测试集：15 个问题，覆盖 10 个 PDF，难度分布：简单 8 / 中等 4 / 困难 3
# ═══════════════════════════════════════════════════════════════

TEST_SET = [
    # ── 简单问题（8 个）：直接问概念，答案在单个 PDF 里 ──
    {
        "query": "PAML 中的 omega 值是什么意思？",
        "expected_pdf": "PAML",
        "expected_keywords": ["omega", "dN", "dS", "selection", "likelihood"],
        "difficulty": "简单",
    },
    {
        "query": "OrthoFinder 的输入格式是什么？",
        "expected_pdf": "OrthoFinder",
        "expected_keywords": ["fasta", "proteome", "input", "orthogroup", "ortholog"],
        "difficulty": "简单",
    },
    {
        "query": "MAFFT 有哪些比对算法？",
        "expected_pdf": "MAFFT",
        "expected_keywords": ["L-INS-i", "FFT-NS-2", "G-INS-i", "alignment", "multiple"],
        "difficulty": "简单",
    },
    {
        "query": "IQ-TREE 是做什么的？",
        "expected_pdf": "IQ-TREE",
        "expected_keywords": ["phylogenetic", "tree", "maximum likelihood", "model", "bootstrap"],
        "difficulty": "简单",
    },
    {
        "query": "Pfam 数据库是什么？",
        "expected_pdf": "Pfam",
        "expected_keywords": ["protein", "family", "domain", "HMM", "annotation"],
        "difficulty": "简单",
    },
    {
        "query": "eggNOG-mapper 的功能是什么？",
        "expected_pdf": "eggNOG",
        "expected_keywords": ["functional", "annotation", "orthology", "eggNOG", "emapper"],
        "difficulty": "简单",
    },
    {
        "query": "CpG 岛是什么？",
        "expected_pdf": "CpG",
        "expected_keywords": ["CpG", "island", "methylation", "GC", "promoter"],
        "difficulty": "简单",
    },
    {
        "query": "mTERF 基因家族的功能是什么？",
        "expected_pdf": "mTERF",
        "expected_keywords": ["mTERF", "mitochondrial", "transcription", "gene family", "plant"],
        "difficulty": "简单",
    },

    # ── 中等问题（4 个）：需要理解上下文，答案需要综合多个段落 ──
    {
        "query": "如何用 OrthoFinder 进行直系同源基因推断？",
        "expected_pdf": "OrthoFinder",
        "expected_keywords": ["ortholog", "orthogroup", "phylogenetic", "species tree", "gene tree"],
        "difficulty": "中等",
    },
    {
        "query": "PAML 的 codeml 程序怎么用？",
        "expected_pdf": "PAML",
        "expected_keywords": ["codeml", "control file", "sequence", "tree", "likelihood"],
        "difficulty": "中等",
    },
    {
        "query": "MAFFT 的 L-INS-i 和 FFT-NS-2 有什么区别？",
        "expected_pdf": "MAFFT",
        "expected_keywords": ["L-INS-i", "FFT-NS-2", "accuracy", "speed", "iterative"],
        "difficulty": "中等",
    },
    {
        "query": "植物表观遗传调控有哪些机制？",
        "expected_pdf": "Epigenetic",
        "expected_keywords": ["DNA methylation", "histone", "small RNA", "epigenetic", "plant"],
        "difficulty": "中等",
    },

    # ── 困难问题（3 个）：跨文档或需要推理，可能命中多个 PDF ──
    {
        "query": "比较基因组分析中，OrthoFinder 和 MAFFT 怎么配合使用？",
        "expected_pdf": "OrthoFinder",  # 主要期望命中 OrthoFinder
        "expected_keywords": ["orthogroup", "alignment", "MAFFT", "phylogenetic", "ortholog"],
        "difficulty": "困难",
    },
    {
        "query": "基因家族进化分析中，PAML 的 dN/dS 分析有什么作用？",
        "expected_pdf": "PAML",
        "expected_keywords": ["dN", "dS", "omega", "selection", "positive", "purifying"],
        "difficulty": "困难",
    },
    {
        "query": "ModelFinder 在系统发育分析中的作用是什么？",
        "expected_pdf": "ModelFinder",
        "expected_keywords": ["model", "selection", "BIC", "AIC", "substitution", "phylogenetic"],
        "difficulty": "困难",
    },
]


# ═══════════════════════════════════════════════════════════════
# 评估逻辑
# ═══════════════════════════════════════════════════════════════

def evaluate_retrieval(vector_store: VectorStore, test_set: List[Dict]) -> Dict:
    """
    评估检索效果。

    Args:
        vector_store: VectorStore 实例
        test_set: 测试集列表

    Returns:
        评估结果字典，包含各项指标和详细结果
    """
    print("=" * 70)
    print("PlantGenome Agent - 检索效果评估")
    print(f"测试集大小: {len(test_set)} 个问题")
    print(f"知识库文档数: {vector_store.count()}")
    print("=" * 70)

    results = []
    hit1 = 0
    hit3 = 0
    keyword_hit = 0
    total_distance = 0

    for idx, item in enumerate(test_set, 1):
        query = item["query"]
        expected_pdf = item["expected_pdf"]
        expected_keywords = item["expected_keywords"]
        difficulty = item["difficulty"]

        print(f"\n[{idx}/{len(test_set)}] ({difficulty}) {query}")
        print(f"  期望 PDF: {expected_pdf}")

        # 检索 top-3
        retrieved = vector_store.search(query, top_k=3)

        # 判断 HitRate@1
        top1_hit = False
        if retrieved and expected_pdf.lower() in retrieved[0]["metadata"]["filename"].lower():
            top1_hit = True
            hit1 += 1

        # 判断 HitRate@3
        top3_hit = False
        for r in retrieved:
            if expected_pdf.lower() in r["metadata"]["filename"].lower():
                top3_hit = True
                hit3 += 1
                break

        # 判断 KeywordHitRate（top-1 包含期望关键词）
        keyword_hit_flag = False
        if retrieved:
            top1_text = retrieved[0]["text"].lower()
            for kw in expected_keywords:
                if kw.lower() in top1_text:
                    keyword_hit_flag = True
                    keyword_hit += 1
                    break

        # 记录 distance
        top1_distance = retrieved[0]["distance"] if retrieved else 2.0
        total_distance += top1_distance

        # 打印检索结果
        for rank, r in enumerate(retrieved, 1):
            filename = r["metadata"]["filename"]
            page = r["metadata"].get("page_num", "?")
            distance = r["distance"]
            is_expected = "✅" if expected_pdf.lower() in filename.lower() else "  "
            print(f"  {is_expected} Top-{rank}: {filename[:40]}... (p.{page}, dist={distance:.4f})")

        # 打印评估标记
        marks = []
        marks.append("Hit@1 ✅" if top1_hit else "Hit@1 ❌")
        marks.append("Hit@3 ✅" if top3_hit else "Hit@3 ❌")
        marks.append("Keyword ✅" if keyword_hit_flag else "Keyword ❌")
        print(f"  评估: {' | '.join(marks)}")

        results.append({
            "index": idx,
            "query": query,
            "expected_pdf": expected_pdf,
            "difficulty": difficulty,
            "hit1": top1_hit,
            "hit3": top3_hit,
            "keyword_hit": keyword_hit_flag,
            "top1_distance": top1_distance,
            "retrieved": [
                {
                    "filename": r["metadata"]["filename"],
                    "page": r["metadata"].get("page_num"),
                    "distance": r["distance"],
                }
                for r in retrieved
            ],
        })

    # 汇总指标
    n = len(test_set)
    metrics = {
        "total_questions": n,
        "HitRate@1": hit1 / n,
        "HitRate@3": hit3 / n,
        "KeywordHitRate": keyword_hit / n,
        "AvgDistance": total_distance / n,
    }

    # 按难度分组统计
    difficulty_stats = {}
    for diff in ["简单", "中等", "困难"]:
        diff_results = [r for r in results if r["difficulty"] == diff]
        if diff_results:
            difficulty_stats[diff] = {
                "count": len(diff_results),
                "HitRate@1": sum(1 for r in diff_results if r["hit1"]) / len(diff_results),
                "HitRate@3": sum(1 for r in diff_results if r["hit3"]) / len(diff_results),
                "KeywordHitRate": sum(1 for r in diff_results if r["keyword_hit"]) / len(diff_results),
                "AvgDistance": sum(r["top1_distance"] for r in diff_results) / len(diff_results),
            }

    # 打印汇总
    print(f"\n{'=' * 70}")
    print("评估汇总")
    print(f"{'=' * 70}")
    print(f"  总问题数: {metrics['total_questions']}")
    print(f"  HitRate@1:      {metrics['HitRate@1']*100:.1f}%  (目标 ≥ 70%)")
    print(f"  HitRate@3:      {metrics['HitRate@3']*100:.1f}%  (目标 ≥ 90%)")
    print(f"  KeywordHitRate: {metrics['KeywordHitRate']*100:.1f}%  (目标 ≥ 60%)")
    print(f"  AvgDistance:    {metrics['AvgDistance']:.4f}  (目标 < 0.6)")

    print(f"\n  按难度分组:")
    for diff, stats in difficulty_stats.items():
        print(f"    {diff} ({stats['count']}题): "
              f"Hit@1={stats['HitRate@1']*100:.0f}%, "
              f"Hit@3={stats['HitRate@3']*100:.0f}%, "
              f"Keyword={stats['KeywordHitRate']*100:.0f}%, "
              f"AvgDist={stats['AvgDistance']:.4f}")

    # 薄弱点分析
    print(f"\n{'=' * 70}")
    print("薄弱点分析")
    print(f"{'=' * 70}")

    failed_hit1 = [r for r in results if not r["hit1"]]
    failed_hit3 = [r for r in results if not r["hit3"]]
    failed_keyword = [r for r in results if not r["keyword_hit"]]

    if failed_hit1:
        print(f"\n  Hit@1 失败的问题 ({len(failed_hit1)} 个):")
        for r in failed_hit1:
            print(f"    - [{r['difficulty']}] {r['query']}")
            print(f"      期望: {r['expected_pdf']}, 实际 Top-1: {r['retrieved'][0]['filename'][:50] if r['retrieved'] else '无'}")

    if failed_hit3:
        print(f"\n  Hit@3 失败的问题 ({len(failed_hit3)} 个):")
        for r in failed_hit3:
            print(f"    - [{r['difficulty']}] {r['query']}")
            print(f"      期望: {r['expected_pdf']}")

    if failed_keyword:
        print(f"\n  Keyword 失败的问题 ({len(failed_keyword)} 个):")
        for r in failed_keyword:
            print(f"    - [{r['difficulty']}] {r['query']}")

    # 改进建议
    print(f"\n  改进建议:")
    if metrics["HitRate@1"] < 0.7:
        print("    ⚠️  HitRate@1 低于目标，可能原因：")
        print("       - 分块不合理（chunk 太大/太小）")
        print("       - 查询扩展不足（中文查询匹配英文文档）")
        print("       - 知识库覆盖不全（某些 PDF 内容少）")
        print("       - 建议：加 reranker、优化分块、查询翻译/扩展")
    if metrics["HitRate@3"] < 0.9:
        print("    ⚠️  HitRate@3 低于目标，说明知识库可能缺少相关内容")
        print("       - 建议：补充相关 PDF，或检查 PDF 解析质量")
    if metrics["AvgDistance"] > 0.6:
        print("    ⚠️  AvgDistance 偏高，说明检索结果与查询语义距离较远")
        print("       - 建议：检查 embedding 模型是否正确加载，查询是否太模糊")

    if metrics["HitRate@1"] >= 0.7 and metrics["HitRate@3"] >= 0.9:
        print("    ✅ 检索质量达标，可以进入下一步（Retriever 封装 + LLM 生成）")

    return {
        "metrics": metrics,
        "difficulty_stats": difficulty_stats,
        "results": results,
        "evaluation_time": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    print("[初始化] 加载 VectorStore...")
    vector_store = VectorStore()
    print(f"  ✅ VectorStore 加载完成，文档数: {vector_store.count()}")

    report = evaluate_retrieval(vector_store, TEST_SET)

    # 保存报告到文件
    report_path = os.path.join(PROJECT_ROOT, "tests", "retrieval_evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 检索效果评估报告\n\n")
        f.write(f"评估时间: {report['evaluation_time']}\n\n")
        f.write("## 总体指标\n\n")
        f.write(f"| 指标 | 值 | 目标 |\n")
        f.write(f"|---|---|---|\n")
        f.write(f"| 总问题数 | {report['metrics']['total_questions']} | - |\n")
        f.write(f"| HitRate@1 | {report['metrics']['HitRate@1']*100:.1f}% | ≥ 70% |\n")
        f.write(f"| HitRate@3 | {report['metrics']['HitRate@3']*100:.1f}% | ≥ 90% |\n")
        f.write(f"| KeywordHitRate | {report['metrics']['KeywordHitRate']*100:.1f}% | ≥ 60% |\n")
        f.write(f"| AvgDistance | {report['metrics']['AvgDistance']:.4f} | < 0.6 |\n")
        f.write("\n## 按难度分组\n\n")
        f.write("| 难度 | 题数 | Hit@1 | Hit@3 | Keyword | AvgDist |\n")
        f.write("|---|---|---|---|---|---|\n")
        for diff, stats in report["difficulty_stats"].items():
            f.write(f"| {diff} | {stats['count']} | {stats['HitRate@1']*100:.0f}% | "
                    f"{stats['HitRate@3']*100:.0f}% | {stats['KeywordHitRate']*100:.0f}% | "
                    f"{stats['AvgDistance']:.4f} |\n")
        f.write("\n## 详细结果\n\n")
        for r in report["results"]:
            f.write(f"### {r['index']}. [{r['difficulty']}] {r['query']}\n\n")
            f.write(f"- 期望 PDF: {r['expected_pdf']}\n")
            f.write(f"- Hit@1: {'✅' if r['hit1'] else '❌'}\n")
            f.write(f"- Hit@3: {'✅' if r['hit3'] else '❌'}\n")
            f.write(f"- Keyword: {'✅' if r['keyword_hit'] else '❌'}\n")
            f.write(f"- Top-1 Distance: {r['top1_distance']:.4f}\n")
            f.write("- 检索结果:\n")
            for rank, ret in enumerate(r["retrieved"], 1):
                f.write(f"  {rank}. {ret['filename']} (p.{ret['page']}, dist={ret['distance']:.4f})\n")
            f.write("\n")

    print(f"\n📄 评估报告已保存到: {report_path}")
