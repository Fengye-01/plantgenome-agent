"""
阶段2-3：重建Gold Set + Dense Baseline
- 标注35条query为answerable/unanswerable/ambiguous
- 对answerable query检索Top-5
- 计算HitRate@1/3/5和MRR
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from tests.eval_dataset import EVAL_DATASET
from app.rag.vector_store import VectorStore


# ═══════════════════════════════════════════════════════════
# Gold Set V2：基于冻结corpus人工标注
# ═══════════════════════════════════════════════════════════
# 知识库14个文档：
# 1. Angermueller 2017 DeepCpG (单细胞甲基化预测)
# 2. Bock 2007 CpG island mapping (CpG岛预测)
# 3. PAML_test_upload (PAML手册测试片段)
# 4. Cain 2022 Intragenic CpG islands (CpG岛调控)
# 5. Ashikawa 2001 (植物CpG岛)
# 6. eggNOG-mapper v2 (功能注释)
# 7. OrthoFinder 2019 (直系同源)
# 8. Epigenetic Regulation in Plants (植物表观调控)
# 9. ModelFinder 2017 (模型选择)
# 10. MAFFT 2013 (多序列比对)
# 11. IQ-TREE 2 2020 (系统发育树)
# 12. Pfam 2021 (蛋白家族数据库)
# 13. Research Progress in Plant Molecular Functions
# 14. PAML 4 2007 (最大似然进化分析)

# relevant_doc_keywords: 用来匹配文件名的关键词
GOLD_V2 = {
    # === ANSWERABLE (18条) ===
    1:  {"label": "answerable", "relevant_docs": ["PAML"], "reason": "Yang 2007 PAML 4手册详细解释omega/dN/dS"},
    2:  {"label": "answerable", "relevant_docs": ["OrthoFinder"], "reason": "OrthoFinder 2019论文描述输入格式"},
    3:  {"label": "answerable", "relevant_docs": ["MAFFT"], "reason": "MAFFT 2013论文描述L-INS-i/G-INS-i"},
    4:  {"label": "answerable", "relevant_docs": ["Research Progress"], "reason": "Research Progress论文提到mTERF"},
    5:  {"label": "answerable", "relevant_docs": ["Bock", "Ashikawa", "Cain", "DeepCpG"], "reason": "多篇CpG岛论文描述定义标准"},
    6:  {"label": "answerable", "relevant_docs": ["PAML"], "reason": "PAML手册描述codeml分支/位点模型"},
    7:  {"label": "answerable", "relevant_docs": ["IQ-TREE", "OrthoFinder", "MAFFT", "ModelFinder"], "reason": "系统发育分析流程在多个工具文档中有描述"},
    8:  {"label": "answerable", "relevant_docs": ["Bock", "Cain", "DeepCpG", "Epigenetic"], "reason": "多篇论文讨论DNA甲基化与CpG岛关系"},
    9:  {"label": "answerable", "relevant_docs": ["OrthoFinder"], "reason": "OrthoFinder论文描述输出结果"},
    10: {"label": "answerable", "relevant_docs": ["OrthoFinder"], "reason": "OrthoFinder论文介绍比较基因组学"},
    12: {"label": "answerable", "relevant_docs": ["IQ-TREE"], "reason": "IQ-TREE 2论文对比其他方法"},
    18: {"label": "answerable", "relevant_docs": ["PAML"], "reason": "PAML手册描述正选择/纯化选择检测"},
    21: {"label": "answerable", "relevant_docs": ["MAFFT"], "reason": "MAFFT论文对比ClustalW"},
    24: {"label": "answerable", "relevant_docs": ["Pfam"], "reason": "Pfam论文描述HMMER/HMM模型"},
    25: {"label": "answerable", "relevant_docs": ["OrthoFinder"], "reason": "OrthoFinder论文定义Orthogroup"},
    28: {"label": "answerable", "relevant_docs": ["DeepCpG", "Bock", "Cain"], "reason": "DeepCpG和CpG岛论文涉及BS-seq/甲基化分析"},
    33: {"label": "answerable", "relevant_docs": ["PAML"], "reason": "PAML手册描述branch-site model"},
    34: {"label": "answerable", "relevant_docs": ["MAFFT", "IQ-TREE"], "reason": "MAFFT和IQ-TREE论文讨论gap处理"},

    # === AMBIGUOUS (5条) ===
    11: {"label": "ambiguous", "reason": "Pfam论文提到HMMER但不专门讲BLAST vs HMMER区别"},
    16: {"label": "ambiguous", "reason": "Epigenetic Regulation可能提到转录因子但不专门讲WRKY"},
    17: {"label": "ambiguous", "reason": "Pfam有NBS-LRR结构域但不专门讲分类"},
    19: {"label": "ambiguous", "reason": "IQ-TREE论文可能提到FastTree对比但不详细"},
    29: {"label": "ambiguous", "reason": "Epigenetic Regulation可能提到小RNA但不专门讲"},

    # === UNANSWERABLE (12条) ===
    13: {"label": "unanswerable", "reason": "知识库无共线性/MCScan相关文档"},
    14: {"label": "unanswerable", "reason": "知识库无密码子偏好性相关文档"},
    15: {"label": "unanswerable", "reason": "知识库无叶绿体基因组专门文档"},
    20: {"label": "unanswerable", "reason": "知识库无MEGA软件文档"},
    22: {"label": "unanswerable", "reason": "知识库无泛基因组相关文档"},
    23: {"label": "unanswerable", "reason": "知识库无复制基因进化专门文档"},
    26: {"label": "unanswerable", "reason": "知识库无群体遗传学/连锁不平衡文档"},
    27: {"label": "unanswerable", "reason": "知识库无GWAS相关文档"},
    30: {"label": "unanswerable", "reason": "知识库无CRISPR相关文档"},
    31: {"label": "unanswerable", "reason": "知识库无结构变异相关文档"},
    32: {"label": "unanswerable", "reason": "知识库无线粒体基因组专门文档"},
    35: {"label": "unanswerable", "reason": "知识库无共线性区块相关文档"},
}


def is_relevant_doc(filename: str, relevant_keywords: list) -> bool:
    """检查检索结果的文件名是否匹配相关文档关键词。"""
    fname_lower = filename.lower()
    for kw in relevant_keywords:
        if kw.lower() in fname_lower:
            return True
    return False


def run():
    print("=" * 70)
    print("阶段2-3：Gold Set V2 + Dense Baseline")
    print("=" * 70)

    vs = VectorStore()
    print(f"\nCorpus: {vs.count()} chunks, {len(json.load(open(PROJECT_ROOT / 'docs' / 'corpus_manifest.json'))['documents'])} documents")

    lit_cases = [c for c in EVAL_DATASET if c["category"] == "literature_search"]

    # 统计标注
    labels = {"answerable": 0, "unanswerable": 0, "ambiguous": 0}
    for case in lit_cases:
        labels[GOLD_V2[case["id"]]["label"]] += 1
    print(f"\nGold Set V2 标注:")
    print(f"  Answerable:   {labels['answerable']} 条")
    print(f"  Unanswerable: {labels['unanswerable']} 条")
    print(f"  Ambiguous:    {labels['ambiguous']} 条")

    # 对answerable query检索Top-5
    answerable_cases = [c for c in lit_cases if GOLD_V2[c["id"]]["label"] == "answerable"]
    unanswerable_cases = [c for c in lit_cases if GOLD_V2[c["id"]]["label"] == "unanswerable"]

    results = []
    print(f"\n{'='*70}")
    print(f"Answerable queries 检索结果（{len(answerable_cases)}条）")
    print(f"{'='*70}")

    for case in answerable_cases:
        query = case["question"]
        gold = GOLD_V2[case["id"]]

        raw = vs.search(query, top_k=5)

        # 判断每个rank是否相关
        ranks = []
        for i, r in enumerate(raw):
            fname = r["metadata"].get("filename", "")
            relevant = is_relevant_doc(fname, gold["relevant_docs"])
            ranks.append({
                "rank": i + 1,
                "filename": fname[:55],
                "distance": round(r["distance"], 4),
                "relevant": relevant,
            })

        # 找第一个相关的rank
        first_relevant = None
        for r in ranks:
            if r["relevant"]:
                first_relevant = r["rank"]
                break

        hit1 = first_relevant == 1
        hit3 = first_relevant is not None and first_relevant <= 3
        hit5 = first_relevant is not None and first_relevant <= 5
        rr = 1.0 / first_relevant if first_relevant else 0.0

        results.append({
            "id": case["id"],
            "question": query[:60],
            "relevant_docs": gold["relevant_docs"],
            "first_relevant_rank": first_relevant,
            "hit@1": hit1,
            "hit@3": hit3,
            "hit@5": hit5,
            "rr": rr,
            "ranks": ranks,
        })

        mark = "✅" if hit3 else ("⚠️" if hit5 else "❌")
        fr = f"R{first_relevant}" if first_relevant else "miss"
        print(f"  [{case['id']:2d}] {mark} {fr:>5} | {query[:45]}...")

    # 汇总指标
    n = len(results)
    h1 = sum(1 for r in results if r["hit@1"])
    h3 = sum(1 for r in results if r["hit@3"])
    h5 = sum(1 for r in results if r["hit@5"])
    mrr = sum(r["rr"] for r in results) / n

    print(f"\n{'='*70}")
    print(f"Dense Retrieval Baseline（仅answerable queries, n={n}）")
    print(f"{'='*70}")
    print(f"  HitRate@1:  {h1}/{n} = {h1/n:.1%}")
    print(f"  HitRate@3:  {h3}/{n} = {h3/n:.1%}")
    print(f"  HitRate@5:  {h5}/{n} = {h5/n:.1%}")
    print(f"  MRR:        {mrr:.3f}")

    # 真正的retrieval failure
    failures = [r for r in results if not r["hit@5"]]
    if failures:
        print(f"\n  真正Retrieval Failure（Top-5无相关文档）:")
        for f in failures:
            print(f"    [{f['id']:2d}] {f['question'][:50]}...")
            for r in f["ranks"]:
                rel_mark = "✓" if r["relevant"] else " "
                print(f"         {rel_mark} R{r['rank']}: d={r['distance']:.3f} {r['filename']}")
    else:
        print(f"\n  无Top-5完全失败case")

    # 保存结果
    output = {
        "gold_set_v2": {str(k): v for k, v in GOLD_V2.items()},
        "summary": {
            "answerable_n": n,
            "hit@1": h1,
            "hit@1_rate": round(h1/n, 4),
            "hit@3": h3,
            "hit@3_rate": round(h3/n, 4),
            "hit@5": h5,
            "hit@5_rate": round(h5/n, 4),
            "mrr": round(mrr, 4),
        },
        "details": results,
    }

    out_path = PROJECT_ROOT / "docs" / "dense_baseline.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {out_path}")

    # Gold Set V2单独保存
    gold_path = PROJECT_ROOT / "docs" / "rag_gold_v2.json"
    gold_data = {
        "description": "RAG Gold Set V2 - 基于冻结corpus人工标注",
        "corpus": "14 documents, 398 chunks",
        "labels": GOLD_V2,
        "statistics": labels,
    }
    with open(gold_path, "w", encoding="utf-8") as f:
        json.dump(gold_data, f, ensure_ascii=False, indent=2)
    print(f"Gold Set V2已保存: {gold_path}")


if __name__ == "__main__":
    run()
