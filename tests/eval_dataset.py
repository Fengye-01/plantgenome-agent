"""
PlantGenome Agent - 评估测试集（20 条）

对应 Hello Agents 第12章（性能评估）。
每条测试标注：question / category / expected_source / expected_keywords / expected_tool

分布：
- literature_search: 10 条（PAML、OrthoFinder、MAFFT、mTERF、CpG、比较基因组）
- fasta_analysis: 3 条
- cpg_scan: 3 条（阳性 2 + 阴性 1）
- pipeline_suggest: 4 条
"""

EVAL_DATASET = [
    # ═══════════════════════════════════════════════════════════
    # literature_search: 10 条
    # ═══════════════════════════════════════════════════════════

    {
        "id": 1,
        "question": "PAML 中的 omega (dN/dS) 值怎么解释？",
        "category": "literature_search",
        "expected_source": "paml",  # 期望来源包含 PAML 相关文档
        "expected_keywords": ["omega", "dN", "dS", "选择", "正选择", "纯化选择"],
        "expected_tool": "search_pdf_knowledge",
        "difficulty": "简单",
    },
    {
        "id": 2,
        "question": "OrthoFinder 的输入需要什么格式？",
        "category": "literature_search",
        "expected_source": "orthofinder",
        "expected_keywords": ["OrthoFinder", "输入", "FASTA", "蛋白", "物种"],
        "expected_tool": "search_pdf_knowledge",
        "difficulty": "简单",
    },
    {
        "id": 3,
        "question": "MAFFT 的 L-INS-i 和 G-INS-i 有什么区别？",
        "category": "literature_search",
        "expected_source": "mafft",
        "expected_keywords": ["MAFFT", "L-INS-i", "G-INS-i", "算法", "比对"],
        "expected_tool": "search_pdf_knowledge",
        "difficulty": "中等",
    },
    {
        "id": 4,
        "question": "mTERF 基因家族在植物中的功能是什么？",
        "category": "literature_search",
        "expected_source": "mterf",
        "expected_keywords": ["mTERF", "植物", "功能", "线粒体", "叶绿体", "基因表达"],
        "expected_tool": "search_pdf_knowledge",
        "difficulty": "中等",
    },
    {
        "id": 5,
        "question": "什么是 CpG 岛？它的定义标准是什么？",
        "category": "literature_search",
        "expected_source": "cpg",
        "expected_keywords": ["CpG", "岛", "定义", "GC", "观察", "期望", "200"],
        "expected_tool": "search_pdf_knowledge",
        "difficulty": "简单",
    },
    {
        "id": 6,
        "question": "PAML codeml 的分支模型和位点模型有什么区别？",
        "category": "literature_search",
        "expected_source": "paml",
        "expected_keywords": ["PAML", "codeml", "分支模型", "位点模型", "选择压力"],
        "expected_tool": "search_pdf_knowledge",
        "difficulty": "困难",
    },
    {
        "id": 7,
        "question": "怎么做基因家族的系统发育分析？",
        "category": "literature_search",
        "expected_source": "phylogenetic",
        "expected_keywords": ["系统发育", "基因家族", "进化树", "MAFFT", "IQ-TREE", "MEGA"],
        "expected_tool": "search_pdf_knowledge",
        "difficulty": "中等",
    },
    {
        "id": 8,
        "question": "DNA 甲基化和 CpG 岛有什么关系？",
        "category": "literature_search",
        "expected_source": "cpg",
        "expected_keywords": ["DNA", "甲基化", "CpG", "岛", "基因表达", "调控"],
        "expected_tool": "search_pdf_knowledge",
        "difficulty": "中等",
    },
    {
        "id": 9,
        "question": "OrthoFinder 输出结果怎么解读？",
        "category": "literature_search",
        "expected_source": "orthofinder",
        "expected_keywords": ["OrthoFinder", "输出", "结果", "正交群", "同源基因", "解读"],
        "expected_tool": "search_pdf_knowledge",
        "difficulty": "中等",
    },
    {
        "id": 10,
        "question": "比较基因组学主要分析哪些内容？",
        "category": "literature_search",
        "expected_source": "comparative",
        "expected_keywords": ["比较基因组", "共线性", "基因家族", "进化", "选择压力", "基因组"],
        "expected_tool": "search_pdf_knowledge",
        "difficulty": "简单",
    },

    # ═══════════════════════════════════════════════════════════
    # fasta_analysis: 3 条
    # ═══════════════════════════════════════════════════════════

    {
        "id": 11,
        "question": "帮我统计这个FASTA序列：\n>seq1\nATCGATCGATCG\n>seq2\nGGCCGGCCGGCC\n>seq3\nATATATATATAT",
        "category": "fasta_analysis",
        "expected_source": None,
        "expected_keywords": ["序列", "数量", "长度", "GC"],
        "expected_tool": "parse_fasta_stats",
        "difficulty": "简单",
    },
    {
        "id": 12,
        "question": "统计一下这些序列的基本信息：\n>gene1\nAAAAATTTTTGGGGGCCCCC\n>gene2\nATGCATGCATGCATGC",
        "category": "fasta_analysis",
        "expected_source": None,
        "expected_keywords": ["序列", "数量", "长度", "GC"],
        "expected_tool": "parse_fasta_stats",
        "difficulty": "简单",
    },
    {
        "id": 13,
        "question": "帮我看看这个FASTA文件有多少条序列，GC含量是多少：\n>s1\nATCG\n>s2\nGCTA\n>s3\nTTAA\n>s4\nCCGG\n>s5\nAGCT",
        "category": "fasta_analysis",
        "expected_source": None,
        "expected_keywords": ["序列", "数量", "长度", "GC"],
        "expected_tool": "parse_fasta_stats",
        "difficulty": "简单",
    },

    # ═══════════════════════════════════════════════════════════
    # cpg_scan: 3 条（阳性 2 + 阴性 1）
    # ═══════════════════════════════════════════════════════════

    {
        "id": 14,
        "question": "这段序列有没有CpG岛？CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG",
        "category": "cpg_scan",
        "expected_source": None,
        "expected_keywords": ["CpG", "岛", "GC", "扫描"],
        "expected_tool": "scan_cpg_islands",
        "difficulty": "简单",
    },
    {
        "id": 15,
        "question": "帮我扫描一下这段DNA序列的CpG岛：CGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGAT",
        "category": "cpg_scan",
        "expected_source": None,
        "expected_keywords": ["CpG", "岛", "GC", "扫描"],
        "expected_tool": "scan_cpg_islands",
        "difficulty": "中等",
    },
    {
        "id": 16,
        "question": "这段序列有没有CpG岛？ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT",
        "category": "cpg_scan",
        "expected_source": None,
        "expected_keywords": ["CpG", "岛", "没有", "未检测"],
        "expected_tool": "scan_cpg_islands",
        "difficulty": "简单",
    },

    # ═══════════════════════════════════════════════════════════
    # pipeline_suggest: 4 条
    # ═══════════════════════════════════════════════════════════

    {
        "id": 17,
        "question": "研究 mTERF 基因家族进化需要什么分析流程？",
        "category": "pipeline_suggest",
        "expected_source": None,
        "expected_keywords": ["流程", "mTERF", "进化", "OrthoFinder", "MAFFT", "PAML"],
        "expected_tool": "suggest_pipeline",
        "difficulty": "中等",
    },
    {
        "id": 18,
        "question": "怎么做基因家族的正选择压力分析？",
        "category": "pipeline_suggest",
        "expected_source": None,
        "expected_keywords": ["流程", "正选择", "压力", "PAML", "codeml", "分支位点"],
        "expected_tool": "suggest_pipeline",
        "difficulty": "困难",
    },
    {
        "id": 19,
        "question": "鉴定一个新的基因家族需要哪些步骤？",
        "category": "pipeline_suggest",
        "expected_source": None,
        "expected_keywords": ["流程", "基因家族", "鉴定", "BLAST", "HMMER", "OrthoFinder"],
        "expected_tool": "suggest_pipeline",
        "difficulty": "中等",
    },
    {
        "id": 20,
        "question": "植物 DNA 甲基化分析的一般流程是什么？",
        "category": "pipeline_suggest",
        "expected_source": None,
        "expected_keywords": ["流程", "DNA", "甲基化", "BS-seq", "CpG", "差异甲基化"],
        "expected_tool": "suggest_pipeline",
        "difficulty": "中等",
    },
]


def get_dataset_summary():
    """返回测试集统计信息。"""
    from collections import Counter
    categories = Counter(item["category"] for item in EVAL_DATASET)
    difficulties = Counter(item["difficulty"] for item in EVAL_DATASET)
    return {
        "total": len(EVAL_DATASET),
        "by_category": dict(categories),
        "by_difficulty": dict(difficulties),
    }


if __name__ == "__main__":
    summary = get_dataset_summary()
    print("=" * 60)
    print("PlantGenome Agent 评估测试集")
    print("=" * 60)
    print(f"总测试数: {summary['total']}")
    print(f"\n按类别分布:")
    for cat, count in summary["by_category"].items():
        print(f"  {cat}: {count} 条")
    print(f"\n按难度分布:")
    for diff, count in summary["by_difficulty"].items():
        print(f"  {diff}: {count} 条")
    print(f"\n测试用例列表:")
    for item in EVAL_DATASET:
        print(f"  [{item['id']:2d}] ({item['category']:20s}) {item['question'][:50]}...")
