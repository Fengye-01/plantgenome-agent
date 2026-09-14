"""
RAG 端到端测试（D7 任务 2）

测试"检索 + 生成"完整链路，验证：
1. 检索到的 sources 是否包含期望 PDF
2. 生成的 answer 是否包含期望关键词
3. 记录 HitRate（5 个中命中几个）

运行：
    cd C:/Users/YeFeng/Desktop/自救计划/plantgenome-agent
    $env:HF_ENDPOINT = "https://hf-mirror.com"
    python tests/test_rag_e2e.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 项目根路径导入
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.rag_generator import RAGGenerator


# ═══════════════════════════════════════════════════════════════
# 测试用例定义
# ═══════════════════════════════════════════════════════════════

TEST_CASES = [
    {
        "id": 1,
        "question": "PAML 中的 omega 值是什么意思？",
        "expected_pdf_keywords": ["PAML", "paml"],
        "expected_answer_keywords": ["omega", "dN", "dS", "选择", "ω"],
        "difficulty": "简单",
    },
    {
        "id": 2,
        "question": "OrthoFinder 的输入格式是什么？",
        "expected_pdf_keywords": ["OrthoFinder", "orthofinder"],
        "expected_answer_keywords": ["FASTA", "fasta", "蛋白", "序列", "输入"],
        "difficulty": "简单",
    },
    {
        "id": 3,
        "question": "MAFFT 有哪些比对算法？",
        "expected_pdf_keywords": ["MAFFT", "mafft"],
        "expected_answer_keywords": ["L-INS-i", "FFT-NS", "算法", "比对"],
        "difficulty": "中等",
    },
    {
        "id": 4,
        "question": "mTERF 基因家族的功能是什么？",
        "expected_pdf_keywords": ["mTERF", "mterf", "MTERF"],
        "expected_answer_keywords": ["线粒体", "转录", "终止", "功能", "叶绿体"],
        "difficulty": "中等",
    },
    {
        "id": 5,
        "question": "什么是 CpG 岛？如何判定？",
        "expected_pdf_keywords": ["CpG", "cpg", "CG"],
        "expected_answer_keywords": ["GC", "Obs/Exp", "判定", "甲基化", "含量"],
        "difficulty": "困难",
    },
]


# ═══════════════════════════════════════════════════════════════
# 测试逻辑
# ═══════════════════════════════════════════════════════════════

def check_pdf_hit(sources: list, expected_keywords: list) -> tuple[bool, str]:
    """
    检查检索到的 sources 是否包含期望的 PDF。

    Args:
        sources: RAGGenerator.generate() 返回的 sources 列表
        expected_keywords: 期望的 PDF 关键词列表

    Returns:
        (是否命中, 命中的来源描述)
    """
    if not sources:
        return False, "无 sources"

    for source in sources:
        source_str = str(source)
        for keyword in expected_keywords:
            if keyword.lower() in source_str.lower():
                return True, source_str[:100]

    return False, f"未找到包含 {expected_keywords} 的来源"


def check_answer_hit(answer: str, expected_keywords: list) -> tuple[bool, list]:
    """
    检查生成的 answer 是否包含期望关键词。

    Args:
        answer: LLM 生成的回答
        expected_keywords: 期望的关键词列表

    Returns:
        (是否命中至少1个关键词, 命中的关键词列表)
    """
    if not answer:
        return False, []

    hit_keywords = []
    for keyword in expected_keywords:
        if keyword.lower() in answer.lower():
            hit_keywords.append(keyword)

    return len(hit_keywords) > 0, hit_keywords


def run_tests():
    """运行所有测试用例，输出测试报告。"""
    print("=" * 80)
    print("RAG 端到端测试（检索 + 生成）")
    print("=" * 80)

    # 初始化 RAGGenerator
    print("\n[初始化] 正在加载 RAGGenerator（BGE-m3 + Chroma + LLM）...")
    rag = RAGGenerator()
    print("[初始化] 完成\n")

    results = []
    pdf_hit_count = 0
    answer_hit_count = 0
    both_hit_count = 0

    for i, case in enumerate(TEST_CASES, 1):
        print(f"--- 测试 {i}/{len(TEST_CASES)} ---")
        print(f"问题: {case['question']}")
        print(f"难度: {case['difficulty']}")

        try:
            # 调用 RAG 生成
            result = rag.generate(case["question"])
            answer = result["answer"]
            sources = result["sources"]
            has_relevant = result["has_relevant"]

            # 检查 PDF 命中
            pdf_hit, pdf_hit_source = check_pdf_hit(sources, case["expected_pdf_keywords"])

            # 检查 answer 关键词命中
            answer_hit, hit_keywords = check_answer_hit(answer, case["expected_answer_keywords"])

            # 两个都命中才算通过
            both_hit = pdf_hit and answer_hit

            if pdf_hit:
                pdf_hit_count += 1
            if answer_hit:
                answer_hit_count += 1
            if both_hit:
                both_hit_count += 1

            # 输出结果
            print(f"  PDF 命中: {'✅' if pdf_hit else '❌'} ({pdf_hit_source})")
            print(f"  Answer 关键词命中: {'✅' if answer_hit else '❌'} (命中: {hit_keywords})")
            print(f"  has_relevant: {has_relevant}")
            print(f"  sources 数量: {len(sources)}")
            print(f"  answer 长度: {len(answer)} 字符")
            print(f"  综合结果: {'✅ 通过' if both_hit else '❌ 未通过'}")

            # 保存结果
            results.append({
                "id": case["id"],
                "question": case["question"],
                "difficulty": case["difficulty"],
                "pdf_hit": pdf_hit,
                "answer_hit": answer_hit,
                "both_hit": both_hit,
                "hit_keywords": hit_keywords,
                "sources_count": len(sources),
                "answer_length": len(answer),
                "has_relevant": has_relevant,
                "answer": answer,
                "sources": sources,
            })

        except Exception as e:
            print(f"  ❌ 测试失败: {str(e)}")
            results.append({
                "id": case["id"],
                "question": case["question"],
                "difficulty": case["difficulty"],
                "pdf_hit": False,
                "answer_hit": False,
                "both_hit": False,
                "error": str(e),
            })

        print()

    # ═══════════════════════════════════════════════════════════
    # 输出汇总报告
    # ═══════════════════════════════════════════════════════════

    total = len(TEST_CASES)
    print("=" * 80)
    print("测试汇总报告")
    print("=" * 80)
    print(f"总测试用例数: {total}")
    print(f"PDF 检索命中: {pdf_hit_count}/{total} ({pdf_hit_count/total*100:.1f}%)")
    print(f"Answer 关键词命中: {answer_hit_count}/{total} ({answer_hit_count/total*100:.1f}%)")
    print(f"综合通过（两者都命中）: {both_hit_count}/{total} ({both_hit_count/total*100:.1f}%)")
    print()

    # 按难度分组
    print("按难度分组:")
    for difficulty in ["简单", "中等", "困难"]:
        diff_results = [r for r in results if r.get("difficulty") == difficulty]
        if diff_results:
            diff_pass = sum(1 for r in diff_results if r.get("both_hit"))
            print(f"  {difficulty}: {diff_pass}/{len(diff_results)} 通过")

    print()

    # 未通过的用例
    failed = [r for r in results if not r.get("both_hit")]
    if failed:
        print("未通过用例分析:")
        for r in failed:
            print(f"  - [{r['id']}] {r['question']}")
            if not r.get("pdf_hit"):
                print(f"    PDF 未命中: 期望 {TEST_CASES[r['id']-1]['expected_pdf_keywords']}")
            if not r.get("answer_hit"):
                print(f"    Answer 未命中: 期望 {TEST_CASES[r['id']-1]['expected_answer_keywords']}")
            if r.get("error"):
                print(f"    错误: {r['error']}")
    else:
        print("所有用例全部通过！🎉")

    print()
    print("=" * 80)

    return results


if __name__ == "__main__":
    results = run_tests()
