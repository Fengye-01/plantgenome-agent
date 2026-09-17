"""
工具 5：search_pubmed（PubMed 文献搜索工具）

通过 NCBI E-utilities API 搜索 PubMed 文献，返回文献元数据列表。
这是 Agent 的工具之一，用于回答"最近有什么关于 XX 的研究"、"找一下 XX 相关的文献"等问题。

与前端的 PubMed 下载入库功能的区别：
- 本工具是同步的，只返回文献元数据（标题/作者/摘要/PMID），不下载 PDF
- 前端面板是异步的，会下载 PDF 并自动入库 RAG
- Agent 调用本工具后，可以在回答中建议用户"在左侧 PubMed 面板输入关键字下载全文入库"

设计要点：
- 同步调用，控制在 5 秒内返回（搜索 + 元数据获取）
- 最多返回 5 篇，避免回答过长
- 返回结构化数据，便于 answer_node 格式化
"""
from __future__ import annotations

from typing import Dict, List, Optional

from app.tools.pubmed_fetcher import search_pubmed, fetch_article_details


def search_pubmed_tool(
    keyword: str,
    max_results: int = 5,
) -> Dict:
    """
    搜索 PubMed 文献，返回文献元数据列表。

    Args:
        keyword: 搜索关键字（支持布尔语法，如 "cancer AND drug"）
        max_results: 返回最大结果数，默认 5，上限 10

    Returns:
        dict，包含：
        - articles: list[dict]，每篇文献的元数据（pmid, title, authors, journal, year, abstract, doi）
        - count: int，返回的文献数量
        - keyword: str，搜索关键字
        - has_result: bool，是否有结果

    失败情况：
        - 网络请求失败 → 返回 {"error": "..."}
        - 搜索结果为空 → 返回 has_result=False
    """
    # 限制 max_results 上限
    max_results = min(max_results, 10)

    try:
        # 步骤 1：搜索 PMID
        pmids = search_pubmed(keyword, max_results=max_results)
    except Exception as e:
        return {"error": f"PubMed 搜索失败: {str(e)}", "articles": [], "count": 0, "has_result": False}

    if not pmids:
        return {
            "articles": [],
            "count": 0,
            "keyword": keyword,
            "has_result": False,
            "message": f"未搜索到与 '{keyword}' 相关的文献",
        }

    try:
        # 步骤 2：获取文献详情
        articles = fetch_article_details(pmids)
    except Exception as e:
        return {"error": f"获取文献详情失败: {str(e)}", "articles": [], "count": 0, "has_result": False}

    # 格式化返回（截断摘要，避免过长）
    formatted_articles = []
    for art in articles:
        abstract = art.get("abstract", "")
        if len(abstract) > 500:
            abstract = abstract[:500] + "..."

        formatted_articles.append({
            "pmid": art.get("pmid", ""),
            "title": art.get("title", ""),
            "authors": art.get("authors", ""),
            "journal": art.get("journal", ""),
            "year": art.get("year", ""),
            "abstract": abstract,
            "doi": art.get("doi"),
            "pmc_id": art.get("pmc_id"),
            "has_full_text": art.get("pmc_id") is not None,
            "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{art.get('pmid', '')}/",
        })

    return {
        "articles": formatted_articles,
        "count": len(formatted_articles),
        "keyword": keyword,
        "has_result": True,
    }


# ═══════════════════════════════════════════════════════════════
# 测试入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    print("=" * 70)
    print("工具测试：search_pubmed")
    print("=" * 70)

    test_keywords = [
        "plant CpG island DNA methylation",
        "mTERF gene family plant evolution",
    ]

    for keyword in test_keywords:
        print(f"\n{'─' * 70}")
        print(f"搜索: {keyword}")
        print(f"{'─' * 70}")

        result = search_pubmed_tool(keyword, max_results=3)

        if "error" in result:
            print(f"❌ 错误: {result['error']}")
            continue

        print(f"找到 {result['count']} 篇文献:")
        for i, art in enumerate(result["articles"], 1):
            print(f"\n  [{i}] PMID: {art['pmid']}")
            print(f"      标题: {art['title'][:80]}")
            print(f"      作者: {art['authors'][:60]}")
            print(f"      期刊: {art['journal']} ({art['year']})")
            print(f"      有全文: {'是' if art['has_full_text'] else '否'}")
            print(f"      摘要: {art['abstract'][:100]}...")

    print("\n" + "=" * 70)
    print("所有测试完成！")
    print("=" * 70)
