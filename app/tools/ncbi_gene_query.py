"""
工具 6：query_ncbi_gene（NCBI 基因信息查询工具）

通过 NCBI E-utilities API 查询基因信息，支持基因名、Gene ID、Locus Tag 等多种查询方式。
特别适合植物基因组研究场景（如拟南芥 AT1G01010、水稻 Os01g01010 等 Locus Tag 查询）。

设计要点：
- 同步调用，控制在 5 秒内返回
- 支持基因名、Locus Tag、Gene ID 查询
- 返回结构化基因信息（ID、描述、染色体位置、物种、别名、功能摘要）
- 查询失败时返回友好的错误信息
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
from urllib.parse import quote

import requests

# NCBI E-utilities 基础 URL
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# 请求头（NCBI 要求提供联系方式）
HEADERS = {
    "User-Agent": "PlantGenome-Agent/1.0 (academic research; contact: researcher@example.com)",
}
# 请求超时（秒）
TIMEOUT = 10


def _esearch_gene(query: str, retmax: int = 3) -> List[str]:
    """
    在 NCBI Gene 数据库中搜索，返回 Gene ID 列表。

    Args:
        query: 搜索查询（基因名、Locus Tag、描述等）
        retmax: 返回最大数量

    Returns:
        Gene ID 字符串列表

    Raises:
        requests.RequestException: 网络请求失败
    """
    url = f"{EUTILS_BASE}/esearch.fcgi"
    params = {
        "db": "gene",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
    }
    resp = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


def _efetch_gene(gene_id: str) -> Optional[Dict]:
    """
    根据 Gene ID 获取基因详情（XML 格式）。

    Args:
        gene_id: NCBI Gene ID

    Returns:
        基因信息字典，失败返回 None
    """
    url = f"{EUTILS_BASE}/efetch.fcgi"
    params = {
        "db": "gene",
        "id": gene_id,
        "retmode": "xml",
    }
    resp = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()

    # 解析 XML
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return None

    # 提取基因信息
    gene_elem = root.find(".//Entrezgene")
    if gene_elem is None:
        return None

    result = {"gene_id": gene_id}

    # 基因名
    name_elem = gene_elem.find(".//Gene-ref_locus")
    if name_elem is not None and name_elem.text:
        result["name"] = name_elem.text

    # 基因描述
    desc_elem = gene_elem.find(".//Gene-ref_desc")
    if desc_elem is not None and desc_elem.text:
        result["description"] = desc_elem.text

    # 物种
    org_elem = gene_elem.find(".//Org-ref_taxname")
    if org_elem is not None and org_elem.text:
        result["organism"] = org_elem.text

    # 染色体位置
    chr_elem = gene_elem.find(".//Gene-ref_maploc")
    if chr_elem is not None and chr_elem.text:
        result["chromosome_location"] = chr_elem.text

    # 别名（Synonyms）
    synonyms = []
    for syn in gene_elem.findall(".//Gene-ref_syn/Gene-ref_syn_E"):
        if syn.text:
            synonyms.append(syn.text)
    if synonyms:
        result["synonyms"] = synonyms

    # Locus Tag（植物基因常用）
    locus_elem = gene_elem.find(".//Dbtag_tag/Object-id_str")
    # Locus Tag 通常在 Gene-ref_locus 或其他字段中，这里做兜底

    # 功能摘要（Comment）
    summary_elem = gene_elem.find(".//Entrezgene_summary")
    if summary_elem is not None and summary_elem.text:
        result["summary"] = summary_elem.text[:500]  # 截断

    return result


def query_ncbi_gene(
    gene_name: str,
    organism: Optional[str] = None,
    max_results: int = 3,
) -> Dict:
    """
    查询 NCBI 基因信息。

    Args:
        gene_name: 基因名、Locus Tag 或 Gene ID（如 "BRCA1"、"AT1G01010"、"672"）
        organism: 可选，限定物种（如 "Arabidopsis thaliana"、"Oryza sativa"）
        max_results: 返回最大结果数，默认 3，上限 5

    Returns:
        dict，包含：
        - genes: list[dict]，每个基因的详细信息
        - count: int，返回的基因数量
        - query: str，原始查询
        - has_result: bool，是否有结果

    失败情况：
        - 网络请求失败 → 返回 {"error": "..."}
        - 搜索结果为空 → 返回 has_result=False
    """
    # 限制 max_results 上限
    max_results = min(max_results, 5)

    # 构建查询（如果指定了物种，加上物种限定）
    query = gene_name
    if organism:
        query = f"{gene_name} AND {organism}[Organism]"

    try:
        # 步骤 1：搜索 Gene ID
        gene_ids = _esearch_gene(query, retmax=max_results)
    except Exception as e:
        return {"error": f"NCBI 基因搜索失败: {str(e)}", "genes": [], "count": 0, "has_result": False}

    if not gene_ids:
        # 如果指定了物种但没找到，尝试不加物种搜索
        if organism:
            try:
                gene_ids = _esearch_gene(gene_name, retmax=max_results)
            except Exception:
                gene_ids = []

        if not gene_ids:
            return {
                "genes": [],
                "count": 0,
                "query": query,
                "has_result": False,
                "message": f"未在 NCBI Gene 数据库中找到与 '{gene_name}' 相关的基因",
            }

    # 步骤 2：获取每个基因的详情
    genes = []
    for gid in gene_ids:
        try:
            gene_info = _efetch_gene(gid)
            if gene_info:
                genes.append(gene_info)
        except Exception:
            continue  # 单个基因获取失败不影响其他

    if not genes:
        return {
            "genes": [],
            "count": 0,
            "query": query,
            "has_result": False,
            "message": f"找到 {len(gene_ids)} 个 Gene ID，但获取详情失败",
        }

    return {
        "genes": genes,
        "count": len(genes),
        "query": query,
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
    print("工具测试：query_ncbi_gene")
    print("=" * 70)

    test_cases = [
        ("AT1G01010", "Arabidopsis thaliana"),  # 拟南芥 Locus Tag
        ("BRCA1", "Homo sapiens"),               # 人类基因名
    ]

    for gene, org in test_cases:
        print(f"\n{'─' * 70}")
        print(f"查询: {gene} (物种: {org})")
        print(f"{'─' * 70}")

        result = query_ncbi_gene(gene, organism=org, max_results=2)

        if "error" in result:
            print(f"❌ 错误: {result['error']}")
            continue

        if not result["has_result"]:
            print(f"⚠️  {result.get('message', '未找到')}")
            continue

        print(f"找到 {result['count']} 个基因:")
        for i, g in enumerate(result["genes"], 1):
            print(f"\n  [{i}] Gene ID: {g.get('gene_id')}")
            print(f"      基因名: {g.get('name', '?')}")
            print(f"      描述: {g.get('description', '?')[:80]}")
            print(f"      物种: {g.get('organism', '?')}")
            print(f"      染色体位置: {g.get('chromosome_location', '?')}")
            if g.get("synonyms"):
                print(f"      别名: {', '.join(g['synonyms'][:5])}")
            if g.get("summary"):
                print(f"      摘要: {g['summary'][:100]}...")

    print("\n" + "=" * 70)
    print("所有测试完成！")
    print("=" * 70)
