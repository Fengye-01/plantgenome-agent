"""
PubMed 文献检索与下载工具。

功能：
1. 通过 NCBI E-utilities API 按关键字搜索 PubMed 文献
2. 获取文献元数据（标题、作者、摘要、期刊、PMID、PMC ID）
3. 尝试从 PMC（PubMed Central）下载开放获取全文 PDF
4. 无全文时用摘要生成 PDF（fpdf2），确保每篇都能入库 RAG
5. 保存到 data/pdfs/ 目录，返回文件路径和元信息

设计要点：
- 遵守 NCBI 速率限制（无 API Key 3 次/秒，加 sleep 控制）
- 每篇文献独立处理，单篇失败不影响整体
- 全文优先，摘要兜底，保证检索结果可入库
- 元数据写入文件名，便于溯源（PMID_第一作者_年份_标题.pdf）
"""
from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import requests

# NCBI E-utilities 基础 URL
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# Europe PMC API（用于获取全文 PDF 直链，比直接访问 PMC 更稳定）
EUROPE_PMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# 请求头（NCBI 要求设置 User-Agent，否则可能限流）
DEFAULT_HEADERS = {
    "User-Agent": "PlantGenome-Agent/1.0 (research; contact: plantgenome@example.com)",
    "Accept": "application/json",
}

# 下载 PDF 时用的浏览器风格请求头（避免 403）
PDF_DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/pdf,application/octet-stream,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

# 文献保存目录
PDF_DIR = Path("data/pdfs")
PDF_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# 1. PubMed 搜索
# ═══════════════════════════════════════════════════════════════

def search_pubmed(keyword: str, max_results: int = 5) -> List[str]:
    """
    在 PubMed 中搜索关键字，返回 PMID 列表。

    使用 NCBI E-utilities esearch 接口。

    Args:
        keyword: 搜索关键字（支持布尔语法，如 "cancer AND drug"）
        max_results: 返回最大结果数，默认 5

    Returns:
        List[str]: PMID 列表

    Raises:
        requests.RequestException: 网络请求失败
    """
    url = f"{EUTILS_BASE}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": keyword,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",  # 按相关度排序
    }

    resp = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    id_list = data.get("esearchresult", {}).get("idlist", [])
    return id_list


# ═══════════════════════════════════════════════════════════════
# 2. 获取文献详情（XML 解析）
# ═══════════════════════════════════════════════════════════════

def fetch_article_details(pmids: List[str]) -> List[Dict]:
    """
    批量获取文献详情，解析 XML 提取元数据。

    使用 NCBI E-utilities efetch 接口，rettype=xml。

    Args:
        pmids: PMID 列表

    Returns:
        List[Dict]: 每篇文献的元数据，包含：
            - pmid: str
            - title: str
            - authors: str（逗号分隔的作者列表）
            - first_author: str（第一作者姓氏）
            - journal: str
            - year: str
            - abstract: str
            - pmc_id: str or None（PMC ID，用于全文下载）
            - doi: str or None
    """
    if not pmids:
        return []

    url = f"{EUTILS_BASE}/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "xml",
        "retmode": "xml",
    }

    resp = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=60)
    resp.raise_for_status()

    # 解析 XML
    root = ET.fromstring(resp.content)
    articles = []

    for article_elem in root.findall(".//PubmedArticle"):
        article = _parse_article_xml(article_elem)
        if article:
            articles.append(article)

    return articles


def _parse_article_xml(article_elem: ET.Element) -> Optional[Dict]:
    """
    解析单篇 PubmedArticle XML 元素。

    Args:
        article_elem: <PubmedArticle> XML 元素

    Returns:
        Dict 或 None（解析失败时）
    """
    try:
        # PMID
        pmid_elem = article_elem.find(".//PMID")
        pmid = pmid_elem.text if pmid_elem is not None else ""

        # 标题
        title_elem = article_elem.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None else "Untitled"
        # 标题可能包含子元素（如斜体），拼接所有文本
        if title_elem is not None and title_elem.text is None:
            title = "".join(title_elem.itertext())
        title = title.strip() if title else "Untitled"

        # 作者
        authors = []
        first_author = ""
        for author_elem in article_elem.findall(".//Author"):
            last_name = author_elem.find("LastName")
            fore_name = author_elem.find("ForeName")
            collective = author_elem.find("CollectiveName")

            if last_name is not None:
                name = last_name.text or ""
                if fore_name is not None and fore_name.text:
                    name = f"{name} {fore_name.text[0]}."
                authors.append(name)
                if not first_author:
                    first_author = last_name.text or ""
            elif collective is not None and collective.text:
                authors.append(collective.text)
                if not first_author:
                    first_author = collective.text.split()[0] if collective.text else ""

        authors_str = ", ".join(authors) if authors else "Unknown"

        # 期刊
        journal_elem = article_elem.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None else "Unknown Journal"

        # 年份
        year_elem = article_elem.find(".//PubDate/Year")
        if year_elem is None:
            year_elem = article_elem.find(".//PubDate/MedlineDate")
        year = year_elem.text[:4] if year_elem is not None and year_elem.text else "Unknown"

        # 摘要（可能有多个 AbstractText，拼接）
        abstract_parts = []
        for abs_elem in article_elem.findall(".//AbstractText"):
            label = abs_elem.get("Label", "")
            text = "".join(abs_elem.itertext()).strip()
            if text:
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
        abstract = "\n\n".join(abstract_parts) if abstract_parts else "（无摘要）"

        # PMC ID（在 ArticleIdList 中找 IdType="pmc"）
        pmc_id = None
        for id_elem in article_elem.findall(".//ArticleId"):
            if id_elem.get("IdType") == "pmc" and id_elem.text:
                pmc_id = id_elem.text
                break

        # DOI
        doi = None
        for id_elem in article_elem.findall(".//ArticleId"):
            if id_elem.get("IdType") == "doi" and id_elem.text:
                doi = id_elem.text
                break

        return {
            "pmid": pmid,
            "title": title,
            "authors": authors_str,
            "first_author": first_author,
            "journal": journal,
            "year": year,
            "abstract": abstract,
            "pmc_id": pmc_id,
            "doi": doi,
        }

    except Exception as e:
        print(f"  ⚠️  解析文献 XML 失败: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# 3. 下载全文 PDF
# ═══════════════════════════════════════════════════════════════

def _get_pdf_url_from_europepmc(pmid: str) -> Optional[str]:
    """
    通过 Europe PMC API 获取文献的全文 PDF 直链。

    Europe PMC 是 EMBL-EBI 维护的开放文献库，提供稳定的 API，
    比直接访问 NCBI PMC 更不容易被反爬拦截。

    Args:
        pmid: PubMed ID

    Returns:
        PDF 直链 URL，找不到返回 None
    """
    try:
        params = {
            "query": f"EXT_ID:{pmid}",
            "resultType": "core",
            "format": "json",
            "pageSize": 1,
        }
        resp = requests.get(
            EUROPE_PMC_SEARCH,
            params=params,
            headers=DEFAULT_HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        results = data.get("resultList", {}).get("result", [])
        if not results:
            return None

        # 查找 PDF 链接
        full_text_urls = results[0].get("fullTextUrlList", {}).get("fullTextUrl", [])
        for url_info in full_text_urls:
            if url_info.get("documentStyle") == "pdf":
                return url_info.get("url")

        return None

    except Exception as e:
        print(f"  ⚠️  Europe PMC 查询失败: {e}")
        return None


def download_full_text_pdf(pmid: str, pmc_id: Optional[str], save_path: Path) -> bool:
    """
    下载文献全文 PDF。

    策略：
    1. 优先通过 Europe PMC API 获取 PDF 直链（最稳定）
    2. 下载时使用浏览器风格请求头，避免 403

    Args:
        pmid: PubMed ID
        pmc_id: PMC ID（备用，当前未直接使用）
        save_path: 保存路径

    Returns:
        bool: 是否下载成功
    """
    # 步骤 1：从 Europe PMC 获取 PDF 直链
    pdf_url = _get_pdf_url_from_europepmc(pmid)
    if not pdf_url:
        print(f"  ⚠️  Europe PMC 未找到 PDF 链接")
        return False

    print(f"  PDF 链接: {pdf_url}")

    # 步骤 2：下载 PDF
    try:
        resp = requests.get(
            pdf_url,
            headers=PDF_DOWNLOAD_HEADERS,
            timeout=60,
            allow_redirects=True,
            stream=True,
        )
        resp.raise_for_status()

        # 验证是否为 PDF
        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type.lower() and "octet-stream" not in content_type.lower():
            print(f"  ⚠️  返回非 PDF 内容: {content_type}")
            return False

        # 写入文件
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        # 验证文件大小
        if save_path.stat().st_size < 1024:
            print(f"  ⚠️  文件过小 ({save_path.stat().st_size} bytes)")
            save_path.unlink(missing_ok=True)
            return False

        return True

    except requests.RequestException as e:
        print(f"  ⚠️  PDF 下载失败: {e}")
        return False
    except Exception as e:
        print(f"  ⚠️  PDF 保存失败: {e}")
        return False


# ═══════════════════════════════════════════════════════════════
# 4. 摘要生成 PDF（兜底方案）
# ═══════════════════════════════════════════════════════════════

def generate_abstract_pdf(article: Dict, save_path: Path) -> bool:
    """
    用文献元数据和摘要生成 PDF 文件（兜底方案）。

    当无法从 PMC 下载全文时，把标题、作者、期刊、摘要、PMID 链接
    生成一个结构化的 PDF，确保文献仍可被 RAG 系统索引。

    使用 fpdf2 库，轻量且不需要系统依赖。

    Args:
        article: 文献元数据 dict
        save_path: 保存路径

    Returns:
        bool: 是否生成成功
    """
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # 标题
        pdf.set_font("Helvetica", "B", 14)
        pdf.multi_cell(0, 7, article["title"])
        pdf.ln(3)

        # 作者
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(60, 60, 60)
        pdf.multi_cell(0, 5, article["authors"])
        pdf.ln(1)

        # 期刊 + 年份
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(80, 80, 80)
        journal_line = f"{article['journal']}, {article['year']}"
        if article.get("doi"):
            journal_line += f"  DOI: {article['doi']}"
        pdf.multi_cell(0, 5, journal_line)
        pdf.ln(1)

        # PMID 链接
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(0, 0, 200)
        pmid_url = f"https://pubmed.ncbi.nlm.nih.gov/{article['pmid']}/"
        pdf.cell(0, 5, f"PMID: {article['pmid']}  ({pmid_url})")
        pdf.ln(8)

        # 分隔线
        pdf.set_draw_color(180, 180, 180)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        # 摘要标题
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 7, "Abstract")
        pdf.ln(8)

        # 摘要正文
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(30, 30, 30)
        abstract = article.get("abstract", "（无摘要）")
        # fpdf2 对 Unicode 支持有限，替换特殊字符
        abstract = _sanitize_for_fpdf(abstract)
        pdf.multi_cell(0, 5, abstract)

        # 页脚标注
        pdf.ln(10)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(120, 120, 120)
        pdf.multi_cell(
            0, 4,
            "This is an abstract-only PDF generated from PubMed metadata. "
            "Full text was not available as open access. "
            "Please visit the PMID link above for the complete article."
        )

        pdf.output(str(save_path))
        return True

    except ImportError:
        print("  ⚠️  fpdf2 未安装，无法生成摘要 PDF。请运行: pip install fpdf2")
        return False
    except Exception as e:
        print(f"  ⚠️  摘要 PDF 生成失败: {e}")
        return False


def _sanitize_for_fpdf(text: str) -> str:
    """
    清理文本，移除 fpdf2 内置 Helvetica 字体不支持的 Unicode 字符。

    fpdf2 默认字体（Helvetica）是 Latin-1 编码，不支持中文和特殊符号。
    这里做基础替换，确保 PDF 能正常生成。

    Args:
        text: 原始文本

    Returns:
        清理后的文本
    """
    # 常见特殊字符替换
    replacements = {
        "\u2013": "-",   # en dash
        "\u2014": "--",  # em dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2026": "...", # ellipsis
        "\u00a0": " ",   # non-breaking space
        "\u00b1": "+/-", # plus-minus
        "\u00d7": "x",   # multiplication sign
        "\u2264": "<=",  # less-than-or-equal
        "\u2265": ">=",  # greater-than-or-equal
        "\u03b1": "alpha",
        "\u03b2": "beta",
        "\u03b3": "gamma",
        "\u03b4": "delta",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # 移除其他非 Latin-1 字符（保留换行和基本 ASCII）
    result = []
    for ch in text:
        if ord(ch) < 256 or ch in "\n\t":
            result.append(ch)
        else:
            result.append("?")
    return "".join(result)


# ═══════════════════════════════════════════════════════════════
# 5. 文件名安全化
# ═══════════════════════════════════════════════════════════════

def _safe_filename(name: str, max_length: int = 80) -> str:
    """
    生成安全的文件名（移除非法字符，限制长度）。

    Args:
        name: 原始文件名
        max_length: 最大长度

    Returns:
        安全的文件名
    """
    # 移除 Windows 文件名非法字符
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    # 移除连续空格
    name = re.sub(r"\s+", " ", name).strip()
    # 限制长度
    if len(name) > max_length:
        name = name[:max_length].rstrip()
    return name


# ═══════════════════════════════════════════════════════════════
# 6. 主入口：搜索 + 下载 + 保存
# ═══════════════════════════════════════════════════════════════

def search_and_download(
    keyword: str,
    max_results: int = 5,
    user_id: Optional[int] = None,
    delay: float = 0.4,
) -> List[Dict]:
    """
    搜索 PubMed 并下载文献，返回保存后的文件信息列表。

    完整流程：
    1. esearch 搜索 PMID 列表
    2. efetch 批量获取文献详情
    3. 逐篇处理：
       a. 有 PMC ID → 尝试下载全文 PDF
       b. 下载失败或无 PMC ID → 用摘要生成 PDF
    4. 保存到 data/pdfs/，文件名含 PMID 和标题
    5. 返回每篇的文件路径和元信息

    Args:
        keyword: 搜索关键字
        max_results: 最大下载数量，默认 5
        user_id: 用户 ID（用于文件名前缀，实现用户隔离）
        delay: 请求间隔（秒），遵守 NCBI 速率限制，默认 0.4s

    Returns:
        List[Dict]: 每篇文献的处理结果，包含：
            - pmid, title, authors, journal, year
            - file_path: 保存的文件路径
            - file_name: 文件名
            - source: "full_text"（PMC 全文）或 "abstract"（摘要生成）
            - success: bool
            - error: str（失败时的错误信息）
    """
    results = []

    print(f"[PubMed] 搜索关键字: {keyword!r}, 最多 {max_results} 篇")

    # 步骤 1：搜索 PMID
    try:
        pmids = search_pubmed(keyword, max_results)
    except Exception as e:
        print(f"[PubMed] ❌ 搜索失败: {e}")
        return [{"success": False, "error": f"PubMed 搜索失败: {e}"}]

    if not pmids:
        print(f"[PubMed] ⚠️  未搜索到相关文献")
        return [{"success": False, "error": "未搜索到相关文献"}]

    print(f"[PubMed] 找到 {len(pmids)} 篇文献: {pmids}")
    time.sleep(delay)

    # 步骤 2：获取文献详情
    try:
        articles = fetch_article_details(pmids)
    except Exception as e:
        print(f"[PubMed] ❌ 获取文献详情失败: {e}")
        return [{"success": False, "error": f"获取文献详情失败: {e}"}]

    print(f"[PubMed] 成功解析 {len(articles)} 篇文献详情")

    # 步骤 3：逐篇下载/生成
    user_prefix = f"user{user_id}_" if user_id is not None else ""

    for i, article in enumerate(articles, 1):
        print(f"\n[PubMed] 处理第 {i}/{len(articles)} 篇: PMID={article['pmid']}")
        print(f"  标题: {article['title'][:80]}...")

        # 生成文件名：PMID_第一作者_年份_标题.pdf
        short_title = _safe_filename(article["title"], max_length=50)
        file_name = f"{user_prefix}PMID{article['pmid']}_{article['first_author']}_{article['year']}_{short_title}.pdf"
        file_path = PDF_DIR / file_name

        source = "abstract"
        success = False
        error = None

        # 3a. 尝试下载全文 PDF（通过 Europe PMC 获取直链）
        if article.get("pmc_id"):
            print(f"  尝试下载全文 (PMC: {article['pmc_id']})")
            if download_full_text_pdf(article["pmid"], article["pmc_id"], file_path):
                source = "full_text"
                success = True
                print(f"  ✅ 全文下载成功: {file_name} ({file_path.stat().st_size} bytes)")
            else:
                print(f"  ⚠️  全文下载失败，改用摘要生成 PDF")
        else:
            print(f"  无 PMC ID，使用摘要生成 PDF")

        # 3b. 兜底：用摘要生成 PDF
        if not success:
            if generate_abstract_pdf(article, file_path):
                source = "abstract"
                success = True
                print(f"  ✅ 摘要 PDF 生成成功: {file_name} ({file_path.stat().st_size} bytes)")
            else:
                error = "全文下载失败且摘要 PDF 生成失败"
                print(f"  ❌ {error}")

        # 记录结果
        result = {
            "pmid": article["pmid"],
            "title": article["title"],
            "authors": article["authors"],
            "journal": article["journal"],
            "year": article["year"],
            "abstract": article["abstract"],
            "pmc_id": article.get("pmc_id"),
            "doi": article.get("doi"),
            "file_path": str(file_path) if success else None,
            "file_name": file_name if success else None,
            "source": source,
            "success": success,
            "error": error,
        }
        results.append(result)

        # 速率控制
        if i < len(articles):
            time.sleep(delay)

    # 汇总
    success_count = sum(1 for r in results if r["success"])
    full_text_count = sum(1 for r in results if r.get("source") == "full_text")
    print(f"\n[PubMed] 完成: {success_count}/{len(results)} 篇成功 "
          f"（全文 {full_text_count} 篇，摘要 {success_count - full_text_count} 篇）")

    return results


# ═══════════════════════════════════════════════════════════════
# 测试入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    print("=" * 70)
    print("PubMed 文献检索下载测试")
    print("=" * 70)

    # 测试搜索
    test_keyword = "plant CpG island DNA methylation"
    print(f"\n测试关键字: {test_keyword}")

    results = search_and_download(test_keyword, max_results=3, delay=0.5)

    print("\n" + "=" * 70)
    print("结果汇总:")
    print("-" * 70)
    for i, r in enumerate(results, 1):
        status = "✅" if r["success"] else "❌"
        source = r.get("source", "N/A")
        print(f"{i}. {status} PMID={r.get('pmid', '?')} | source={source}")
        print(f"   标题: {r.get('title', '?')[:70]}")
        if r.get("file_name"):
            print(f"   文件: {r['file_name']}")
        if r.get("error"):
            print(f"   错误: {r['error']}")
    print("=" * 70)
