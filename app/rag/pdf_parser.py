"""
PDF 文本提取模块（RAG 流水线第 1 步：文档加载）

对应 Hello Agents 第 8 章 RAG 的「Document Loading」阶段。
职责：把二进制 PDF 转换成带页码的纯文本，为后续清洗、分块、向量化做准备。

为什么用 PyMuPDF (fitz) 而不是 pypdf？
- PyMuPDF 提取速度更快，对双栏论文的文本顺序处理更好
- 支持提取页码、字体、位置等元信息（以后做结构化解析用得上）
- 社区活跃，维护频繁
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Optional

import pymupdf as fitz  # PyMuPDF（官方推荐用 pymupdf，保留 fitz 别名兼容代码）
import pymupdf4llm  # PyMuPDF 官方 Markdown 转换库


class PDFParser:
    """
    PDF 文本提取器。

    输入：PDF 文件路径
    输出：List[Dict]，每页一个 dict，格式：
        {
            "page_num": int,           # 页码（从 1 开始）
            "text": str,                # 该页提取的纯文本
            "metadata": {
                "filename": str,        # 文件名
                "file_path": str,       # 完整路径
                "page_count": int,      # 总页数
            }
        }
    """

    def __init__(self, min_page_text_length: int = 20):
        """
        Args:
            min_page_text_length: 单页最小文本长度，低于此值认为该页无有效内容
                                  （比如封面页、纯图片页）
        """
        self.min_page_text_length = min_page_text_length

    def parse(self, pdf_path: str | Path) -> List[Dict]:
        """
        解析 PDF，提取每页文本。

        Args:
            pdf_path: PDF 文件路径

        Returns:
            每页文本的列表

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不是 PDF 文件或文件损坏
        """
        pdf_path = Path(pdf_path)

        # ── 前置校验 ──
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"不是 PDF 文件: {pdf_path.suffix}")

        # ── 打开 PDF 并提取文本 ──
        results: List[Dict] = []
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            raise ValueError(f"无法打开 PDF（可能损坏或加密）: {e}") from e

        try:
            page_count = len(doc)
            metadata = {
                "filename": pdf_path.name,
                "file_path": str(pdf_path.resolve()),
                "page_count": page_count,
            }

            for page_idx in range(page_count):
                page = doc[page_idx]
                text = page.get_text("text")  # 按阅读顺序提取纯文本

                # 跳过内容过少的页（封面、纯图片页等）
                if len(text.strip()) < self.min_page_text_length:
                    continue

                results.append({
                    "page_num": page_idx + 1,  # 页码从 1 开始，符合人类习惯
                    "text": text,
                    "metadata": metadata,
                })
        finally:
            doc.close()  # 确保文件句柄释放

        if not results:
            raise ValueError(f"PDF 没有提取到有效文本（可能是扫描件，需要 OCR）: {pdf_path.name}")

        return results

    def parse_to_markdown(self, pdf_path: str | Path) -> List[Dict]:
        """
        解析 PDF，提取每页的 Markdown 文本（保留标题、列表、表格结构）。

        与 parse() 的区别：
        - parse() 用 fitz.get_text("text") 提取纯文本，丢失格式结构
        - parse_to_markdown() 用 pymupdf4llm 转换为 Markdown，保留 # 标题、- 列表、| 表格 |
        - 输出格式与 parse() 完全一致（List[Dict]），下游可无缝切换

        Args:
            pdf_path: PDF 文件路径

        Returns:
            每页 Markdown 文本的列表，格式同 parse()

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不是 PDF 文件或文件损坏
        """
        pdf_path = Path(pdf_path)

        # ── 前置校验（与 parse() 共享）──
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"不是 PDF 文件: {pdf_path.suffix}")

        # ── 先获取总页数（用 fitz 打开，轻量）──
        try:
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()
        except Exception as e:
            raise ValueError(f"无法打开 PDF（可能损坏或加密）: {e}") from e

        # ── 构建 metadata（与 parse() 共享）──
        metadata = {
            "filename": pdf_path.name,
            "file_path": str(pdf_path.resolve()),
            "page_count": page_count,
        }

        # ── 逐页转换为 Markdown ──
        results: List[Dict] = []
        for page_idx in range(page_count):
            try:
                # pymupdf4llm.to_markdown 支持 pages 参数指定页码（从 0 开始）
                md_text = pymupdf4llm.to_markdown(
                    str(pdf_path),
                    pages=[page_idx],
                )
            except Exception as e:
                # 单页转换失败时跳过该页，不中断整个解析
                print(f"  ⚠️ 第 {page_idx + 1} 页 Markdown 转换失败: {e}")
                continue

            # 跳过内容过少的页（封面、纯图片页等）
            if len(md_text.strip()) < self.min_page_text_length:
                continue

            results.append({
                "page_num": page_idx + 1,
                "text": md_text,
                "metadata": metadata,
            })

        if not results:
            raise ValueError(f"PDF 没有提取到有效 Markdown（可能是扫描件，需要 OCR）: {pdf_path.name}")

        return results

    def parse_directory(self, dir_path: str | Path) -> Dict[str, List[Dict]]:
        """
        批量解析目录下所有 PDF。

        Args:
            dir_path: 包含 PDF 的目录

        Returns:
            {文件名: 解析结果列表} 的字典
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"目录不存在: {dir_path}")

        all_results: Dict[str, List[Dict]] = {}
        pdf_files = sorted(dir_path.glob("*.pdf"))

        for pdf_file in pdf_files:
            try:
                pages = self.parse(pdf_file)
                all_results[pdf_file.name] = pages
                print(f"  ✅ {pdf_file.name}: {len(pages)} 页")
            except Exception as e:
                print(f"  ❌ {pdf_file.name}: {e}")

        return all_results


# ── 测试：python -m app.rag.pdf_parser ──
if __name__ == "__main__":
    print("=" * 60)
    print("PlantGenome Agent - PDF 解析模块测试")
    print("=" * 60)

    parser = PDFParser()

    # 测试 1：解析单个 PDF（选 PAML 手册，排版相对简单）
    test_pdf = "data/pdfs/Yang - 2007 - PAML 4 Phylogenetic analysis by maximum likelihood.pdf"
    if os.path.exists(test_pdf):
        print(f"\n[测试 1] 解析单个 PDF: {os.path.basename(test_pdf)}")
        print("-" * 40)
        pages = parser.parse(test_pdf)
        print(f"总页数: {len(pages)}")
        print(f"第 1 页页码: {pages[0]['page_num']}")
        print(f"第 1 页文本长度: {len(pages[0]['text'])} 字符")
        print(f"metadata: {pages[0]['metadata']}")
        print(f"\n第 1 页前 300 字符预览:")
        print(pages[0]['text'][:300])
        print("...")
    else:
        print(f"测试文件不存在: {test_pdf}")

    # 测试 2：批量解析 data/pdfs/ 目录
    print(f"\n[测试 2] 批量解析 data/pdfs/ 目录")
    print("-" * 40)
    all_pdfs = parser.parse_directory("data/pdfs")
    print(f"\n成功解析 {len(all_pdfs)} 个 PDF")
    total_pages = sum(len(pages) for pages in all_pdfs.values())
    print(f"总页数: {total_pages}")

    # 测试 3：Markdown 解析（与纯文本对比）
    if os.path.exists(test_pdf):
        print(f"\n[测试 3] Markdown 解析对比: {os.path.basename(test_pdf)}")
        print("-" * 40)
        md_pages = parser.parse_to_markdown(test_pdf)
        print(f"Markdown 总页数: {len(md_pages)}")
        print(f"第 1 页 Markdown 长度: {len(md_pages[0]['text'])} 字符")

        # 对比纯文本和 Markdown 的第 1 页前 500 字符
        print(f"\n【纯文本 vs Markdown 对比（第 1 页前 500 字符）】")
        print("-" * 40)
        print("【纯文本】")
        print(pages[0]['text'][:500])
        print("\n【Markdown】")
        print(md_pages[0]['text'][:500])

        # 统计 Markdown 中的结构元素
        md_full = "\n".join(p["text"] for p in md_pages)
        heading_count = md_full.count("\n# ") + md_full.count("\n## ") + md_full.count("\n### ")
        table_count = md_full.count("\n| ")
        list_count = md_full.count("\n- ") + md_full.count("\n* ")
        print(f"\n【Markdown 结构统计】")
        print(f"  标题数（#/##/###）: {heading_count}")
        print(f"  表格行数（|）: {table_count}")
        print(f"  列表项数（-/*）: {list_count}")

    print("\n" + "=" * 60)
    print("PDF 解析模块测试完成（纯文本 + Markdown 双模式）。")
    print("=" * 60)
