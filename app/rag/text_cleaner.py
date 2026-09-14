"""
文本清洗模块（RAG 流水线第 2 步：文本清洗）

对应 Hello Agents 第 8 章 RAG 的「Document Loading」后续预处理阶段。
职责：清洗 PDF 提取出的原始文本，去除页眉页脚、页码、重复行、多余空白，
     避免这些噪声进入向量库影响检索准确率。

为什么需要清洗？
- PDF 提取的文本包含大量噪声：每页都有的期刊名、页码、DOI、版权声明
- 如果不清洗，这些噪声会被分块、向量化，检索时可能命中"第 12 页"这种无意义内容
- 清洗是 RAG 质量的第一道防线，比后面加 reranker 成本低得多
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List, Dict


class TextCleaner:
    """
    PDF 文本清洗器。

    输入：List[Dict]（pdf_parser.PDFParser.parse() 的输出）
    输出：List[Dict]（清洗后的文本，保留 page_num 和 metadata）
    """

    def __init__(
        self,
        min_line_length: int = 3,
        max_repeated_line_ratio: float = 0.3,
    ):
        """
        Args:
            min_line_length: 单行最小字符数，低于此值的行被丢弃（页眉符号、孤立标点）
            max_repeated_line_ratio: 跨页重复行比例阈值。如果某行出现在超过这个比例的页面中，
                                      认为是页眉页脚，全部去除。默认 30%（10 页里出现 3 次以上）。
        """
        self.min_line_length = min_line_length
        self.max_repeated_line_ratio = max_repeated_line_ratio

        # 常见页眉页脚正则模式
        self.noise_patterns = [
            re.compile(r'^\s*\d+\s*$'),                              # 纯数字页码
            re.compile(r'^\s*page\s*\d+\s*$', re.IGNORECASE),       # "Page 12"
            re.compile(r'^\s*\d+\s*[/\\]\s*\d+\s*$'),               # "12 / 100"
            re.compile(r'^\s*doi\s*:', re.IGNORECASE),               # DOI 行
            re.compile(r'^\s*copyright\b', re.IGNORECASE),           # 版权声明
            re.compile(r'^\s*©\s*\d{4}', re.IGNORECASE),             # © 2021
            re.compile(r'^\s*(downloaded|received|accepted|published)\b', re.IGNORECASE),  # 期刊元信息
            re.compile(r'^\s*https?://\S+\s*$'),                     # 孤立 URL
            re.compile(r'^\s*[\-\_=*#]+\s*$'),                       # 分隔线
        ]

    def clean(self, pages: List[Dict]) -> List[Dict]:
        """
        清洗所有页面的文本。

        Args:
            pages: PDFParser.parse() 的输出

        Returns:
            清洗后的页面列表
        """
        if not pages:
            return []

        # ── 第 1 步：识别跨页重复行（页眉页脚）──
        header_footer_lines = self._find_header_footer_lines(pages)

        # ── 第 2 步：逐页清洗 ──
        cleaned_pages: List[Dict] = []
        for page in pages:
            cleaned_text = self._clean_single_page(
                page["text"],
                header_footer_lines,
            )
            # 清洗后如果文本过短，跳过该页
            if len(cleaned_text.strip()) < 50:
                continue

            cleaned_pages.append({
                "page_num": page["page_num"],
                "text": cleaned_text,
                "metadata": page["metadata"],
            })

        return cleaned_pages

    def _find_header_footer_lines(self, pages: List[Dict]) -> set:
        """
        识别跨页重复出现的行（大概率是页眉页脚）。

        逻辑：统计每一行在多少页中出现，如果出现比例超过 max_repeated_line_ratio，
        就认为是页眉页脚，加入黑名单。
        """
        total_pages = len(pages)
        line_page_counter: Counter = Counter()

        for page in pages:
            # 用 set 去重，同一页里重复出现的行只算一次
            lines_in_page = set()
            for line in page["text"].split("\n"):
                stripped = line.strip()
                if len(stripped) >= self.min_line_length:
                    lines_in_page.add(stripped)
            for line in lines_in_page:
                line_page_counter[line] += 1

        # 出现比例超过阈值的行，认为是页眉页脚
        threshold = max(2, int(total_pages * self.max_repeated_line_ratio))
        header_footer = {
            line for line, count in line_page_counter.items()
            if count >= threshold
        }
        return header_footer

    def _clean_single_page(self, text: str, header_footer_lines: set) -> str:
        """
        清洗单页文本。

        清洗步骤：
        1. 按行拆分
        2. 去除首尾空白
        3. 去除过短行
        4. 去除匹配噪声模式的行
        5. 去除页眉页脚黑名单中的行
        6. 合并连续空行
        """
        lines = text.split("\n")
        cleaned_lines: List[str] = []
        prev_empty = False

        for line in lines:
            stripped = line.strip()

            # 过短行直接丢弃
            if len(stripped) < self.min_line_length:
                continue

            # 匹配噪声模式的行丢弃
            if any(pattern.match(stripped) for pattern in self.noise_patterns):
                continue

            # 页眉页脚黑名单中的行丢弃
            if stripped in header_footer_lines:
                continue

            # 合并连续空行
            if stripped == "":
                if not prev_empty:
                    cleaned_lines.append("")
                prev_empty = True
            else:
                cleaned_lines.append(stripped)
                prev_empty = False

        return "\n".join(cleaned_lines).strip()


# ── 测试：python -m app.rag.text_cleaner ──
if __name__ == "__main__":
    import os
    from app.rag.pdf_parser import PDFParser

    print("=" * 60)
    print("PlantGenome Agent - 文本清洗模块测试")
    print("=" * 60)

    parser = PDFParser()
    cleaner = TextCleaner()

    test_pdf = "data/pdfs/Yang - 2007 - PAML 4 Phylogenetic analysis by maximum likelihood.pdf"
    if not os.path.exists(test_pdf):
        print(f"测试文件不存在: {test_pdf}")
        exit(1)

    # 解析 PDF
    print(f"\n[步骤 1] 解析 PDF: {os.path.basename(test_pdf)}")
    pages = parser.parse(test_pdf)
    print(f"原始页数: {len(pages)}")
    original_total_chars = sum(len(p["text"]) for p in pages)
    print(f"原始总字符数: {original_total_chars}")

    # 清洗文本
    print(f"\n[步骤 2] 清洗文本")
    cleaned_pages = cleaner.clean(pages)
    print(f"清洗后页数: {len(cleaned_pages)}")
    cleaned_total_chars = sum(len(p["text"]) for p in cleaned_pages)
    print(f"清洗后总字符数: {cleaned_total_chars}")
    print(f"去除字符数: {original_total_chars - cleaned_total_chars} "
          f"({(original_total_chars - cleaned_total_chars) / original_total_chars * 100:.1f}%)")

    # 对比第 1 页清洗前后
    print(f"\n[对比] 第 1 页清洗前后前 500 字符")
    print("-" * 40)
    print("【清洗前】")
    print(pages[0]["text"][:500])
    print("\n【清洗后】")
    print(cleaned_pages[0]["text"][:500])

    print("\n" + "=" * 60)
    print("文本清洗模块测试完成。")
    print("=" * 60)
