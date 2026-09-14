"""
文本分块模块（RAG 流水线第 3 步：文本分块 / Text Splitting）

对应 Hello Agents 第 8 章 RAG 的「Text Splitting」阶段。
采用教程的 Markdown 智能分块 4 大策略：
  1. 标题层次感知 - 利用 #、##、### 结构，维护标题路径
  2. 段落语义保持 - 按段落（空行）拆分，不在段落中间切分
  3. Token 精确控制 - 用 BGE-m3 tokenizer 计算 token 数，适配嵌入模型
  4. 智能重叠策略 - overlap 取上一个 chunk 的最后几个段落，包含标题路径

与之前版本的区别（自行修改说明）：
  - 之前：固定字符数滑动窗口 + 句子边界查找，输入为纯文本
  - 现在：基于 Markdown 标题层次的智能分块 + Token 精确控制，输入为 Markdown
  - 原因：教程明确要求基于 Markdown 结构做智能分块，且我们已升级到 parse_to_markdown()
"""
from __future__ import annotations

import os
import re
import uuid
from typing import List, Dict, Tuple

from dotenv import load_dotenv

load_dotenv()


class Chunker:
    """
    Markdown 智能分块器。

    输入：List[Dict]（PDFParser.parse_to_markdown() 的输出，每页带 page_num + text（Markdown） + metadata）
    输出：List[Dict]，每个 chunk 格式：
        {
            "chunk_id": str,
            "text": str,               # chunk 文本（开头注入标题路径）
            "page_num": int,
            "chunk_index": int,
            "metadata": {
                "filename": str,
                "file_path": str,
                "page_count": int,
                "source": str,
                "heading_path": str,   # 标题路径（如 "# 标题 > ## 子标题"）
            }
        }
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        embedding_model: str = None,
    ):
        """
        Args:
            chunk_size: 每个 chunk 的最大 token 数，默认从 .env 读 CHUNK_SIZE（800）
            chunk_overlap: 相邻 chunk 的重叠 token 数，默认从 .env 读 CHUNK_OVERLAP（100）
            embedding_model: embedding 模型名，用于加载对应 tokenizer，默认 BAAI/bge-m3
        """
        self.chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "800"))
        self.chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "100"))
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

        # 校验参数
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size 必须大于 0，当前: {self.chunk_size}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) 必须小于 chunk_size ({self.chunk_size})"
            )

        # 标题正则：匹配 # 到 ###### 开头的标题行
        self.heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$')

        # 加载 tokenizer（用于 Token 精确控制）
        self.tokenizer = None
        self._load_tokenizer()

    def _load_tokenizer(self):
        """加载 BGE-m3 tokenizer，失败时用字符数近似作为 fallback"""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
            print(f"  ✅ Tokenizer 加载成功: {self.embedding_model}")
        except Exception as e:
            print(f"  ⚠️ Tokenizer 加载失败: {e}")
            print(f"     使用字符数近似（中文约 1 字符=1 token，英文约 4 字符=1 token，平均约 2）")
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """
        计算文本的 token 数。

        Args:
            text: 输入文本

        Returns:
            token 数量
        """
        if not text.strip():
            return 0
        if self.tokenizer:
            # add_special_tokens=False：不计算 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> [SEP] 等特殊标记
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        else:
            # Fallback：字符数近似（平均约 2 字符=1 token）
            return max(1, len(text) // 2)

    def chunk(self, pages: List[Dict]) -> List[Dict]:
        """
        主入口：把所有页的 Markdown 文本切成 chunks。

        Args:
            pages: PDFParser.parse_to_markdown() 的输出

        Returns:
            所有 chunk 的列表
        """
        if not pages:
            return []

        all_chunks: List[Dict] = []

        for page in pages:
            page_text = page["text"]
            page_num = page["page_num"]
            metadata = page["metadata"]

            if not page_text.strip():
                continue

            # 对单页 Markdown 进行智能分块
            page_chunks = self._chunk_markdown_page(page_text)

            for chunk_idx, (chunk_text, heading_path_str) in enumerate(page_chunks):
                source = f"{metadata['filename']}, 第 {page_num} 页"

                chunk_metadata = {
                    "filename": metadata["filename"],
                    "file_path": metadata["file_path"],
                    "page_count": metadata["page_count"],
                    "source": source,
                    "heading_path": heading_path_str,
                }

                all_chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "text": chunk_text,
                    "page_num": page_num,
                    "chunk_index": chunk_idx,
                    "metadata": chunk_metadata,
                })

        return all_chunks

    def _chunk_markdown_page(self, markdown_text: str) -> List[Tuple[str, str]]:
        """
        对单页 Markdown 文本进行智能分块。

        流程：
          1. 按标题拆分 → sections 列表
          2. 对每个 section：
             - 标题路径 + 内容 token 数 ≤ chunk_size → 整个 section 作为一个 chunk
             - 超过 chunk_size → 按段落细分（带 overlap）
          3. 每个 chunk 开头注入标题路径

        Args:
            markdown_text: 单页 Markdown 文本

        Returns:
            List[Tuple[chunk_text, heading_path_str]]
        """
        # 1. 按标题拆分，得到 sections 列表
        sections = self._split_by_headings(markdown_text)

        # 2. 对每个 section 进行分块
        chunks = []
        for section in sections:
            heading_path_list = section["heading_path"]
            heading_path_str = " > ".join(heading_path_list) if heading_path_list else ""
            heading_path_text = "\n".join(heading_path_list) if heading_path_list else ""
            content = section["content"]

            if not content.strip():
                continue

            heading_tokens = self.count_tokens(heading_path_text)
            content_tokens = self.count_tokens(content)

            if heading_tokens + content_tokens + 2 <= self.chunk_size:
                # 整个 section 作为一个 chunk（未超过 chunk_size）
                if heading_path_text:
                    chunk_text = heading_path_text + "\n\n" + content
                else:
                    chunk_text = content
                chunks.append((chunk_text.strip(), heading_path_str))
            else:
                # section 太大，按段落细分
                sub_chunks = self._split_large_section(
                    heading_path_text, heading_path_str, content, heading_tokens
                )
                chunks.extend(sub_chunks)

        return chunks

    def _split_by_headings(self, markdown_text: str) -> List[Dict]:
        """
        按标题拆分 Markdown 文本，得到 sections 列表。

        维护一个标题栈：遇到标题时，弹出级别 >= 当前级别的标题，压入当前标题。
        每个 section 包含标题路径（从根到当前节点的所有标题）和内容。

        Args:
            markdown_text: Markdown 文本

        Returns:
            [{"heading_path": ["# 标题", "## 子标题"], "content": "..."}]
        """
        lines = markdown_text.split("\n")
        sections = []
        heading_stack: List[Tuple[int, str]] = []  # [(level, full_title_text)]
        current_content: List[str] = []

        for line in lines:
            match = self.heading_pattern.match(line.strip())
            if match:
                # 遇到标题，先保存当前 section
                if current_content:
                    content_text = "\n".join(current_content).strip()
                    if content_text:
                        sections.append({
                            "heading_path": [h[1] for h in heading_stack],
                            "content": content_text,
                        })
                    current_content = []

                # 更新标题栈：弹出级别 >= 当前级别的标题
                level = len(match.group(1))
                title = match.group(0).strip()  # 保留完整的 "# 标题" 格式
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))
            else:
                current_content.append(line)

        # 保存最后一个 section
        if current_content:
            content_text = "\n".join(current_content).strip()
            if content_text:
                sections.append({
                    "heading_path": [h[1] for h in heading_stack],
                    "content": content_text,
                })

        return sections

    def _split_large_section(
        self,
        heading_path_text: str,
        heading_path_str: str,
        content: str,
        heading_tokens: int,
    ) -> List[Tuple[str, str]]:
        """
        对过大的 section 按段落进行细分，带 overlap。

        流程：
          1. 按段落（空行分隔）拆分
          2. 累积段落，当 token 数超过 chunk_size 时保存当前 chunk
          3. 下一个 chunk 从上一个 chunk 的最后几个段落开始（overlap）
          4. 每个 chunk 开头注入标题路径

        Args:
            heading_path_text: 标题路径文本（用于注入 chunk 开头）
            heading_path_str: 标题路径字符串（用于 metadata）
            content: section 内容
            heading_tokens: 标题路径的 token 数

        Returns:
            List[Tuple[chunk_text, heading_path_str]]
        """
        # 按段落拆分（一个或多个空行分隔）
        paragraphs = re.split(r'\n\s*\n', content)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return []

        chunks = []
        current_chunk: List[str] = []
        current_tokens = heading_tokens

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            if current_tokens + para_tokens + 2 <= self.chunk_size:
                # 当前段落加入当前 chunk（+2 for \n\n 分隔符）
                current_chunk.append(para)
                current_tokens += para_tokens + 2
            else:
                # 当前 chunk 满了，保存
                if current_chunk:
                    chunk_text = self._build_chunk_text(heading_path_text, current_chunk)
                    chunks.append((chunk_text, heading_path_str))

                # 开始新 chunk，包含 overlap（取上一个 chunk 的最后几个段落）
                if chunks and self.chunk_overlap > 0:
                    overlap_paras = self._get_overlap_paragraphs(current_chunk)
                    current_chunk = overlap_paras
                    current_tokens = heading_tokens + sum(
                        self.count_tokens(p) + 2 for p in overlap_paras
                    )
                else:
                    current_chunk = []
                    current_tokens = heading_tokens

                # 加入当前段落
                current_chunk.append(para)
                current_tokens += para_tokens + 2

        # 保存最后一个 chunk
        if current_chunk:
            chunk_text = self._build_chunk_text(heading_path_text, current_chunk)
            chunks.append((chunk_text, heading_path_str))

        return chunks

    def _build_chunk_text(self, heading_path_text: str, paragraphs: List[str]) -> str:
        """
        构建 chunk 文本：标题路径 + 段落内容。

        Args:
            heading_path_text: 标题路径文本
            paragraphs: 段落列表

        Returns:
            完整的 chunk 文本
        """
        content_text = "\n\n".join(paragraphs)
        if heading_path_text:
            return (heading_path_text + "\n\n" + content_text).strip()
        else:
            return content_text.strip()

    def _get_overlap_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """
        从段落列表末尾取若干段落，总 token 数不超过 chunk_overlap。

        Args:
            paragraphs: 上一个 chunk 的段落列表

        Returns:
            作为 overlap 的段落列表（保持原顺序）
        """
        overlap = []
        total_tokens = 0
        for para in reversed(paragraphs):
            para_tokens = self.count_tokens(para)
            if total_tokens + para_tokens <= self.chunk_overlap:
                overlap.insert(0, para)
                total_tokens += para_tokens
            else:
                break
        return overlap


# ── 测试：python -m app.rag.chunker ──
if __name__ == "__main__":
    from app.rag.pdf_parser import PDFParser
    from app.rag.text_cleaner import TextCleaner

    print("=" * 60)
    print("PlantGenome Agent - Markdown 智能分块模块测试")
    print("（对应 Hello Agents 第 8 章：标题层次感知 + Token 精确控制）")
    print("=" * 60)

    parser = PDFParser()
    cleaner = TextCleaner()
    chunker = Chunker()

    print(f"\n配置：")
    print(f"  chunk_size = {chunker.chunk_size} tokens")
    print(f"  chunk_overlap = {chunker.chunk_overlap} tokens")
    print(f"  embedding_model = {chunker.embedding_model}")
    print(f"  tokenizer 已加载: {chunker.tokenizer is not None}")

    test_pdf = "data/pdfs/Yang - 2007 - PAML 4 Phylogenetic analysis by maximum likelihood.pdf"
    if not os.path.exists(test_pdf):
        print(f"测试文件不存在: {test_pdf}")
        exit(1)

    # 步骤 1：解析 PDF（用 Markdown 模式）
    print(f"\n[步骤 1] 解析 PDF（Markdown 模式）: {os.path.basename(test_pdf)}")
    pages = parser.parse_to_markdown(test_pdf)
    print(f"  页数: {len(pages)}")

    # 步骤 2：清洗文本
    print(f"\n[步骤 2] 清洗文本")
    cleaned_pages = cleaner.clean(pages)
    print(f"  清洗后页数: {len(cleaned_pages)}")

    # 步骤 3：Markdown 智能分块
    print(f"\n[步骤 3] Markdown 智能分块")
    chunks = chunker.chunk(cleaned_pages)
    print(f"  总 chunk 数: {len(chunks)}")

    # 统计 chunk token 长度分布
    chunk_tokens = [chunker.count_tokens(c["text"]) for c in chunks]
    print(f"  chunk token 长度: 最小={min(chunk_tokens)}, 最大={max(chunk_tokens)}, "
          f"平均={sum(chunk_tokens)//len(chunk_tokens)} tokens")

    # 按页统计
    from collections import Counter
    page_counts = Counter(c["page_num"] for c in chunks)
    print(f"  每页 chunk 数: {dict(sorted(page_counts.items()))}")

    # 统计有标题路径的 chunk 比例
    with_heading = sum(1 for c in chunks if c["metadata"]["heading_path"])
    print(f"  含标题路径的 chunk: {with_heading}/{len(chunks)} "
          f"({with_heading/len(chunks)*100:.1f}%)")

    # 展示前 3 个 chunk
    print(f"\n【前 3 个 chunk 预览】")
    print("-" * 40)
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1} (page {chunk['page_num']}, index {chunk['chunk_index']}):")
        print(f"  chunk_id: {chunk['chunk_id'][:8]}...")
        print(f"  source: {chunk['metadata']['source']}")
        print(f"  heading_path: {chunk['metadata']['heading_path'] or '(无)'}")
        print(f"  token 长度: {chunker.count_tokens(chunk['text'])}")
        print(f"  内容前 300 字符:")
        print(f"  {chunk['text'][:300]}...")

    print("\n" + "=" * 60)
    print("Markdown 智能分块模块测试完成。")
    print("=" * 60)
