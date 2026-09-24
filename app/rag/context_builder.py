"""
ContextBuilder 上下文构建器（参考 Hello Agents 第九章 9.3 节）

实现 GSSC 流水线（Gather-Select-Structure-Compress），把检索结果组织成结构化的 LLM 上下文。

根据我们项目的简化说明：
- Gather：只从 RAG 检索 + 系统指令汇集（单轮 RAG，没有记忆和多轮对话）
- Select：只用 distance 作为相关性评分（单轮场景不需要新近性）
- Structure：用 RAG 场景化模板 [Role]/[Evidence]/[Task]/[Output]
- Compress：截断低相关性 chunk（避免额外延迟和 API 调用）
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional

from dotenv import load_dotenv

load_dotenv()


# ═══════════════════════════════════════════════════════════════
# 核心数据结构（参考教程 9.3.2 ContextPacket / ContextConfig）
# ═══════════════════════════════════════════════════════════════

@dataclass
class ContextPacket:
    """
    候选信息包（参考教程 9.3.2 ContextPacket）。

    每个候选信息都会被封装为一个 ContextPacket，包含内容、token 数量和相关性分数。
    """
    content: str
    token_count: int
    relevance_score: float = 0.5  # 0.0-1.0，越高越相关
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        # 确保相关性分数在有效范围内
        self.relevance_score = max(0.0, min(1.0, self.relevance_score))


@dataclass
class ContextConfig:
    """
    上下文构建配置（参考教程 9.3.2 ContextConfig）。

    简化版：只保留 RAG 场景需要的配置。
    """
    max_tokens: int = 3000                    # 最大 token 数量（上下文窗口限制）
    reserve_ratio: float = 0.2                 # 为系统指令预留的比例
    min_relevance: float = 0.25                # 最低相关性阈值（对应 distance < 1.5，与 Retriever 对齐）
    enable_compression: bool = True            # 是否启用压缩（超限时截断低相关性）

    def __post_init__(self):
        assert 0.0 <= self.reserve_ratio <= 1.0, "reserve_ratio 必须在 [0, 1] 范围内"
        assert 0.0 <= self.min_relevance <= 1.0, "min_relevance 必须在 [0, 1] 范围内"


# ═══════════════════════════════════════════════════════════════
# RAG 上下文模板（参考教程 9.3.1 分区组织，简化为 RAG 场景）
# ═══════════════════════════════════════════════════════════════

RAG_CONTEXT_TEMPLATE = """[Role & Policies]
你是一个植物基因组学研究助手，专门回答植物比较基因组、基因家族分析、生信工具使用等问题。

【核心规则 - 必须严格遵守】
1. 【基于证据】必须完全基于下方的 [Evidence] 参考资料回答，绝对不要使用参考资料以外的知识，不要编造。
2. 【无法回答时】如果参考资料中没有相关信息，请明确说"根据现有参考资料，无法回答这个问题"，并说明缺少什么信息。
3. 【引用标注】回答时在相关句子末尾标注引用来源，格式为 [1]、[2]，对应参考资料编号。每个主要观点都必须有引用。
4. 【简洁性】回答必须简洁，控制在 300-800 字以内。不要重复同样的内容，不要啰嗦，不要展开无关细节。
5. 【禁止重复】绝对不要重复同一个词或句子。如果发现自己在重复，立即停止。
6. 【语言】使用中文回答，专业术语（如 dN/dS、omega、FASTA、OrthoFinder）保留英文原文。
7. 【结构清晰】按"定义/概念 → 原理/方法 → 应用/意义"的结构组织回答，用短句，避免长句。
8. 【矛盾处理】如果参考资料之间有矛盾，指出矛盾并分别说明不同来源的观点。

[Evidence]
{evidence}

[Task]
请基于以上参考资料，简洁、准确地回答以下问题（300-800字，不要重复）：
{query}

[Output]
回答：
"""


# ═══════════════════════════════════════════════════════════════
# ContextBuilder 主类（参考教程 9.3 GSSC 流水线）
# ═══════════════════════════════════════════════════════════════

class ContextBuilder:
    """
    上下文构建器（参考教程 9.3 ContextBuilder）。

    实现 GSSC 流水线：
      Gather（汇集）→ Select（选择）→ Structure（结构化）→ Compress（压缩）

    输入：用户 query + Retriever 返回的检索结果
    输出：结构化上下文字符串 + 选中的 sources 列表
    """

    def __init__(self, config: ContextConfig = None, tokenizer=None):
        """
        Args:
            config: 上下文构建配置（可选，不传则用默认值）
            tokenizer: BGE-m3 tokenizer 实例（可选，不传则自动加载）
        """
        self.config = config or ContextConfig()
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
                print("  ✅ ContextBuilder Tokenizer 加载成功: BAAI/bge-m3")
            except Exception as e:
                print(f"  ⚠️ ContextBuilder Tokenizer 加载失败: {e}，回退到字符数估算")
                self.tokenizer = None

    def build(
        self,
        query: str,
        retrieve_result: Dict,
    ) -> Dict:
        """
        构建上下文主入口（GSSC 流水线）。

        Args:
            query: 用户问题
            retrieve_result: Retriever.retrieve() 的返回结果

        Returns:
            Dict：
                {
                    "context": str,           # 结构化上下文字符串
                    "selected_sources": List[str],  # 选中的 sources 列表
                    "selected_packets": List[ContextPacket],  # 选中的信息包
                    "total_tokens": int,       # 总 token 数
                    "compressed": bool,        # 是否被压缩
                }
        """
        # ── Gather：汇集候选信息 ──
        packets = self._gather(query, retrieve_result)

        # ── Select：智能信息选择 ──
        selected_packets, compressed = self._select(packets)

        # ── Structure：结构化组织 ──
        context, selected_sources = self._structure(query, selected_packets)

        # ── Compress：已在 Select 阶段处理（超限时截断低相关性）──
        # 教程的 Compress 是用 LLM 总结，我们简化为在 Select 阶段截断

        total_tokens = sum(p.token_count for p in selected_packets)

        return {
            "context": context,
            "selected_sources": selected_sources,
            "selected_packets": selected_packets,
            "total_tokens": total_tokens,
            "compressed": compressed,
        }

    # ═══════════════════════════════════════════════════════════
    # Gather：多源信息汇集（参考教程 9.3.3 (1)）
    # ═══════════════════════════════════════════════════════════

    def _gather(self, query: str, retrieve_result: Dict) -> List[ContextPacket]:
        """
        Gather 阶段：把检索结果转成 ContextPacket 列表。

        简化版：只从 RAG 检索结果汇集（教程还包括系统指令、记忆、对话历史，
        我们是单轮 RAG，不需要这些）。

        Args:
            query: 用户问题
            retrieve_result: Retriever.retrieve() 的返回结果

        Returns:
            ContextPacket 列表
        """
        packets = []

        for r in retrieve_result.get("results", []):
            # 把 distance 转成相关性分数（distance ∈ [0, 2] → relevance ∈ [0, 1]）
            # distance=0 → relevance=1（完全相同）
            # distance=2 → relevance=0（完全相反）
            relevance = 1.0 - r["distance"] / 2.0

            packet = ContextPacket(
                content=r["text"],
                token_count=self._count_tokens(r["text"]),
                relevance_score=relevance,
                metadata={
                    "rank": r["rank"],
                    "filename": r["filename"],
                    "page_num": r["page_num"],
                    "heading_path": r["heading_path"],
                    "distance": r["distance"],
                    "source": r["source"],
                },
            )
            packets.append(packet)

        return packets

    # ═══════════════════════════════════════════════════════════
    # Select：智能信息选择（参考教程 9.3.3 (2)）
    # ═══════════════════════════════════════════════════════════

    def _select(self, packets: List[ContextPacket]) -> tuple[List[ContextPacket], bool]:
        """
        Select 阶段：按相关性评分 + token 预算，贪心选择信息包。

        简化版：
        - 只用相关性评分（教程用相关性+新近性双维度，单轮场景不需要新近性）
        - 过滤低于 min_relevance 的信息包
        - 按相关性降序排序
        - 贪心选择，直到达到 token 上限
        - 超限时截断低相关性信息包（即 Compress 阶段的简化版）

        Args:
            packets: 候选信息包列表

        Returns:
            (选中的信息包列表, 是否被压缩)
        """
        if not packets:
            return [], False

        # 1. 过滤低于最低相关性阈值的信息包
        filtered = [p for p in packets if p.relevance_score >= self.config.min_relevance]

        # 2. 按相关性降序排序
        filtered.sort(key=lambda p: p.relevance_score, reverse=True)

        # 3. 计算可用 token 预算（扣除系统指令预留）
        available_tokens = int(self.config.max_tokens * (1 - self.config.reserve_ratio))

        # 4. 贪心选择：按相关性从高到低填充，直到达到 token 上限
        selected = []
        current_tokens = 0
        compressed = False

        for packet in filtered:
            if current_tokens + packet.token_count <= available_tokens:
                selected.append(packet)
                current_tokens += packet.token_count
            else:
                # Token 预算已满，标记为被压缩
                compressed = True
                break

        # 5. 如果所有信息包都被选中但仍超限（理论上不会，因为贪心选择会停），
        # 或者没有任何信息包被选中（所有都低于阈值），处理边界情况
        if not selected and filtered:
            # 所有信息包都超过 token 预算？选相关性最高的一个并截断
            selected.append(filtered[0])
            compressed = True

        return selected, compressed

    # ═══════════════════════════════════════════════════════════
    # Structure：结构化组织（参考教程 9.3.1 分区组织 + 9.3.3 (3)）
    # ═══════════════════════════════════════════════════════════

    def _structure(
        self,
        query: str,
        selected_packets: List[ContextPacket],
    ) -> tuple[str, List[str]]:
        """
        Structure 阶段：按分区模板组织成结构化上下文。

        使用 RAG 场景化模板：[Role & Policies] / [Evidence] / [Task] / [Output]
        （教程用 [Role]/[Task]/[State]/[Evidence]/[Context]/[Output]，
        我们简化为 RAG 场景需要的分区）

        Args:
            query: 用户问题
            selected_packets: 选中的信息包列表

        Returns:
            (结构化上下文字符串, 选中的 sources 列表)
        """
        if not selected_packets:
            # 没有选中任何信息包，返回空证据
            evidence = "（未检索到相关参考资料）"
            sources = []
        else:
            # 按原始 rank 排序（Select 阶段按相关性排序了，这里恢复原始顺序）
            selected_packets.sort(key=lambda p: p.metadata.get("rank", 0))

            # 组装 Evidence 部分
            evidence_parts = []
            sources = []

            for i, packet in enumerate(selected_packets, 1):
                rank = packet.metadata.get("rank", i)
                filename = packet.metadata.get("filename", "未知")
                page_num = packet.metadata.get("page_num", "?")
                heading_path = packet.metadata.get("heading_path", "")

                # 来源标签
                source_label = f"参考资料 {i}：{filename}, 第 {page_num} 页"
                if heading_path:
                    # 标题路径只取最后两级
                    heading_parts = heading_path.split(" > ")
                    if len(heading_parts) > 2:
                        short_heading = " > ".join(heading_parts[-2:])
                    else:
                        short_heading = heading_path
                    source_label += f"（{short_heading}）"

                evidence_parts.append(f"{source_label}\n{packet.content}")
                sources.append(f"[{i}] {filename}, 第 {page_num} 页" + (f"（{short_heading}）" if heading_path else ""))

            evidence = "\n\n".join(evidence_parts)

        # 用模板组装完整上下文
        context = RAG_CONTEXT_TEMPLATE.format(
            evidence=evidence,
            query=query,
        )

        return context, sources

    # ═══════════════════════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════════════════════

    def _count_tokens(self, text: str) -> int:
        """
        计算文本的 token 数。

        优先使用 BGE-m3 tokenizer 精确计算，失败时回退到字符数估算。

        Args:
            text: 文本

        Returns:
            token 数
        """
        if not text:
            return 0
        if self.tokenizer:
            # add_special_tokens=False：不计算 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> [SEP] 等特殊标记
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        else:
            # Fallback：字符数近似（中文约 1 字符=1 token，英文约 4 字符=1 token，平均约 2）
            chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            english_words = len([w for w in text.split() if any(c.isascii() and c.isalpha() for c in w)])
            return int(chinese_chars + english_words * 1.3) + 10


# ── 测试：python -m app.rag.context_builder ──
if __name__ == "__main__":
    from app.rag.retriever import Retriever
    from app.rag.vector_store import VectorStore

    print("=" * 60)
    print("PlantGenome Agent - ContextBuilder 测试")
    print("=" * 60)

    # 初始化
    print("\n[初始化] 加载 VectorStore + Retriever + ContextBuilder...")
    vector_store = VectorStore()
    retriever = Retriever(vector_store)
    context_builder = ContextBuilder(ContextConfig(max_tokens=2000, reserve_ratio=0.2))
    print(f"  ✅ VectorStore 文档数: {vector_store.count()}")
    print(f"  ✅ ContextConfig: max_tokens={context_builder.config.max_tokens}, "
          f"reserve_ratio={context_builder.config.reserve_ratio}")

    # 测试问题
    test_queries = [
        "PAML 中的 omega 值是什么意思？",
        "MAFFT 有哪些比对算法？",
    ]

    for query in test_queries:
        print(f"\n{'─' * 60}")
        print(f"查询: {query}")
        print(f"{'─' * 60}")

        # 1. Retriever 检索
        retrieve_result = retriever.retrieve(query)
        print(f"  检索到 {retrieve_result['total_count']} 个结果")

        # 2. ContextBuilder 构建上下文
        build_result = context_builder.build(query, retrieve_result)
        print(f"  选中 {len(build_result['selected_packets'])} 个信息包")
        print(f"  总 token 数: {build_result['total_tokens']}")
        print(f"  是否被压缩: {build_result['compressed']}")
        print(f"  上下文长度: {len(build_result['context'])} 字符")

        print(f"\n  选中的 sources:")
        for s in build_result["selected_sources"]:
            print(f"    {s}")

        print(f"\n  上下文前 500 字符:")
        print(f"    {build_result['context'][:500]}...")

    print("\n" + "=" * 60)
    print("ContextBuilder 测试完成。")
    print("=" * 60)
