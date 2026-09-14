"""
Retriever 检索封装模块（RAG 流水线第 6 步：检索封装）

对应 Hello Agents 第 8 章 RAG 的「Retrieval（检索）」阶段。
职责：在 VectorStore.search 基础上增加 query 预处理、结果格式化、sources 组装、空结果处理，
     为 LLM 生成提供统一格式的检索结果。

为什么需要再封装一层 Retriever，而不是直接用 VectorStore.search？
1. 解耦：LLM 生成层不依赖 VectorStore 内部格式，以后换向量数据库不需要改 LLM 层
2. query 预处理：统一处理用户输入的多余空白、特殊字符
3. 结果格式化：增加 rank、similarity 等字段
4. sources 组装：把 metadata 组装成用户友好的引用格式
5. 空结果处理：检索结果为空或 distance 过大时返回明确提示，避免 LLM 幻觉
"""
from __future__ import annotations

import os
import re
from typing import List, Dict, Optional

from dotenv import load_dotenv

load_dotenv()


class Retriever:
    """
    检索封装器。

    输入：用户 query（字符串）
    输出：格式化的检索结果 Dict：
        {
            "query": str,                    # 预处理后的 query
            "results": List[Dict],           # 格式化后的检索结果（含 rank、similarity）
            "sources": List[str],             # sources 列表（用户友好格式）
            "has_relevant": bool,             # 是否有相关结果（distance < threshold）
            "total_count": int,               # 检索到的结果数量
        }
    """

    def __init__(
        self,
        vector_store,
        top_k: int = None,
        distance_threshold: float = 1.5,
        enable_mqe: bool = False,
        enable_hyde: bool = False,
    ):
        """
        Args:
            vector_store: VectorStore 实例
            top_k: 默认返回数量，默认从 .env 读 RETRIEVE_TOP_K（3）
            distance_threshold: 余弦距离阈值，超过此值认为不相关（默认 1.5，范围 [0, 2]）
            enable_mqe: 是否启用多查询扩展（MQE，Multi-Query Expansion）。
                        原理：用 LLM 生成多个语义等价查询，并行检索后合并去重。
                        当前为预留接口，暂未实现，开启时回退到基础检索并打印提示。
                        对应 Hello Agents 第 8 章 8.3.5 高级检索策略。
            enable_hyde: 是否启用假设文档嵌入（HyDE，Hypothetical Document Embedding）。
                         原理：用 LLM 先生成假设性答案文档，再用该文档向量检索。
                         当前为预留接口，暂未实现，开启时回退到基础检索并打印提示。
                         对应 Hello Agents 第 8 章 8.3.5 高级检索策略。
        """
        self.vector_store = vector_store
        self.top_k = top_k or int(os.getenv("RETRIEVE_TOP_K", "3"))
        self.distance_threshold = distance_threshold
        self.enable_mqe = enable_mqe
        self.enable_hyde = enable_hyde

        # 预留接口提示：MQE/HyDE 暂未实现，MVP 完成后作为 V2 优化项
        if enable_mqe or enable_hyde:
            enabled = []
            if enable_mqe:
                enabled.append("MQE")
            if enable_hyde:
                enabled.append("HyDE")
            print(f"[Retriever] 注意：{' + '.join(enabled)} 为预留接口，当前暂未实现，"
                  f"已回退到基础向量检索。MVP 完成后将作为 V2 优化项实现。")

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filter_metadata: Dict = None,
        enable_mqe: bool = None,
        enable_hyde: bool = None,
    ) -> Dict:
        """
        检索主入口。

        流程：
          1. query 预处理
          2. （预留）MQE 多查询扩展 / HyDE 假设文档嵌入
          3. 调用 VectorStore 检索
          4. 结果格式化（增加 rank、similarity）
          5. sources 组装
          6. 判断是否有相关结果（distance < threshold）

        Args:
            query: 用户问题
            top_k: 返回数量（可选，默认用 self.top_k）
            filter_metadata: 按 metadata 过滤（可选）
            enable_mqe: 单次调用是否启用 MQE（可选，默认用 self.enable_mqe）
            enable_hyde: 单次调用是否启用 HyDE（可选，默认用 self.enable_hyde）

        Returns:
            格式化的检索结果 Dict
        """
        # 确定本次调用的高级检索策略（单次参数优先于初始化参数）
        use_mqe = enable_mqe if enable_mqe is not None else self.enable_mqe
        use_hyde = enable_hyde if enable_hyde is not None else self.enable_hyde

        # 1. query 预处理
        cleaned_query = self._preprocess_query(query)

        # 2. （预留）高级检索策略
        # MQE：用 LLM 生成多个语义等价查询，并行检索后合并去重
        # HyDE：用 LLM 先生成假设性答案文档，再用该文档向量检索
        # 当前暂未实现，回退到基础检索。MVP 完成后作为 V2 优化项。
        if use_mqe or use_hyde:
            # 这里未来会插入 MQE/HyDE 逻辑
            pass

        # 3. 调用 VectorStore 检索
        raw_results = self.vector_store.search(
            cleaned_query,
            top_k=top_k or self.top_k,
            filter_metadata=filter_metadata,
        )

        # 3. 结果格式化
        formatted_results = self._format_results(raw_results)

        # 4. sources 组装
        sources = self._build_sources(formatted_results)

        # 5. 判断是否有相关结果（至少有一个结果的 distance < threshold）
        has_relevant = any(
            r["distance"] < self.distance_threshold for r in formatted_results
        )

        return {
            "query": cleaned_query,
            "results": formatted_results,
            "sources": sources,
            "has_relevant": has_relevant,
            "total_count": len(formatted_results),
        }

    def _preprocess_query(self, query: str) -> str:
        """
        query 预处理：去除多余空白、统一标点、去除首尾空白。

        Args:
            query: 原始用户输入

        Returns:
            预处理后的 query
        """
        if not query:
            return ""

        # 去除首尾空白
        query = query.strip()

        # 合并连续空白（多个空格/制表符/换行合并为一个空格）
        query = re.sub(r'\s+', ' ', query)

        # 统一全角空格为半角
        query = query.replace('\u3000', ' ')

        # 去除控制字符（除了常见的可打印字符）
        query = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', query)

        return query

    def _format_results(self, raw_results: List[Dict]) -> List[Dict]:
        """
        格式化检索结果：增加 rank、similarity 字段，统一字段名。

        Args:
            raw_results: VectorStore.search() 的原始输出

        Returns:
            格式化后的检索结果列表
        """
        formatted = []
        for i, r in enumerate(raw_results):
            # 余弦距离转相似度（distance ∈ [0, 2] → similarity ∈ [0, 1]）
            # distance=0 → similarity=1（完全相同）
            # distance=2 → similarity=0（完全相反）
            similarity = 1.0 - r["distance"] / 2.0

            formatted.append({
                "rank": i + 1,
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "metadata": r["metadata"],
                "distance": round(r["distance"], 4),
                "similarity": round(similarity, 4),
                # 便捷字段：从 metadata 中提取常用信息
                "filename": r["metadata"].get("filename", "未知"),
                "page_num": r["metadata"].get("page_num", "?"),
                "source": r["metadata"].get("source", "未知来源"),
                "heading_path": r["metadata"].get("heading_path", ""),
            })

        return formatted

    def _build_sources(self, results: List[Dict]) -> List[str]:
        """
        组装 sources 列表（用户友好格式），供 LLM 生成回答时引用。

        格式示例：
            "[1] PAML 4.pdf, 第 3 页（## Methods > ### Dataset）"
            "[2] OrthoFinder.pdf, 第 5 页"

        Args:
            results: 格式化后的检索结果

        Returns:
            sources 字符串列表
        """
        sources = []
        for r in results:
            # 基础来源：[rank] filename, 第 X 页
            source = f"[{r['rank']}] {r['filename']}, 第 {r['page_num']} 页"

            # 如果有标题路径，追加在括号里
            if r["heading_path"]:
                # 标题路径可能很长，只取最后两级
                heading_parts = r["heading_path"].split(" > ")
                if len(heading_parts) > 2:
                    short_heading = " > ".join(heading_parts[-2:])
                else:
                    short_heading = r["heading_path"]
                source += f"（{short_heading}）"

            sources.append(source)

        return sources

    def get_context_for_llm(self, retrieve_result: Dict, max_context_chars: int = 3000) -> str:
        """
        把检索结果组装成 LLM 能用的上下文文本（供 D5 RAG 生成使用）。

        格式：
            【参考资料 1】PAML 4.pdf, 第 3 页
            （内容）...

            【参考资料 2】...

        Args:
            retrieve_result: retrieve() 的返回结果
            max_context_chars: 最大上下文字符数（避免超过 LLM 上下文窗口）

        Returns:
            组装好的上下文字符串
        """
        if not retrieve_result["has_relevant"] or not retrieve_result["results"]:
            return "（未检索到相关参考资料）"

        context_parts = []
        total_chars = 0

        for r in retrieve_result["results"]:
            # 只包含 distance < threshold 的相关结果
            if r["distance"] >= self.distance_threshold:
                continue

            source_label = f"【参考资料 {r['rank']}】{r['source']}"
            if r["heading_path"]:
                source_label += f"（{r['heading_path']}）"

            content = r["text"]

            # 控制总长度
            if total_chars + len(source_label) + len(content) > max_context_chars:
                # 截断当前内容
                remaining = max_context_chars - total_chars - len(source_label) - 10
                if remaining > 100:
                    content = content[:remaining] + "..."
                else:
                    break

            context_parts.append(f"{source_label}\n{content}")
            total_chars += len(source_label) + len(content)

        return "\n\n".join(context_parts)


# ── 测试：python -m app.rag.retriever ──
if __name__ == "__main__":
    from app.rag.vector_store import VectorStore

    print("=" * 60)
    print("PlantGenome Agent - Retriever 检索封装测试")
    print("=" * 60)

    # 初始化
    print("\n[初始化] 加载 VectorStore + Retriever...")
    vector_store = VectorStore()
    retriever = Retriever(vector_store, top_k=3, distance_threshold=1.5)
    print(f"  ✅ VectorStore 文档数: {vector_store.count()}")
    print(f"  ✅ Retriever top_k: {retriever.top_k}")
    print(f"  ✅ distance_threshold: {retriever.distance_threshold}")

    # 测试问题
    test_queries = [
        "PAML 中的 omega 值是什么意思？",
        "OrthoFinder 的输入格式是什么？",
        "MAFFT 有哪些比对算法？",
        "这是一个完全不相关的问题：今天天气怎么样？",  # 测试空结果处理
    ]

    for query in test_queries:
        print(f"\n{'─' * 60}")
        print(f"查询: {query}")
        print(f"{'─' * 60}")

        result = retriever.retrieve(query)

        print(f"  预处理后 query: {result['query']}")
        print(f"  检索结果数: {result['total_count']}")
        print(f"  有相关结果: {result['has_relevant']}")

        print(f"\n  Sources:")
        for source in result["sources"]:
            print(f"    {source}")

        print(f"\n  Top-1 详情:")
        if result["results"]:
            r = result["results"][0]
            print(f"    rank: {r['rank']}")
            print(f"    distance: {r['distance']}")
            print(f"    similarity: {r['similarity']}")
            print(f"    filename: {r['filename']}")
            print(f"    page: {r['page_num']}")
            print(f"    heading_path: {r['heading_path'] or '(无)'}")
            print(f"    内容前 200 字符: {r['text'][:200]}...")

        # 测试 LLM 上下文组装
        context = retriever.get_context_for_llm(result, max_context_chars=1000)
        print(f"\n  LLM 上下文（前 300 字符）:")
        print(f"    {context[:300]}...")

    print("\n" + "=" * 60)
    print("Retriever 检索封装测试完成。")
    print("=" * 60)
