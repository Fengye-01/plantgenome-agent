"""
工具 1：search_pdf_knowledge（PDF 文献检索工具）

封装第一周的 RAG 检索能力，供 Agent 调用。
检索 PDF 文献知识库，返回相关片段 + sources。

参考 Hello Agents 第 4 章工具调用 + 第 8 章 RAG 检索。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from app.core.config import settings
from app.rag.vector_store import VectorStore
from app.rag.bm25_store import hybrid_search


def search_pdf_knowledge(
    query: str,
    top_k: int = 3,
    user_id: Optional[int] = None,
    enable_hybrid: bool = True,
) -> Dict:
    """
    检索 PDF 文献知识库，返回相关片段 + sources。

    这是 Agent 的核心工具之一，用于回答文献、方法、概念、生信工具用法等问题。

    检索策略：
    - 混合检索（默认开启）：向量检索召回 top_k*5 候选 → BM25 关键词重排 → RRF 融合
    - 向量检索擅长语义匹配，BM25 擅长专业术语精确匹配，融合后兼顾两者
    - 可通过 enable_hybrid=False 回退到纯向量检索

    Args:
        query: 检索关键词或问题
        top_k: 返回最相关的 top_k 个文档块，默认 3
        user_id: 当前登录用户 ID，用于用户级数据隔离——只检索该用户上传的文档；
                 None 时不过滤（独立脚本/测试场景）
        enable_hybrid: 是否启用混合检索（BM25 + 向量 + RRF），默认 True

    Returns:
        dict，包含：
        - context: str，拼接后的检索结果文本（带编号和来源）
        - sources: list[dict]，来源列表，每个包含 filename、page、snippet、heading_path
        - count: int，检索到的文档块数量
        - has_result: bool，是否检索到结果
        - retrieval_mode: str，检索模式（"hybrid" 或 "vector_only"）

    失败情况：
        - 向量库加载失败
        - 检索结果为空（返回 has_result=False）
    """
    # 初始化向量库（每次调用创建新实例，避免状态污染）
    vs = VectorStore()

    # 用户级数据隔离：只检索当前用户的文档（user_id 来自 JWT 认证，非 LLM 输出）
    filter_metadata = {"user_id": user_id} if user_id is not None else None

    # 向量检索：混合检索时召回更多候选（top_k * 5），供 BM25 重排
    recall_k = top_k * 5 if enable_hybrid else top_k
    raw_results = vs.search(query, top_k=recall_k, filter_metadata=filter_metadata)

    retrieval_mode = "vector_only"

    if enable_hybrid and len(raw_results) > top_k:
        # 混合检索：BM25 + 向量 + RRF 融合
        fused_results = hybrid_search(query, raw_results, top_k=top_k)
        if fused_results:
            raw_results = fused_results
            retrieval_mode = "hybrid"

    if not raw_results:
        return {
            "context": "",
            "sources": [],
            "count": 0,
            "has_result": False,
            "evidence_status": "insufficient",
            "evidence_reason": "no_results",
            "best_distance": None,
            "distance_threshold": settings.rag_max_distance,
            "retrieval_mode": retrieval_mode,
        }

    best_distance = min(
        (r.get("distance", float("inf")) for r in raw_results),
        default=float("inf"),
    )
    relevant_results = [
        r
        for r in raw_results
        if r.get("distance", float("inf")) <= settings.rag_max_distance
        and bool((r.get("text") or r.get("content", "")).strip())
    ]

    if not relevant_results:
        return {
            "context": "",
            "sources": [],
            "count": 0,
            "has_result": False,
            "evidence_status": "insufficient",
            "evidence_reason": "low_relevance_or_empty_content",
            "best_distance": round(best_distance, 4),
            "distance_threshold": settings.rag_max_distance,
            "retrieval_mode": retrieval_mode,
        }

    raw_results = relevant_results

    # 格式化检索结果
    context_parts = []
    sources = []

    for i, r in enumerate(raw_results):
        # 检索结果正文字段：vector_store 返回 "text"（兼容历史命名 "content"）
        content = r.get("text") or r.get("content", "")
        metadata = r.get("metadata", {})
        distance = r.get("distance", 0.0)

        # 从 metadata 提取来源信息
        filename = metadata.get("filename", "未知文件")
        source = metadata.get("source", "")
        heading_path = metadata.get("heading_path", "")
        page_num = metadata.get("page_num", "")

        # 构建 context 片段（带编号和来源）
        context_part = f"[{i+1}] 来源: {filename}"
        if page_num:
            context_part += f" (第 {page_num} 页)"
        if heading_path:
            context_part += f" [{heading_path}]"
        context_part += f"\n{content}"
        context_parts.append(context_part)

        # 构建 source（供前端展示和 answer_node 使用）
        source_info = {
            "index": i + 1,
            "filename": filename,
            "page_num": page_num,
            "heading_path": heading_path,
            "snippet": content[:200] + "..." if len(content) > 200 else content,
            "distance": round(distance, 4),
            "similarity": round(max(0.0, min(1.0, 1 - distance / 2)), 4),
        }
        sources.append(source_info)

    return {
        "context": "\n\n".join(context_parts),
        "sources": sources,
        "count": len(raw_results),
        "has_result": True,
        "evidence_status": "sufficient",
        "evidence_reason": None,
        "best_distance": round(best_distance, 4),
        "distance_threshold": settings.rag_max_distance,
        "retrieval_mode": retrieval_mode,
    }


# ═══════════════════════════════════════════════════════════════
# 测试入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # 项目根路径导入
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    print("=" * 70)
    print("工具测试：search_pdf_knowledge")
    print("=" * 70)

    test_queries = [
        "PAML omega 值是什么意思？",
        "OrthoFinder 输入格式",
        "MAFFT 比对算法",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─' * 70}")
        print(f"测试 {i}: {query}")
        print(f"{'─' * 70}")

        result = search_pdf_knowledge(query, top_k=3)

        print(f"检索到 {result['count']} 个文档块")
        print(f"has_result: {result['has_result']}")
        print(f"\nContext（前 500 字符）:")
        print(result["context"][:500])
        print(f"\nSources:")
        for s in result["sources"]:
            print(f"  [{s['index']}] {s['filename']} (p.{s['page_num']}) - 相似度: {s['similarity']}")

    print("\n" + "=" * 70)
    print("所有测试完成！")
    print("=" * 70)
