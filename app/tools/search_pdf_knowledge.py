"""
工具 1：search_pdf_knowledge（PDF 文献检索工具）

封装第一周的 RAG 检索能力，供 Agent 调用。
检索 PDF 文献知识库，返回相关片段 + sources。

参考 Hello Agents 第 4 章工具调用 + 第 8 章 RAG 检索。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from app.rag.vector_store import VectorStore


def search_pdf_knowledge(
    query: str,
    top_k: int = 3,
    user_id: Optional[int] = None,
) -> Dict:
    """
    检索 PDF 文献知识库，返回相关片段 + sources。

    这是 Agent 的核心工具之一，用于回答文献、方法、概念、生信工具用法等问题。

    Args:
        query: 检索关键词或问题
        top_k: 返回最相关的 top_k 个文档块，默认 3
        user_id: 当前登录用户 ID，用于用户级数据隔离——只检索该用户上传的文档；
                 None 时不过滤（独立脚本/测试场景）

    Returns:
        dict，包含：
        - context: str，拼接后的检索结果文本（带编号和来源）
        - sources: list[dict]，来源列表，每个包含 filename、page、snippet、heading_path
        - count: int，检索到的文档块数量
        - has_result: bool，是否检索到结果

    失败情况：
        - 向量库加载失败
        - 检索结果为空（返回 has_result=False）
    """
    # 初始化向量库（每次调用创建新实例，避免状态污染）
    vs = VectorStore()

    # 用户级数据隔离：只检索当前用户的文档（user_id 来自 JWT 认证，非 LLM 输出）
    filter_metadata = {"user_id": user_id} if user_id is not None else None

    # 执行检索
    results = vs.search(query, top_k=top_k, filter_metadata=filter_metadata)

    if not results:
        return {
            "context": "",
            "sources": [],
            "count": 0,
            "has_result": False,
        }

    # 格式化检索结果
    context_parts = []
    sources = []

    for i, r in enumerate(results):
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
            "similarity": round(1 - distance, 4),  # 余弦距离转相似度
        }
        sources.append(source_info)

    return {
        "context": "\n\n".join(context_parts),
        "sources": sources,
        "count": len(results),
        "has_result": True,
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
