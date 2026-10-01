"""
工具 1：search_pdf_knowledge（PDF 知识库检索）

支持三种可配置检索模式（RETRIEVAL_MODE），可随时回滚：
  - dense_only    ：纯 BGE-m3 向量召回 + distance 门控；
  - candidate_rrf ：Chroma 先召回 top_k*5 候选，在候选内构建 BM25 并用 RRF
                    融合（旧线上默认，BM25 被候选集框死）；
  - global_rrf    ：用户全库 Dense / BM25 各自独立 TopN 召回，按共同 chunk_id
                    去重后 RRF 融合，再做分通道证据门控（本 MVP 新增）。

证据门控（确定性，不依赖 LLM）：
  - Dense 证据：Chroma distance <= rag_max_distance；
  - Sparse 证据：BM25 得分 > 0（存在有效查询词匹配）；
  - 满足任一即进入证据集合；无证据时 has_result=False，由 answer_node 拒答。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import get_settings
from app.rag.bm25_snapshot import get_snapshot_manager
from app.rag.bm25_store import BM25Index, rrf_fuse
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

_VALID_MODES = ("dense_only", "candidate_rrf", "global_rrf")


def _dense_recall(store, query: str, n: int, user_id: Optional[int]) -> List[dict]:
    """统一 Dense 召回：user_id 存在时按 metadata 过滤，否则不过滤。"""
    flt = {"user_id": user_id} if user_id is not None else None
    return store.search(query, top_k=n, filter_metadata=flt)


# ═══════════════════════════════════════════════════════════════
# 统一 chunk 表示 + context / sources 构造
# ═══════════════════════════════════════════════════════════════

def _assemble_item(
    chunk_id: str,
    dense_result: Optional[dict],
    bm25_score: Optional[float],
    snap,
    user_id: Optional[int],
) -> Dict[str, Any]:
    """
    将一路或两路命中合并为统一 chunk item。

    BM25-only 的 chunk 不在 Dense 结果中，需从快照取正文与元数据。
    """
    if dense_result is not None:
        md = dense_result["metadata"] or {}
        return {
            "chunk_id": chunk_id,
            "text": dense_result["text"],
            "filename": md.get("filename", "未知文件"),
            "page_num": md.get("page_num", 0),
            "section": md.get("section", ""),
            "chunk_index": md.get("chunk_index", 0),
            "document_id": md.get("document_id"),
            "user_id": md.get("user_id", user_id),
            "distance": dense_result["distance"],
            "bm25_score": bm25_score,
            "from_dense": True,
            "from_sparse": bm25_score is not None,
        }

    # BM25-only：正文/元数据来自全库快照
    info = snap.chunks[chunk_id]
    return {
        "chunk_id": chunk_id,
        "text": info["text"],
        "filename": info["filename"] or "未知文件",
        "page_num": info["page_num"] or 0,
        "section": info["section"] or "",
        "chunk_index": info["chunk_index"],
        "document_id": info["document_id"],
        "user_id": user_id,
        "distance": None,
        "bm25_score": bm25_score,
        "from_dense": False,
        "from_sparse": True,
    }


def _build_context(items: List[dict]) -> str:
    parts = []
    for i, it in enumerate(items, 1):
        loc = f"{it['filename']} (p.{it['page_num']})"
        if it["section"]:
            loc += f" · {it['section']}"
        parts.append(f"[{i}] {loc}\n{it['text']}")
    return "\n\n".join(parts)


def _build_sources(items: List[dict]) -> List[dict]:
    sources = []
    for i, it in enumerate(items, 1):
        sources.append({
            "index": i,
            "filename": it["filename"],
            "page_num": it["page_num"],
            "section": it["section"],
            "snippet": it["text"][:150].replace("\n", " "),
            "chunk_id": it["chunk_id"],
        })
    return sources


# ═══════════════════════════════════════════════════════════════
# 模式一：dense_only
# ═══════════════════════════════════════════════════════════════

def _dense_only_search(
    query: str, user_id: Optional[int], top_k: int, settings
) -> Tuple[List[dict], Optional[str]]:
    store = VectorStore()
    results = _dense_recall(store, query, top_k, user_id)
    if not results:
        return [], "no_results"
    gated = [r for r in results if r["distance"] <= settings.rag_max_distance]
    if not gated:
        return [], "low_relevance_or_empty_content"
    items = [_assemble_item(r["chunk_id"], r, None, None, user_id)
             for r in gated[:top_k]]
    return items, None


# ═══════════════════════════════════════════════════════════════
# 模式二：candidate_rrf（Dense 候选集内 BM25 重排）
# ═══════════════════════════════════════════════════════════════

def _candidate_rrf_search(
    query: str, user_id: Optional[int], top_k: int, settings
) -> Tuple[List[dict], Optional[str]]:
    store = VectorStore()
    results = _dense_recall(store, query, top_k * 5, user_id)
    if not results:
        return [], "no_results"

    # distance 门控先过滤
    gated = [r for r in results if r["distance"] <= settings.rag_max_distance]
    if not gated:
        return [], "low_relevance_or_empty_content"

    # 在候选内构建 BM25
    bm25 = BM25Index([r["text"] for r in gated])
    bm25_hits = bm25.search(query, top_k=min(len(gated), top_k * 2))

    dense_ids = [r["chunk_id"] for r in gated]
    bm25_ids = [gated[pos]["chunk_id"] for pos, _ in bm25_hits]

    # rrf_fuse 对元素类型无要求，直接传 chunk_id 列表（复用，不另写算法）
    fused_ids = rrf_fuse(dense_ids, bm25_ids, k=settings.rrf_k, top_k=top_k)

    dense_map = {r["chunk_id"]: r for r in gated}
    bm25_map = {gated[pos]["chunk_id"]: score for pos, score in bm25_hits}
    items = [
        _assemble_item(fid, dense_map.get(fid), bm25_map.get(fid), None, user_id)
        for fid in fused_ids
    ]
    return items, None


# ═══════════════════════════════════════════════════════════════
# 模式三：global_rrf（全库 Dense / BM25 独立召回 + RRF）
# ═══════════════════════════════════════════════════════════════

def _global_rrf_search(
    query: str, user_id: Optional[int], top_k: int, settings
) -> Tuple[List[dict], str, Optional[str], Optional[str]]:
    """
    Returns:
        (items, actual_mode, evidence_reason, degrade_reason)。
        快照构建失败且无旧快照时降级 candidate_rrf；用户无语料时 Sparse 为空、
        仍使用 Dense 结果（mode 保持 global_rrf）。
    """
    store = VectorStore()

    # 1) Dense 独立召回（不依赖 BM25）
    dense_results = _dense_recall(
        store, query, settings.dense_top_n, user_id
    )
    dense_map = {r["chunk_id"]: r for r in dense_results}

    # 2) Sparse 独立召回（用户全库 BM25 快照）
    sparse_map: Dict[str, float] = {}
    snap = None
    if user_id is not None:
        try:
            snap = get_snapshot_manager().get_snapshot(user_id)
            if snap is not None:
                for cid, score in snap.search(query, top_n=settings.sparse_top_n):
                    sparse_map[cid] = score
        except Exception as exc:
            # 无旧快照可用：降级 candidate_rrf，保证检索不中断
            logger.warning("global_rrf 快照不可用，降级 candidate_rrf: %s", exc)
            items, ev_reason = _candidate_rrf_search(
                query, user_id, top_k, settings
            )
            return items, "candidate_rrf", ev_reason, "bm25_snapshot_failed"

    if not dense_results and not sparse_map:
        return [], "global_rrf", "no_results", None

    # 3) RRF 融合（按共同 chunk_id 去重）
    fused_ids = rrf_fuse(
        [r["chunk_id"] for r in dense_results],
        list(sparse_map.keys()),
        k=settings.rrf_k,
    )

    # 4) 分通道证据门控 + 组装
    items: List[dict] = []
    for fid in fused_ids:
        d = dense_map.get(fid)
        bm25 = sparse_map.get(fid)
        dense_ok = d is not None and d["distance"] <= settings.rag_max_distance
        sparse_ok = bm25 is not None  # search 已过滤 0 分
        if not (dense_ok or sparse_ok):
            continue
        items.append(_assemble_item(fid, d, bm25, snap, user_id))
        if len(items) >= top_k:
            break

    if not items:
        return [], "global_rrf", "low_relevance_or_empty_content", None
    return items, "global_rrf", None, None


# ═══════════════════════════════════════════════════════════════
# 主工具函数
# ═══════════════════════════════════════════════════════════════

def search_pdf_knowledge(
    query: str,
    top_k: Optional[int] = None,
    user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    检索用户知识库，返回给 Agent / RAG 生成使用。

    Args:
        query: 用户问题或检索关键词
        top_k: 返回片段数量，默认取 settings.retrieve_top_k
        user_id: 用户 ID（由 JWT 认证态注入），None 时不做用户过滤

    Returns:
        dict: context, sources, count, has_result, evidence_status,
              best_distance, retrieval_mode
    """
    settings = get_settings()
    top_k = top_k or settings.retrieve_top_k

    mode = (settings.retrieval_mode or "").strip().lower()
    if mode not in _VALID_MODES:
        logger.warning("非法 RETRIEVAL_MODE=%s，保守回退 candidate_rrf", mode)
        mode = "candidate_rrf"

    actual_mode = mode
    degrade_reason: Optional[str] = None
    evidence_reason: Optional[str] = None
    try:
        if mode == "dense_only":
            items, evidence_reason = _dense_only_search(
                query, user_id, top_k, settings
            )
        elif mode == "global_rrf":
            (items, actual_mode, evidence_reason,
             degrade_reason) = _global_rrf_search(query, user_id, top_k, settings)
        else:
            items, evidence_reason = _candidate_rrf_search(
                query, user_id, top_k, settings
            )
    except Exception as exc:
        # 整体异常：最后兜底 dense_only，避免检索接口直接崩溃
        logger.warning("检索模式 %s 异常(%s)，兜底 dense_only", mode, exc)
        try:
            items, evidence_reason = _dense_only_search(
                query, user_id, top_k, settings
            )
            actual_mode = "dense_only"
            degrade_reason = f"{mode}_failed"
        except Exception as inner:
            logger.error("dense_only 兜底仍失败: %s", inner)
            items = []

    context = _build_context(items)
    sources = _build_sources(items)
    distances = [it["distance"] for it in items if it["distance"] is not None]
    best_distance = min(distances) if distances else None

    has_result = len(items) > 0
    result = {
        "context": context,
        "sources": sources,
        "count": len(items),
        "has_result": has_result,
        "evidence_status": "sufficient" if has_result else "insufficient",
        "evidence_reason": None if has_result else evidence_reason,
        "best_distance": best_distance,
        "retrieval_mode": actual_mode,
    }
    if degrade_reason:
        result["degrade_reason"] = degrade_reason
    return result


# ═══════════════════════════════════════════════════════════════
# 测试入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    print("=" * 70)
    print("工具测试：search_pdf_knowledge")
    print("=" * 70)

    test_queries = [
        "PAML 中如何设置分支模型进行选择压力分析？",
        "什么是 dN/dS 比值？怎么判断正选择？",
        "MAFFT 多序列比对有哪些算法？",
    ]

    for q in test_queries:
        print(f"\n{'─' * 70}")
        print(f"查询: {q}")
        result = search_pdf_knowledge(q, top_k=3, user_id=1)
        print(f"模式: {result['retrieval_mode']} | 命中: {result['count']} | "
              f"best_distance: {result['best_distance']}")
        for s in result["sources"]:
            print(f"  [{s['index']}] {s['filename']} (p.{s['page_num']})")

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
