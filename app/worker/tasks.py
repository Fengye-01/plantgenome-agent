"""
ARQ Worker 异步任务定义。

包含：
- process_pdf_document: PDF 文档解析入库异步任务
- process_pubmed_search: PubMed 文献检索下载 + 批量入库异步任务
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional, Tuple

from arq import Retry

from app.core.database import get_session
from app.models import Document, DocumentChunk, Task
from app.rag.bm25_snapshot import bump_bm25_version
from app.rag.pdf_parser import PDFParser
from app.rag.text_cleaner import TextCleaner
from app.rag.chunker import Chunker
from app.rag.vector_store import VectorStore


def _chunk_content(chunk: dict) -> str:
    """Normalize the Chunker output while retaining legacy compatibility."""
    return str(chunk.get("text") or chunk.get("content") or "")


def _chunk_page_number(chunk: dict) -> int | None:
    metadata = chunk.get("metadata") or {}
    return metadata.get("page_start") or chunk.get("page_num")


def _chunk_section(chunk: dict) -> str | None:
    heading_path = (chunk.get("metadata") or {}).get("heading_path")
    if isinstance(heading_path, list):
        heading_path = " > ".join(str(item) for item in heading_path)
    if not heading_path:
        return None
    return str(heading_path)[:500]


def _ingest_pdf_to_db(
    document_id: int,
    user_id: int,
    db,
    task: Optional[Task] = None,
) -> Tuple[int, int]:
    """
    共用的 PDF 入库核心逻辑：解析 → 清洗 → 分块 → 向量化 → 保存元数据。

    被 process_pdf_document 和 process_pubmed_search 共用。

    Args:
        document_id: 文档 ID
        user_id: 用户 ID
        db: 数据库 session
        task: 关联的 Task 对象（用于更新进度，可为 None）

    Returns:
        (chunk_count, page_count): 分块数量和页数

    Raises:
        各种解析/入库异常由调用方处理
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise ValueError(f"Document {document_id} 不存在")

    # 1. PDF 解析
    print(f"[Ingest] 步骤 1/5: 解析 PDF - {document.filename}")
    parser = PDFParser()
    pages = parser.parse_to_markdown(document.file_path)
    if task:
        task.progress = 30
        db.commit()

    # 2. 文本清洗
    print(f"[Ingest] 步骤 2/5: 清洗文本 - {len(pages)} 页")
    cleaner = TextCleaner()
    cleaned_pages = cleaner.clean(pages)
    if task:
        task.progress = 50
        db.commit()

    # 3. 智能分块
    print(f"[Ingest] 步骤 3/5: 智能分块")
    chunker = Chunker()
    chunks = chunker.chunk(cleaned_pages)
    if task:
        task.progress = 70
        db.commit()

    # 检测：扫描版 PDF
    if not chunks:
        error_msg = (
            "PDF 解析后未提取到有效文本（0 个 chunks）。"
            "该 PDF 可能是扫描版/图片型 PDF，需要 OCR 支持。"
        )
        document.status = "failed"
        document.error_message = error_msg
        if task:
            task.status = "failed"
            task.error_message = error_msg
            task.completed_at = datetime.utcnow()
        db.commit()
        raise ValueError(error_msg)

    # 4. 向量化 + Chroma 入库
    print(f"[Ingest] 步骤 4/5: 向量化入库 - {len(chunks)} chunks")
    vector_store = VectorStore()

    chunks_for_vectordb = []
    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {}).copy()
        meta["document_id"] = document_id
        meta["user_id"] = user_id
        chunks_for_vectordb.append({
            "chunk_id": f"doc{document_id}_chunk{i}",
            "text": _chunk_content(chunk),
            "page_num": _chunk_page_number(chunk) or 1,
            "chunk_index": i,
            "metadata": meta,
        })

    added_ids = vector_store.add_documents(chunks_for_vectordb)

    # 5. 保存 DocumentChunk 记录（先清除旧记录，保证幂等）
    print(f"[Ingest] 步骤 5/5: 保存元数据")
    try:
        db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
        for i, (chunk, chroma_id) in enumerate(zip(chunks, added_ids)):
            page_number = _chunk_page_number(chunk)
            doc_chunk = DocumentChunk(
                document_id=document_id,
                chunk_index=i,
                content=_chunk_content(chunk),
                page_start=page_number,
                page_end=(chunk.get("metadata") or {}).get("page_end") or page_number,
                section=_chunk_section(chunk),
                chroma_id=chroma_id,
            )
            db.add(doc_chunk)

        # 更新文档状态
        document.status = "completed"
        document.chunk_count = len(chunks)
        document.processed_at = datetime.utcnow()
        document.error_message = None

        if task:
            task.progress = 100

        db.commit()
    except Exception:
        db.rollback()
        vector_store.delete_documents(added_ids)
        raise
    print(f"[Ingest] ✅ 完成: {document.filename}, {len(chunks)} chunks")

    return len(chunks), len(pages)


async def process_pdf_document(
    ctx: dict,
    document_id: int,
    user_id: int,
    task_id: int,
) -> dict[str, Any]:
    """
    异步处理 PDF 文档：解析 → 清洗 → 分块 → 向量化 → 入库。

    Args:
        ctx: ARQ 上下文
        document_id: 文档 ID
        user_id: 用户 ID
        task_id: 任务 ID

    Returns:
        dict: 处理结果（chunk_count, filename, status）
    """
    print(f"[Worker] 开始处理 PDF 文档: document_id={document_id}, task_id={task_id}")

    with get_session() as db:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise Retry(defer=10)

        # 幂等检查：如果文档已经处理完成，直接返回，不重复处理
        if document.status == "completed":
            print(f"[Worker] 文档 {document_id} 已处理完成，跳过")
            return {
                "status": "already_completed",
                "document_id": document_id,
                "chunk_count": document.chunk_count or 0,
                "filename": document.filename,
            }

        task = db.query(Task).filter(Task.id == task_id).first()

        document.status = "processing"
        if task:
            task.status = "running"
            task.started_at = datetime.utcnow()
            task.progress = 10
        db.commit()

        try:
            chunk_count, page_count = _ingest_pdf_to_db(document_id, user_id, db, task)

            if task:
                task.status = "completed"
                task.progress = 100
                task.completed_at = datetime.utcnow()
                task.result = {
                    "chunk_count": chunk_count,
                    "filename": document.filename,
                    "pages": page_count,
                }
            db.commit()

            # SQL 与 Chroma 均已确认成功，递增用户 BM25 版本，通知 API 进程重建快照
            if bump_bm25_version(user_id) is None:
                print("[Worker] ⚠️ BM25 版本递增失败（Redis 不可用），"
                      "API 将在 TTL 后兜底重建")

            return {
                "status": "completed",
                "document_id": document_id,
                "chunk_count": chunk_count,
                "filename": document.filename,
            }

        except Exception as e:
            print(f"[Worker] ❌ PDF 处理失败: {str(e)}")
            db.rollback()
            failed_document = db.query(Document).filter(Document.id == document_id).first()
            failed_task = db.query(Task).filter(Task.id == task_id).first()
            if failed_document:
                failed_document.status = "failed"
                failed_document.error_message = str(e)[:500]
            if failed_task:
                failed_task.status = "failed"
                failed_task.error_message = str(e)[:500]
                failed_task.completed_at = datetime.utcnow()
            db.commit()
            raise


async def process_pubmed_search(
    ctx: dict,
    keyword: str,
    max_results: int,
    user_id: int,
    task_id: int,
) -> dict[str, Any]:
    """
    异步处理 PubMed 文献检索：搜索 → 下载 → 逐篇入库。

    流程：
    1. 调用 NCBI E-utilities 搜索 PMID
    2. 获取文献详情（标题/作者/摘要/PMC ID）
    3. 逐篇：尝试 PMC 下载全文 PDF，失败则用摘要生成 PDF
    4. 逐篇创建 Document 记录并走入库流水线
    5. 汇总结果，更新 Task

    Args:
        ctx: ARQ 上下文
        keyword: 搜索关键字
        max_results: 最大下载数量
        user_id: 用户 ID
        task_id: 任务 ID

    Returns:
        dict: 处理结果（total, success, failed, articles）
    """
    from app.tools.pubmed_fetcher import search_and_download

    print(f"[Worker] PubMed 检索任务启动: keyword={keyword!r}, "
          f"max_results={max_results}, user_id={user_id}, task_id={task_id}")

    with get_session() as db:
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            task.status = "running"
            task.started_at = datetime.utcnow()
            task.progress = 5
            db.commit()

        try:
            # 步骤 1：搜索并下载文献
            print(f"[Worker] 步骤 1/2: 搜索 PubMed 并下载文献")
            articles = search_and_download(
                keyword=keyword,
                max_results=max_results,
                user_id=user_id,
                delay=0.4,
            )

            if task:
                task.progress = 40
                db.commit()

            # 步骤 2：逐篇入库
            print(f"[Worker] 步骤 2/2: 逐篇入库，共 {len(articles)} 篇")
            success_count = 0
            failed_count = 0
            article_results = []

            for i, article in enumerate(articles):
                if not article.get("success"):
                    failed_count += 1
                    article_results.append({
                        "pmid": article.get("pmid"),
                        "title": article.get("title", "")[:100],
                        "status": "download_failed",
                        "error": article.get("error"),
                    })
                    continue

                # 创建 Document 记录
                file_path = article["file_path"]
                file_name = article["file_name"]
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

                document = Document(
                    user_id=user_id,
                    filename=file_name,
                    file_path=file_path,
                    file_size=file_size,
                    status="processing",
                    chunk_count=0,
                )
                db.add(document)
                db.flush()
                document_id = document.id
                db.commit()

                # 入库
                try:
                    chunk_count, page_count = _ingest_pdf_to_db(
                        document.id, user_id, db, task=None
                    )
                    success_count += 1
                    # 该篇 SQL+Chroma 已成功，递增版本（多篇则递增多次，无副作用）
                    if bump_bm25_version(user_id) is None:
                        print("[Worker] ⚠️ BM25 版本递增失败（Redis 不可用）")
                    article_results.append({
                        "pmid": article.get("pmid"),
                        "title": article.get("title", "")[:100],
                        "source": article.get("source"),
                        "document_id": document.id,
                        "chunk_count": chunk_count,
                        "status": "completed",
                    })
                    print(f"[Worker]   ✅ [{i+1}/{len(articles)}] 入库成功: "
                          f"PMID={article.get('pmid')}, {chunk_count} chunks")
                except Exception as e:
                    failed_count += 1
                    db.rollback()
                    failed_document = (
                        db.query(Document).filter(Document.id == document_id).first()
                    )
                    if failed_document:
                        failed_document.status = "failed"
                        failed_document.error_message = str(e)[:500]
                    db.commit()
                    article_results.append({
                        "pmid": article.get("pmid"),
                        "title": article.get("title", "")[:100],
                        "status": "ingest_failed",
                        "error": str(e)[:200],
                    })
                    print(f"[Worker]   ❌ [{i+1}/{len(articles)}] 入库失败: "
                          f"PMID={article.get('pmid')}, {e}")

                # 更新整体进度（40% → 90%）
                if task:
                    progress = 40 + int((i + 1) / len(articles) * 50)
                    task.progress = min(progress, 90)
                    db.commit()

            # 完成
            if task:
                task.status = "completed"
                task.progress = 100
                task.completed_at = datetime.utcnow()
                task.result = {
                    "keyword": keyword,
                    "total": len(articles),
                    "success": success_count,
                    "failed": failed_count,
                    "articles": article_results,
                }
            db.commit()

            print(f"[Worker] ✅ PubMed 检索完成: 成功 {success_count}/{len(articles)}, "
                  f"失败 {failed_count}")

            return {
                "status": "completed",
                "keyword": keyword,
                "total": len(articles),
                "success": success_count,
                "failed": failed_count,
                "articles": article_results,
            }

        except Exception as e:
            print(f"[Worker] ❌ PubMed 检索任务失败: {str(e)}")
            db.rollback()
            failed_task = db.query(Task).filter(Task.id == task_id).first()
            if failed_task:
                failed_task.status = "failed"
                failed_task.error_message = str(e)[:500]
                failed_task.completed_at = datetime.utcnow()
            db.commit()
            raise
