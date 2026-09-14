"""
ARQ Worker 异步任务定义。

包含：
- process_pdf_document: PDF 文档解析入库异步任务
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from arq import Retry

from app.core.database import get_session
from app.models import Document, DocumentChunk, Task
from app.rag.pdf_parser import PDFParser
from app.rag.text_cleaner import TextCleaner
from app.rag.chunker import Chunker
from app.rag.vector_store import VectorStore


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

    处理流程：
    1. 从数据库读取文档信息
    2. 更新任务状态为 running
    3. PDF 解析（PyMuPDF）
    4. 文本清洗
    5. Markdown 智能分块
    6. BGE-m3 向量化 + Chroma 入库
    7. 更新文档状态和 chunk_count
    8. 更新任务状态为 completed
    """
    print(f"[Worker] 开始处理 PDF 文档: document_id={document_id}, task_id={task_id}")

    with get_session() as db:
        # 1. 读取文档信息
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise Retry(defer=10)  # 文档还没创建好，10秒后重试

        task = db.query(Task).filter(Task.id == task_id).first()

        # 2. 更新状态
        document.status = "processing"
        if task:
            task.status = "running"
            task.started_at = datetime.utcnow()
            task.progress = 10
        db.commit()

        try:
            # 3. PDF 解析
            print(f"[Worker] 步骤 1/5: 解析 PDF - {document.filename}")
            parser = PDFParser()
            pages = parser.parse_to_markdown(document.file_path)
            if task:
                task.progress = 30
                db.commit()

            # 4. 文本清洗
            print(f"[Worker] 步骤 2/5: 清洗文本 - {len(pages)} 页")
            cleaner = TextCleaner()
            cleaned_pages = cleaner.clean(pages)
            if task:
                task.progress = 50
                db.commit()

            # 5. 智能分块
            print(f"[Worker] 步骤 3/5: 智能分块")
            chunker = Chunker()
            chunks = chunker.chunk(cleaned_pages)
            if task:
                task.progress = 70
                db.commit()

            # 检测：如果分块数量为 0，说明 PDF 可能是扫描版（无文本层）
            if not chunks:
                error_msg = (
                    "PDF 解析后未提取到有效文本（0 个 chunks）。"
                    "该 PDF 可能是扫描版/图片型 PDF，需要 OCR 支持。"
                    "请上传包含文本层的 PDF。"
                )
                print(f"[Worker] ⚠️ {error_msg}")
                document.status = "failed"
                document.error_message = error_msg
                if task:
                    task.status = "failed"
                    task.error_message = error_msg
                    task.completed_at = datetime.utcnow()
                db.commit()
                return {"status": "failed", "error": error_msg}

            # 6. 向量化 + Chroma 入库
            print(f"[Worker] 步骤 4/5: 向量化入库 - {len(chunks)} chunks")
            vector_store = VectorStore()

            # 格式转换：chunker 输出格式 → vector_store 期望格式
            # chunker 输出: {"content": "...", "metadata": {page_start, page_end, heading_path, ...}}
            # vector_store 期望: {"chunk_id": "...", "text": "...", "page_num": ..., "chunk_index": ..., "metadata": {...}}
            chunks_for_vectordb = []
            for i, chunk in enumerate(chunks):
                meta = chunk.get("metadata", {}).copy()
                meta["document_id"] = document_id
                meta["user_id"] = user_id
                chunks_for_vectordb.append({
                    "chunk_id": f"doc{document_id}_chunk{i}",
                    "text": chunk.get("content", ""),
                    "page_num": meta.get("page_start", 1),
                    "chunk_index": i,
                    "metadata": meta,
                })

            added_ids = vector_store.add_documents(chunks_for_vectordb)

            # 7. 保存 DocumentChunk 记录
            print(f"[Worker] 步骤 5/5: 保存元数据")
            for i, (chunk, chroma_id) in enumerate(zip(chunks, added_ids)):
                doc_chunk = DocumentChunk(
                    document_id=document_id,
                    chunk_index=i,
                    content=chunk.get("content", ""),
                    page_start=chunk.get("metadata", {}).get("page_start"),
                    page_end=chunk.get("metadata", {}).get("page_end"),
                    section=chunk.get("metadata", {}).get("heading_path"),
                    chroma_id=chroma_id,
                )
                db.add(doc_chunk)

            # 8. 更新文档状态
            document.status = "completed"
            document.chunk_count = len(chunks)
            document.processed_at = datetime.utcnow()
            document.error_message = None

            if task:
                task.status = "completed"
                task.progress = 100
                task.completed_at = datetime.utcnow()
                task.result = {
                    "chunk_count": len(chunks),
                    "filename": document.filename,
                    "pages": len(pages),
                }

            db.commit()
            print(f"[Worker] ✅ PDF 处理完成: {document.filename}, {len(chunks)} chunks")

            return {
                "status": "completed",
                "document_id": document_id,
                "chunk_count": len(chunks),
                "filename": document.filename,
            }

        except Exception as e:
            # 处理失败
            print(f"[Worker] ❌ PDF 处理失败: {str(e)}")
            document.status = "failed"
            document.error_message = str(e)[:500]

            if task:
                task.status = "failed"
                task.error_message = str(e)[:500]
                task.completed_at = datetime.utcnow()

            db.commit()
            raise  # 重新抛出，让 ARQ 重试
