"""
文档接口：上传 PDF、查询文档列表、删除文档。
"""
from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.core.database import get_db
from app.models import Document, DocumentChunk, Task, User
from app.schemas.document import DocumentResponse, DocumentUploadResponse, PubMedSearchRequest, PubMedSearchResponse

settings = get_settings()
router = APIRouter(prefix="/api/documents", tags=["文档"])

# 上传目录
UPLOAD_DIR = Path("data/pdfs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    上传 PDF 文档（异步处理）。

    流程：
    1. 保存 PDF 文件到 data/pdfs/
    2. 创建 Document 记录（status=pending）
    3. 创建 Task 记录（status=pending）
    4. 提交异步任务到 ARQ 队列
    5. 返回 task_id 和 document_id

    前端通过 GET /api/tasks/{task_id} 轮询处理进度。
    """
    # 验证文件类型
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="只支持 PDF 文件",
        )

    # 保存文件
    safe_filename = f"{current_user.id}_{int(datetime.utcnow().timestamp())}_{file.filename}"
    file_path = UPLOAD_DIR / safe_filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_size = file_path.stat().st_size

    # 创建 Document 记录
    document = Document(
        user_id=current_user.id,
        filename=file.filename,
        file_path=str(file_path),
        file_size=file_size,
        status="pending",
        chunk_count=0,
    )
    db.add(document)
    db.flush()  # 获取 document.id

    # 创建 Task 记录
    task = Task(
        user_id=current_user.id,
        task_type="pdf_ingest",
        status="pending",
        progress=0,
        params={
            "document_id": document.id,
            "filename": file.filename,
            "file_size": file_size,
        },
    )
    db.add(task)
    db.flush()  # 获取 task.id

    db.commit()
    db.refresh(document)
    db.refresh(task)

    # 提交异步任务到 ARQ 队列
    try:
        from urllib.parse import urlparse
        parsed = urlparse(settings.redis_url)
        redis_settings = RedisSettings(
            host=parsed.hostname or "localhost",
            port=parsed.port or 6379,
            password=parsed.password,
            database=int(parsed.path.lstrip("/")) if parsed.path else 0,
        )
        redis = await create_pool(redis_settings)
        await redis.enqueue_job(
            "process_pdf_document",
            document_id=document.id,
            user_id=current_user.id,
            task_id=task.id,
        )
        await redis.close()
    except Exception as e:
        # ARQ 连接失败，标记任务失败（但文档已保存）
        task.status = "failed"
        task.error_message = f"ARQ 队列连接失败: {str(e)}"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"任务队列不可用: {str(e)}",
        )

    return DocumentUploadResponse(
        task_id=task.id,
        document_id=document.id,
        filename=file.filename,
        status="pending",
        message="文档已提交，正在后台处理",
    )


@router.get("", response_model=list[DocumentResponse])
def list_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 50,
):
    """获取当前用户的文档列表。"""
    documents = (
        db.query(Document)
        .filter(Document.user_id == current_user.id)
        .order_by(Document.created_at.desc())
        .limit(limit)
        .all()
    )
    return documents


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """获取单个文档信息。"""
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id,
    ).first()
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")
    return document


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    删除文档（同时删除文件、数据库记录和 Chroma 向量）。
    """
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id,
    ).first()
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")

    # 删除文件
    try:
        if os.path.exists(document.file_path):
            os.remove(document.file_path)
    except Exception:
        pass

    # 从 Chroma 删除向量
    try:
        chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).all()
        chroma_ids = [c.chroma_id for c in chunks if c.chroma_id]
        if chroma_ids:
            from app.rag.vector_store import VectorStore

            vector_store = VectorStore()
            vector_store.delete_documents(chroma_ids)
    except Exception:
        pass

    # 删除数据库记录（级联删除 DocumentChunk）
    db.delete(document)
    db.commit()

    return None


@router.post("/pubmed-search", response_model=PubMedSearchResponse, status_code=status.HTTP_202_ACCEPTED)
async def pubmed_search(
    request: PubMedSearchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    按关键字从 PubMed 检索文献并自动下载入库（异步处理）。

    流程：
    1. 校验关键字和数量
    2. 创建 Task 记录（status=pending, task_type=pubmed_search）
    3. 提交异步任务到 ARQ 队列
    4. 返回 task_id，前端轮询 GET /api/tasks/{task_id}

    异步任务内部会：
    - 调用 NCBI E-utilities 搜索 PMID
    - 获取文献详情（标题/作者/摘要/PMC ID）
    - 尝试 PMC 下载全文 PDF，失败则用摘要生成 PDF
    - 逐篇复用现有 PDF 入库流水线（解析→分块→向量化）
    """
    # 校验
    keyword = request.keyword.strip()
    if not keyword:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="搜索关键字不能为空",
        )
    if not (1 <= request.max_results <= 20):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="max_results 必须在 1-20 之间",
        )

    # 创建 Task 记录
    task = Task(
        user_id=current_user.id,
        task_type="pubmed_search",
        status="pending",
        progress=0,
        params={
            "keyword": keyword,
            "max_results": request.max_results,
        },
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    # 提交异步任务到 ARQ 队列
    try:
        from urllib.parse import urlparse
        parsed = urlparse(settings.redis_url)
        redis_settings = RedisSettings(
            host=parsed.hostname or "localhost",
            port=parsed.port or 6379,
            password=parsed.password,
            database=int(parsed.path.lstrip("/")) if parsed.path else 0,
        )
        redis = await create_pool(redis_settings)
        await redis.enqueue_job(
            "process_pubmed_search",
            keyword=keyword,
            max_results=request.max_results,
            user_id=current_user.id,
            task_id=task.id,
        )
        await redis.close()
    except Exception as e:
        task.status = "failed"
        task.error_message = f"ARQ 队列连接失败: {str(e)}"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"任务队列不可用: {str(e)}",
        )

    return PubMedSearchResponse(
        task_id=task.id,
        keyword=keyword,
        max_results=request.max_results,
        status="pending",
        message="PubMed 检索任务已提交，正在后台下载并入库",
    )
