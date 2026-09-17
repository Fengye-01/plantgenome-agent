"""文档相关 Pydantic 模型。"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    """文档响应。"""
    id: int
    filename: str
    file_size: int
    status: str
    chunk_count: int
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class DocumentUploadResponse(BaseModel):
    """文档上传响应（异步任务）。"""
    task_id: int
    document_id: int
    filename: str
    status: str = "pending"
    message: str = "文档已提交，正在后台处理"


class PubMedSearchRequest(BaseModel):
    """PubMed 文献搜索请求。"""
    keyword: str
    max_results: int = 5


class PubMedSearchResponse(BaseModel):
    """PubMed 文献搜索响应（异步任务）。"""
    task_id: int
    keyword: str
    max_results: int
    status: str = "pending"
    message: str = "PubMed 检索任务已提交，正在后台下载并入库"
