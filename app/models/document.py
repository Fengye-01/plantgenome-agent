"""
PlantGenome Agent - Document + DocumentChunk 模型

对应 Hello Agents 第8章（RAG）：文档元数据管理。
Document 表存储 PDF 元数据，DocumentChunk 表存储分块信息，与 Chroma 通过 chroma_id 关联。
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Document(Base):
    """
    文档元数据表

    存储上传的 PDF 文献的元信息，与 Chroma 向量库通过 document_id 关联。

    状态流转：
    pending（待处理）→ processing（处理中）→ completed（完成）/ failed（失败）
    """
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)

    filename: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, default=0)

    # 状态：pending / processing / completed / failed
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    processed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # 关系
    user = relationship("User", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename={self.filename!r}, status={self.status!r})>"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "filename": self.filename,
            "file_size": self.file_size,
            "status": self.status,
            "chunk_count": self.chunk_count,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
        }


class DocumentChunk(Base):
    """
    文档块表

    存储每个 chunk 的元信息，与 Chroma 向量库通过 chroma_id 关联。
    Chroma 中存储向量和 content，PostgreSQL 中存储元数据（页码、章节等）。

    为什么需要这张表？
    - Chroma 的 metadata 查询能力有限
    - PostgreSQL 可以做复杂查询（按文档、按页码、按章节筛选）
    - 两者通过 chroma_id 关联，查询时先查 PostgreSQL 拿到 chroma_id，再去 Chroma 取向量
    """
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)

    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # 页码范围（一个 chunk 可能跨页）
    page_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    page_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # 章节路径（Markdown 标题层次，如 "方法 > 序列比对"）
    section: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Chroma 中的 ID（用于关联向量库）
    chroma_id: Mapped[Optional[str]] = mapped_column(String(100), unique=True, index=True, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # 关系
    document = relationship("Document", back_populates="chunks")

    def __repr__(self) -> str:
        return f"<DocumentChunk(id={self.id}, doc_id={self.document_id}, index={self.chunk_index})>"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "content_preview": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section": self.section,
            "chroma_id": self.chroma_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
