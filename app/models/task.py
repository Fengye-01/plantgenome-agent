"""
PlantGenome Agent - Task 模型

异步任务表，对应 Redis 异步任务队列。
存储任务状态、进度和结果，支持前端轮询查询。
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Task(Base):
    """
    异步任务表

    存储后台异步任务的状态和结果，如 PDF 解析入库。

    状态流转：
    pending（待执行）→ running（执行中）→ completed（完成）/ failed（失败）

    前端通过 GET /tasks/{task_id} 轮询查询任务状态。
    """
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)

    # 任务类型：pdf_ingest / 其他
    task_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # 状态：pending / running / completed / failed
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True, nullable=False)

    # 进度（0-100）
    progress: Mapped[int] = mapped_column(Integer, default=0)

    # 任务输入参数（JSONB）
    params: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # 任务结果（JSONB）
    result: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # 错误信息
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # 关系
    user = relationship("User", back_populates="tasks")

    def __repr__(self) -> str:
        return f"<Task(id={self.id}, type={self.task_type!r}, status={self.status!r}, progress={self.progress}%)>"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "task_type": self.task_type,
            "status": self.status,
            "progress": self.progress,
            "params": self.params,
            "result": self.result,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
