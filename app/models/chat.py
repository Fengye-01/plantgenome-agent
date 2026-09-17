"""
PlantGenome Agent - ChatSession + Message 模型

聊天历史存储，对应 Hello Agents 第5章（Memory）。
ChatSession 表存储会话，Message 表存储每条消息。
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class ChatSession(Base):
    """
    聊天会话表

    每个用户可以有多个会话，每个会话包含多条消息。
    会话标题通常取第一条用户消息的前 50 个字符。
    """
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)

    title: Mapped[str] = mapped_column(String(200), default="新会话", nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # 关系
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan",
                             order_by="Message.created_at")

    def __repr__(self) -> str:
        return f"<ChatSession(id={self.id}, user_id={self.user_id}, title={self.title!r})>"

    def to_dict(self, include_messages: bool = False) -> dict:
        result = {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "message_count": len(self.messages),
        }
        if include_messages:
            result["messages"] = [m.to_dict() for m in self.messages]
        return result


class Message(Base):
    """
    聊天消息表

    存储每条消息的内容和元数据。
    - role: user / assistant
    - sources: 文献来源引用（JSON，兼容 SQLite 和 PostgreSQL）
    - tool_name: 调用的工具名
    - tool_result: 工具执行结果（JSON）

    为什么用 JSON？
    - SQLAlchemy 通用 JSON 类型，兼容 SQLite 和 PostgreSQL
    - 可以存储结构化数据（字典、列表）
    - 比 Text 存储 JSON 更专业
    """
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)

    role: Mapped[str] = mapped_column(String(20), nullable=False, index=True)  # user / assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # 文献来源引用（JSON 数组）
    sources: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    # 工具调用信息
    tool_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    tool_result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Agent 执行日志（JSON，用于前端展示执行过程）
    execution_log: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # 关系
    session = relationship("ChatSession", back_populates="messages")

    def __repr__(self) -> str:
        return f"<Message(id={self.id}, session_id={self.session_id}, role={self.role!r})>"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "sources": self.sources,
            "tool_name": self.tool_name,
            "tool_result": self.tool_result,
            "execution_log": self.execution_log,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
