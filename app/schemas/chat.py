"""聊天相关 Pydantic 模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天请求。"""

    message: str = Field(..., description="用户消息")
    session_id: Optional[int] = Field(None, description="会话 ID，不传则创建新会话")


class Source(BaseModel):
    """文献来源。"""

    filename: Optional[str] = None
    page_num: Optional[int] = None
    snippet: Optional[str] = None
    heading_path: Optional[str] = None
    similarity: Optional[float] = None


class ChatResponse(BaseModel):
    """聊天响应。"""

    answer: str
    session_id: int
    sources: list[Source] = Field(default_factory=list)
    tool_name: Optional[str] = None
    tool_result: Optional[Any] = None
    execution_log: Optional[list[dict]] = None


class MessageResponse(BaseModel):
    """消息响应。"""

    id: int
    session_id: int
    role: str
    content: str
    sources: Optional[list] = None
    tool_name: Optional[str] = None
    tool_result: Optional[Any] = None
    execution_log: Optional[list] = None
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class ChatSessionResponse(BaseModel):
    """会话响应。"""

    id: int
    title: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    message_count: int = 0

    model_config = {"from_attributes": True}
