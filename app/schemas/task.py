"""任务相关 Pydantic 模型。"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class TaskResponse(BaseModel):
    """异步任务响应。"""
    id: int
    task_type: str
    status: str
    progress: int
    params: Optional[dict] = None
    result: Optional[Any] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}
