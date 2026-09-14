"""
任务接口：查询异步任务状态。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.database import get_db
from app.models import Task, User
from app.schemas.task import TaskResponse

router = APIRouter(prefix="/api/tasks", tags=["任务"])


@router.get("/{task_id}", response_model=TaskResponse)
def get_task(
    task_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    查询异步任务状态。

    前端轮询此接口获取任务进度。
    """
    task = db.query(Task).filter(Task.id == task_id, Task.user_id == current_user.id).first()
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="任务不存在",
        )
    return task


@router.get("", response_model=list[TaskResponse])
def list_tasks(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 20,
):
    """
    获取当前用户的任务列表。
    """
    tasks = (
        db.query(Task)
        .filter(Task.user_id == current_user.id)
        .order_by(Task.created_at.desc())
        .limit(limit)
        .all()
    )
    return tasks
