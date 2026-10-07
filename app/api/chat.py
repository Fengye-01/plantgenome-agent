"""
聊天接口：发送消息、获取会话列表、获取历史消息。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.agents.graph import run_agent
from app.api.deps import get_current_user
from app.core.config import get_settings
from app.core.database import get_db
from app.models import ChatSession, Message, User
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatSessionResponse,
    MessageResponse,
    Source,
)

router = APIRouter(prefix="/api/chat", tags=["聊天"])
logger = logging.getLogger(__name__)
settings = get_settings()


def _load_conversation_history(db: Session, session_id: int) -> list[dict[str, str]]:
    """Load a bounded, chronological history for one already-authorized session."""
    rows = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(settings.chat_history_limit)
        .all()
    )
    rows.reverse()
    max_chars = settings.chat_history_message_max_chars
    return [
        {"role": row.role, "content": row.content[:max_chars]}
        for row in rows
        if row.role in {"user", "assistant"} and row.content
    ]


def _last_executed_tool(result: dict[str, Any]) -> str | None:
    """Return the last tool that actually reached tool history, not the final router choice."""
    for call in reversed(result.get("tool_history", []) or []):
        if call.get("tool"):
            return call["tool"]
    return result.get("tool_name")


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    发送聊天消息（走完整 Agent 链路）。

    流程：
    1. 获取或创建会话
    2. 读取有限历史并持久化用户消息
    3. 结束数据库事务后调用 Agent（router → tool → answer）
    4. 单独保存 AI 回答或失败记录
    5. 返回回答
    """
    # 1. 获取或创建会话
    if request.session_id:
        session = (
            db.query(ChatSession)
            .filter(
                ChatSession.id == request.session_id,
                ChatSession.user_id == current_user.id,
            )
            .first()
        )
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
    else:
        # 创建新会话，标题取用户消息前 50 字符
        session = ChatSession(
            user_id=current_user.id,
            title=request.message[:50] if request.message else "新会话",
        )
        db.add(session)
        db.flush()

    # 2. 只读取已通过所有权校验的当前会话历史。
    history = _load_conversation_history(db, session.id)

    # 3. 保存用户消息
    user_message = Message(
        session_id=session.id,
        role="user",
        content=request.message,
    )
    db.add(user_message)
    session.updated_at = datetime.now(timezone.utc)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise
    db.refresh(session)

    # 4. 数据库事务已经结束；Agent 的网络调用和工具执行不会长期占用事务。
    try:
        result = run_agent(
            request.message,
            messages=history,
            user_id=current_user.id,
        )
    except Exception as exc:
        logger.exception(
            "Agent request failed for user=%s session=%s", current_user.id, session.id
        )
        failure_answer = "Agent 执行失败，本次未产生可信结果，请稍后重试。"
        failure_log = [
            {
                "node": "agent",
                "status": "failed",
                "error_type": "agent_execution_error",
                "error": type(exc).__name__,
            }
        ]
        db.add(
            Message(
                session_id=session.id,
                role="assistant",
                content=failure_answer,
                execution_log=failure_log,
            )
        )
        session.updated_at = datetime.now(timezone.utc)
        try:
            db.commit()
        except Exception:
            db.rollback()
            raise
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=failure_answer,
        ) from exc

    answer = result.get("final_answer", "抱歉，未能生成回答。")
    sources = result.get("sources", [])
    tool_name = _last_executed_tool(result)
    tool_result = result.get("tool_result")
    execution_log = result.get("execution_log", [])

    # 5. 保存 AI 回答
    ai_message = Message(
        session_id=session.id,
        role="assistant",
        content=answer,
        sources=sources if sources else None,
        tool_name=tool_name,
        tool_result=tool_result if isinstance(tool_result, (dict, list)) else None,
        execution_log=execution_log if execution_log else None,
    )
    db.add(ai_message)

    # 更新会话时间
    session.updated_at = datetime.now(timezone.utc)

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise
    db.refresh(session)
    db.refresh(ai_message)

    # 6. 返回响应
    source_list = []
    for s in sources:
        if isinstance(s, dict):
            source_list.append(
                Source(
                    filename=s.get("filename"),
                    page_num=s.get("page_num"),
                    snippet=s.get("snippet"),
                    heading_path=s.get("heading_path"),
                    similarity=s.get("similarity"),
                )
            )

    return ChatResponse(
        answer=answer,
        session_id=session.id,
        sources=source_list,
        tool_name=tool_name,
        tool_result=tool_result,
        execution_log=execution_log if execution_log else None,
    )


@router.get("/sessions", response_model=list[ChatSessionResponse])
def list_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 20,
):
    """获取当前用户的会话列表。"""
    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.updated_at.desc())
        .limit(limit)
        .all()
    )
    result = []
    for s in sessions:
        result.append(
            ChatSessionResponse(
                id=s.id,
                title=s.title,
                created_at=s.created_at,
                updated_at=s.updated_at,
                message_count=len(s.messages),
            )
        )
    return result


@router.get("/sessions/{session_id}/messages", response_model=list[MessageResponse])
def get_session_messages(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """获取指定会话的历史消息。"""
    session = (
        db.query(ChatSession)
        .filter(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id,
        )
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    messages = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
        .all()
    )
    return messages


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """删除会话（级联删除所有消息）。"""
    session = (
        db.query(ChatSession)
        .filter(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id,
        )
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    db.delete(session)
    db.commit()
    return None
