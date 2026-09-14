"""
FastAPI 依赖注入。

提供：
- get_db: 数据库会话
- get_current_user: 当前登录用户（从 JWT 令牌解析）
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_user_id_from_token
from app.models import User

# OAuth2 密码流，tokenUrl 指向登录接口
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    获取当前登录用户。

    从 Authorization: Bearer <token> 中提取 JWT 令牌，
    解码得到用户 ID，然后从数据库查询用户。

    如果令牌无效或用户不存在，抛出 401 未授权错误。

    用法：
        @app.get("/api/chat/sessions")
        def get_sessions(current_user: User = Depends(get_current_user)):
            ...
    """
    user_id = get_user_id_from_token(token)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="用户已被禁用",
        )

    return user


def get_current_user_optional(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User | None:
    """
    获取当前登录用户（可选）。

    与 get_current_user 不同，如果没有令牌或令牌无效，返回 None 而不是抛出异常。
    用于可选认证的接口。
    """
    try:
        return get_current_user(token, db)
    except HTTPException:
        return None
