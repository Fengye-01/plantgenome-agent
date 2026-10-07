"""
PlantGenome Agent - 安全模块

密码哈希（直接使用 bcrypt 库，避免 passlib 版本冲突）+ JWT 令牌生成与验证。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import bcrypt
from jose import JWTError, jwt

from app.core.config import get_settings

settings = get_settings()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    验证密码是否正确。

    Args:
        plain_password: 明文密码
        hashed_password: 哈希后的密码（bcrypt 格式）

    Returns:
        bool: 密码是否匹配
    """
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )


def get_password_hash(password: str) -> str:
    """
    获取密码的哈希值。

    使用 bcrypt 算法，自动生成 salt。
    bcrypt 限制密码最长 72 字节，超过部分自动截断。

    Args:
        password: 明文密码

    Returns:
        str: 哈希后的密码（包含 salt）
    """
    # bcrypt 限制 72 字节，超长密码截断
    password_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def create_access_token(
    subject: str | Any, expires_delta: Optional[timedelta] = None
) -> str:
    """
    创建 JWT 访问令牌。

    Args:
        subject: 令牌主题（通常是用户 ID）
        expires_delta: 过期时间增量，默认使用配置中的时间

    Returns:
        str: JWT 令牌字符串

    JWT Payload 结构：
    {
        "sub": "user_id",      # 主题（用户 ID）
        "exp": 1234567890,     # 过期时间
        "iat": 1234567890,     # 签发时间
        "type": "access"       # 令牌类型
    }
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.access_token_expire_minutes
        )

    to_encode = {
        "sub": str(subject),
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access",
    }
    encoded_jwt = jwt.encode(
        to_encode, settings.secret_key, algorithm=settings.algorithm
    )
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    """
    解码并验证 JWT 访问令牌。

    Args:
        token: JWT 令牌字符串

    Returns:
        Optional[dict]: 解码后的 payload，如果令牌无效则返回 None
    """
    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.algorithm]
        )
        if payload.get("type") != "access" or not payload.get("sub"):
            return None
        return payload
    except JWTError:
        return None


def get_user_id_from_token(token: str) -> Optional[int]:
    """
    从 JWT 令牌中提取用户 ID。

    Args:
        token: JWT 令牌字符串

    Returns:
        Optional[int]: 用户 ID，如果令牌无效则返回 None
    """
    payload = decode_access_token(token)
    if payload and "sub" in payload:
        try:
            return int(payload["sub"])
        except (ValueError, TypeError):
            return None
    return None
