"""
Redis 连接管理。

提供 Redis 客户端连接，用于：
- ARQ 异步任务队列
- 缓存
- 会话存储（可选）
"""
from __future__ import annotations

import redis
from app.core.config import get_settings

settings = get_settings()

# Redis 客户端（连接池）
redis_client = redis.Redis.from_url(
    settings.redis_url,
    decode_responses=True,  # 自动解码 bytes 为 str
    socket_connect_timeout=5,
    socket_timeout=5,
)


def get_redis() -> redis.Redis:
    """获取 Redis 客户端。"""
    return redis_client


def check_redis_connection() -> bool:
    """检查 Redis 连接是否正常。"""
    try:
        return redis_client.ping()
    except Exception:
        return False
