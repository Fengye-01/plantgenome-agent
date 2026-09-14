"""
ARQ Worker 配置。

用法：
    arq app.worker.settings.WorkerSettings
"""
from __future__ import annotations

from arq.connections import RedisSettings

from app.core.config import get_settings

settings = get_settings()


# 解析 Redis URL 为 ARQ 需要的格式
def _parse_redis_url(url: str) -> dict:
    """解析 redis://user:password@host:port/db 格式。"""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 6379,
        "password": parsed.password,
        "database": int(parsed.path.lstrip("/")) if parsed.path else 0,
    }


_redis_config = _parse_redis_url(settings.redis_url)


class WorkerSettings:
    """ARQ Worker 配置类。"""
    # Redis 连接
    redis_settings = RedisSettings(
        host=_redis_config["host"],
        port=_redis_config["port"],
        password=_redis_config["password"],
        database=_redis_config["database"],
    )

    # 任务函数列表（直接引用函数对象，ARQ 不支持 module:attr 字符串格式）
    from app.worker.tasks import process_pdf_document
    functions = [
        process_pdf_document,
    ]

    # Worker 配置
    max_jobs = 5  # 最大并发任务数
    job_timeout = 600  # 任务超时时间（秒）
    retry_jobs = True  # 失败重试
    max_tries = 3  # 最大重试次数

    # 日志
    # log_level = "INFO"
