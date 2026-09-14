"""
PlantGenome Agent - 数据库核心模块

对应 Hello Agents 第8章（RAG）：文档元数据管理。
使用 SQLAlchemy 2.0 + PostgreSQL，提供数据库连接和会话管理。

表设计：
- users: 用户表
- documents: 文档元数据表
- document_chunks: 文档块表（与 Chroma 通过 chroma_id 关联）
- chat_sessions: 聊天会话表
- messages: 聊天消息表
- tasks: 异步任务表
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    """SQLAlchemy 声明式基类，所有模型继承自此类。"""
    pass


# 数据库连接 URL
# 格式: postgresql+psycopg://user:password@host:port/dbname
# 使用 psycopg v3（而非 psycopg2），解决 Windows 中文系统编码问题
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://plantgenome:plantgenome@localhost:5432/plantgenome"
).replace("postgresql+psycopg2://", "postgresql+psycopg://")

# 创建数据库引擎
# - pool_pre_ping: 连接前先 ping，避免使用已断开的连接
# - pool_recycle: 连接回收时间（秒），避免 PostgreSQL 超时断开
# - echo: 是否打印 SQL 语句（调试用，生产环境关闭）
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False,
)

# 会话工厂
SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    expire_on_commit=False,
    autoflush=False,
)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI 依赖注入：获取数据库会话。

    用法：
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            ...

    每次请求创建一个会话，请求结束后自动关闭。
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    上下文管理器：在非 FastAPI 环境（脚本、Worker）中使用数据库会话。

    用法：
        with get_session() as db:
            db.add(...)
            db.commit()

    自动提交和回滚：
    - 正常退出时自动 commit
    - 发生异常时自动 rollback
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """
    初始化数据库：创建所有表。

    用法：
        from app.core.database import init_db
        init_db()

    注意：
    - 只创建不存在的表（CREATE TABLE IF NOT EXISTS）
    - 不会删除或修改已存在的表
    - 生产环境建议用 Alembic 做迁移
    """
    # 导入所有模型，确保 Base.metadata 包含所有表
    from app.models import User, Document, DocumentChunk, ChatSession, Message, Task

    Base.metadata.create_all(bind=engine)
    print("✅ 数据库表创建完成")


def get_db_url() -> str:
    """获取当前数据库连接 URL（隐藏密码，用于日志）。"""
    from urllib.parse import urlparse, urlunparse
    parsed = urlparse(DATABASE_URL)
    # 隐藏密码
    if parsed.password:
        netloc = f"{parsed.username}:***@{parsed.hostname}:{parsed.port}"
        return urlunparse(parsed._replace(netloc=netloc))
    return DATABASE_URL
