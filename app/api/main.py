"""
PlantGenome Agent - FastAPI 主入口

整合所有路由：
- /api/auth: 认证（注册、登录、当前用户）
- /api/chat: 聊天（发送消息、会话列表、历史消息）
- /api/documents: 文档（上传、列表、删除）
- /api/tasks: 任务（查询异步任务状态）
- /health: 健康检查
"""
from __future__ import annotations

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.tasks import router as tasks_router
from app.core.config import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理。

    startup: 初始化数据库表
    shutdown: 清理资源
    """
    # Startup
    print("=" * 60)
    print("🌱 PlantGenome Agent 启动中...")
    print("=" * 60)

    # 初始化数据库表
    try:
        from app.core.database import init_db
        init_db()
        print("✅ 数据库初始化完成")
    except Exception as e:
        print(f"⚠️  数据库初始化失败（将使用无存储模式）: {e}")

    # 检查 Redis
    try:
        from app.core.redis import check_redis_connection
        if check_redis_connection():
            print("✅ Redis 连接正常")
        else:
            print("⚠️  Redis 连接失败（异步任务将不可用）")
    except Exception as e:
        print(f"⚠️  Redis 检查失败: {e}")

    print("=" * 60)
    print(f"✅ PlantGenome Agent 已启动: http://{settings.app_host}:{settings.app_port}")
    print(f"📖 API 文档: http://{settings.app_host}:{settings.app_port}/docs")
    print("=" * 60)

    yield

    # Shutdown
    print("🌱 PlantGenome Agent 已关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title="PlantGenome Agent",
    description="面向植物比较基因组研究的 PDF-RAG + Tool Calling 智能体",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS 中间件（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(tasks_router)


@app.get("/health", tags=["系统"])
def health_check():
    """健康检查接口。"""
    return {
        "status": "ok",
        "service": "PlantGenome Agent",
        "version": "2.0.0",
    }


@app.get("/", tags=["系统"])
def root():
    """根路径，重定向到 API 文档。"""
    return {
        "message": "欢迎使用 PlantGenome Agent",
        "docs": "/docs",
        "health": "/health",
        "api": {
            "auth": "/api/auth",
            "chat": "/api/chat",
            "documents": "/api/documents",
            "tasks": "/api/tasks",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
    )
