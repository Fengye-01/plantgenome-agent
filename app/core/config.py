"""
PlantGenome Agent 全局配置
从环境变量或 .env 文件读取，保持配置与代码分离。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
load_dotenv()


@dataclass
class Settings:
    # ── LLM 配置 ──
    llm_model_id: str = os.getenv("LLM_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1")
    llm_timeout: int = int(os.getenv("LLM_TIMEOUT", "120"))

    # 生成参数
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))

    # ── 向量库配置 ──
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "data/chroma")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

    # ── RAG 参数 ──
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "600"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    retrieve_top_k: int = int(os.getenv("RETRIEVE_TOP_K", "3"))

    # ── 数据库配置 ──
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://plantgenome:plantgenome@localhost:5432/plantgenome"
    )

    # ── Redis 配置 ──
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # ── JWT / 安全配置 ──
    secret_key: str = os.getenv("SECRET_KEY", "plantgenome-agent-secret-key-change-in-production")
    algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24小时

    # ── 应用配置 ──
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8000"))


# 全局单例
settings = Settings()


def get_settings() -> Settings:
    """获取全局配置单例。"""
    return settings
