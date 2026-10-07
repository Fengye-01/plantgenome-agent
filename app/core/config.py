"""
PlantGenome Agent 全局配置
从环境变量或 .env 文件读取，保持配置与代码分离。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
load_dotenv()

DEFAULT_SECRET_KEY = "plantgenome-agent-secret-key-change-in-production"


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
    rag_max_distance: float = float(os.getenv("RAG_MAX_DISTANCE", "0.8"))

    # ── 检索模式与独立召回参数 ──
    # dense_only：纯向量；candidate_rrf：Dense候选内BM25重排（当前线上默认）；
    # global_rrf：用户全库 Dense / BM25 独立召回后 RRF
    retrieval_mode: str = os.getenv("RETRIEVAL_MODE", "candidate_rrf")
    dense_top_n: int = int(os.getenv("DENSE_TOP_N", "20"))
    sparse_top_n: int = int(os.getenv("SPARSE_TOP_N", "20"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    bm25_snapshot_ttl: int = int(os.getenv("BM25_SNAPSHOT_TTL", "300"))

    # ── 数据库配置 ──
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://plantgenome:plantgenome@localhost:5432/plantgenome",
    )

    # ── Redis 配置 ──
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # ── JWT / 安全配置 ──
    secret_key: str = os.getenv("SECRET_KEY", DEFAULT_SECRET_KEY)
    algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(
        os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440")
    )  # 24小时

    # ── 应用配置 ──
    app_env: str = os.getenv("APP_ENV", "development").lower()
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8000"))
    cors_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:8501,http://127.0.0.1:8501",
        ).split(",")
        if origin.strip()
    )
    chat_history_limit: int = int(os.getenv("CHAT_HISTORY_LIMIT", "12"))
    chat_history_message_max_chars: int = int(
        os.getenv("CHAT_HISTORY_MESSAGE_MAX_CHARS", "2000")
    )


# 全局单例
settings = Settings()


def get_settings() -> Settings:
    """获取全局配置单例。"""
    return settings


def validate_security_settings(value: Settings | None = None) -> None:
    """Fail fast when production is started with an unsafe JWT secret."""
    current = value or settings
    if current.app_env not in {"production", "prod"}:
        return

    if current.secret_key == DEFAULT_SECRET_KEY or len(current.secret_key) < 32:
        raise RuntimeError(
            "生产环境必须通过 SECRET_KEY 配置至少 32 个字符的随机 JWT 密钥"
        )
    if "*" in current.cors_origins:
        raise RuntimeError("生产环境 CORS_ORIGINS 不能使用通配符 *")
