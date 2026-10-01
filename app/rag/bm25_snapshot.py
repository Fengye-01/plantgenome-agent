"""
用户级全库 BM25 快照管理器（global_rrf 的稀疏检索通道）。

解决的问题
----------
旧链路（candidate_rrf）先由 Chroma 召回 top_k*5 个候选，再在这 15 个候选里
临时构建 BM25。BM25 被候选集框死，无法补回 Dense 压根没召回的 chunk。

本模块为每个用户在进程内维护一份"全库 BM25 快照"：
  - 首次查询时懒加载：从 SQL（SQLite/PostgreSQL）按 user_id 读取该用户
    completed 文档的全部合格 chunk，用已有 BM25Index 构建一次；
  - 构建成功后原子替换，构建失败保留旧快照；
  - 用户级锁（SingleFlight），同一进程同一用户并发只构建一次；
  - 跨进程失效：Worker（独立进程）入库/删除成功后递增 Redis 版本键，
    API 使用快照前比较版本；Redis 不可用时退化为进程内 TTL 兜底。

设计边界（MVP，不是生产架构）
----------------------------
  - 融合键为 DocumentChunk.chroma_id（当前为 doc{document_id}_chunk{index}，
    在当前文档代次内有效；重新切分后不保证内容身份稳定，稳定 UUID 属后续架构）；
  - 不做分布式锁、不跨进程共享 Python BM25 对象、不引入 OpenSearch；
  - 只加载 SQL 中有记录的 chunk，历史遗留的孤儿向量不会进入快照。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from sqlalchemy import text

from app.rag.bm25_store import BM25Index

# 版本键前缀（与设计文档一致；集中在此处便于统一命名）
BM25_VERSION_KEY = "rag:bm25:version:{user_id}"


# ═══════════════════════════════════════════════════════════════
# 快照数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class BM25Snapshot:
    """单个用户的不可变 BM25 快照。构建完成后整体替换，不就地修改。"""

    user_id: int
    corpus_version: int                      # 构建时对应的 Redis 版本（Redis 不可用时为 0）
    built_at: float                          # 构建时间戳（TTL 判断用）
    index: BM25Index                         # 全库 BM25 索引
    chunk_ids: List[str]                     # 位置 -> chunk_id（与 BM25 文档位置一一对应）
    chunks: Dict[str, dict] = field(default_factory=dict)  # chunk_id -> 正文/元数据
    stale: bool = False                      # 构建失败、沿用旧快照时标记

    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    def search(self, query: str, top_n: int) -> List[tuple]:
        """
        全库 BM25 检索。

        Returns:
            List[(chunk_id, bm25_score)]，已按分数降序、过滤 0 分。
        """
        hits = self.index.search(query, top_k=None)
        out: List[tuple] = []
        for pos, score in hits:
            if score <= 0:
                continue  # 无有效词匹配的结果不作为有效 Sparse 证据
            out.append((self.chunk_ids[pos], score))
            if len(out) >= top_n:
                break
        return out


# ═══════════════════════════════════════════════════════════════
# 从 SQL 加载用户 chunk
# ═══════════════════════════════════════════════════════════════

# DocumentChunk 不直接含 user_id，通过 JOIN documents 显式取得所有权。
# 只包含：completed 文档、非空正文、非空 chroma_id（共同融合键）。
# 当前模型无软删除字段（删除为硬删除），因此无需排除软删除。
_LOAD_SQL = text(
    """
    SELECT c.document_id, c.chunk_index, c.content, c.page_start, c.page_end,
           c.section, c.chroma_id, d.filename
    FROM document_chunks AS c
    JOIN documents AS d ON c.document_id = d.id
    WHERE d.user_id = :uid
      AND d.status = 'completed'
      AND c.chroma_id IS NOT NULL AND TRIM(c.chroma_id) <> ''
      AND TRIM(COALESCE(c.content, '')) <> ''
    ORDER BY c.document_id, c.chunk_index
    """
)


def _build_snapshot(session, user_id: int, corpus_version: int) -> BM25Snapshot:
    """从 SQL 加载用户合格 chunk 并构建全库 BM25 快照。"""
    rows = session.execute(_LOAD_SQL, {"uid": user_id}).fetchall()

    documents: List[str] = []
    chunk_ids: List[str] = []
    chunks: Dict[str, dict] = {}

    for row in rows:
        chroma_id = row.chroma_id
        # 防御：同一 chroma_id 不重复加入（理论上 chroma_id 唯一）
        if chroma_id in chunks:
            continue
        content = row.content or ""
        documents.append(content)
        chunk_ids.append(chroma_id)
        chunks[chroma_id] = {
            "chunk_id": chroma_id,
            "text": content,
            "document_id": row.document_id,
            "filename": row.filename,
            "chunk_index": row.chunk_index,
            "page_num": row.page_start,
            "section": row.section,
        }

    index = BM25Index(documents)
    return BM25Snapshot(
        user_id=user_id,
        corpus_version=corpus_version or 0,
        built_at=time.time(),
        index=index,
        chunk_ids=chunk_ids,
        chunks=chunks,
    )


# ═══════════════════════════════════════════════════════════════
# Redis 版本键（跨进程失效）
# ═══════════════════════════════════════════════════════════════

def _redis_client():
    """惰性创建 redis-py 客户端（复用 settings.redis_url，不另建连接配置）。"""
    import redis

    from app.core.config import get_settings

    return redis.Redis.from_url(get_settings().redis_url, decode_responses=True)


def get_bm25_version(user_id: int) -> Optional[int]:
    """
    读取用户 BM25 语料版本。

    Returns:
        int 版本（键不存在返回 0）；Redis 不可用返回 None（调用方改用 TTL）。
    """
    try:
        client = _redis_client()
        value = client.get(BM25_VERSION_KEY.format(user_id=user_id))
        return int(value) if value is not None else 0
    except Exception:
        return None


def bump_bm25_version(user_id: int) -> Optional[int]:
    """
    原子递增用户 BM25 版本（Worker 在入库/删除确认成功后调用）。

    Returns:
        新版本号；Redis 不可用返回 None（主流程不因此回滚）。
    """
    try:
        client = _redis_client()
        return int(client.incr(BM25_VERSION_KEY.format(user_id=user_id)))
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
# 快照管理器（进程内缓存 + 用户级 SingleFlight）
# ═══════════════════════════════════════════════════════════════

class BM25SnapshotManager:
    """
    每个 API 进程一个实例，维护 {user_id: BM25Snapshot}。

    项目当前为同步调用链（Tool 函数是普通 def），因此使用 threading.Lock
    而非 asyncio.Lock。
    """

    def __init__(
        self,
        session_factory: Callable,
        ttl_seconds: float = 300.0,
        version_provider: Callable[[int], Optional[int]] = get_bm25_version,
    ):
        """
        Args:
            session_factory: 无参可调用，返回 SQLAlchemy Session 上下文管理器
                            （生产用 get_session；测试可传入临时 session 工厂）
            ttl_seconds: Redis 不可用 / 版本缺失时的进程内兜底 TTL
            version_provider: 返回用户当前 Redis 版本（None 表示 Redis 不可用）
        """
        self._session_factory = session_factory
        self._ttl = ttl_seconds
        self._version_provider = version_provider
        self._snapshots: Dict[int, BM25Snapshot] = {}
        self._locks: Dict[int, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _user_lock(self, user_id: int) -> threading.Lock:
        """获取（必要时创建）用户级构建锁；不同用户互不阻塞。"""
        with self._locks_guard:
            lock = self._locks.get(user_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[user_id] = lock
            return lock

    def _needs_rebuild(
        self, snap: Optional[BM25Snapshot], expected_version: Optional[int]
    ) -> bool:
        if snap is None:
            return True
        # 能拿到 Redis 版本：版本不一致即重建
        if expected_version is not None:
            return snap.corpus_version != expected_version
        # Redis 不可用：用进程内 TTL 兜底
        return (time.time() - snap.built_at) > self._ttl

    def get_snapshot(self, user_id: int) -> Optional[BM25Snapshot]:
        """
        获取用户快照（懒加载 / 按版本或 TTL 重建）。

        降级规则：
          - 构建成功：原子替换旧快照并返回；
          - 构建失败但有旧快照：返回标记 stale 的旧快照；
          - 构建失败且无旧快照：抛给调用方（由检索层降级到 candidate/dense）。
        """
        expected_version = self._version_provider(user_id)
        snap = self._snapshots.get(user_id)

        if not self._needs_rebuild(snap, expected_version):
            return snap

        lock = self._user_lock(user_id)
        with lock:
            # Double-check：等待锁期间可能已被其他线程构建
            expected_version = self._version_provider(user_id)
            snap = self._snapshots.get(user_id)
            if not self._needs_rebuild(snap, expected_version):
                return snap

            version_for_build = expected_version if expected_version is not None else 0
            try:
                with self._session_factory() as session:
                    new_snap = _build_snapshot(session, user_id, version_for_build)
            except Exception:
                # 构建失败：保留旧快照并标记 stale；无旧快照则向上抛
                if snap is not None:
                    snap.stale = True
                    return snap
                raise

            self._snapshots[user_id] = new_snap  # 原子替换引用
            return new_snap

    def invalidate(self, user_id: int) -> None:
        """
        清除本进程内用户快照（用于同进程删除/测试）。

        注意：Worker 是独立进程，无法通过本方法通知 API，跨进程失效必须
        依赖 Redis 版本键。
        """
        self._snapshots.pop(user_id, None)


# ═══════════════════════════════════════════════════════════════
# 进程级单例（生产检索使用；测试自行实例化独立 Manager）
# ═══════════════════════════════════════════════════════════════

_default_manager: Optional[BM25SnapshotManager] = None
_default_lock = threading.Lock()


def get_snapshot_manager() -> BM25SnapshotManager:
    """返回进程级默认快照管理器（惰性单例）。"""
    global _default_manager
    if _default_manager is None:
        with _default_lock:
            if _default_manager is None:
                from app.core.config import get_settings
                from app.core.database import get_session

                settings = get_settings()
                _default_manager = BM25SnapshotManager(
                    session_factory=get_session,
                    ttl_seconds=settings.bm25_snapshot_ttl,
                    version_provider=get_bm25_version,
                )
    return _default_manager
