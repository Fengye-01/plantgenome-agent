"""
用户级全库 BM25 独立召回 MVP 的单元测试。

不依赖真实 Chroma / Redis / LLM：
  - 用内存 SQLite + 真实 ORM 模型构造受控语料；
  - BM25Index / BM25SnapshotManager 为纯计算，真实运行；
  - VectorStore 用 Fake（返回可控 Dense 结果）；
  - Redis 版本通过可控 version_provider 模拟。
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.core.config import get_settings
from app.models import Document, DocumentChunk, User
from app.rag.bm25_snapshot import BM25SnapshotManager
from app.tools import search_pdf_knowledge as search_module


# ═══════════════════════════════════════════════════════════════
# 测试语料 fixture
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def db_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def factory():
        s = Session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # 填充数据
    with factory() as db:
        db.add_all([User(id=1, username="a", email="a@x.com",
                         hashed_password="x"),
                    User(id=2, username="b", email="b@x.com",
                         hashed_password="x")])

        # user1: doc1 completed
        d1 = Document(id=1, user_id=1, filename="paml.pdf", file_path="x",
                      file_size=1, status="completed", chunk_count=4)
        # user1: doc2 processing（不应入快照）
        d2 = Document(id=2, user_id=1, filename="processing.pdf", file_path="x",
                      file_size=1, status="processing", chunk_count=1)
        # user2: doc3 completed（隔离）
        d3 = Document(id=3, user_id=2, filename="user2.pdf", file_path="x",
                      file_size=1, status="completed", chunk_count=1)
        db.add_all([d1, d2, d3])
        db.flush()

        db.add_all([
            DocumentChunk(document_id=1, chunk_index=0,
                          content="PAML branch site model positive selection dN dS analysis",
                          page_start=1, page_end=1, section="Methods",
                          chroma_id="doc1_chunk0"),
            DocumentChunk(document_id=1, chunk_index=1,
                          content="MAFFT multiple sequence alignment algorithm guide",
                          page_start=2, page_end=2, section="Tools",
                          chroma_id="doc1_chunk1"),
            DocumentChunk(document_id=1, chunk_index=2,
                          content="   ", page_start=3, page_end=3,
                          section="Empty", chroma_id="doc1_chunk2"),  # 空正文
            DocumentChunk(document_id=1, chunk_index=3,
                          content="missing fusion id chunk", page_start=4,
                          page_end=4, section="X", chroma_id=None),  # 空ID
            DocumentChunk(document_id=2, chunk_index=0,
                          content="should not load processing document",
                          page_start=1, page_end=1, section="X",
                          chroma_id="doc2_chunk0"),
            DocumentChunk(document_id=3, chunk_index=0,
                          content="user two private content PAML secret",
                          page_start=1, page_end=1, section="X",
                          chroma_id="doc3_chunk0"),
        ])

    return factory


def _manager(db_factory, versions, ttl=300.0):
    """构造版本可控的快照管理器。versions 可为 int 或 callable。"""
    provider = versions if callable(versions) else (lambda uid: versions)
    return BM25SnapshotManager(db_factory, ttl_seconds=ttl,
                               version_provider=provider)


# ═══════════════════════════════════════════════════════════════
# 一、快照加载 / 隔离 / 过滤
# ═══════════════════════════════════════════════════════════════

def test_snapshot_filters_and_isolates(db_factory):
    snap = _manager(db_factory, 0).get_snapshot(1)
    # user1 仅 doc1 的 2 个合格 chunk（排除空正文/空ID/processing）
    assert set(snap.chunk_ids) == {"doc1_chunk0", "doc1_chunk1"}

    snap2 = _manager(db_factory, 0).get_snapshot(2)
    assert snap2.chunk_ids == ["doc3_chunk0"]  # user2 隔离


def test_snapshot_excludes_empty_and_missing_id(db_factory):
    snap = _manager(db_factory, 0).get_snapshot(1)
    assert "doc1_chunk2" not in snap.chunks  # 空正文
    assert all(cid for cid in snap.chunk_ids)  # 无空融合ID


def test_empty_corpus_returns_zero_snapshot(db_factory):
    # user 99 无语料
    snap = _manager(db_factory, 0).get_snapshot(99)
    assert snap.size == 0


# ═══════════════════════════════════════════════════════════════
# 二、版本缓存 / 重建 / TTL / Redis 不可用
# ═══════════════════════════════════════════════════════════════

def test_same_version_no_rebuild(db_factory):
    calls = {"n": 0}

    @contextmanager
    def counting():
        calls["n"] += 1
        with db_factory() as s:
            yield s

    mgr = _manager(counting, 5)
    mgr.get_snapshot(1)
    mgr.get_snapshot(1)
    assert calls["n"] == 1  # 同版本只构建一次


def test_version_change_triggers_rebuild(db_factory):
    versions = iter([0, 0, 1, 1])
    mgr = _manager(db_factory, lambda uid: next(versions))
    s1 = mgr.get_snapshot(1)
    s2 = mgr.get_snapshot(1)
    assert s2.corpus_version == 1
    assert s2 is not s1


def test_redis_unavailable_uses_ttl(db_factory):
    # version_provider 始终 None（Redis 不可用）
    mgr = _manager(db_factory, lambda uid: None, ttl=300)
    snap = mgr.get_snapshot(1)
    assert snap.size == 2  # 仍能构建
    assert mgr.get_snapshot(1) is snap  # TTL 内不重建


def test_ttl_expiry_rebuild(db_factory):
    import time as _t
    mgr = _manager(db_factory, lambda uid: None, ttl=300)
    snap = mgr.get_snapshot(1)
    snap.built_at = _t.time() - 400  # 模拟 TTL 过期
    assert mgr.get_snapshot(1) is not snap


def test_build_failure_keeps_stale_snapshot(db_factory):
    state = {"v": 1}
    mgr = _manager(db_factory, lambda uid: state["v"])
    good = mgr.get_snapshot(1)
    state["v"] = 2  # 触发下次重建

    @contextmanager
    def broken():
        raise RuntimeError("db down")
        yield  # pragma: no cover

    mgr._session_factory = broken
    stale = mgr.get_snapshot(1)  # 新版本触发重建，但失败
    assert stale is good and stale.stale is True


# ═══════════════════════════════════════════════════════════════
# 三、独立双路召回 + RRF（核心：补回 Dense 漏召回）
# ═══════════════════════════════════════════════════════════════

class FakeDenseStore:
    """可控 Dense：只返回 doc1_chunk1（漏掉真正相关的 doc1_chunk0）。"""
    def __init__(self, records):
        self.records = records

    def search(self, query, top_k=None, filter_metadata=None):
        out = self.records
        if filter_metadata and "user_id" in filter_metadata:
            out = [r for r in out
                   if r["metadata"].get("user_id") == filter_metadata["user_id"]]
        return out[:top_k] if top_k else out


_DENSE_MISS = [
    {
        "chunk_id": "doc1_chunk1",
        "text": "MAFFT multiple sequence alignment algorithm guide",
        "metadata": {"user_id": 1, "filename": "paml.pdf", "page_num": 2,
                     "section": "Tools", "chunk_index": 1, "document_id": 1},
        "distance": 0.3,
    }
]


def test_global_rrf_recovers_dense_miss(monkeypatch, db_factory):
    monkeypatch.setattr(get_settings(), "retrieval_mode", "global_rrf")
    mgr = _manager(db_factory, 0)
    monkeypatch.setattr(search_module, "get_snapshot_manager", lambda: mgr)
    monkeypatch.setattr(search_module, "VectorStore",
                        lambda *a, **k: FakeDenseStore(_DENSE_MISS))

    result = search_module.search_pdf_knowledge(
        "PAML positive selection dN dS", top_k=3, user_id=1
    )
    ids = [s["chunk_id"] for s in result["sources"]]
    assert "doc1_chunk0" in ids  # 全库 BM25 补回
    assert result["retrieval_mode"] == "global_rrf"


def test_candidate_rrf_cannot_recover_miss(monkeypatch, db_factory):
    # candidate 的 Dense 候选里没有 doc1_chunk0，BM25 无法补回
    monkeypatch.setattr(get_settings(), "retrieval_mode", "candidate_rrf")
    monkeypatch.setattr(search_module, "VectorStore",
                        lambda *a, **k: FakeDenseStore(_DENSE_MISS))
    result = search_module.search_pdf_knowledge(
        "PAML positive selection dN dS", top_k=3, user_id=1
    )
    ids = [s["chunk_id"] for s in result["sources"]]
    assert "doc1_chunk0" not in ids  # candidate 补不回


# ═══════════════════════════════════════════════════════════════
# 四、分通道证据门控
# ═══════════════════════════════════════════════════════════════

def test_bm25_only_passes_without_dense_distance(monkeypatch, db_factory):
    monkeypatch.setattr(get_settings(), "retrieval_mode", "global_rrf")
    mgr = _manager(db_factory, 0)
    monkeypatch.setattr(search_module, "get_snapshot_manager", lambda: mgr)
    monkeypatch.setattr(search_module, "VectorStore",
                        lambda *a, **k: FakeDenseStore(_DENSE_MISS))
    result = search_module.search_pdf_knowledge(
        "PAML positive selection", top_k=3, user_id=1
    )
    chunk0 = [s for s in result["sources"] if s["chunk_id"] == "doc1_chunk0"]
    assert chunk0  # BM25-only 不因缺 distance 报错


def test_dense_over_threshold_excluded_without_sparse(monkeypatch, db_factory):
    # Dense 返回距离超标的 chunk，且该词不被 BM25 命中
    records = [{
        "chunk_id": "doc1_chunk1",
        "text": "MAFFT multiple sequence alignment algorithm guide",
        "metadata": {"user_id": 1, "filename": "paml.pdf", "page_num": 2,
                     "section": "Tools", "chunk_index": 1, "document_id": 1},
        "distance": 0.95,  # 超阈值
    }]
    mgr = _manager(db_factory, 0)
    monkeypatch.setattr(get_settings(), "retrieval_mode", "global_rrf")
    monkeypatch.setattr(search_module, "get_snapshot_manager", lambda: mgr)
    monkeypatch.setattr(search_module, "VectorStore",
                        lambda *a, **k: FakeDenseStore(records))
    # query 与语料无语义/词法匹配：Dense 无候选，BM25 0 分 → 无证据
    result = search_module.search_pdf_knowledge(
        "zzzz unrelated xyz qqq", top_k=3, user_id=1
    )
    assert result["has_result"] is False  # 无证据 → 拒答


def test_no_evidence_triggers_abstention_shape(monkeypatch, db_factory):
    monkeypatch.setattr(get_settings(), "retrieval_mode", "global_rrf")
    mgr = _manager(db_factory, 0)
    monkeypatch.setattr(search_module, "get_snapshot_manager", lambda: mgr)
    monkeypatch.setattr(search_module, "VectorStore",
                        lambda *a, **k: FakeDenseStore([]))
    result = search_module.search_pdf_knowledge(
        "nothing matches here", top_k=3, user_id=1
    )
    assert result["has_result"] is False
    assert result["evidence_status"] == "insufficient"


# ═══════════════════════════════════════════════════════════════
# 五、模式选择 / 非法值 / 降级
# ═══════════════════════════════════════════════════════════════

def test_illegal_mode_falls_back(monkeypatch, db_factory):
    import app.core.config as cfg
    monkeypatch.setattr(cfg.get_settings(), "retrieval_mode", "weird_mode")
    monkeypatch.setattr(search_module, "VectorStore",
                        lambda *a, **k: FakeDenseStore(_DENSE_MISS))
    result = search_module.search_pdf_knowledge("MAFFT", top_k=3, user_id=1)
    assert result["retrieval_mode"] == "candidate_rrf"


def test_global_snapshot_failure_degrades(monkeypatch, db_factory):
    monkeypatch.setattr(get_settings(), "retrieval_mode", "global_rrf")

    class DeadManager:
        def get_snapshot(self, uid):
            raise RuntimeError("snapshot build failed")

    monkeypatch.setattr(search_module, "get_snapshot_manager", DeadManager)
    monkeypatch.setattr(search_module, "VectorStore",
                        lambda *a, **k: FakeDenseStore(_DENSE_MISS))
    result = search_module.search_pdf_knowledge("MAFFT", top_k=3, user_id=1)
    assert result["retrieval_mode"] == "candidate_rrf"
    assert result.get("degrade_reason") == "bm25_snapshot_failed"


def test_user_isolation_in_global(monkeypatch, db_factory):
    monkeypatch.setattr(get_settings(), "retrieval_mode", "global_rrf")
    mgr = _manager(db_factory, 0)
    monkeypatch.setattr(search_module, "get_snapshot_manager", lambda: mgr)

    # Dense 只放 user2 的内容，user1 检索不应得到
    other = [{
        "chunk_id": "doc3_chunk0",
        "text": "user two private content PAML secret",
        "metadata": {"user_id": 2, "filename": "user2.pdf", "page_num": 1,
                     "section": "X", "chunk_index": 0, "document_id": 3},
        "distance": 0.2,
    }]
    monkeypatch.setattr(search_module, "VectorStore",
                        lambda *a, **k: FakeDenseStore(other))
    result = search_module.search_pdf_knowledge("PAML", top_k=3, user_id=1)
    assert "doc3_chunk0" not in [s["chunk_id"] for s in result["sources"]]
