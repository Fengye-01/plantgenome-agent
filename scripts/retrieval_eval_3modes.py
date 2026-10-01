"""
三模式检索评估：dense_only / candidate_rrf / global_rrf
=====================================================
在【隔离临时环境】（独立 Chroma 目录 + 独立 SQLite）中重新规范入库一批 PDF，
随后用 Gold Set V2 的 18 条 answerable query 在同一数据集上分别跑三种模式，
计算 HitRate@1/3/5、MRR、查询延迟，并单独记录 case 18。

不读取、不修改、不删除历史 Chroma（data/chroma 的 447 条）与正式库。
结果写入 docs/retrieval_eval_3modes_<时间戳>.json，不覆盖历史评估。

运行：
    python scripts/retrieval_eval_3modes.py
"""
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# ── 1. 环境隔离（必须在 import app 之前设置）────────────────────────
EVAL_DIR = Path(tempfile.mkdtemp(prefix="pg_eval_"))
(EVAL_DIR / "chroma").mkdir(parents=True, exist_ok=True)
_db_path = (EVAL_DIR / "eval.db").as_posix()
os.environ["CHROMA_PERSIST_DIR"] = str(EVAL_DIR / "chroma")
os.environ["DATABASE_URL"] = f"sqlite:///{_db_path}"
os.environ.setdefault("RETRIEVAL_MODE", "candidate_rrf")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.database import SessionLocal, init_db  # noqa: E402
from app.models import Document, User  # noqa: E402
from app.rag.bm25_snapshot import get_snapshot_manager  # noqa: E402
from app.rag.vector_store import VectorStore  # noqa: E402
from app.tools.search_pdf_knowledge import search_pdf_knowledge  # noqa: E402
from app.worker.tasks import _ingest_pdf_to_db  # noqa: E402
from tests.eval_dataset import EVAL_DATASET  # noqa: E402

# ── 2. 短名 → 文件名片段（用于定位 PDF 与相关性判定）────────────────
DOC_FILES = {
    "PAML": "Yang - 2007 - PAML 4",
    "OrthoFinder": "Emms和Kelly - 2019 - OrthoFinder",
    "MAFFT": "Katoh和Standley - 2013 - MAFFT",
    "Research Progress": "Research Progress in the Molecular Functions",
    "Bock": "Bock_2007",
    "Ashikawa": "Ashikawa - 2001",
    "Cain": "Cain_2022_Intragenic",
    "DeepCpG": "Angermueller_2017_DeepCpG",
    "IQ-TREE": "Minh 等 - 2020 - IQ-TREE",
    "ModelFinder": "Kalyaanamoorthy 等 - 2017 - ModelFinder",
    "Epigenetic": "Epigenetic Regulation in Plants",
    "Pfam": "Mistry 等 - 2021 - Pfam",
}


def rank_of_relevant(sources, relevant_docs):
    """返回第一个相关结果的 1-based rank，无命中返回 None。"""
    for idx, s in enumerate(sources):
        fn = s.get("filename", "") or ""
        if any(short in fn for short in relevant_docs):
            return idx + 1
    return None


def main():
    print("隔离评估目录:", EVAL_DIR)
    settings = get_settings()

    # ── 3. 建表 + 用户 ────────────────────────────────────────────
    init_db()
    db = SessionLocal()
    db.add(User(id=1, username="admin", email="admin@eval.com",
                 hashed_password="x"))
    db.commit()

    # ── 4. 重新规范入库 PDF（写隔离 Chroma + SQL）─────────────────
    print("\n=== 入库 PDF ===")
    ingested = []
    for short, frag in DOC_FILES.items():
        matches = list((PROJECT_ROOT / "data/pdfs").glob(f"*{frag}*.pdf"))
        if not matches:
            print(f"  [跳过] 找不到 {short}: {frag}")
            continue
        pdf_path = matches[0]
        doc = Document(
            user_id=1, filename=pdf_path.name, file_path=str(pdf_path),
            file_size=pdf_path.stat().st_size, status="processing", chunk_count=0,
        )
        db.add(doc)
        db.flush()
        did = doc.id
        _ingest_pdf_to_db(did, 1, db)
        db.commit()
        ingested.append((short, did, pdf_path.name))
        print(f"  [doc {did:2d}] {short}: {pdf_path.name[:60]}")
    db.close()

    # ── 5. 加载 18 answerable query ───────────────────────────────
    gold = json.load(
        open(PROJECT_ROOT / "docs/rag_gold_v2.json", encoding="utf-8")
    )["labels"]
    answerable = []
    for i in range(1, 36):
        lab = gold[str(i)]
        if lab["label"] == "answerable":
            # 只保留相关文档已实际入库的 query
            if all(any(s == d[0] for d in ingested) for s in lab["relevant_docs"]):
                answerable.append(
                    (i, EVAL_DATASET[i - 1]["question"], lab["relevant_docs"])
                )
    print(f"\n参与评估的 answerable query: {len(answerable)}")

    # ── 6. 三模式评估 ─────────────────────────────────────────────
    by_mode = {}
    for mode in ["dense_only", "candidate_rrf", "global_rrf"]:
        settings.retrieval_mode = mode
        cases, latencies = [], []
        cold_ms, degrade, first = None, 0, True
        dense_recall, sparse_recall = [], []
        for qid, q, rel in answerable:
            t0 = time.perf_counter()
            r = search_pdf_knowledge(q, top_k=5, user_id=1)
            dt = (time.perf_counter() - t0) * 1000
            latencies.append(dt)
            if mode == "global_rrf" and first:
                cold_ms = dt  # 首次含快照冷构建
            first = False
            if r.get("degrade_reason"):
                degrade += 1
            rank = rank_of_relevant(r["sources"], rel)
            case = {"query_id": qid, "query": q, "rank": rank,
                    "relevant_docs": rel,
                    "top5_files": [s.get("filename", "")[:45]
                                   for s in r["sources"]]}
            # global：额外统计两路召回数量（不计入延迟）
            if mode == "global_rrf":
                store = VectorStore()
                dn = len(store.search(q, top_k=settings.dense_top_n,
                                      filter_metadata={"user_id": 1}))
                sn = len(get_snapshot_manager().get_snapshot(1).search(
                    q, top_n=settings.sparse_top_n))
                dense_recall.append(dn)
                sparse_recall.append(sn)
                case["dense_recall_n"] = dn
                case["sparse_recall_n"] = sn
            cases.append(case)

        n = len(cases)

        def hr(k):
            return sum(1 for c in cases if c["rank"] and c["rank"] <= k) / n

        mrr = sum(1 / c["rank"] for c in cases if c["rank"]) / n
        hot = (sum(latencies[1:]) / len(latencies[1:])
               if mode == "global_rrf" and len(latencies) > 1 else None)
        by_mode[mode] = {
            "n": n,
            "hits": sum(1 for c in cases if c["rank"]),
            "HitRate@1": round(hr(1), 4),
            "HitRate@3": round(hr(3), 4),
            "HitRate@5": round(hr(5), 4),
            "MRR": round(mrr, 4),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
            "cold_build_ms": round(cold_ms, 1) if cold_ms else None,
            "hot_avg_ms": round(hot, 1) if hot else None,
            "degrade_count": degrade,
            "avg_dense_recall": round(sum(dense_recall) / len(dense_recall), 1)
            if dense_recall else None,
            "avg_sparse_recall": round(sum(sparse_recall) / len(sparse_recall), 1)
            if sparse_recall else None,
            "cases": cases,
        }

    # ── 7. case 18 详细审计 ───────────────────────────────────────
    q18 = "什么是正向选择和纯化选择？怎么检测？"
    store = VectorStore()
    dense18 = store.search(q18, top_k=settings.dense_top_n,
                           filter_metadata={"user_id": 1})
    snap = get_snapshot_manager().get_snapshot(1)
    sparse18 = snap.search(q18, top_n=settings.sparse_top_n)

    def paml_rank_dense():
        for i, d in enumerate(dense18):
            if "PAML" in (d.get("metadata", {}).get("filename", "")):
                return i + 1, d["distance"], d["chunk_id"]
        return None

    def paml_rank_sparse():
        for i, (cid, sc) in enumerate(sparse18):
            ch = snap.chunks.get(cid)
            fn = (ch or {}).get("filename", "") if ch else ""
            if "PAML" in fn:
                return i + 1, sc, cid
        return None

    case18 = {
        "query": q18,
        "dense_channel_paml": paml_rank_dense(),
        "sparse_channel_paml": paml_rank_sparse(),
        "dense_top5_ids": [d["chunk_id"] for d in dense18[:5]],
        "sparse_top5": [(c, round(s, 3)) for c, s in sparse18[:5]],
    }
    for mode in ["dense_only", "candidate_rrf", "global_rrf"]:
        settings.retrieval_mode = mode
        r = search_pdf_knowledge(q18, top_k=5, user_id=1)
        case18[f"{mode}_rank"] = rank_of_relevant(r["sources"], ["PAML"])
        case18[f"{mode}_top5"] = [s.get("filename", "")[:40]
                                  for s in r["sources"]]

    # ── 8. 保存结果 ───────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "generated_at": datetime.now().isoformat(),
        "eval_dir": str(EVAL_DIR),
        "ingested_documents": [{"short": s, "document_id": d, "filename": f}
                               for s, d, f in ingested],
        "gold_source": "docs/rag_gold_v2.json (answerable only)",
        "note": "隔离临时环境重新入库语料；历史 data/chroma 447 条未参与。",
        "modes": by_mode,
        "case18": case18,
    }
    out_path = PROJECT_ROOT / f"docs/retrieval_eval_3modes_{ts}.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # ── 9. 打印汇总 ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("三模式检索指标对比（同一数据集，answerable n=%d）" %
          by_mode["dense_only"]["n"])
    print("=" * 72)
    print(f"{'模式':<16}{'HR@1':>8}{'HR@3':>8}{'HR@5':>8}{'MRR':>8}"
          f"{'均延迟ms':>10}")
    for mode in ["dense_only", "candidate_rrf", "global_rrf"]:
        m = by_mode[mode]
        print(f"{mode:<16}{m['HitRate@1']:>8.3f}{m['HitRate@3']:>8.3f}"
              f"{m['HitRate@5']:>8.3f}{m['MRR']:>8.3f}"
              f"{m['avg_latency_ms']:>10.1f}")
    g = by_mode["global_rrf"]
    print(f"\nglobal_rrf 冷构建: {g['cold_build_ms']}ms，"
          f"热查询均: {g['hot_avg_ms']}ms，降级次数: {g['degrade_count']}")
    print(f"global 平均 Dense 召回: {g['avg_dense_recall']}，"
          f"Sparse 召回: {g['avg_sparse_recall']}")
    print("\ncase 18:")
    print("  Dense通道 PAML rank:", case18["dense_channel_paml"])
    print("  Sparse通道 PAML rank:", case18["sparse_channel_paml"])
    for mode in ["dense_only", "candidate_rrf", "global_rrf"]:
        print(f"  {mode} 最终 rank:", case18[f"{mode}_rank"])
    print("\n结果已保存:", out_path)


if __name__ == "__main__":
    main()
