"""
阶段5：Hybrid Retrieval (Dense + BM25 + RRF)
- Dense: BGE-m3 + Chroma
- BM25: 对所有chunk文本建立BM25索引
- RRF: Reciprocal Rank Fusion, k=60
"""
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from rank_bm25 import BM25Okapi
from tests.eval_dataset import EVAL_DATASET
from app.rag.vector_store import VectorStore

# 复用Gold V2标注
GOLD_V2_PATH = PROJECT_ROOT / "docs" / "rag_gold_v2.json"
with open(GOLD_V2_PATH, "r", encoding="utf-8") as f:
    GOLD_V2 = json.load(f)["labels"]


def tokenize(text: str) -> list:
    """简单分词：英文按单词，中文按字，保留数字和专业术语。"""
    # 小写化
    text = text.lower()
    # 英文单词+数字
    tokens = re.findall(r'[a-z0-9_]+', text)
    # 中文字符单字切分
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    tokens.extend(chinese_chars)
    return tokens


def is_relevant_doc(filename: str, relevant_keywords: list) -> bool:
    fname_lower = filename.lower()
    for kw in relevant_keywords:
        if kw.lower() in fname_lower:
            return True
    return False


def rrf_fuse(dense_results: list, bm25_results: list, k: int = 60, top_n: int = 10) -> list:
    """
    Reciprocal Rank Fusion.
    score(d) = sum_i 1 / (k + rank_i(d))
    """
    scores = {}
    chunk_map = {}

    for rank, item in enumerate(dense_results):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        chunk_map[cid] = item

    for rank, item in enumerate(bm25_results):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        if cid not in chunk_map:
            chunk_map[cid] = item

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused = []
    for i, cid in enumerate(sorted_ids[:top_n]):
        item = chunk_map[cid]
        fused.append({
            "rank": i + 1,
            "chunk_id": cid,
            "filename": item["metadata"].get("filename", "")[:55],
            "distance": item.get("distance", 0),
            "rrf_score": round(scores[cid], 6),
        })

    return fused


def run():
    print("=" * 70)
    print("阶段5：Hybrid Retrieval (Dense + BM25 + RRF)")
    print("=" * 70)

    vs = VectorStore()

    # 获取所有chunk用于BM25索引
    all_data = vs.collection.get(include=["metadatas", "documents"])
    all_ids = all_data["ids"]
    all_texts = all_data["documents"]
    all_metas = all_data["metadatas"]

    print(f"\n建立BM25索引: {len(all_texts)} chunks...")
    tokenized_corpus = [tokenize(text) for text in all_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    print("  BM25索引构建完成")

    # 只评估answerable queries
    lit_cases = [c for c in EVAL_DATASET if c["category"] == "literature_search"]
    answerable_cases = [c for c in lit_cases if GOLD_V2[str(c["id"])]["label"] == "answerable"]

    results = []
    print(f"\n{'='*70}")
    print(f"Hybrid检索结果（{len(answerable_cases)}条answerable queries）")
    print(f"{'='*70}")

    for case in answerable_cases:
        query = case["question"]
        gold = GOLD_V2[str(case["id"])]
        relevant_docs = gold["relevant_docs"]

        # Dense检索 Top-10
        t0 = time.time()
        dense_raw = vs.search(query, top_k=10)
        dense_time = time.time() - t0

        # BM25检索 Top-10
        t0 = time.time()
        query_tokens = tokenize(query)
        bm25_scores = bm25.get_scores(query_tokens)
        # 取Top-10
        bm25_top_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
        bm25_raw = []
        for idx in bm25_top_idx:
            bm25_raw.append({
                "chunk_id": all_ids[idx],
                "metadata": all_metas[idx],
                "text": all_texts[idx][:200],
            })
        bm25_time = time.time() - t0

        # RRF融合
        fused = rrf_fuse(dense_raw, bm25_raw, k=60, top_n=10)

        # 判断相关性
        first_relevant = None
        for item in fused[:5]:
            if is_relevant_doc(item["filename"], relevant_docs):
                first_relevant = item["rank"]
                break

        hit1 = first_relevant == 1
        hit3 = first_relevant is not None and first_relevant <= 3
        hit5 = first_relevant is not None and first_relevant <= 5
        rr = 1.0 / first_relevant if first_relevant else 0.0

        results.append({
            "id": case["id"],
            "question": query[:60],
            "first_relevant_rank": first_relevant,
            "hit@1": hit1,
            "hit@3": hit3,
            "hit@5": hit5,
            "rr": rr,
            "total_time_ms": round((dense_time + bm25_time) * 1000, 1),
            "top5": [{"rank": f["rank"], "filename": f["filename"], "rrf": f["rrf_score"]} for f in fused[:5]],
        })

        mark = "✅" if hit3 else ("⚠️" if hit5 else "❌")
        fr = f"R{first_relevant}" if first_relevant else "miss"
        print(f"  [{case['id']:2d}] {mark} {fr:>5} | {query[:45]}...")

    # 汇总
    n = len(results)
    h1 = sum(1 for r in results if r["hit@1"])
    h3 = sum(1 for r in results if r["hit@3"])
    h5 = sum(1 for r in results if r["hit@5"])
    mrr = sum(r["rr"] for r in results) / n
    avg_latency = sum(r["total_time_ms"] for r in results) / n

    print(f"\n{'='*70}")
    print(f"Hybrid Retrieval (Dense + BM25 + RRF) Baseline（n={n}）")
    print(f"{'='*70}")
    print(f"  HitRate@1:  {h1}/{n} = {h1/n:.1%}")
    print(f"  HitRate@3:  {h3}/{n} = {h3/n:.1%}")
    print(f"  HitRate@5:  {h5}/{n} = {h5/n:.1%}")
    print(f"  MRR:        {mrr:.3f}")
    print(f"  平均延迟:   {avg_latency:.0f}ms")

    # 保存
    output = {
        "method": "Dense (BGE-m3) + BM25 + RRF(k=60)",
        "summary": {
            "answerable_n": n,
            "hit@1": h1,
            "hit@1_rate": round(h1/n, 4),
            "hit@3": h3,
            "hit@3_rate": round(h3/n, 4),
            "hit@5": h5,
            "hit@5_rate": round(h5/n, 4),
            "mrr": round(mrr, 4),
            "avg_latency_ms": round(avg_latency, 1),
        },
        "details": results,
    }

    out_path = PROJECT_ROOT / "docs" / "hybrid_baseline.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    run()
