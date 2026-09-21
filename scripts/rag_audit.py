"""
RAG Evaluation Audit
=====================
对35条literature_search case做检索审计：
1. 直接调用VectorStore检索Top-5（不经过Agent）
2. 导出每条case的query/gold/Top-5结果详情
3. 分析当前HitRate@3判定规则的问题
4. 用语义相关性判断重新计算HitRate@1/3/5和MRR
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from tests.eval_dataset import EVAL_DATASET
from app.rag.vector_store import VectorStore


def run_audit():
    print("=" * 70)
    print("RAG Evaluation Audit")
    print("=" * 70)

    # 初始化
    print("\n[初始化] 加载 VectorStore...")
    vs = VectorStore()
    total_chunks = vs.count()
    print(f"  Chroma chunk 总数: {total_chunks}")

    # 只取 literature_search 类
    lit_cases = [c for c in EVAL_DATASET if c["category"] == "literature_search"]
    print(f"  literature_search case 数: {len(lit_cases)}")

    # 获取知识库中所有文件名列表（用于gold mapping）
    print("\n[知识库文档列表]")
    all_docs = vs.collection.get(include=["metadatas"])
    filenames = set()
    for meta in all_docs["metadatas"]:
        fname = meta.get("filename", "")
        if fname:
            filenames.add(fname)
    for f in sorted(filenames):
        print(f"  - {f}")

    # 对每条case检索Top-5
    results = []
    for case in lit_cases:
        query = case["question"]
        gold_keyword = case["expected_source"]

        raw = vs.search(query, top_k=5)

        top5 = []
        for i, r in enumerate(raw):
            top5.append({
                "rank": i + 1,
                "chunk_id": r["chunk_id"],
                "filename": r["metadata"].get("filename", ""),
                "page_num": r["metadata"].get("page_num", "?"),
                "heading_path": r["metadata"].get("heading_path", ""),
                "distance": round(r["distance"], 4),
                "similarity": round(1.0 - r["distance"] / 2.0, 4),
                "text_snippet": r["text"][:200].replace("\n", " "),
            })

        # 当前HitRate@3判定（旧规则：关键词出现在filename/snippet/heading_path中）
        old_hit = False
        for item in top5[:3]:
            searchable = (item["filename"] + " " + item["text_snippet"] + " " + item["heading_path"]).lower()
            if gold_keyword.lower() in searchable:
                old_hit = True
                break

        results.append({
            "id": case["id"],
            "question": query,
            "gold_keyword": gold_keyword,
            "expected_keywords": case["expected_keywords"],
            "old_hit_at_3": old_hit,
            "top5": top5,
        })

        # 打印简要结果
        hit_mark = "✅" if old_hit else "❌"
        print(f"\n  [{case['id']:2d}] {hit_mark} Q: {query[:50]}...")
        print(f"       gold_keyword: {gold_keyword}")
        for t in top5:
            print(f"       R{t['rank']}: d={t['distance']:.3f} sim={t['similarity']:.3f} "
                  f"| {t['filename'][:40]} | p{t['page_num']}")

    # 保存完整审计结果
    audit_path = PROJECT_ROOT / "docs" / "rag_audit_detail.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_chunks_in_store": total_chunks,
            "knowledge_base_files": sorted(filenames),
            "cases": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n完整审计结果已保存: {audit_path}")

    # 旧规则统计
    old_hits = sum(1 for r in results if r["old_hit_at_3"])
    print(f"\n[旧规则] HitRate@3: {old_hits}/{len(results)} = {old_hits/len(results):.1%}")

    return results, filenames


if __name__ == "__main__":
    run_audit()
