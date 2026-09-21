"""
阶段1：Corpus审计与去重
1. 列出Chroma中所有文档
2. 识别重复（相同文件名但不同document_id）
3. 删除重复chunk
4. 输出corpus_manifest
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from app.rag.vector_store import VectorStore


def run():
    print("=" * 70)
    print("阶段1：Corpus 审计")
    print("=" * 70)

    vs = VectorStore()
    total_chunks = vs.count()
    print(f"\n当前总chunk数: {total_chunks}")

    # 获取所有文档
    all_data = vs.collection.get(include=["metadatas", "documents"])
    ids = all_data["ids"]
    metadatas = all_data["metadatas"]

    # 按文件名分组
    by_filename = defaultdict(list)
    for i, meta in enumerate(metadatas):
        fname = meta.get("filename", "unknown")
        by_filename[fname].append({
            "chunk_id": ids[i],
            "page_num": meta.get("page_num"),
            "chunk_index": meta.get("chunk_index"),
            "document_id": meta.get("document_id", "N/A"),
            "heading_path": meta.get("heading_path", ""),
        })

    print(f"\n独立文件名数: {len(by_filename)}")
    print(f"\n{'='*70}")
    print(f"{'文件名':<60} {'chunk数':>6} {'doc_id数':>8}")
    print(f"{'='*70}")

    duplicates_to_remove = []
    unique_docs = []

    for fname in sorted(by_filename.keys()):
        chunks = by_filename[fname]
        doc_ids = set(c["document_id"] for c in chunks)
        n_chunks = len(chunks)
        n_docs = len(doc_ids)

        short_name = fname[:58]
        print(f"{short_name:<60} {n_chunks:>6} {n_docs:>8}")

        if n_docs > 1:
            # 重复文档：保留第一个document_id，删除其余
            doc_id_list = sorted(doc_ids)
            keep_doc_id = doc_id_list[0]
            remove_doc_ids = doc_id_list[1:]
            print(f"  ⚠️  重复! 保留 doc_id={keep_doc_id}, 删除 doc_ids={remove_doc_ids}")

            for c in chunks:
                if c["document_id"] in remove_doc_ids:
                    duplicates_to_remove.append(c["chunk_id"])

            unique_docs.append({
                "filename": fname,
                "document_id": keep_doc_id,
                "chunk_count": sum(1 for c in chunks if c["document_id"] == keep_doc_id),
            })
        else:
            unique_docs.append({
                "filename": fname,
                "document_id": list(doc_ids)[0],
                "chunk_count": n_chunks,
            })

    print(f"\n{'='*70}")
    print(f"需要删除的重复chunk数: {len(duplicates_to_remove)}")
    print(f"删除后预计chunk数: {total_chunks - len(duplicates_to_remove)}")

    # 执行删除
    if duplicates_to_remove:
        print(f"\n正在删除重复chunk...")
        vs.collection.delete(ids=duplicates_to_remove)
        remaining = vs.count()
        print(f"删除完成，剩余chunk数: {remaining}")
    else:
        remaining = total_chunks
        print("无重复需要删除")

    # 输出corpus_manifest
    manifest = {
        "total_chunks_after_dedup": remaining,
        "total_unique_documents": len(unique_docs),
        "documents": unique_docs,
        "removed_duplicate_chunks": len(duplicates_to_remove),
    }

    manifest_path = PROJECT_ROOT / "docs" / "corpus_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\ncorpus_manifest 已保存: {manifest_path}")
    print(f"\n最终独立文档数: {len(unique_docs)}")
    for d in unique_docs:
        print(f"  - {d['filename'][:60]} (doc_id={d['document_id']}, {d['chunk_count']} chunks)")


if __name__ == "__main__":
    run()
