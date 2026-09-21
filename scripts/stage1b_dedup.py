"""
阶段1b：按内容去重
识别实际内容相同的重复文档并删除
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


# 手动确认的重复组：保留哪个document_id，删除哪些
# 基于审计输出确认
DUPLICATE_GROUPS = {
    # Bock 2007: 保留doc_id=2，删除doc_id=3
    "bock_2007": {
        "keep_doc_id": 2,
        "remove_doc_ids": [3],
        "description": "Bock_2007_CpG island mapping",
    },
    # PAML_test_upload: 保留doc_id=4，删除doc_id=7,8
    "paml_test": {
        "keep_doc_id": 4,
        "remove_doc_ids": [7, 8],
        "description": "PAML_test_upload.pdf",
    },
    # Cain 2022: 保留doc_id=6，删除无前缀版本
    # 无前缀版本的metadata中document_id不是6，需要用filename匹配
    "cain_2022": {
        "keep_filename_pattern": "1_1789201769_Cain_2022",
        "remove_filename_pattern": "Cain_2022_Intragenic",
        "description": "Cain_2022_Intragenic CpG islands",
    },
}


def run():
    print("=" * 70)
    print("阶段1b：按内容去重")
    print("=" * 70)

    vs = VectorStore()
    before = vs.count()
    print(f"\n去重前chunk数: {before}")

    all_data = vs.collection.get(include=["metadatas"])
    ids = all_data["ids"]
    metadatas = all_data["metadatas"]

    to_delete = []

    # 按document_id删除重复
    for group_name, group_info in DUPLICATE_GROUPS.items():
        if "remove_doc_ids" in group_info:
            for did in group_info["remove_doc_ids"]:
                for i, meta in enumerate(metadatas):
                    if meta.get("document_id") == did:
                        to_delete.append(ids[i])
            print(f"  {group_info['description']}: 删除 doc_ids={group_info['remove_doc_ids']}")

    # 按filename删除重复（Cain无前缀版本）
    for group_name, group_info in DUPLICATE_GROUPS.items():
        if "remove_filename_pattern" in group_info:
            pattern = group_info["remove_filename_pattern"]
            for i, meta in enumerate(metadatas):
                fname = meta.get("filename", "")
                if pattern in fname and "1_1789" not in fname:
                    to_delete.append(ids[i])
            print(f"  {group_info['description']}: 删除filename包含'{pattern}'的无前缀版本")

    print(f"\n将删除 {len(to_delete)} 个重复chunk")

    if to_delete:
        vs.collection.delete(ids=to_delete)
        after = vs.count()
        print(f"删除完成。去重后chunk数: {after}")
    else:
        after = before
        print("无重复需要删除")

    # 输出最终corpus_manifest
    all_data2 = vs.collection.get(include=["metadatas"])
    metadatas2 = all_data2["metadatas"]

    by_filename = defaultdict(list)
    for i, meta in enumerate(metadatas2):
        fname = meta.get("filename", "unknown")
        by_filename[fname].append(all_data2["ids"][i])

    unique_docs = []
    for fname in sorted(by_filename.keys()):
        unique_docs.append({
            "filename": fname,
            "chunk_count": len(by_filename[fname]),
        })

    manifest = {
        "total_chunks": after,
        "total_unique_documents": len(unique_docs),
        "documents": unique_docs,
        "removed_duplicate_chunks": len(to_delete),
    }

    manifest_path = PROJECT_ROOT / "docs" / "corpus_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"最终 Corpus Manifest:")
    print(f"  独立文档数: {len(unique_docs)}")
    print(f"  总chunk数: {after}")
    print(f"  删除重复chunk: {len(to_delete)}")
    print(f"{'='*70}")
    for d in unique_docs:
        print(f"  - {d['filename'][:65]} ({d['chunk_count']} chunks)")
    print(f"\nmanifest已保存: {manifest_path}")


if __name__ == "__main__":
    run()
