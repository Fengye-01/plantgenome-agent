"""
数据迁移脚本：给历史 chunks 补 user_id 归属。

背景：
    用户系统（JWT）上线前，scripts/ingest_all_pdfs.py 批量入库的 chunks
    没有 user_id 标签。开启向量库用户隔离后，where={"user_id": x} 会把
    这些无标签数据全部过滤掉，导致 admin 也检索不到旧知识库。

做法：
    把所有缺少 user_id 的 chunks 归属给指定用户（默认 admin，user_id=1）。
    Chroma 的 update 会整体替换 metadata，所以必须保留原有字段后再补 user_id。

运行：
    python scripts/migrate_chroma_userid.py            # 预览（dry-run，不写入）
    python scripts/migrate_chroma_userid.py --apply     # 真正写入
    python scripts/migrate_chroma_userid.py --apply --user-id 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.vector_store import VectorStore  # noqa: E402


def migrate(target_user_id: int, apply: bool) -> None:
    vs = VectorStore()
    collection = vs.collection

    all_data = collection.get(include=["metadatas"])
    ids = all_data["ids"]
    metadatas = all_data["metadatas"]

    missing_ids = []
    new_metadatas = []
    for cid, meta in zip(ids, metadatas):
        meta = dict(meta or {})
        if "user_id" not in meta:
            meta["user_id"] = target_user_id
            missing_ids.append(cid)
            new_metadatas.append(meta)

    total = len(ids)
    print("=" * 60)
    print(f"集合总 chunks: {total}")
    print(f"缺 user_id、待迁移: {len(missing_ids)}")
    print(f"已有 user_id、跳过: {total - len(missing_ids)}")
    print(f"目标归属 user_id: {target_user_id}")
    print("=" * 60)

    if not missing_ids:
        print("没有需要迁移的数据，退出。")
        return

    # 分批更新，避免一次性 payload 过大
    BATCH = 100
    if not apply:
        print("\n[dry-run] 未写入。确认无误后加 --apply 执行。")
        print("样例（前 3 条更新后的 metadata）：")
        for m in new_metadatas[:3]:
            print("  ", {k: (str(v)[:30]) for k, v in m.items()})
        return

    for start in range(0, len(missing_ids), BATCH):
        batch_ids = missing_ids[start:start + BATCH]
        batch_meta = new_metadatas[start:start + BATCH]
        collection.update(ids=batch_ids, metadatas=batch_meta)
        print(f"  已更新 {start + len(batch_ids)} / {len(missing_ids)}")

    # 校验
    after = collection.get(include=["metadatas"])
    still_missing = sum(1 for m in after["metadatas"] if not m or "user_id" not in m)
    print(f"\n迁移完成。迁移后仍缺 user_id 的 chunks: {still_missing}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="给历史 chunks 补 user_id")
    parser.add_argument("--user-id", type=int, default=1, help="归属目标用户 ID（默认 1=admin）")
    parser.add_argument("--apply", action="store_true", help="真正写入；不加则为预览")
    args = parser.parse_args()
    migrate(args.user_id, args.apply)
