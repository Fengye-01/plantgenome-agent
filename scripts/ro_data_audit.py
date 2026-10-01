"""只读数据审计：统计 SQL / Chroma 数据情况与 ID 交集。不修改任何数据。"""
import sqlite3
import sys
from pathlib import Path

import chromadb

DB = "data/plantgenome.db"
CHROMA = "data/chroma"

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

print("=" * 70)
print("【1】用户列表")
print("=" * 70)
for r in cur.execute("SELECT id, username FROM users"):
    print(f"  user {r['id']}: {r['username']}")

print("\n" + "=" * 70)
print("【2】每个用户的文档数 / 各状态")
print("=" * 70)
for r in cur.execute("""
    SELECT user_id, status, COUNT(*) c FROM documents GROUP BY user_id, status ORDER BY user_id
"""):
    print(f"  user {r['user_id']} | status={r['status']} | {r['c']} 篇")

print("\n" + "=" * 70)
print("【3】SQL DocumentChunk 统计")
print("=" * 70)
total = cur.execute("SELECT COUNT(*) FROM document_chunks").fetchone()[0]
empty = cur.execute(
    "SELECT COUNT(*) FROM document_chunks WHERE TRIM(COALESCE(content,''))=''"
).fetchone()[0]
no_cid = cur.execute(
    "SELECT COUNT(*) FROM document_chunks WHERE chroma_id IS NULL OR TRIM(chroma_id)=''"
).fetchone()[0]
print(f"  SQL chunk 总数: {total}")
print(f"  空正文: {empty}")
print(f"  空 chroma_id: {no_cid}")

# SQL chunk 按文档归属用户
print("\n  各用户拥有的 SQL chunk 数:")
for r in cur.execute("""
    SELECT d.user_id, COUNT(c.id) cnt
    FROM document_chunks c JOIN documents d ON c.document_id=d.id
    GROUP BY d.user_id
"""):
    print(f"    user {r['user_id']}: {r['cnt']} chunks")

print("\n" + "=" * 70)
print("【4】Chroma 统计与 ID 交集")
print("=" * 70)
client = chromadb.PersistentClient(path=CHROMA)
for col in client.list_collections():
    cc = client.get_collection(col.name)
    chroma_ids = set(cc.get()["ids"])
    print(f"  集合 {col.name}: {len(chroma_ids)} 条")

    sql_ids = set(
        r[0] for r in cur.execute(
            "SELECT chroma_id FROM document_chunks WHERE chroma_id IS NULL OR chroma_id<>''"
        )
    )
    inter = sql_ids & chroma_ids
    only_sql = sql_ids - chroma_ids
    only_chroma = chroma_ids - sql_ids
    print(f"  SQL∩Chroma 交集: {len(inter)}")
    print(f"  仅 SQL: {len(only_sql)}")
    print(f"  仅 Chroma: {len(only_chroma)}")

    # Chroma 中按 user_id metadata 分布
    got = cc.get()
    from collections import Counter
    users = Counter((m or {}).get("user_id") for m in got["metadatas"])
    print(f"  Chroma metadata user_id 分布: {dict(users)}")

conn.close()
print("\n只读审计完成（未修改任何数据）。")
