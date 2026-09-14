"""
批量入库脚本：把 data/pdfs/ 下所有 PDF 全部解析、清洗、分块、入库到 Chroma。

对应 Hello Agents 第 8 章 RAG 的「Indexing（索引构建）」阶段。
用法：
    $env:HF_ENDPOINT = "https://hf-mirror.com"  # 第一次需要，模型缓存后不需要
    python scripts/ingest_all_pdfs.py

流程：
    清空旧数据 → 遍历 PDF → Markdown 解析 → 文本清洗 → 智能分块 → 向量化入库 → 生成报告
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from typing import List, Dict

# 确保项目根目录在 sys.path 中（脚本在 scripts/ 子目录下）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.rag.pdf_parser import PDFParser
from app.rag.text_cleaner import TextCleaner
from app.rag.chunker import Chunker
from app.rag.vector_store import VectorStore


def ingest_all_pdfs(pdf_dir: str = "data/pdfs") -> Dict:
    """
    批量入库所有 PDF。

    Args:
        pdf_dir: PDF 目录路径

    Returns:
        入库报告字典
    """
    # 初始化各模块
    print("=" * 60)
    print("PlantGenome Agent - 批量入库脚本")
    print("（Markdown 解析 → 文本清洗 → 智能分块 → BGE-m3 embedding → Chroma 入库）")
    print("=" * 60)

    print("\n[初始化] 加载各模块...")
    parser = PDFParser()
    cleaner = TextCleaner()
    chunker = Chunker()
    vector_store = VectorStore()
    print("  ✅ 所有模块加载完成")

    # 清空旧数据（保证幂等）
    print(f"\n[清空] 清空旧数据...")
    vector_store.clear()
    print(f"  ✅ 已清空，当前文档数: {vector_store.count()}")

    # 遍历 PDF
    if not os.path.isdir(pdf_dir):
        print(f"\n❌ PDF 目录不存在: {pdf_dir}")
        sys.exit(1)

    pdf_files = sorted([
        f for f in os.listdir(pdf_dir)
        if f.lower().endswith(".pdf")
    ])

    if not pdf_files:
        print(f"\n❌ 目录下没有 PDF 文件: {pdf_dir}")
        sys.exit(1)

    print(f"\n[发现] 找到 {len(pdf_files)} 个 PDF")

    # 处理每个 PDF
    report = {
        "start_time": datetime.now().isoformat(),
        "pdf_dir": pdf_dir,
        "total_pdfs": len(pdf_files),
        "results": [],
        "total_pages": 0,
        "total_chunks": 0,
        "failed_pdfs": [],
    }

    total_start_time = time.time()

    for idx, fname in enumerate(pdf_files, 1):
        fpath = os.path.join(pdf_dir, fname)
        print(f"\n{'─' * 60}")
        print(f"[{idx}/{len(pdf_files)}] 处理: {fname}")
        print(f"{'─' * 60}")

        pdf_start_time = time.time()

        try:
            # ① Markdown 解析
            print(f"  ① Markdown 解析...")
            pages = parser.parse_to_markdown(fpath)
            print(f"     ✅ {len(pages)} 页")

            # ② 文本清洗
            print(f"  ② 文本清洗...")
            cleaned_pages = cleaner.clean(pages)
            print(f"     ✅ {len(cleaned_pages)} 页（清洗后）")

            # ③ 智能分块
            print(f"  ③ 智能分块...")
            chunks = chunker.chunk(cleaned_pages)
            print(f"     ✅ {len(chunks)} 个 chunk")

            # 统计 chunk token 长度
            if chunks:
                chunk_tokens = [chunker.count_tokens(c["text"]) for c in chunks]
                avg_tokens = sum(chunk_tokens) // len(chunk_tokens)
                print(f"     平均 token 长度: {avg_tokens}")

            # ④ 向量化入库
            print(f"  ④ 向量化入库（BGE-m3 embedding + Chroma）...")
            added = vector_store.add_documents(chunks)
            print(f"     ✅ 入库 {added} 个文档")

            # 记录结果
            pdf_elapsed = time.time() - pdf_start_time
            result = {
                "filename": fname,
                "pages": len(pages),
                "cleaned_pages": len(cleaned_pages),
                "chunks": len(chunks),
                "avg_tokens": avg_tokens if chunks else 0,
                "elapsed_seconds": round(pdf_elapsed, 2),
                "status": "success",
            }
            report["results"].append(result)
            report["total_pages"] += len(pages)
            report["total_chunks"] += len(chunks)

            print(f"     ⏱️  耗时: {pdf_elapsed:.2f} 秒")

        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            report["failed_pdfs"].append({
                "filename": fname,
                "error": str(e),
            })
            report["results"].append({
                "filename": fname,
                "status": "failed",
                "error": str(e),
            })

    # 汇总
    total_elapsed = time.time() - total_start_time
    report["end_time"] = datetime.now().isoformat()
    report["total_elapsed_seconds"] = round(total_elapsed, 2)
    report["final_document_count"] = vector_store.count()

    print(f"\n{'=' * 60}")
    print("批量入库完成！")
    print(f"{'=' * 60}")
    print(f"  总 PDF 数: {report['total_pdfs']}")
    print(f"  成功: {report['total_pdfs'] - len(report['failed_pdfs'])} 个")
    print(f"  失败: {len(report['failed_pdfs'])} 个")
    print(f"  总页数: {report['total_pages']}")
    print(f"  总 chunk 数: {report['total_chunks']}")
    print(f"  Chroma 文档数: {report['final_document_count']}")
    print(f"  总耗时: {total_elapsed:.2f} 秒")

    if report["failed_pdfs"]:
        print(f"\n⚠️  失败的 PDF:")
        for f in report["failed_pdfs"]:
            print(f"  - {f['filename']}: {f['error']}")

    # 打印每个 PDF 的详细结果
    print(f"\n📊 入库报告:")
    print(f"{'文件名':<50} {'页数':>6} {'chunk数':>8} {'平均token':>10} {'耗时(s)':>8}")
    print(f"{'─' * 90}")
    for r in report["results"]:
        if r["status"] == "success":
            print(f"{r['filename']:<50} {r['pages']:>6} {r['chunks']:>8} {r['avg_tokens']:>10} {r['elapsed_seconds']:>8.2f}")
        else:
            print(f"{r['filename']:<50} {'失败':>6} {r['error'][:30]}")

    return report


if __name__ == "__main__":
    report = ingest_all_pdfs("data/pdfs")
