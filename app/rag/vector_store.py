"""
向量存储模块（RAG 流水线第 4-5 步：向量化 + 向量存储）

对应 Hello Agents 第 8 章 RAG 的「Embedding」+「Vector Store」阶段。
职责：用 BGE-m3 把 chunk 转成向量，存入 Chroma 向量数据库，提供检索接口。

为什么用 BGE-m3？
- 多语言支持（中文+英文），对生物专有名词支持好
- 1024 维向量，检索精度高
- 开源免费，可本地运行，不需要 API Key

为什么用 Chroma？
- 轻量级，Python 原生，安装简单
- 支持持久化到磁盘，适合 MVP
- API 简洁，上手快
- 以后可以无缝迁移到 Milvus（企业级）
"""
from __future__ import annotations

import os
import time
from typing import List, Dict, Optional

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()


def _is_lock_error(e: Exception) -> bool:
    """
    判断异常是否为数据库锁相关错误。

    Chroma 底层用 SQLite，并发读写时会触发：
    - OperationalError: database is locked
    - sqlite3.OperationalError
    - 包含 "lock" / "locked" / "busy" 关键字的错误
    """
    error_msg = str(e).lower()
    lock_keywords = ["lock", "locked", "busy", "database is locked", "operationalerror"]
    return any(kw in error_msg for kw in lock_keywords)


def _retry_on_lock(func, max_retries: int = 3, base_delay: float = 1.0):
    """
    重试装饰器：遇到数据库锁错误时自动重试。

    重试策略：指数退避
    - 第 1 次重试：等待 1 秒
    - 第 2 次重试：等待 2 秒
    - 第 3 次重试：等待 4 秒

    Args:
        func: 要执行的函数
        max_retries: 最大重试次数
        base_delay: 基础延迟（秒）

    Returns:
        函数执行结果
    """
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if _is_lock_error(e) and attempt < max_retries:
                delay = base_delay * (2 ** attempt)  # 指数退避：1s, 2s, 4s
                print(f"  ⚠️  数据库锁冲突，{delay}s 后重试（第 {attempt+1}/{max_retries} 次）...")
                time.sleep(delay)
                continue
            raise  # 非锁错误或重试次数耗尽，直接抛出
    raise last_exception


class VectorStore:
    """
    向量存储管理器。

    封装 Chroma 向量数据库，提供文档入库、检索、统计、清空等接口。
    使用 BGE-m3 作为 embedding 模型。
    """

    def __init__(
        self,
        collection_name: str = "plantgenome_docs",
        embedding_model: str = None,
        persist_dir: str = None,
    ):
        """
        初始化向量存储。

        Args:
            collection_name: Chroma 集合名称，默认 plantgenome_docs
            embedding_model: embedding 模型名，默认从 .env 读 EMBEDDING_MODEL（BAAI/bge-m3）
            persist_dir: Chroma 持久化目录，默认从 .env 读 CHROMA_PERSIST_DIR（data/chroma）
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model or os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "data/chroma")

        # 确保持久化目录存在
        os.makedirs(self.persist_dir, exist_ok=True)

        # 初始化 BGE-m3 embedding 函数（Chroma 内置的 SentenceTransformer 包装）
        # 第一次运行会自动下载模型（约 2GB），后续从缓存加载
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name,
        )

        # 初始化 Chroma 持久化客户端
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},  # 使用余弦相似度
        )

    def add_documents(self, chunks: List[Dict]) -> List[str]:
        """
        把 chunks 转成向量并存入 Chroma。

        Args:
            chunks: 每个 chunk 必须包含：
                - chunk_id: 文档唯一 ID
                - text: 文档内容
                - page_num: 页码
                - chunk_index: chunk 序号
                - metadata: 元数据 dict

        Returns:
            List[str]: 成功入库的文档 ID 列表
        """
        if not chunks:
            return []

        # 提取 ids、documents、metadatas
        ids = [str(chunk["chunk_id"]) for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            # Chroma 的 metadata 只支持 str/int/float/bool，需要把复杂类型转成 str
            meta = chunk["metadata"].copy()
            meta["page_num"] = chunk["page_num"]
            meta["chunk_index"] = chunk["chunk_index"]
            metadatas.append(meta)

        # 批量入库（Chroma 会自动用 embedding_function 做向量化）
        # 加重试逻辑，处理并发写入时的 SQLite 锁冲突
        _retry_on_lock(
            lambda: self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            ),
            max_retries=3,
            base_delay=1.0,
        )

        return ids

    def search(
        self,
        query: str,
        top_k: int = None,
        filter_metadata: Dict = None,
    ) -> List[Dict]:
        """
        检索与 query 最相关的 top_k 个文档。

        Args:
            query: 用户问题
            top_k: 返回数量，默认从 .env 读 RETRIEVE_TOP_K（3）
            filter_metadata: 按 metadata 过滤（如 {"filename": "xxx.pdf"}），可选

        Returns:
            检索结果列表，每个结果包含：
            - chunk_id: 文档 ID
            - text: 文档内容
            - metadata: 元数据（filename、page_num、source 等）
            - distance: 与 query 的距离（余弦距离，越小越相关）
        """
        if not query.strip():
            return []

        top_k = top_k or int(os.getenv("RETRIEVE_TOP_K", "3"))

        # 调用 Chroma 检索（加重试逻辑，处理并发读写时的 SQLite 锁冲突）
        results = _retry_on_lock(
            lambda: self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=filter_metadata,
            ),
            max_retries=3,
            base_delay=1.0,
        )

        # 解析结果（Chroma 返回的是嵌套列表，因为支持多 query）
        if not results["ids"] or not results["ids"][0]:
            return []

        retrieved = []
        for i in range(len(results["ids"][0])):
            retrieved.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

        return retrieved

    def count(self) -> int:
        """返回集合中的文档数量。"""
        return self.collection.count()

    def clear(self) -> None:
        """清空集合中的所有文档（用于重新入库）。"""
        # 删除并重新创建集合
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def get_stats(self) -> Dict:
        """返回集合统计信息。"""
        return {
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model_name,
            "persist_dir": self.persist_dir,
            "document_count": self.count(),
        }


# ── 测试：python -m app.rag.vector_store ──
if __name__ == "__main__":
    from app.rag.pdf_parser import PDFParser
    from app.rag.text_cleaner import TextCleaner
    from app.rag.chunker import Chunker

    print("=" * 60)
    print("PlantGenome Agent - 完整 RAG 流水线测试")
    print("（Markdown 解析 → 智能分块 → BGE-m3 embedding → Chroma 入库 → 检索）")
    print("=" * 60)

    # 初始化各模块
    parser = PDFParser()
    cleaner = TextCleaner()
    chunker = Chunker()

    test_pdf = "data/pdfs/Yang - 2007 - PAML 4 Phylogenetic analysis by maximum likelihood.pdf"
    if not os.path.exists(test_pdf):
        print(f"测试文件不存在: {test_pdf}")
        exit(1)

    # 步骤 1-3：Markdown 解析 → 清洗 → 智能分块
    print(f"\n[步骤 1] 解析 PDF（Markdown 模式）")
    pages = parser.parse_to_markdown(test_pdf)
    print(f"  页数: {len(pages)}")

    print(f"\n[步骤 2] 清洗文本")
    cleaned_pages = cleaner.clean(pages)
    print(f"  清洗后页数: {len(cleaned_pages)}")

    print(f"\n[步骤 3] 文本分块")
    chunks = chunker.chunk(cleaned_pages)
    print(f"  chunk 数: {len(chunks)}")

    # 步骤 4：初始化向量存储（第一次会下载 BGE-m3 模型，可能需要几分钟）
    print(f"\n[步骤 4] 初始化向量存储（BGE-m3 embedding + Chroma）")
    print("  注意：第一次运行会自动下载 BGE-m3 模型（约 2GB），请耐心等待...")
    vector_store = VectorStore()
    print(f"  集合: {vector_store.collection_name}")
    print(f"  持久化目录: {vector_store.persist_dir}")

    # 步骤 5：清空旧数据（测试用）
    print(f"\n[步骤 5] 清空旧数据（测试用）")
    vector_store.clear()
    print(f"  已清空，当前文档数: {vector_store.count()}")

    # 步骤 6：入库
    print(f"\n[步骤 6] 向量化入库（BGE-m3 embedding + Chroma 存储）")
    added = vector_store.add_documents(chunks)
    print(f"  成功入库: {added} 个文档")
    print(f"  当前文档数: {vector_store.count()}")

    # 步骤 7：检索测试
    print(f"\n[步骤 7] 检索测试")
    test_queries = [
        "PAML 中的 omega 值是什么意思？",
        "codeml 怎么用？",
        "最大似然法在系统发育分析中的作用？",
    ]
    for query in test_queries:
        print(f"\n  查询: {query}")
        results = vector_store.search(query, top_k=2)
        for i, r in enumerate(results):
            print(f"    结果 {i+1}: page {r['metadata']['page_num']}, "
                  f"distance={r['distance']:.4f}")
            print(f"      source: {r['metadata']['source']}")
            print(f"      内容前 100 字符: {r['text'][:100]}...")

    # 统计信息
    print(f"\n[统计信息]")
    stats = vector_store.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("向量存储模块测试完成。")
    print("=" * 60)
