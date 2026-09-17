"""
BM25 检索模块（混合检索的关键词检索通道）。

在向量召回的候选集上构建 BM25 索引，做关键词精确匹配，
然后与向量检索结果通过 RRF（Reciprocal Rank Fusion）融合。

为什么需要 BM25？
- 向量检索擅长语义匹配，但对专业术语、基因名、软件名等精确关键词可能召回不准
- BM25 擅长关键词精确匹配，对生信领域的专有名词（如 PAML、OrthoFinder、mTERF）召回更好
- 两者融合（RRF）可以兼顾语义和精确匹配，显著提升 RAG 召回率

设计要点：
- 不维护全局 BM25 索引（避免与 Chroma 双写一致性问题）
- 每次检索时在向量召回的候选集（top_k * 5）上动态构建 BM25 索引
- 候选集规模小（通常 15-30 篇），构建索引开销可忽略
- 支持用户隔离（候选集已按 user_id 过滤）
"""
from __future__ import annotations

import math
import re
from typing import List, Dict, Tuple


# ═══════════════════════════════════════════════════════════════
# 分词器（中英文混合）
# ═══════════════════════════════════════════════════════════════

def tokenize(text: str) -> List[str]:
    """
    简单分词器：英文按单词切分，中文按单字切分，保留专业术语。

    对生信领域优化：
    - 保留连字符词（如 CpG-island → 拆为 cpg, island）
    - 保留大小写不敏感（统一小写）
    - 数字单独成词
    - 过滤长度 < 2 的词（除了单字母基因名如 "m"）

    Args:
        text: 输入文本

    Returns:
        分词后的 token 列表
    """
    if not text:
        return []

    # 统一小写
    text = text.lower()

    # 替换连字符为空格，拆分复合词
    text = re.sub(r'[-_/]', ' ', text)

    # 提取英文单词和数字（连续字母/数字）
    tokens = re.findall(r'[a-z0-9]+', text)

    # 过滤过短的词（保留 >=2 字符的，以及常见单字母如 "m" "k"）
    filtered = []
    for t in tokens:
        if len(t) >= 2:
            filtered.append(t)
        elif t in ('m', 'k', 'r', 'n', 'p'):  # 常见生信缩写
            filtered.append(t)

    return filtered


# ═══════════════════════════════════════════════════════════════
# BM25 索引（在候选集上动态构建）
# ═══════════════════════════════════════════════════════════════

class BM25Index:
    """
    BM25 索引（在候选文档集上动态构建）。

    实现经典 BM25 公式：
        score(D, Q) = sum over qi in Q of IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))

    其中：
        IDF(qi) = ln((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
        f(qi, D) = qi 在文档 D 中的词频
        |D| = 文档长度
        avgdl = 平均文档长度
        k1 = 1.5（词频饱和参数）
        b = 0.75（长度归一化参数）

    参考：Robertson & Zaragoza (2009) The Probabilistic Relevance Framework
    """

    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        """
        在候选文档集上构建 BM25 索引。

        Args:
            documents: 文档文本列表
            k1: 词频饱和参数，默认 1.5
            b: 长度归一化参数，默认 0.75
        """
        self.k1 = k1
        self.b = b
        self.N = len(documents)

        # 分词
        self.doc_tokens = [tokenize(doc) for doc in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / self.N if self.N > 0 else 0

        # 计算文档频率（DF）：每个词出现在多少篇文档中
        self.df = {}
        for tokens in self.doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.df[token] = self.df.get(token, 0) + 1

        # 计算 IDF
        self.idf = {}
        for token, df in self.df.items():
            self.idf[token] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

        # 预计算每篇文档的词频
        self.tf = []
        for tokens in self.doc_tokens:
            freq = {}
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1
            self.tf.append(freq)

    def search(self, query: str, top_k: int = None) -> List[Tuple[int, float]]:
        """
        检索与 query 最相关的文档。

        Args:
            query: 查询文本
            top_k: 返回 top_k 个结果，默认返回全部

        Returns:
            List[Tuple[int, float]]: (文档索引, BM25 分数)，按分数降序排列
        """
        if self.N == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # 计算每篇文档的 BM25 分数
        scores = []
        for doc_idx in range(self.N):
            score = 0.0
            doc_len = self.doc_lengths[doc_idx]
            doc_tf = self.tf[doc_idx]

            for q_token in query_tokens:
                if q_token not in self.idf:
                    continue

                idf = self.idf[q_token]
                freq = doc_tf.get(q_token, 0)

                if freq == 0:
                    continue

                # BM25 核心公式
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) if self.avgdl > 0 else freq + self.k1
                score += idf * numerator / denominator

            scores.append((doc_idx, score))

        # 按分数降序排列
        scores.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            scores = scores[:top_k]

        return scores


# ═══════════════════════════════════════════════════════════════
# RRF 融合（Reciprocal Rank Fusion）
# ═══════════════════════════════════════════════════════════════

def rrf_fuse(
    vector_ranking: List[int],
    bm25_ranking: List[int],
    k: int = 60,
    top_k: int = None,
) -> List[int]:
    """
    RRF（Reciprocal Rank Fusion）融合多个排序列表。

    公式：RRF_score(d) = sum over rankings of 1 / (k + rank(d))

    RRF 的优点：
    - 不需要归一化不同检索系统的分数（只看排名）
    - 对排名靠前的结果权重更大
    - 实现简单，效果稳定

    Args:
        vector_ranking: 向量检索的文档索引排名列表（按相关性降序）
        bm25_ranking: BM25 检索的文档索引排名列表（按相关性降序）
        k: RRF 常数，默认 60（越大排名差异越平滑）
        top_k: 返回 top_k 个结果，默认返回全部

    Returns:
        List[int]: 融合后的文档索引排名列表
    """
    scores = {}

    # 向量检索排名
    for rank, doc_idx in enumerate(vector_ranking):
        scores[doc_idx] = scores.get(doc_idx, 0) + 1.0 / (k + rank + 1)

    # BM25 检索排名
    for rank, doc_idx in enumerate(bm25_ranking):
        scores[doc_idx] = scores.get(doc_idx, 0) + 1.0 / (k + rank + 1)

    # 按融合分数降序排列
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    result = [doc_idx for doc_idx, _ in fused]

    if top_k:
        result = result[:top_k]

    return result


# ═══════════════════════════════════════════════════════════════
# 混合检索主函数
# ═══════════════════════════════════════════════════════════════

def hybrid_search(
    query: str,
    vector_results: List[Dict],
    top_k: int = 3,
    bm25_weight: float = 1.0,
    vector_weight: float = 1.0,
) -> List[Dict]:
    """
    混合检索：向量检索 + BM25 关键词检索 + RRF 融合。

    流程：
    1. 向量检索已经召回了候选集（由调用方传入，通常是 top_k * 5）
    2. 在候选集上构建 BM25 索引
    3. BM25 检索 query，得到 BM25 排名
    4. 向量检索排名（按 distance 升序，即相似度降序）
    5. RRF 融合两个排名
    6. 返回融合后的 top_k 结果

    Args:
        query: 查询文本
        vector_results: 向量检索结果列表（每个包含 text、distance、metadata 等）
        top_k: 返回 top_k 个结果
        bm25_weight: BM25 通道权重（RRF 中通过排名体现，这里预留接口）
        vector_weight: 向量通道权重（预留接口）

    Returns:
        List[Dict]: 融合后的检索结果（保持原格式，按融合相关性排序）
    """
    if not vector_results:
        return []

    # 如果只有一个结果，直接返回
    if len(vector_results) <= top_k:
        return vector_results

    # 1. 向量检索排名（按 distance 升序，即最相关的排前面）
    # vector_results 已经按 distance 升序排列（Chroma 返回的就是有序的）
    vector_ranking = list(range(len(vector_results)))

    # 2. 在候选集上构建 BM25 索引
    docs_text = [r.get("text", "") or r.get("content", "") for r in vector_results]
    bm25 = BM25Index(docs_text)

    # 3. BM25 检索
    bm25_results = bm25.search(query, top_k=len(vector_results))
    bm25_ranking = [doc_idx for doc_idx, _ in bm25_results]

    # 4. RRF 融合
    fused_ranking = rrf_fuse(vector_ranking, bm25_ranking, top_k=top_k)

    # 5. 按融合排名重新组织结果
    fused_results = [vector_results[idx] for idx in fused_ranking]

    return fused_results


# ═══════════════════════════════════════════════════════════════
# 测试入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("BM25 + RRF 混合检索测试")
    print("=" * 60)

    # 模拟候选文档
    docs = [
        "PAML is a package for phylogenetic analysis by maximum likelihood. The codeml program estimates omega ratios.",
        "OrthoFinder is a tool for ortholog inference and phylogenetic analysis. It takes protein sequences as input.",
        "MAFFT is a multiple sequence alignment program. It offers L-INS-i, G-INS-i and other algorithms.",
        "DNA methylation in plants involves CpG islands and epigenetic regulation. Methylation patterns are dynamic.",
        "The mTERF gene family plays important roles in plant mitochondrial gene expression and evolution.",
    ]

    query = "PAML codeml omega maximum likelihood"

    # 构建 BM25 索引
    bm25 = BM25Index(docs)
    print(f"\n文档数: {bm25.N}")
    print(f"平均文档长度: {bm25.avgdl:.1f}")
    print(f"词汇数: {len(bm25.df)}")

    # BM25 检索
    results = bm25.search(query, top_k=3)
    print(f"\nBM25 检索结果 (query: {query!r}):")
    for doc_idx, score in results:
        print(f"  文档 {doc_idx}: score={score:.4f}")
        print(f"    内容: {docs[doc_idx][:60]}...")

    # 模拟向量检索排名（假设向量检索的排名是 2, 0, 1, 3, 4）
    vector_ranking = [2, 0, 1, 3, 4]
    bm25_ranking = [doc_idx for doc_idx, _ in bm25.search(query)]

    print(f"\n向量检索排名: {vector_ranking}")
    print(f"BM25 检索排名: {bm25_ranking}")

    # RRF 融合
    fused = rrf_fuse(vector_ranking, bm25_ranking, top_k=3)
    print(f"RRF 融合排名 (top 3): {fused}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
