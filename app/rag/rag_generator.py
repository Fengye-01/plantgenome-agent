"""
RAG 生成模块（RAG 流水线第 7 步：生成）

对应 Hello Agents 第 8 章 RAG 的「Generation（生成）」阶段 + 第 9 章上下文工程。
职责：通过 ContextBuilder（GSSC 流水线）把检索结果组织成结构化上下文，传给 LLM 生成回答。

核心设计：
1. ContextBuilder：GSSC 流水线（Gather-Select-Structure-Compress），参考第九章 9.3 节
2. 结构化上下文模板：[Role & Policies] / [Evidence] / [Task] / [Output] 分区组织
3. 空结果处理：检索不到相关内容时，LLM 明确说"无法回答"，不幻觉
4. 引用标注：回答中标注 [1][2]，对应用户友好的 sources 列表
"""
from __future__ import annotations

import os
from typing import Dict, Optional

from dotenv import load_dotenv

load_dotenv()

from app.core.llm import HelloAgentsLLM
from app.rag.context_builder import ContextBuilder, ContextConfig
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore


class RAGGenerator:
    """
    RAG 生成器。

    输入：用户 query（字符串）
    输出：Dict：
        {
            "answer": str,                    # LLM 生成的回答（带 [1][2] 引用标注）
            "sources": List[str],              # sources 列表（用户友好格式，对应 [1][2]）
            "has_relevant": bool,              # 是否检索到相关结果
            "query": str,                      # 预处理后的用户问题
            "retrieve_result": Dict,           # 完整的检索结果（调试用）
            "llm_model": str,                  # 使用的 LLM 模型
        }
    """

    def __init__(
        self,
        retriever: Retriever = None,
        llm: HelloAgentsLLM = None,
        context_builder: ContextBuilder = None,
        context_config: ContextConfig = None,
    ):
        """
        Args:
            retriever: Retriever 实例（可选，不传则自动创建）
            llm: HelloAgentsLLM 实例（可选，不传则自动创建）
            context_builder: ContextBuilder 实例（可选，不传则自动创建）
            context_config: ContextConfig 配置（可选，不传则用默认值）
        """
        # 自动创建 Retriever（如果没传）
        if retriever is None:
            vector_store = VectorStore()
            self.retriever = Retriever(vector_store)
        else:
            self.retriever = retriever

        # 自动创建 LLM（如果没传）
        if llm is None:
            self.llm = HelloAgentsLLM()
        else:
            self.llm = llm

        # 自动创建 ContextBuilder（如果没传）
        if context_builder is None:
            self.context_builder = ContextBuilder(context_config or ContextConfig())
        else:
            self.context_builder = context_builder

    def generate(self, query: str, top_k: int = None) -> Dict:
        """
        RAG 生成主入口。

        流程：
          1. Retriever 检索（query 预处理 → 向量检索 → 结果格式化 → sources 组装）
          2. 拼接上下文（Retriever.get_context_for_llm）
          3. 构建 Prompt（系统提示 + 上下文 + 用户问题）
          4. 调用 LLM 生成（HelloAgentsLLM.invoke）
          5. 返回回答 + sources + 检索结果

        Args:
            query: 用户问题
            top_k: 检索返回数量（可选，默认用 Retriever 的 top_k）

        Returns:
            RAG 生成结果 Dict
        """
        # 1. Retriever 检索
        retrieve_result = self.retriever.retrieve(query, top_k=top_k)

        # 2. ContextBuilder 构建结构化上下文（GSSC 流水线）
        # Gather：把检索结果转成 ContextPacket
        # Select：按相关性评分 + token 预算贪心选择
        # Structure：按 [Role]/[Evidence]/[Task]/[Output] 分区组织
        # Compress：超限时截断低相关性 chunk
        build_result = self.context_builder.build(
            query=retrieve_result["query"],
            retrieve_result=retrieve_result,
        )
        context = build_result["context"]
        selected_sources = build_result["selected_sources"]

        # 3. 调用 LLM 生成
        # context 已经包含了 [Role & Policies] 系统提示和 [Task] 用户问题，
        # 所以直接作为 prompt 传给 LLM，不需要额外传 system 参数
        answer = self.llm.invoke(prompt=context)

        # 4. 返回结果
        return {
            "answer": answer,
            "sources": selected_sources,  # 用 ContextBuilder 选中的 sources
            "has_relevant": retrieve_result["has_relevant"],
            "query": retrieve_result["query"],
            "retrieve_result": retrieve_result,
            "context_builder_result": build_result,  # 调试用
            "llm_model": os.getenv("LLM_MODEL_ID", "unknown"),
        }

    def generate_with_debug(self, query: str, top_k: int = None) -> Dict:
        """
        带调试信息的生成（开发调试用，打印检索结果和 Prompt）。

        Args:
            query: 用户问题
            top_k: 检索返回数量

        Returns:
            RAG 生成结果 Dict（含 debug 信息）
        """
        print("=" * 60)
        print("RAG 生成调试模式（ContextBuilder GSSC 流水线）")
        print("=" * 60)

        # 1. 检索
        print(f"\n[1/5] 检索用户问题: {query}")
        retrieve_result = self.retriever.retrieve(query, top_k=top_k)
        print(f"  检索到 {retrieve_result['total_count']} 个结果")
        print(f"  has_relevant: {retrieve_result['has_relevant']}")
        print(f"  Sources:")
        for s in retrieve_result["sources"]:
            print(f"    {s}")

        # 2. ContextBuilder GSSC 流水线
        print(f"\n[2/5] ContextBuilder 构建结构化上下文（GSSC 流水线）")
        build_result = self.context_builder.build(
            query=retrieve_result["query"],
            retrieve_result=retrieve_result,
        )
        context = build_result["context"]
        selected_sources = build_result["selected_sources"]

        print(f"  Gather: 汇集了 {len(retrieve_result['results'])} 个候选信息包")
        print(f"  Select: 选中了 {len(build_result['selected_packets'])} 个信息包")
        print(f"  Structure: 按 [Role]/[Evidence]/[Task]/[Output] 分区组织")
        print(f"  Compress: {'是（截断低相关性）' if build_result['compressed'] else '否'}")
        print(f"  总 token 数: {build_result['total_tokens']}")
        print(f"  上下文长度: {len(context)} 字符")
        print(f"  选中的 sources:")
        for s in selected_sources:
            print(f"    {s}")

        # 3. 展示上下文结构
        print(f"\n[3/5] 上下文结构预览（前 500 字符）:")
        print(f"  {context[:500]}...")

        # 4. 调用 LLM
        print(f"\n[4/5] 调用 LLM 生成（模型: {os.getenv('LLM_MODEL_ID')}）")
        answer = self.llm.invoke(prompt=context)
        print(f"  回答长度: {len(answer)} 字符")

        # 5. 最终回答
        print(f"\n[5/5] 最终回答:")
        print(answer)
        print(f"{'=' * 60}")

        return {
            "answer": answer,
            "sources": selected_sources,
            "has_relevant": retrieve_result["has_relevant"],
            "query": retrieve_result["query"],
            "retrieve_result": retrieve_result,
            "context_builder_result": build_result,
            "llm_model": os.getenv("LLM_MODEL_ID", "unknown"),
            "debug": {
                "context": context,
                "context_length": len(context),
                "total_tokens": build_result["total_tokens"],
                "compressed": build_result["compressed"],
            },
        }


# ── 测试：python -m app.rag.rag_generator ──
if __name__ == "__main__":
    print("=" * 60)
    print("PlantGenome Agent - RAG 生成模块测试")
    print("=" * 60)

    # 初始化
    print("\n[初始化] 加载 VectorStore + Retriever + ContextBuilder + RAGGenerator...")
    vector_store = VectorStore()
    retriever = Retriever(vector_store)
    context_config = ContextConfig(max_tokens=3000, reserve_ratio=0.2)
    generator = RAGGenerator(retriever=retriever, context_config=context_config)
    print(f"  ✅ VectorStore 文档数: {vector_store.count()}")
    print(f"  ✅ ContextConfig: max_tokens={context_config.max_tokens}, reserve_ratio={context_config.reserve_ratio}")
    print(f"  ✅ LLM 模型: {os.getenv('LLM_MODEL_ID')}")

    # 测试问题
    test_queries = [
        "PAML 中的 omega 值是什么意思？",
        "OrthoFinder 的输入格式是什么？",
        "MAFFT 有哪些比对算法？",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─' * 60}")
        print(f"测试 {i}/{len(test_queries)}: {query}")
        print(f"{'─' * 60}")

        result = generator.generate(query)

        print(f"\n回答:")
        print(result["answer"])
        print(f"\n引用来源:")
        for s in result["sources"]:
            print(f"  {s}")
        print(f"\nhas_relevant: {result['has_relevant']}")
        print(f"LLM 模型: {result['llm_model']}")

    print("\n" + "=" * 60)
    print("RAG 生成模块测试完成。")
    print("=" * 60)
