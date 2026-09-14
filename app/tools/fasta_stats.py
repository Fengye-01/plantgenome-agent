"""
工具 2：parse_fasta_stats（FASTA 序列统计工具）

解析 FASTA 格式的序列文本，返回序列统计信息。
这是纯计算工具，不依赖 LLM 或外部 API，执行速度快。

参考 Hello Agents 第 4 章工具调用（Tool Calling）。
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple


def parse_fasta(fasta_text: str) -> List[Tuple[str, str]]:
    """
    解析 FASTA 文本，返回 (序列ID, 序列内容) 的列表。

    FASTA 格式说明：
    - 以 > 开头的行是序列 ID 行
    - 后续行是序列内容（可以跨多行）
    - 以 ; 开头的行是注释行，忽略
    - 空行忽略

    Args:
        fasta_text: FASTA 格式的文本

    Returns:
        [(seq_id, sequence), ...] 列表
    """
    sequences = []
    current_id = None
    current_seq = []

    for line in fasta_text.strip().split('\n'):
        line = line.strip()

        # 跳过空行和注释行
        if not line or line.startswith(';'):
            continue

        # ID 行：以 > 开头
        if line.startswith('>'):
            # 保存上一条序列
            if current_id is not None:
                sequences.append((current_id, ''.join(current_seq)))
            # 提取 ID（> 后面的第一个词）
            current_id = line[1:].split()[0] if len(line) > 1 else "unknown"
            current_seq = []
        else:
            # 序列内容行，转为大写并追加
            current_seq.append(line.upper())

    # 保存最后一条序列
    if current_id is not None:
        sequences.append((current_id, ''.join(current_seq)))

    return sequences


def calculate_gc_content(sequence: str) -> float:
    """
    计算序列的 GC 含量。

    GC 含量 = (G 的数量 + C 的数量) / 序列总长度

    Args:
        sequence: DNA/RNA 序列

    Returns:
        GC 含量（0.0 - 1.0）
    """
    if not sequence:
        return 0.0
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence)


def calculate_length_distribution(lengths: List[int]) -> Dict[str, int]:
    """
    计算序列长度分布。

    分为 5 个区间：
    - <1kb: 小于 1000 bp
    - 1-2kb: 1000-1999 bp
    - 2-3kb: 2000-2999 bp
    - 3-5kb: 3000-4999 bp
    - >5kb: 大于等于 5000 bp

    Args:
        lengths: 序列长度列表

    Returns:
        各区间的序列数量
    """
    dist = {
        "<1kb": 0,
        "1-2kb": 0,
        "2-3kb": 0,
        "3-5kb": 0,
        ">5kb": 0,
    }

    for l in lengths:
        if l < 1000:
            dist["<1kb"] += 1
        elif l < 2000:
            dist["1-2kb"] += 1
        elif l < 3000:
            dist["2-3kb"] += 1
        elif l < 5000:
            dist["3-5kb"] += 1
        else:
            dist[">5kb"] += 1

    return dist


def parse_fasta_stats(fasta_text: str) -> Dict:
    """
    解析 FASTA 文本，返回序列统计信息。

    这是 Agent 的工具之一，用于分析用户提供的 FASTA 序列。

    Args:
        fasta_text: FASTA 格式的文本

    Returns:
        dict，包含：
        - num_sequences: 序列数量
        - total_length: 总长度
        - avg_length: 平均长度
        - min_length: 最短序列长度
        - max_length: 最长序列长度
        - gc_content: 整体 GC 含量
        - length_distribution: 长度分布
        - sequence_ids: 序列 ID 列表（最多显示 20 个）
        - per_sequence: 每条序列的详细统计（ID、长度、GC含量）

    失败情况：
        - 输入不是有效的 FASTA 格式 → 返回 {"error": "..."}
        - 序列为空 → 返回 {"error": "..."}
    """
    # 第一步：解析 FASTA
    sequences = parse_fasta(fasta_text)

    if not sequences:
        return {"error": "未检测到有效的 FASTA 序列，请检查输入格式（序列 ID 应以 > 开头）"}

    # 第二步：计算统计信息
    lengths = [len(seq) for _, seq in sequences]
    all_seq = ''.join(seq for _, seq in sequences)

    # 整体 GC 含量
    overall_gc = calculate_gc_content(all_seq)

    # 长度分布
    length_dist = calculate_length_distribution(lengths)

    # 每条序列的详细统计
    per_sequence = []
    for seq_id, seq in sequences:
        per_sequence.append({
            "id": seq_id,
            "length": len(seq),
            "gc_content": round(calculate_gc_content(seq), 4),
        })

    return {
        "num_sequences": len(sequences),
        "total_length": sum(lengths),
        "avg_length": round(sum(lengths) / len(lengths), 1),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "gc_content": round(overall_gc, 4),
        "length_distribution": length_dist,
        "sequence_ids": [sid for sid, _ in sequences[:20]],  # 最多显示 20 个
        "per_sequence": per_sequence[:20],  # 最多显示 20 条
        "has_more": len(sequences) > 20,
    }


# ═══════════════════════════════════════════════════════════════
# 测试入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # 项目根路径导入
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    print("=" * 70)
    print("工具测试：parse_fasta_stats")
    print("=" * 70)

    # 测试 1：正常的 3 条序列
    print("\n--- 测试 1：正常的 3 条序列 ---")
    fasta1 = """>gene1 ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA
>gene2
ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA
ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA
>gene3
ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA
ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA
ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA
"""
    result1 = parse_fasta_stats(fasta1)
    print(f"序列数量: {result1['num_sequences']}")
    print(f"总长度: {result1['total_length']}")
    print(f"平均长度: {result1['avg_length']}")
    print(f"最短: {result1['min_length']}, 最长: {result1['max_length']}")
    print(f"GC 含量: {result1['gc_content']}")
    print(f"长度分布: {result1['length_distribution']}")
    print(f"序列 IDs: {result1['sequence_ids']}")

    # 测试 2：空输入
    print("\n--- 测试 2：空输入 ---")
    result2 = parse_fasta_stats("")
    print(f"结果: {result2}")

    # 测试 3：无效格式（没有 > 开头）
    print("\n--- 测试 3：无效格式 ---")
    result3 = parse_fasta_stats("ATGCGTACGTAGCTA")
    print(f"结果: {result3}")

    # 测试 4：含注释行
    print("\n--- 测试 4：含注释行 ---")
    fasta4 = """; 这是注释行
>seq1
ATGCGTACGTAGCTA
; 另一条注释
>seq2
ATGCGTACGTAGCTAGCTA
"""
    result4 = parse_fasta_stats(fasta4)
    print(f"序列数量: {result4['num_sequences']}")
    print(f"序列 IDs: {result4['sequence_ids']}")

    print("\n" + "=" * 70)
    print("所有测试完成！")
    print("=" * 70)
