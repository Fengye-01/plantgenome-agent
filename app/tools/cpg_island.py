"""
工具 3：scan_cpg_islands（CpG 岛扫描工具）

用滑动窗口扫描 DNA 序列，识别 CpG 岛。
采用 Gardiner-Garden & Frommer (1987) 经典标准：
- GC 含量 > 50%
- CpG Obs/Exp 比值 > 0.6
- 长度 > 200 bp

这是纯计算工具，不依赖 LLM 或外部 API，执行速度快。

参考 Hello Agents 第 4 章工具调用（Tool Calling）。
"""
from __future__ import annotations

from typing import Dict, List, Optional


def clean_sequence(sequence: str) -> str:
    """
    清洗 DNA 序列：去除换行、空格，转为大写。

    Args:
        sequence: 原始 DNA 序列

    Returns:
        清洗后的序列
    """
    return sequence.upper().replace('\n', '').replace('\r', '').replace(' ', '').replace('\t', '')


def calculate_gc_content(sequence: str) -> float:
    """
    计算序列的 GC 含量。

    GC 含量 = (G 的数量 + C 的数量) / 序列总长度

    Args:
        sequence: DNA 序列

    Returns:
        GC 含量（0.0 - 1.0）
    """
    if not sequence:
        return 0.0
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence)


def calculate_cpg_obs_exp(sequence: str) -> float:
    """
    计算 CpG Obs/Exp 比值。

    公式：(CpG数 × 序列长度) / (C数 × G数)

    生物学意义：
    - 在大多数基因组中，CpG 二核苷酸的出现频率远低于预期（因为 CpG 中的 C 容易甲基化并脱氨变成 T）
    - CpG 岛是例外，其中 CpG 的出现频率接近或超过预期
    - Obs/Exp > 0.6 是 CpG 岛的经典判定标准之一

    Args:
        sequence: DNA 序列

    Returns:
        CpG Obs/Exp 比值
    """
    if not sequence:
        return 0.0

    seq_len = len(sequence)
    cpg_count = sequence.count('CG')
    c_count = sequence.count('C')
    g_count = sequence.count('G')

    # 避免除以 0
    if c_count == 0 or g_count == 0:
        return 0.0

    return (cpg_count * seq_len) / (c_count * g_count)


def scan_cpg_islands(
    sequence: str,
    window_size: int = 200,
    step: int = 100,
    gc_threshold: float = 0.5,
    oe_threshold: float = 0.6,
) -> List[Dict]:
    """
    滑动窗口扫描 CpG 岛，合并重叠阳性窗口。

    算法流程：
    1. 用滑动窗口遍历序列，每个窗口计算 GC 含量和 CpG Obs/Exp 比值
    2. 标记同时满足 GC > threshold 和 Obs/Exp > threshold 的窗口为阳性
    3. 合并重叠或相邻的阳性窗口，形成完整的 CpG 岛
    4. 计算每个 CpG 岛的平均 GC 含量和平均 Obs/Exp 比值

    Args:
        sequence: DNA 序列
        window_size: 滑动窗口大小（bp），默认 200（Gardiner-Garden 标准）
        step: 滑动步长（bp），默认 100（50% 重叠）
        gc_threshold: GC 含量阈值，默认 0.5（50%）
        oe_threshold: CpG Obs/Exp 比值阈值，默认 0.6

    Returns:
        list[dict]，每个 CpG 岛包含：
        - start: 起始位置（0-based）
        - end: 结束位置（exclusive）
        - length: 长度（bp）
        - gc_content: 平均 GC 含量
        - obs_exp_ratio: 平均 CpG Obs/Exp 比值
        - num_windows: 由多少个阳性窗口合并而成

        如果序列太短或没有检测到 CpG 岛，返回包含 warning/result 的列表。
    """
    # 第一步：清洗序列
    seq = clean_sequence(sequence)

    if not seq:
        return [{"warning": "输入序列为空"}]

    if len(seq) < window_size:
        return [{
            "warning": f"序列长度({len(seq)}bp)小于窗口大小({window_size}bp)，无法进行滑动窗口扫描",
            "sequence_length": len(seq),
            "window_size": window_size,
        }]

    # 第二步：滑动窗口标记阳性区域
    positive_windows = []

    for i in range(0, len(seq) - window_size + 1, step):
        window = seq[i:i + window_size]

        # 计算窗口的 GC 含量和 CpG Obs/Exp 比值
        gc_content = calculate_gc_content(window)
        obs_exp = calculate_cpg_obs_exp(window)

        # 判断是否为阳性窗口（同时满足两个条件）
        if gc_content > gc_threshold and obs_exp > oe_threshold:
            positive_windows.append({
                "start": i,
                "end": i + window_size,
                "gc_content": gc_content,
                "obs_exp": obs_exp,
            })

    if not positive_windows:
        return [{
            "result": "未检测到符合标准的 CpG 岛",
            "sequence_length": len(seq),
            "window_size": window_size,
            "step": step,
            "gc_threshold": gc_threshold,
            "oe_threshold": oe_threshold,
            "num_windows_scanned": len(range(0, len(seq) - window_size + 1, step)),
        }]

    # 第三步：合并重叠或相邻的阳性窗口
    islands = []

    # 按起始位置排序（已经是有序的，但保险起见）
    positive_windows.sort(key=lambda x: x["start"])

    # 初始化第一个岛
    current = {
        "start": positive_windows[0]["start"],
        "end": positive_windows[0]["end"],
        "gc_sum": positive_windows[0]["gc_content"],
        "oe_sum": positive_windows[0]["obs_exp"],
        "num_windows": 1,
    }

    for w in positive_windows[1:]:
        # 判断是否重叠或相邻（当前窗口的 start <= 当前岛的 end）
        if w["start"] <= current["end"]:
            # 合并：扩展结束位置，累加 GC 和 Obs/Exp
            current["end"] = max(current["end"], w["end"])
            current["gc_sum"] += w["gc_content"]
            current["oe_sum"] += w["obs_exp"]
            current["num_windows"] += 1
        else:
            # 不重叠：保存当前岛，开始新岛
            islands.append({
                "start": current["start"],
                "end": current["end"],
                "length": current["end"] - current["start"],
                "gc_content": round(current["gc_sum"] / current["num_windows"], 4),
                "obs_exp_ratio": round(current["oe_sum"] / current["num_windows"], 4),
                "num_windows": current["num_windows"],
            })
            current = {
                "start": w["start"],
                "end": w["end"],
                "gc_sum": w["gc_content"],
                "oe_sum": w["obs_exp"],
                "num_windows": 1,
            }

    # 保存最后一个岛
    islands.append({
        "start": current["start"],
        "end": current["end"],
        "length": current["end"] - current["start"],
        "gc_content": round(current["gc_sum"] / current["num_windows"], 4),
        "obs_exp_ratio": round(current["oe_sum"] / current["num_windows"], 4),
        "num_windows": current["num_windows"],
    })

    return islands


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
    print("工具测试：scan_cpg_islands")
    print("=" * 70)

    # 测试 1：构造一段含 CpG 岛的序列（高 GC + 高 CpG 密度）
    print("\n--- 测试 1：含 CpG 岛的序列 ---")
    # 构造：前 100bp 低 GC，中间 400bp 高 GC + 高 CpG，后 100bp 低 GC
    low_gc = "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT"
    high_cpg = "CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG"
    # 高 GC 但 CpG 密度适中的序列
    high_gc = "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC"
    test_seq1 = low_gc + high_cpg * 4 + high_gc * 2 + low_gc
    print(f"序列长度: {len(test_seq1)}")
    result1 = scan_cpg_islands(test_seq1)
    print(f"检测到 {len(result1)} 个 CpG 岛:")
    for i, island in enumerate(result1, 1):
        if "warning" in island or "result" in island:
            print(f"  {island}")
        else:
            print(f"  岛 {i}: 位置 {island['start']}-{island['end']} ({island['length']}bp), "
                  f"GC={island['gc_content']}, Obs/Exp={island['obs_exp_ratio']}, "
                  f"由 {island['num_windows']} 个窗口合并")

    # 测试 2：低 GC 序列（不应检测到 CpG 岛）
    print("\n--- 测试 2：低 GC 序列（AT 丰富）---")
    test_seq2 = "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT" * 10
    print(f"序列长度: {len(test_seq2)}")
    result2 = scan_cpg_islands(test_seq2)
    print(f"结果: {result2}")

    # 测试 3：序列太短
    print("\n--- 测试 3：序列太短 ---")
    test_seq3 = "ATCGATCG" * 10  # 80bp
    print(f"序列长度: {len(test_seq3)}")
    result3 = scan_cpg_islands(test_seq3)
    print(f"结果: {result3}")

    # 测试 4：空序列
    print("\n--- 测试 4：空序列 ---")
    result4 = scan_cpg_islands("")
    print(f"结果: {result4}")

    print("\n" + "=" * 70)
    print("所有测试完成！")
    print("=" * 70)
