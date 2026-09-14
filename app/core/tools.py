"""
工具执行器：参考 Hello Agents 教程 chapter4/tools.py 的 ToolExecutor 实现

与教程的区别：
- 教程的 ToolExecutor 是基础版（registerTool / getTool / getAvailableTools）
- 本文件保留教程的核心接口，增加了工具描述格式化、异常处理、执行日志
- 后续生信工具（parse_fasta_stats / scan_cpg_islands / suggest_pipeline）都通过这里注册

为什么需要 ToolExecutor？
- Agent（ReAct / LangGraph）需要知道有哪些工具可用、每个工具做什么
- ToolExecutor 统一管理工具的注册、查询、执行，避免工具散落在各处
- 工具描述会被拼接到 Agent 的 Prompt 中，让 LLM 知道该调用哪个工具
"""
from __future__ import annotations

import time
from typing import Dict, Any, Callable, List


class ToolExecutor:
    """
    工具执行器，负责管理和执行工具。
    参考 Hello Agents 教程 chapter4/tools.py 的实现。
    """

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.execution_log: List[Dict[str, Any]] = []

    def registerTool(self, name: str, description: str, func: Callable):
        """
        向工具箱中注册一个新工具。

        Args:
            name: 工具名称（Agent 调用时用这个名字）
            description: 工具描述（会被拼接到 Agent Prompt 中，告诉 LLM 这个工具做什么）
            func: 工具执行函数，接收一个字符串参数，返回字符串结果
        """
        if name in self.tools:
            print(f"警告：工具 '{name}' 已存在，将被覆盖。")

        self.tools[name] = {"description": description, "func": func}
        print(f"✅ 工具 '{name}' 已注册。")

    def getTool(self, name: str) -> Callable:
        """根据名称获取工具的执行函数。"""
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串。
        这个字符串会被拼接到 Agent 的 Prompt 中，让 LLM 知道有哪些工具可用。

        Returns:
            格式化的工具列表，例如：
            - Search: 一个网页搜索引擎...
            - Calculator: 一个计算器工具...
        """
        return "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])

    def execute(self, name: str, tool_input: str) -> str:
        """
        执行指定工具，记录执行日志，处理异常。

        Args:
            name: 工具名称
            tool_input: 工具输入参数（字符串）

        Returns:
            工具执行结果（字符串）
        """
        start_time = time.time()
        tool_func = self.getTool(name)

        if not tool_func:
            result = f"错误：未找到名为 '{name}' 的工具。"
            status = "not_found"
        else:
            try:
                result = tool_func(tool_input)
                status = "success"
            except Exception as e:
                result = f"工具执行时发生错误: {e}"
                status = "error"

        latency = round(time.time() - start_time, 3)
        self.execution_log.append({
            "tool": name,
            "input": tool_input[:100],  # 只记录前100字符
            "status": status,
            "latency": latency,
        })

        return result

    def getExecutionLog(self) -> List[Dict[str, Any]]:
        """获取工具执行历史日志。"""
        return self.execution_log


# --- 工具初始化与使用示例：python -m app.core.tools ---
if __name__ == "__main__":
    print("=" * 60)
    print("PlantGenome Agent - ToolExecutor 测试（参考 Hello Agents 教程）")
    print("=" * 60)
    print()

    # 1. 初始化工具执行器
    toolExecutor = ToolExecutor()

    # 2. 注册示例工具（模拟教程的 Search 工具，但不依赖外部 API）
    def mock_search(query: str) -> str:
        """模拟搜索工具，返回固定结果（测试用）。"""
        return f"[模拟搜索结果] 关于 '{query}' 的信息：这是一个测试结果，实际使用时会调用真实搜索引擎。"

    def calculator(expression: str) -> str:
        """简单计算器工具，计算数学表达式。"""
        try:
            result = eval(expression)
            return f"计算结果: {expression} = {result}"
        except Exception as e:
            return f"计算错误: {e}"

    toolExecutor.registerTool(
        "Search",
        "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。",
        mock_search,
    )
    toolExecutor.registerTool(
        "Calculator",
        "一个数学计算器。当你需要进行数学计算时，应使用此工具。输入为数学表达式，如 '2 + 3 * 4'。",
        calculator,
    )

    # 3. 打印可用工具
    print("\n--- 可用的工具 ---")
    print(toolExecutor.getAvailableTools())

    # 4. 执行工具测试
    print("\n--- 执行工具测试 ---")

    print("\n[测试 1] Search['什么是基因组学']")
    result = toolExecutor.execute("Search", "什么是基因组学")
    print(f"结果: {result}")

    print("\n[测试 2] Calculator['15 * 23 + 7']")
    result = toolExecutor.execute("Calculator", "15 * 23 + 7")
    print(f"结果: {result}")

    print("\n[测试 3] 不存在的工具 UnknownTool['test']")
    result = toolExecutor.execute("UnknownTool", "test")
    print(f"结果: {result}")

    # 5. 打印执行日志
    print("\n--- 工具执行日志 ---")
    for log in toolExecutor.getExecutionLog():
        print(f"  {log['tool']}: status={log['status']}, latency={log['latency']}s")

    print()
    print("=" * 60)
    print("ToolExecutor 测试通过。")
    print("=" * 60)
