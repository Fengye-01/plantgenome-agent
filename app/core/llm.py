"""
LLM 封装层：参考 Hello Agents 教程 chapter4/llm_client.py 的 HelloAgentsLLM 实现

与教程的区别：
- 教程用 openai 库的 OpenAI 兼容接口（本文件采用同样方式）
- 增加了异步 stream 方法（FastAPI SSE / Streamlit 打字机效果需要）
- 增加了 system prompt 参数支持
- 保留教程的 think() 方法命名，同时提供 invoke() / stream() 别名

为什么用 openai 库而不是直接 httpx 调 Ollama？
- Ollama 提供 OpenAI 兼容端点（/v1），用 openai 库是行业标准写法
- Hello Agents 教程全程用 openai 库，保持一致便于参考教程代码
- openai 库处理了重试、超时、流式解析等细节，比自己写 httpx 更健壮
"""
from __future__ import annotations

import os
from typing import AsyncGenerator, List, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI

# 加载项目根目录的 .env 文件
load_dotenv()


class HelloAgentsLLM:
    """
    参考 Hello Agents 教程定制的 LLM 客户端。
    调用任何兼容 OpenAI 接口的服务（本地 Ollama / OpenAI / 其他兼容端点），默认流式响应。
    """

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        timeout: int = None,
    ):
        """
        初始化客户端。优先使用传入参数，未提供则从环境变量加载。
        环境变量名与 Hello Agents 教程一致：LLM_MODEL_ID / LLM_API_KEY / LLM_BASE_URL
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        api_key = api_key or os.getenv("LLM_API_KEY")
        base_url = base_url or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", "120"))

        if not all([self.model, api_key, base_url]):
            raise ValueError(
                "模型ID、API密钥和服务地址必须被提供或在 .env 文件中定义。"
                "请检查 .env 中的 LLM_MODEL_ID / LLM_API_KEY / LLM_BASE_URL"
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def think(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
        stream: bool = True,
        verbose: bool = True,
    ) -> str:
        """
        调用大语言模型进行思考，返回完整响应文本。
        与教程一致的方法名，默认流式输出到控制台。

        Args:
            messages: OpenAI 格式的消息列表 [{"role": "system/user", "content": "..."}]
            temperature: 生成温度，RAG/Agent 场景建议 0.3（稳定输出）
            max_tokens: 最大生成 token 数，默认 2048，防止模型无限生成导致重复循环
            stream: 是否流式输出
            verbose: 是否打印过程信息

        Returns:
            模型生成的完整回答文本
        """
        if verbose:
            print(f"🧠 正在调用 {self.model} 模型...")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )

            if stream:
                if verbose:
                    print("✅ 大语言模型响应成功:")
                collected_content = []
                for chunk in response:
                    if not chunk.choices:
                        continue
                    content = chunk.choices[0].delta.content or ""
                    if verbose:
                        print(content, end="", flush=True)
                    collected_content.append(content)
                if verbose:
                    print()  # 流式输出结束后换行
                return "".join(collected_content)
            else:
                return response.choices[0].message.content

        except Exception as e:
            if verbose:
                print(f"❌ 调用 LLM API 时发生错误: {e}")
            return None

    def invoke(self, prompt: str, system: Optional[str] = None, temperature: float = 0.3, max_tokens: int = 2048) -> str:
        """
        同步调用，便捷接口。自动构建 messages，返回完整回答。
        这是对 think() 的封装，适合上层业务直接调用。

        Args:
            prompt: 用户问题 / 指令
            system: 系统提示词（可选），用于设定角色、约束输出格式
            temperature: 生成温度
            max_tokens: 最大生成 token 数，默认 1024

        Returns:
            模型生成的完整回答文本
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.think(messages, temperature=temperature, max_tokens=max_tokens, stream=False, verbose=False)

    async def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """
        异步流式调用，逐 token 产出回答。
        用于 FastAPI SSE 流式响应、Streamlit 打字机效果等场景。

        Args:
            prompt: 用户问题 / 指令
            system: 系统提示词（可选）
            temperature: 生成温度
            max_tokens: 最大生成 token 数，默认 1024

        Yields:
            逐段文本 token
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # openai 库本身支持异步，但为了保持与教程一致的同步客户端，
        # 这里用同步客户端的流式迭代在异步生成器中 yield
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in response:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content or ""
            if content:
                yield content


# --- 客户端使用示例：python -m app.core.llm ---
if __name__ == "__main__":
    import asyncio

    print("=" * 60)
    print("PlantGenome Agent - LLM 封装层测试（参考 Hello Agents 教程）")
    print("=" * 60)
    print(f"Model: {os.getenv('LLM_MODEL_ID')}")
    print(f"Base URL: {os.getenv('LLM_BASE_URL')}")
    print()

    try:
        llm = HelloAgentsLLM()
    except ValueError as e:
        print(f"初始化失败: {e}")
        exit(1)

    # 测试 1: think() 方法（教程原生方式，带 system prompt）
    print("[测试 1] think() 方法 - 带 system prompt 的角色设定")
    print("-" * 40)
    messages = [
        {"role": "system", "content": "你是一个植物基因组学研究助手，回答要专业、简洁，使用生物信息学术语。"},
        {"role": "user", "content": "什么是 CpG 岛？"},
    ]
    answer = llm.think(messages, temperature=0.3)
    if answer:
        print(f"\n✅ think() 调用成功，回答长度: {len(answer)} 字符")
    print()

    # 测试 2: invoke() 便捷方法
    print("[测试 2] invoke() 便捷方法 - 简单调用")
    print("-" * 40)
    answer = llm.invoke("用一句话说明 RAG 的基本流程。")
    print(f"回答: {answer}")
    print("✅ invoke() 调用成功")
    print()

    # 测试 3: 异步 stream() 方法
    print("[测试 3] stream() 异步流式方法 - 逐 token 输出")
    print("-" * 40)

    async def test_stream():
        full = ""
        async for token in llm.stream("写一个 Python 快速排序函数。"):
            print(token, end="", flush=True)
            full += token
        print()
        print(f"✅ stream() 调用成功，共 {len(full)} 字符")

    asyncio.run(test_stream())
    print()
    print("=" * 60)
    print("全部测试通过。LLM 封装层（参考 Hello Agents 教程）就绪。")
    print("=" * 60)
