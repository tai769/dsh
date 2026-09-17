"""第 01 章 demo：依次运行三种调用方式。

在项目根目录运行：
    .venv/bin/python 01-streaming-agent/demo.py

需要项目根目录的 .env 中包含 DEEPSEEK_API_KEY。
"""

from __future__ import annotations

import asyncio

from client import DeepSeekClient, Message


HISTORY = [
    Message(role="system", content="你是一个简洁的助手，回答不超过三句话。"),
    Message(role="user", content="什么是流式输出？用一句话回答。"),
]


def demo_chat() -> None:
    """非流式：等待生成完成后一次性打印。"""
    print("=" * 60)
    print("演示 1：非流式调用（一次拿回完整回答）")
    print("=" * 60)
    client = DeepSeekClient()
    answer = client.chat(HISTORY)
    print(f"完整回答：{answer}")


async def demo_stream() -> None:
    """流式：每收到一个分片便立即打印。"""
    print()
    print("=" * 60)
    print("演示 2：流式调用（边生成边显示）")
    print("=" * 60)
    client = DeepSeekClient()
    print("逐分片输出：", end="")
    async for piece in client.stream(HISTORY):
        print(piece, end="", flush=True)
    print()


async def demo_stream_message() -> None:
    """流式展示分片，并组装成一条完整历史消息。"""
    print()
    print("=" * 60)
    print("演示 3：流式 + 组装成完整历史消息")
    print("=" * 60)
    client = DeepSeekClient()
    message = await client.stream_message(HISTORY)
    print(f"进入历史的消息：role={message.role!r}，长度={len(message.content)} 字")
    print(f"消息内容：{message.content}")


async def main() -> None:
    demo_chat()
    await demo_stream()
    await demo_stream_message()


if __name__ == "__main__":
    asyncio.run(main())
