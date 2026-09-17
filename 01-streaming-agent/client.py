"""第 01 章：最小流式 Agent。

这一章实现三样东西：
1. ``load_api_key()`` —— 从项目根目录的 .env 读 API Key
2. ``Message`` —— 一条不可变的对话消息
3. ``DeepSeekClient`` —— 同时支持完整响应与流式响应的模型客户端
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import httpx
from dotenv import dotenv_values
from httpx_sse import aconnect_sse


def load_api_key() -> str:
    """先读取环境变量，再读取项目根目录的 .env。"""
    from_env = os.getenv("DEEPSEEK_API_KEY")
    if from_env:
        return from_env

    # 本仓库将每章源码直接放在章节目录下，因此向上一级就是项目根目录。
    env_path = Path(__file__).resolve().parents[1] / ".env"
    from_file = dotenv_values(env_path).get("DEEPSEEK_API_KEY")
    if from_file:
        return from_file

    raise RuntimeError(
        "找不到 DEEPSEEK_API_KEY：请在项目根目录创建 .env，"
        "写入一行 DEEPSEEK_API_KEY=你的key"
    )


@dataclass(frozen=True)
class Message:
    """一条创建后不可修改的对话消息。"""

    role: str
    content: str


class DeepSeekClient:
    """基于 httpx 与 httpx-sse 的 DeepSeek OpenAI 兼容客户端。"""

    BASE_URL = "https://api.deepseek.com"
    MODEL = "deepseek-v4-flash"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or load_api_key()

    def chat(self, messages: list[Message]) -> str:
        """把整段对话发给模型，并在生成结束后返回完整回答。"""
        with httpx.Client(timeout=60) as client:
            response = client.post(
                f"{self.BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.MODEL,
                    "messages": [
                        {"role": message.role, "content": message.content}
                        for message in messages
                    ],
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            return cast(str, data["choices"][0]["message"]["content"])

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """模型每生成一小段，就立即产出一个分片。"""
        completed = False
        async with httpx.AsyncClient(timeout=60) as client:
            async with aconnect_sse(
                client,
                "POST",
                f"{self.BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.MODEL,
                    "messages": [
                        {"role": message.role, "content": message.content}
                        for message in messages
                    ],
                    "stream": True,
                },
            ) as event_source:
                event_source.response.raise_for_status()
                async for event in event_source.aiter_sse():
                    if event.data == "[DONE]":
                        completed = True
                        break
                    payload = json.loads(event.data)
                    delta = payload["choices"][0].get("delta", {})
                    piece = delta.get("content")
                    if piece:
                        yield piece
        if not completed:
            raise RuntimeError("流式响应在 [DONE] 之前中断，拒绝保存不完整消息")

    async def stream_message(self, messages: list[Message]) -> Message:
        """流式收集所有分片，并组装成一条完整的 assistant 消息。"""
        pieces: list[str] = []
        async for piece in self.stream(messages):
            pieces.append(piece)
        return Message(role="assistant", content="".join(pieces))
