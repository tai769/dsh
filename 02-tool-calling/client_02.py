from ast import arguments
from __future__ import annotations
import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx
from dotenv import dotenv_values
from httpx_sse import aconnect_sse


def load_api_key() -> str:
    """按「环境变量优先，其次 .env 文件」的顺序找 DeepSeek API Key。"""
    from_env = os.getenv("DEEPSEEK_API_KEY")
    if from_env:
        return from_env
    env_path = Path(__file__).resolve().parents[3] / ".env"
    from_file = dotenv_values(env_path).get("DEEPSEEK_API_KEY")
    if from_file:
        return from_file
    raise RuntimeError("找不到 DEEPSEEK_API_KEY：请参考 .env.example 创建 .env")

@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str

@dataclass
class Message:
    role: str
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    execute: Callable[[dict[str, Any]], str]

class DeepSeekClient:
    BASE_URL = "https://api.deepseek.com"
    MODEL = "deepseek-chat"

    @staticmethod
    def _wire_message(m: Message) -> dict[str, Any]:
        wire: dict[str, Any] = {"role": m.role}
        if m.content is not None:
            wire["content"] = m.content
        elif m.role == "assistant":
            wire["content"] = ""
        if m.reasoning_content:
            wire["reasoning_content"] = m.reasoning_content
        if m.tool_calls:
            wire["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in m.tool_calls
            ]
        if m.tool_call_id is not None:
            wire["tool_call_id"] = m.tool_call_id
        return wire        









async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """流式调用：只处理纯文本回答的展示。工具调用场景见 README 对照。"""
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
                    "messages": [self._wire_message(m) for m in messages],
                    "stream": True,
                },
            ) as event_source:
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