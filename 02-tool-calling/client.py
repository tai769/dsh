"""第 02 章：支持工具调用的客户端与消息模型。"""

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
    """先读取环境变量，再读取项目根目录的 .env。"""
    from_env = os.getenv("DEEPSEEK_API_KEY")
    if from_env:
        return from_env
    env_path = Path(__file__).resolve().parents[1] / ".env"
    from_file = dotenv_values(env_path).get("DEEPSEEK_API_KEY")
    if from_file:
        return from_file
    raise RuntimeError("找不到 DEEPSEEK_API_KEY：请在项目根目录配置 .env")


@dataclass(frozen=True)
class ToolCall:
    """模型发起的一次工具调用请求。"""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Message:
    """支持思考内容、工具调用和工具结果的对话消息。"""

    role: str
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True)
class Tool:
    """Agent 可用工具的模型描述和本地执行函数。"""

    name: str
    description: str
    parameters: dict[str, Any]
    execute: Callable[[dict[str, Any]], str]


class DeepSeekClient:
    BASE_URL = "https://api.deepseek.com"
    MODEL = "deepseek-v4-flash"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or load_api_key()

    @staticmethod
    def _wire_message(message: Message) -> dict[str, Any]:
        """把内部消息转换成 DeepSeek OpenAI 兼容协议字典。"""
        wire: dict[str, Any] = {"role": message.role}
        if message.content is not None:
            wire["content"] = message.content
        elif message.role == "assistant":
            wire["content"] = ""
        if message.reasoning_content:
            wire["reasoning_content"] = message.reasoning_content
        if message.tool_calls:
            wire["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                }
                for call in message.tool_calls
            ]
        if message.tool_call_id is not None:
            wire["tool_call_id"] = message.tool_call_id
        return wire

    def chat(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
    ) -> Message:
        """执行非流式调用，返回含工具调用信息的完整消息。"""
        payload: dict[str, Any] = {
            "model": self.MODEL,
            "messages": [self._wire_message(message) for message in messages],
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]

        with httpx.Client(timeout=60) as client:
            response = client.post(
                f"{self.BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        raw_message = data["choices"][0]["message"]
        tool_calls = tuple(
            ToolCall(
                id=raw_call["id"],
                name=raw_call["function"]["name"],
                arguments=raw_call["function"]["arguments"],
            )
            for raw_call in raw_message.get("tool_calls") or []
        )
        return Message(
            role="assistant",
            content=raw_message.get("content"),
            reasoning_content=raw_message.get("reasoning_content"),
            tool_calls=tool_calls,
        )

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """流式调用：只处理纯文本回答。"""
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
                        self._wire_message(message) for message in messages
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
