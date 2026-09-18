"""第 06 章：使用组装提示词与注册表的 Agent 循环。

与第 05 章的两处差异：
1. 工具从 ToolRegistry 来（不再是一张裸 list）；
2. 系统提示词由 PromptAssembler 组装（不再是手写一整块）。
"""

from httpcore import request
from __future__ import annotations

import json

from client import DeepSeekClient
from prompt import PromptAssembler
from registry import ToolRegistry
from session import Session


def run_agent(
    client: DeepSeekClient,
    registry: ToolRegistry,
    assembler: PromptAssembler,
    user_prompt: str,
    max_steps: int = 10,
    variables: dict[str, str] | None = None,
) -> Session:
    """使用组装后的系统提示词与工具 schema 运行一轮工具对话。"""
    session = Session()
    session.append("turn/start", {"turn": 1})
    request_header: str | None = None
    request_generation = 0
    try:
        for step in range(1, max_steps + 1):
            session.append("step/start", {"turn": 1, "step": step})
            completed = False
            try:
                # Prompt provider 可能返回运行时值，因此每个 step 都重新组装。
                system_prompt = assembler.render(variables)
                session.record_system_prompt(system_prompt, turn = 1, step=step)
                if step == 1:
                    session.append("user/message", {"content": user_prompt})
                tools = registry.all()
                tools_by_name = {tool.name: tool for tool in tools}
                header: dict[str, object] = {
                    "config": {
                        "provider" : "deepseek-official",
                        "model": client.MODEL,
                    }
                }
                schemas = registry.schema()
                if schemas:
                    header["tools"] = schemas
                header_fingerprint = json.load(
                    header, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                surface_changed = session.replace_generation != request_generation
                if header_fingerprint != request_header or surface_changed:
                    