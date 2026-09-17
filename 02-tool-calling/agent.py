"""第 02 章：工具调用循环 —— DeepSeek Harness Agent Loop 的最小骨架。

流程：请求模型 → 模型请求工具 → 执行工具 → 结果回灌 → 再请求 → 最终作答。
"""

from __future__ import annotations

import json

from client import DeepSeekClient, Message, Tool


def run_agent(
    client: DeepSeekClient,
    tools: list[Tool],
    system_prompt: str,
    user_prompt: str,
    max_steps: int = 10,
) -> list[Message]:
    """运行一轮带工具调用的对话，返回包含最终回答的完整历史。"""
    tools_by_name = {tool.name: tool for tool in tools}
    history: list[Message] = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt),
    ]

    for _step in range(max_steps):
        reply = client.chat(history, tools)
        history.append(reply)

        if not reply.tool_calls:
            return history

        # 工具失败也作为结果回灌，让模型有机会在下一轮自行修正。
        for call in reply.tool_calls:
            tool = tools_by_name.get(call.name)
            if tool is None:
                result = f"Error: 模型请求了未注册的工具 {call.name!r}"
            else:
                try:
                    args = json.loads(call.arguments)
                    result = tool.execute(args)
                except Exception as error:
                    result = f"工具执行出错: {error}"
            history.append(
                Message(role="tool", content=result, tool_call_id=call.id)
            )

    raise RuntimeError(f"Agent 在 {max_steps} 个 step 内没有结束")
