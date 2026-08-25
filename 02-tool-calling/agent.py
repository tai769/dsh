from subprocess import TimeoutExpired
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

    tools_by_name = {tool.name: tool for tool in tools}
    history: list[Message] = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt),
    ]
    for _step in range(max_steps):
        reply = client.chat(history,tools)
        history.append(reply)
        if not reply.tool_calls:
            return history
        for call in reply.tool_calls:
            tool = tools_by_name.get(call.name)
            if tool is None:
                result = f"Error: 模型请求了未注册的工具{call.name!r}"
            else:
                try:
                    args = json.loads(call.arguments)
                    result = tool.execute(args)
                except Exception as error:
                    result = f"工具执行出错： {error}"
                history.append(
                    Message(role = "tool", content = result, tool_call_id=call.id)
                )
    raise RuntimeError(f"Agent 在 {max_steps} 个 setp内没有结束")
