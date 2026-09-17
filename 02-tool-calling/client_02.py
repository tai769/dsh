"""旧文件名的兼容入口；第 02 章正式实现位于 client.py。"""

from client import DeepSeekClient, Message, Tool, ToolCall, load_api_key

__all__ = ["DeepSeekClient", "Message", "Tool", "ToolCall", "load_api_key"]
