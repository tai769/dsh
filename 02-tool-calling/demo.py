import os
from  pathlib  import Path

from dotenv import dotenv_values


from dataclasses import dataclass

def load_api_key() -> str:
    from_env = os.getenv("DEEPSEEK_API_KEY")
    if from_env:
        return from_env

    env_path = Path(__file__).resolve().parents[3] / ".env"
    from_file = dotenv_values(env_path).get("DEEPSEEK_API_KEY")
    if from_file:
        return from_file
    raise RuntimeError("找不到 DEEPSEEK_API_KEY：请参考 .env.example 创建 .env")

@dataclass(frozen=True)
class Message:
    role: str
    content: str

m = Message(role = "user", content="你好")

