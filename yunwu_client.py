import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# 读取与 app.py 同一级目录中的 .env
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def call_yunwu(prompt: str) -> str:
    """调用云雾 API，并返回模型回复。"""

    api_key = os.getenv("YUNWU_API_KEY", "").strip()
    base_url = os.getenv("YUNWU_BASE_URL", "").strip().rstrip("/")
    model = os.getenv("YUNWU_MODEL", "").strip()

    if not api_key:
        raise RuntimeError("没有读取到 YUNWU_API_KEY")

    if not base_url:
        raise RuntimeError("没有读取到 YUNWU_BASE_URL")

    if not model:
        raise RuntimeError("没有读取到 YUNWU_MODEL")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=60.0,
        max_retries=1,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "你是 WebSim 推荐系统中的大模型辅助模块。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
        max_tokens=200,
    )

    answer = response.choices[0].message.content

    if not answer:
        return "云雾 API 返回成功，但回复内容为空。"

    return answer.strip()