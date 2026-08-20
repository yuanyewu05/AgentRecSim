import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(
    BASE_DIR / ".env",
    override=True,
)


def call_bailian(prompt: str) -> str:
    """调用阿里云百炼API，并返回模型回复。"""

    api_key = os.getenv(
        "DASHSCOPE_API_KEY",
        "",
    ).strip()
    base_url = os.getenv(
        "DASHSCOPE_BASE_URL",
        "",
    ).strip().rstrip("/")
    model = os.getenv(
        "DASHSCOPE_MODEL",
        "",
    ).strip()

    if not api_key:
        raise RuntimeError(
            "没有读取到DASHSCOPE_API_KEY"
        )
    if not base_url:
        raise RuntimeError(
            "没有读取到DASHSCOPE_BASE_URL"
        )
    if not model:
        raise RuntimeError(
            "没有读取到DASHSCOPE_MODEL"
        )

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
                "content": (
                    "你是WebSim推荐系统中的"
                    "阿里云百炼大模型辅助模块。"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
        max_tokens=200,
        extra_body={
            "enable_thinking": False,
        },
    )

    answer = response.choices[0].message.content
    if not answer:
        return "阿里云百炼API返回成功，但回复内容为空。"
    return answer.strip()
