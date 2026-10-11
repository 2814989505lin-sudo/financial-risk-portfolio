"""Shared LLM configuration for Alibaba Cloud Model Studio (Bailian).

The project still imports ``langchain_openai.ChatOpenAI`` because Bailian
provides an OpenAI-compatible Chat Completions endpoint.
"""

import os


BAILIAN_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()

BAILIAN_BASE_URL = os.getenv(
    "BAILIAN_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

BAILIAN_MODEL = os.getenv("BAILIAN_MODEL", "kimi-k2.5")


def get_bailian_api_key() -> str:
    if not BAILIAN_API_KEY:
        raise RuntimeError(
            "Bailian API key is missing. Set DASHSCOPE_API_KEY in the environment."
        )
    return BAILIAN_API_KEY
