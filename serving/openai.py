# serving/openai.py

from __future__ import annotations

import time
import uuid
from typing import Any


def completion_id() -> str:
    """
    Generate a unique OpenAI-style completion ID.
    """
    return f"chatcmpl-{uuid.uuid4().hex}"


def created_timestamp() -> int:
    """
    Unix timestamp used by OpenAI responses.
    """
    return int(time.time())


def chat_completion_response(
    *,
    response_id: str,
    created: int,
    model: str,
    text: str,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    """
    Standard (non-streaming) OpenAI Chat Completion response.
    """

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def chat_completion_chunk(
    *,
    response_id: str,
    created: int,
    model: str,
    content: str = "",
    finish_reason: str | None = None,
    include_role: bool = False,
) -> dict[str, Any]:
    """
    Streaming ChatCompletion chunk.

    The first chunk should be sent with include_role=True.

    Example:
        {
            "choices": [
                {
                    "delta": {
                        "role": "assistant"
                    }
                }
            ]
        }

    After that, send only content deltas.
    """

    delta: dict[str, Any] = {}

    if include_role:
        delta["role"] = "assistant"

    if content:
        delta["content"] = content

    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }