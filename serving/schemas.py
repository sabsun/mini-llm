from pydantic import BaseModel, Field
from typing import Literal

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt")

    temperature: float = Field(default=0.8, ge=0.0)
    top_k: int = Field(default=40, ge=0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)

    repetition_penalty: float = Field(default=1.0, ge=1.0)

    max_new_tokens: int = Field(default=128, gt=0)

    stop_tokens: list[str] = Field(default_factory=list)


class GenerateResponse(BaseModel):
    text: str

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "MiniLlama"

    messages: list[ChatMessage]

    temperature: float = Field(default=0.8, ge=0.0)

    top_k: int = Field(default=40, ge=0)

    top_p: float = Field(default=0.95, gt=0.0, le=1.0)

    repetition_penalty: float = Field(
        default=1.0,
        ge=1.0,
    )

    max_tokens: int = Field(
        default=128,
        gt=0,
    )

    stop: list[str] | str | None = None

    stream: bool = False