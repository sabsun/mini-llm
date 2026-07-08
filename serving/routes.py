from fastapi import APIRouter, Request

from serving.generation_config import GenerationConfig
from serving.schemas import (
    GenerateRequest,
    GenerateResponse,
)
from serving.streaming import sse_response

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/generate", response_model=GenerateResponse)
def generate(
    body: GenerateRequest,
    request: Request,
):
    generator = request.app.state.llm.generator

    config = GenerationConfig(
        temperature=body.temperature,
        top_k=body.top_k,
        top_p=body.top_p,
        repetition_penalty=body.repetition_penalty,
        max_new_tokens=body.max_new_tokens,
        stop_tokens=body.stop_tokens,
    )

    text = "".join(
        generator.stream(
            prompt=body.prompt,
            config=config,
        )
    )

    return GenerateResponse(text=text)

@router.post("/generate/stream")
def generate_stream(
    body: GenerateRequest,
    request: Request,
):
    generator = request.app.state.llm.generator

    config = GenerationConfig(
        temperature=body.temperature,
        top_k=body.top_k,
        top_p=body.top_p,
        repetition_penalty=body.repetition_penalty,
        max_new_tokens=body.max_new_tokens,
        stop_tokens=body.stop_tokens,
    )

    stream = generator.stream(
        prompt=body.prompt,
        config=config,
    )

    return sse_response(stream)