from serving.generation_config import GenerationConfig
from serving.schemas import ChatCompletionRequest, GenerateRequest


def to_generation_config(
    request: GenerateRequest | ChatCompletionRequest,
) -> GenerationConfig:
    stop_tokens = request.stop if hasattr(request, "stop") else request.stop_tokens

    if stop_tokens is None:
        stop_tokens = []
    elif isinstance(stop_tokens, str):
        stop_tokens = [stop_tokens]

    max_new_tokens = (
        request.max_tokens
        if hasattr(request, "max_tokens")
        else request.max_new_tokens
    )

    return GenerationConfig(
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        max_new_tokens=max_new_tokens,
        stop_tokens=stop_tokens,
    )