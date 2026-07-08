from __future__ import annotations

import json

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from serving.converters import to_generation_config
from serving.openai import (
    completion_id,
    created_timestamp,
    chat_completion_chunk,
    chat_completion_response,
)
from serving.prompt_formatter import PromptFormatter
from serving.schemas import ChatCompletionRequest

router = APIRouter(
    prefix="/v1",
    tags=["OpenAI Compatible API"],
)


@router.post("/chat/completions")
def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
):
    """
    OpenAI-compatible Chat Completions endpoint.
    """

    generator = request.app.state.llm.generator

    prompt = PromptFormatter.format(body.messages)

    config = to_generation_config(body)

    response_id = completion_id()

    created = created_timestamp()

    #
    # Non-streaming response
    #
    if not body.stream:

        output = []

        for chunk in generator.stream(
            prompt=prompt,
            config=config,
        ):
            output.append(chunk)

        text = "".join(output)

        return chat_completion_response(
            response_id=response_id,
            created=created,
            model=body.model,
            text=text,
        )

    #
    # Streaming response
    #
    async def event_generator():

        #
        # Initial assistant role chunk
        #
        yield {
            "data": json.dumps(
                chat_completion_chunk(
                    response_id=response_id,
                    created=created,
                    model=body.model,
                    include_role=True,
                )
            )
        }

        #
        # Token streaming
        #
        for chunk in generator.stream(
            prompt=prompt,
            config=config,
        ):

            yield {
                "data": json.dumps(
                    chat_completion_chunk(
                        response_id=response_id,
                        created=created,
                        model=body.model,
                        content=chunk,
                    )
                )
            }

        #
        # Final chunk
        #
        yield {
            "data": json.dumps(
                chat_completion_chunk(
                    response_id=response_id,
                    created=created,
                    model=body.model,
                    finish_reason="stop",
                )
            )
        }

        #
        # OpenAI stream terminator
        #
        yield {
            "data": "[DONE]"
        }

    return EventSourceResponse(event_generator())