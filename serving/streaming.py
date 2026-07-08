from collections.abc import Iterator

from sse_starlette.sse import EventSourceResponse


def sse_response(generator: Iterator[str]) -> EventSourceResponse:
    """
    Wrap a text generator as an SSE response.
    """

    async def event_generator():
        for chunk in generator:

            if chunk.finish_reason is not None:
                break

            yield {
                "event": "token",
                "data": chunk.text,
            }

        yield {
            "event": "done",
            "data": "[DONE]",
        }

    return EventSourceResponse(event_generator())