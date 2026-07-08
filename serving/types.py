from dataclasses import dataclass


@dataclass(slots=True)
class GenerationChunk:
    """
    A single generated chunk emitted by the generator.
    """

    text: str
    token_id: int
    finish_reason: str | None = None