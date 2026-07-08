from dataclasses import dataclass, field


@dataclass(slots=True)
class GenerationConfig:
    """
    Configuration controlling text generation.
    """

    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    max_new_tokens: int = 128
    stop_tokens: list[str] = field(default_factory=list)