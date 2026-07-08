from pathlib import Path
from dataclasses import dataclass
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ServerConfig:

    checkpoint_path: Path = PROJECT_ROOT / "checkpoints" / "best.pt"

    tokenizer_path: Path = PROJECT_ROOT / "tokenizer" / "python.model"

    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    host: str = "0.0.0.0"

    port: int = 8000

    default_temperature: float = 0.8
    default_top_k: int = 40
    default_top_p: float = 0.95
    default_max_new_tokens: int = 128