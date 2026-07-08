import serving.bootstrap

from model import MiniLlama
from config import D_MODEL, N_LAYERS, N_HEADS, HIDDEN_DIM

from pathlib import Path

import sentencepiece as spm
import torch

from serving.config import ServerConfig

class ModelLoader:
    """
    Loads the MiniLlama model and tokenizer once.

    This class is intended to be instantiated a single time during
    server startup and reused for all inference requests.
    """

    def __init__(self, config: ServerConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

    def _load_model(self) -> MiniLlama:
        """
        Load checkpoint and return model in evaluation mode.
        """

        checkpoint_path = Path(self.config.checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        tokenizer = self._load_tokenizer()
        vocab_size = tokenizer.get_piece_size()

        # Optional safety check
        if (
            "vocab_size" in checkpoint
            and checkpoint["vocab_size"] != vocab_size
        ):
            raise ValueError(
                f"Tokenizer vocab ({vocab_size}) does not match "
                f"checkpoint vocab ({checkpoint['vocab_size']})."
            )

        model = MiniLlama(
            vocab_size=vocab_size,
            d_model=D_MODEL,
            n_layers=N_LAYERS,
            n_heads=N_HEADS,
            hidden_dim=HIDDEN_DIM,
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        model.to(self.device)

        model.eval()

        return model

    def _load_tokenizer(self):
        """
        Load SentencePiece tokenizer.
        """

        tokenizer_path = Path(self.config.tokenizer_path)

        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found: {tokenizer_path}"
            )

        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(str(tokenizer_path))

        return tokenizer