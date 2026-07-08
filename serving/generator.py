from __future__ import annotations

from typing import Iterator

import torch

from serving.generation_config import GenerationConfig
from serving.loader import ModelLoader
from serving.sampling import sample_next_token


class TextGenerator:
    """
    Production-style text generator.

    Responsible only for inference.

    No HTTP.
    No FastAPI.
    No SSE.
    """

    def __init__(self, loader: ModelLoader):
        self.model = loader.model
        self.tokenizer = loader.tokenizer
        self.device = loader.device

    @torch.inference_mode()
    def stream(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> Iterator[str]:
        """
        Generate text incrementally.

        Yields decoded text chunks.
        """

        input_ids = self.tokenizer.encode(prompt)

        input_ids = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)

        generated = input_ids.clone()

        for _ in range(config.max_new_tokens):

            logits = self.model(generated)

            next_logits = logits[0, -1]

            next_token = sample_next_token(
                logits=next_logits,
                generated_tokens=generated[0],
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
            )

            next_token = torch.tensor(
                [[next_token]],
                device=self.device,
            )

            generated = torch.cat(
                [generated, next_token],
                dim=1,
            )

            token_id = next_token.item()

            piece = self.tokenizer.decode([token_id])

            if piece:
                yield piece

            if piece in config.stop_tokens:
                break