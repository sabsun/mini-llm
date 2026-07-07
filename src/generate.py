
import torch


class Generator:
    """
    Greedy text generator for Mini-LLaMA.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device,
        max_seq_len=128,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_len = max_seq_len

    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_new_tokens=100,
    ):

        self.model.eval()

        tokens = self.tokenizer.encode(prompt)

        ids = torch.tensor(
            [tokens],
            dtype=torch.long,
            device=self.device,
        )

        for _ in range(max_new_tokens):

            # Keep only the last max_seq_len tokens
            context = ids[:, -self.max_seq_len:]

            logits = self.model(context)

            next_token = logits[:, -1].argmax(dim=-1)

            ids = torch.cat(
                [
                    ids,
                    next_token.unsqueeze(1),
                ],
                dim=1,
            )

        return self.tokenizer.decode(
            ids[0].tolist()
        )
