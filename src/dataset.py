
import torch
from torch.utils.data import Dataset


class LlamaDataset(Dataset):
    """
    Dataset for causal language modeling.

    Input:
        tokens: list[int]
        seq_len: int

    Returns:
        x: (seq_len,)
        y: (seq_len,)
    """

    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, index):
        x = self.tokens[index:index + self.seq_len]
        y = self.tokens[index + 1:index + self.seq_len + 1]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )
