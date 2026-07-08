
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

    def __init__(self, tokens, seq_len, stride=None):
        self.tokens = tokens
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len

    def __len__(self):
        return (len(self.tokens) - self.seq_len - 1) // self.stride + 1

    def __getitem__(self, index):

        start = index * self.stride
        x = self.tokens[start:start + self.seq_len]
        y = self.tokens[start + 1:start + self.seq_len + 1]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )
