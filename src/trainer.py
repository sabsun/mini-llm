
import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    """
    Trainer for Mini-LLaMA.
    """

    def __init__(
        self,
        model,
        optimizer,
        device,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device

        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, dataloader):

        self.model.train()

        total_loss = 0.0

        progress = tqdm(dataloader)

        for x, y in progress:

            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(x)

            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        return total_loss / len(dataloader)
