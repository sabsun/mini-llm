import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sentencepiece as spm

ROOT = Path(__file__).parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from config import *
from dataset import LlamaDataset
from model import MiniLlama
from trainer import Trainer


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# --------------------------------------------------
# Tokenizer
# --------------------------------------------------

tokenizer = spm.SentencePieceProcessor()
tokenizer.load(str(ROOT / TOKENIZER_PATH))

vocab_size = tokenizer.get_piece_size()

print(f"Vocabulary Size : {vocab_size}")


# --------------------------------------------------
# Load Pre-tokenized Dataset
# --------------------------------------------------

train_tokens = np.load(ROOT / TRAIN_TOKENS_PATH)
val_tokens = np.load(ROOT / VAL_TOKENS_PATH)

print(f"Train Tokens      : {len(train_tokens):,}")
print(f"Validation Tokens : {len(val_tokens):,}")
print(f"Vocabulary Size      : {vocab_size:,}")

train_dataset = LlamaDataset(train_tokens, SEQ_LEN)

print("Dataset Length :", len(train_dataset))
print("Expected Batches :", len(train_dataset) // BATCH_SIZE)

# --------------------------------------------------
# DataLoaders
# --------------------------------------------------

train_loader = DataLoader(
    LlamaDataset(train_tokens, SEQ_LEN),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    LlamaDataset(val_tokens, SEQ_LEN),
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = MiniLlama(
    vocab_size=vocab_size,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    hidden_dim=HIDDEN_DIM,
).to(device)


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


trainer = Trainer(
    model=model,
    optimizer=optimizer,
    device=device,
)


# --------------------------------------------------
# Checkpoints
# --------------------------------------------------

ckpt_dir = ROOT / CHECKPOINT_DIR
ckpt_dir.mkdir(exist_ok=True)

latest = ckpt_dir / "latest.pt"
best = ckpt_dir / "best.pt"

start_epoch = 1
best_val = float("inf")

if latest.exists():

    checkpoint = torch.load(
        latest,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1

    best_val = checkpoint.get(
        "best_val_loss",
        float("inf"),
    )

    print(f"Resuming from epoch {start_epoch}")


# --------------------------------------------------
# Validation
# --------------------------------------------------

def validate():

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )

            total_loss += loss.item()

    return total_loss / len(val_loader)


# --------------------------------------------------
# Training
# --------------------------------------------------

for epoch in range(start_epoch, EPOCHS + 1):

    print("=" * 60)
    print(f"Epoch {epoch}/{EPOCHS}")

    train_loss = trainer.train_epoch(train_loader)

    val_loss = validate()

    print(f"Train Loss : {train_loss:.4f}")
    print(f"Val Loss   : {val_loss:.4f}")

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": min(best_val, val_loss),
        "vocab_size": vocab_size,
    }

    torch.save(state, latest)

    if val_loss < best_val:

        best_val = val_loss

        torch.save(state, best)

        print("Saved best checkpoint.")


print("Training completed.")