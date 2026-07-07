
import sys
from pathlib import Path
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

tokenizer = spm.SentencePieceProcessor()
tokenizer.load(str(ROOT / TOKENIZER_PATH))
vocab_size = tokenizer.get_piece_size()

with open(ROOT / DATASET_PATH, "r", encoding="utf-8") as f:
    text = f.read()

tokens = tokenizer.encode(text)
split = int(len(tokens) * 0.9)
train_tokens = tokens[:split]
val_tokens = tokens[split:]

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

model = MiniLlama(
    vocab_size=vocab_size,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    hidden_dim=HIDDEN_DIM,
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

trainer = Trainer(model, optimizer, device)

ckpt_dir = ROOT / CHECKPOINT_DIR
ckpt_dir.mkdir(exist_ok=True)
latest = ckpt_dir / "latest.pt"
best = ckpt_dir / "best.pt"

start_epoch = 1
best_val = float("inf")

if latest.exists():
    ckpt = torch.load(latest, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    best_val = ckpt.get("best_val_loss", float("inf"))
    print(f"Resuming from epoch {start_epoch}")

def validate():
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )
            total += loss.item()
    return total / len(val_loader)

for epoch in range(start_epoch, EPOCHS + 1):
    print("="*60)
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
