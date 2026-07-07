
import sys
from pathlib import Path
import torch
import sentencepiece as spm

ROOT = Path(__file__).parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from config import *
from model import MiniLlama

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = spm.SentencePieceProcessor()
tokenizer.load(str(ROOT / TOKENIZER_PATH))
vocab_size = tokenizer.get_piece_size()

model = MiniLlama(
    vocab_size=vocab_size,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    hidden_dim=HIDDEN_DIM,
).to(device)

ckpt = torch.load(ROOT / CHECKPOINT_DIR / "best.pt", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

MAX_NEW_TOKENS = globals().get("MAX_NEW_TOKENS", 200)
TEMPERATURE = globals().get("TEMPERATURE", 0.8)

@torch.no_grad()
def generate(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=40):
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        context = ids[:, -SEQ_LEN:]
        logits = model(context)[:, -1, :] / temperature

        if top_k is not None:
            values, indices = torch.topk(logits, top_k)
            probs = torch.softmax(values, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            next_token = indices.gather(-1, next_idx)
        else:
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

        ids = torch.cat([ids, next_token], dim=1)

    return tokenizer.decode(ids[0].tolist())

print("="*60)
print("MiniLlama Inference")
print("Type 'exit' to quit.")
print("="*60)

while True:
    prompt = input("\nPrompt: ")
    if prompt.lower() == "exit":
        break
    print("\n" + generate(prompt))
