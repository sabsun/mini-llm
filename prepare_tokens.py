from pathlib import Path

import numpy as np
import sentencepiece as spm
from tqdm import tqdm


# =====================================================
# Paths
# =====================================================

ROOT = Path(__file__).parent

CORPUS_PATH = ROOT / "data" / "general_corpus.txt"

TOKENIZER_PATH = ROOT / "tokenizer" / "general.model"

GENERAL_TOKENS_PATH = ROOT / "data" / "general_tokens.npy"

TRAIN_TOKENS_PATH = ROOT / "data" / "train_tokens.npy"

VAL_TOKENS_PATH = ROOT / "data" / "val_tokens.npy"


# =====================================================
# Load Tokenizer
# =====================================================

print("=" * 60)
print("Loading tokenizer...")

tokenizer = spm.SentencePieceProcessor()
tokenizer.load(str(TOKENIZER_PATH))

print(f"Vocabulary Size : {tokenizer.get_piece_size()}")


# =====================================================
# Tokenize Corpus
# =====================================================

print("=" * 60)
print("Tokenizing corpus...")

tokens = []

with open(CORPUS_PATH, "r", encoding="utf-8") as f:

    for line in tqdm(f):

        line = line.strip()

        if not line:
            continue

        ids = tokenizer.encode(line)

        tokens.extend(ids)

tokens = np.array(tokens, dtype=np.uint16)

print(f"\nTotal Tokens : {len(tokens):,}")

np.save(GENERAL_TOKENS_PATH, tokens)

print(f"Saved : {GENERAL_TOKENS_PATH}")


# =====================================================
# Train / Validation Split
# =====================================================

print("=" * 60)
print("Creating train/validation split...")

split = int(len(tokens) * 0.9)

train_tokens = tokens[:split]

val_tokens = tokens[split:]

print(f"Train Tokens      : {len(train_tokens):,}")
print(f"Validation Tokens : {len(val_tokens):,}")

np.save(TRAIN_TOKENS_PATH, train_tokens)

np.save(VAL_TOKENS_PATH, val_tokens)

print(f"Saved : {TRAIN_TOKENS_PATH}")
print(f"Saved : {VAL_TOKENS_PATH}")

print("=" * 60)
print("Dataset preparation completed successfully.")