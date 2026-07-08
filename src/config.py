D_MODEL = 256
N_LAYERS = 6
N_HEADS = 8
HIDDEN_DIM = 1024

SEQ_LEN = 128
BATCH_SIZE = 16

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1

EPOCHS = 1

DATASET_PATH = "data/python/train.txt"
TOKENIZER_PATH = "tokenizer/general.model"
CHECKPOINT_DIR = "checkpoints"

MAX_NEW_TOKENS = 200
TEMPERATURE = 0.8

TRAIN_TOKENS_PATH = "data/train_tokens.npy"
VAL_TOKENS_PATH = "data/val_tokens.npy"