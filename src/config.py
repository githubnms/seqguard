"""
config.py

Central configuration for SeqGuard. All hyperparameters and paths live here
so that pretrain.py, finetune.py, and evaluate.py stay in sync — change a
value once, here, instead of editing multiple scripts separately.
"""

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED_DIR = "data/processed"
SEQUENCES_PATH = f"{PROCESSED_DIR}/sequences.pt"
LABELS_PATH = f"{PROCESSED_DIR}/labels.pt"
VOCAB_PATH = f"{PROCESSED_DIR}/vocab.pt"

CHECKPOINT_DIR = "checkpoints"
PRETRAINED_ENCODER_PATH = f"{CHECKPOINT_DIR}/pretrained_encoder.pt"
FINETUNED_MODEL_PATH = f"{CHECKPOINT_DIR}/finetuned_model.pt"
BASELINE_MODEL_PATH = f"{CHECKPOINT_DIR}/baseline_model.pt"

# ---------------------------------------------------------------------------
# Data / sequence settings
# ---------------------------------------------------------------------------
SEQ_LEN = 50          # matches MAX_LEN used in prepare_data.py
VOCAB_SIZE = 5672     # actual vocab size from your run (5670) + buffer for
                      # padding token (0) and mask token — set below

# Reserved special tokens
PAD_TOKEN = 0
MASK_TOKEN = VOCAB_SIZE - 1   # last index reserved for the [MASK] token

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1
TABULAR_FEATURE_DIM = 16   # size of tabular embedding used in FusionClassifier

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 256
LR = 3e-4
MASK_PROB = 0.15       # fraction of tokens masked during self-supervised pretraining
PRETRAIN_EPOCHS = 10
FINETUNE_EPOCHS = 5
WEIGHT_DECAY = 1e-5

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42