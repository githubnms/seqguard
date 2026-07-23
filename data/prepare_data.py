# 1. Load raw CSV in chunks (it's large — don't load all at once)
# 2. Group rows by device_id
# 3. Sort each group by 'hour' (timestamp)
# 4. Convert each row's categorical features into a single integer token
#    using a vocabulary you build (site_id + app_id + device_type -> token_id)
# 5. Truncate/pad each user's sequence to a fixed length (e.g. 50 events)
# 6. Save as a processed tensor file (sequences.pt) + labels (click/no-click as weak label)

"""
prepare_data.py

Builds per-device event sequences from the raw Avazu CTR CSV, tokenizes
categorical context, pads/truncates to a fixed length, and saves the result
as PyTorch tensors for training.

Run from the project root:
    python data/prepare_data.py
"""

import os
import pandas as pd
import torch
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW_PATH = "data/raw/train.csv"
OUT_DIR = "data/processed"
MAX_LEN = 50          # fixed sequence length per device
CHUNK_SIZE = 500_000  # rows per chunk when reading the large CSV
N_ROWS_LIMIT = 2_000_000  # start small while testing; increase later or set to None

os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: Build a vocabulary mapping (site_id, app_id, device_type) -> token id
# ---------------------------------------------------------------------------
def build_vocab(df):
    """Creates a unique integer token for each distinct
    (site_id, app_id, device_type) combination seen in the data."""
    combo = (
        df["site_id"].astype(str)
        + "_"
        + df["app_id"].astype(str)
        + "_"
        + df["device_type"].astype(str)
    )
    unique_values = combo.unique()
    # token 0 is reserved for padding, so real tokens start at 1
    vocab = {val: idx + 1 for idx, val in enumerate(unique_values)}
    return vocab, combo


# ---------------------------------------------------------------------------
# Step 2: Pad or truncate a sequence to a fixed length
# ---------------------------------------------------------------------------
def pad_sequence(seq, max_len=MAX_LEN, pad_token=0):
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [pad_token] * (max_len - len(seq)) + seq


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    print(f"Loading up to {N_ROWS_LIMIT} rows from {RAW_PATH} ...")

    # Read only the columns we actually need (saves memory on a large file)
    usecols = [
        "device_id", "hour", "click",
        "site_id", "app_id", "device_type",
    ]
    df = pd.read_csv(RAW_PATH, usecols=usecols, nrows=N_ROWS_LIMIT)
    print(f"Loaded {len(df)} rows.")

    # Step: build vocabulary + tokenize each row
    print("Building vocabulary...")
    vocab, combo = build_vocab(df)
    df["token"] = combo.map(vocab)
    print(f"Vocabulary size: {len(vocab)}")

    # Step: group rows by device_id, sort each group by time
    print("Grouping by device_id and sorting by hour...")
    df = df.sort_values(["device_id", "hour"])
    grouped = df.groupby("device_id")

    sequences = []
    labels = []

    print("Building per-device sequences...")
    for device_id, group in grouped:
        token_seq = group["token"].tolist()
        click_seq = group["click"].tolist()

        padded_seq = pad_sequence(token_seq)
        # weak label: did this device click at least once in its history
        label = int(any(click_seq))

        sequences.append(padded_seq)
        labels.append(label)

    print(f"Built {len(sequences)} device sequences.")

    # Step: convert to tensors and save
    sequences_tensor = torch.tensor(sequences, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    torch.save(sequences_tensor, os.path.join(OUT_DIR, "sequences.pt"))
    torch.save(labels_tensor, os.path.join(OUT_DIR, "labels.pt"))
    torch.save(vocab, os.path.join(OUT_DIR, "vocab.pt"))

    print("Saved sequences.pt, labels.pt, and vocab.pt to data/processed/")
    print(f"Sequences shape: {sequences_tensor.shape}")
    print(f"Labels shape: {labels_tensor.shape}")


if __name__ == "__main__":
    main()