"""
dataset.py

PyTorch Dataset classes for SeqGuard.

MaskedSequenceDataset: used for self-supervised pretraining. Randomly masks
~15% of real (non-padding) tokens in each sequence and returns both the
masked sequence and the original targets, so the model can be trained to
predict the hidden tokens (BERT-style masked-language-modeling, applied to
ad-event sequences instead of text).
"""

import random
import torch
from torch.utils.data import Dataset

import sys
sys.path.append("src")
import config


class MaskedSequenceDataset(Dataset):
    def __init__(self, sequences, mask_prob=config.MASK_PROB,
                 mask_token=config.MASK_TOKEN, pad_token=config.PAD_TOKEN):
        """
        sequences: torch.LongTensor of shape [num_devices, seq_len]
                   (this is exactly what sequences.pt contains)
        """
        self.sequences = sequences
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.pad_token = pad_token

    def __len__(self):
        return len(self.sequences)

    def mask_tokens(self, seq):
        """
        Randomly replaces ~15% of non-padding tokens with the [MASK] token.
        Returns:
            masked_seq: the sequence with some tokens replaced by MASK_TOKEN
            labels: same length as seq; contains the ORIGINAL token at masked
                    positions, and -100 everywhere else (ignored by the loss
                    function during training)
        """
        seq = seq.clone()
        labels = torch.full_like(seq, fill_value=-100)  # -100 = "ignore this position"

        for i in range(len(seq)):
            token = seq[i].item()

            # never mask padding — there's nothing real to predict there
            if token == self.pad_token:
                continue

            if random.random() < self.mask_prob:
                labels[i] = token          # remember the real token as the target
                seq[i] = self.mask_token   # hide it from the model

        return seq, labels

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        masked_seq, labels = self.mask_tokens(seq)
        return masked_seq, labels


if __name__ == "__main__":
    # Quick manual test: load real preprocessed sequences and check masking
    # behaves correctly before wiring it into full training.
    sequences = torch.load(config.SEQUENCES_PATH)
    print(f"Loaded sequences: {sequences.shape}")

    dataset = MaskedSequenceDataset(sequences)
    masked_seq, labels = dataset[0]

    print("Original sequence:  ", sequences[0].tolist())
    print("Masked sequence:    ", masked_seq.tolist())
    print("Labels (targets):   ", labels.tolist())

    num_masked = (labels != -100).sum().item()
    print(f"\nNumber of positions masked in this example: {num_masked}")