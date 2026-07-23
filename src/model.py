"""
model.py

Model architecture for SeqGuard:
- SequenceEncoder: Transformer encoder over tokenized ad-event sequences.
- MaskedEventHead: prediction head used only during self-supervised pretraining.
- FusionClassifier: combines the pretrained sequence embedding with tabular
  features to output a bot-vs-human probability.
"""

import math
import torch
import torch.nn as nn

import sys
sys.path.append("src")
import config


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding, added to token embeddings so
    the Transformer knows the ORDER of events, not just which events occurred."""

    def __init__(self, embed_dim, max_len=config.SEQ_LEN):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # shape: [1, max_len, embed_dim]

    def forward(self, x):
        # x: [batch, seq_len, embed_dim]
        return x + self.pe[:, : x.size(1), :]


class SequenceEncoder(nn.Module):
    """
    Embeds tokens, adds positional information, and runs them through a
    Transformer encoder stack. Output: one contextual embedding per position
    in the input sequence.
    """

    def __init__(
        self,
        vocab_size=config.VOCAB_SIZE,
        embed_dim=config.EMBED_DIM,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        max_len=config.SEQ_LEN,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=config.PAD_TOKEN)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, padding_mask=None):
        """
        x: [batch, seq_len] token ids
        padding_mask: [batch, seq_len] bool tensor, True where token is PAD
                      (tells the Transformer to ignore those positions)
        Returns: [batch, seq_len, embed_dim] contextual embeddings
        """
        emb = self.embedding(x)                       # [batch, seq_len, embed_dim]
        emb = self.positional_encoding(emb)
        out = self.transformer(emb, src_key_padding_mask=padding_mask)
        return out


class MaskedEventHead(nn.Module):
    """
    Used only during self-supervised pretraining. Projects the encoder's
    output at each position back to vocabulary size, so the model can
    predict which token was originally at each masked position.
    """

    def __init__(self, embed_dim=config.EMBED_DIM, vocab_size=config.VOCAB_SIZE):
        super().__init__()
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, encoder_output):
        # encoder_output: [batch, seq_len, embed_dim]
        # returns: [batch, seq_len, vocab_size] logits over the vocabulary
        return self.linear(encoder_output)


class FusionClassifier(nn.Module):
    """
    Used after pretraining, for the actual bot-detection task. Combines a
    pooled sequence embedding (from the pretrained encoder) with tabular
    features, and outputs a single bot-probability score.
    """

    def __init__(
        self,
        embed_dim=config.EMBED_DIM,
        tabular_dim=config.TABULAR_FEATURE_DIM,
        hidden_dim=64,
    ):
        super().__init__()
        self.tabular_proj = nn.Linear(tabular_dim, tabular_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + tabular_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sequence_embedding, tabular_features):
        """
        sequence_embedding: [batch, embed_dim]  (already pooled, e.g. mean over seq_len)
        tabular_features:   [batch, tabular_dim]
        Returns: [batch, 1] raw logits (apply sigmoid outside, or use
                 BCEWithLogitsLoss which does it internally)
        """
        tab = self.tabular_proj(tabular_features)
        combined = torch.cat([sequence_embedding, tab], dim=1)
        logits = self.classifier(combined)
        return logits


def pool_sequence_embedding(encoder_output, padding_mask=None):
    """
    Helper: collapses [batch, seq_len, embed_dim] into [batch, embed_dim] by
    mean-pooling over real (non-padding) positions only.
    """
    if padding_mask is None:
        return encoder_output.mean(dim=1)

    mask = (~padding_mask).unsqueeze(-1).float()  # 1 where real token, 0 where pad
    summed = (encoder_output * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


if __name__ == "__main__":
    # Quick smoke test with dummy data, using real config values, to confirm
    # shapes flow correctly before wiring into pretrain.py / finetune.py.
    batch_size = 4

    dummy_tokens = torch.randint(1, config.VOCAB_SIZE, (batch_size, config.SEQ_LEN))
    dummy_tokens[:, :10] = config.PAD_TOKEN  # simulate some padding
    padding_mask = dummy_tokens == config.PAD_TOKEN

    encoder = SequenceEncoder()
    mlm_head = MaskedEventHead()
    classifier = FusionClassifier()

    encoder_out = encoder(dummy_tokens, padding_mask=padding_mask)
    print("Encoder output shape:      ", encoder_out.shape)  # [batch, seq_len, embed_dim]

    mlm_logits = mlm_head(encoder_out)
    print("MaskedEventHead output:    ", mlm_logits.shape)   # [batch, seq_len, vocab_size]

    pooled = pool_sequence_embedding(encoder_out, padding_mask)
    print("Pooled sequence embedding: ", pooled.shape)        # [batch, embed_dim]

    dummy_tabular = torch.randn(batch_size, config.TABULAR_FEATURE_DIM)
    class_logits = classifier(pooled, dummy_tabular)
    print("FusionClassifier output:   ", class_logits.shape)  # [batch, 1]

    print("\nAll shapes correct — model architecture is wired up properly.")