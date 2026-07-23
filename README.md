# SeqGuard: Self-Supervised Sequence Learning for Bot Traffic Detection

Detecting invalid/bot traffic in online advertising by modeling user ad-interaction
history as a sequence — pretrained with a self-supervised masked-event objective,
then fine-tuned for classification, and served as a real-time scoring pipeline.

> Inspired by representation-learning approaches used in production ad-fraud systems,
> including *Scaling Generative Pre-training for User Ad Activity Sequences* (AdKDD 2023)
> and *Self-supervised Representation Learning Across Sequential and Tabular Features
> Using Transformers* (NeurIPS 2022). This project is an independent, small-scale
> re-implementation built to explore the same class of methods on public data.

---

## Table of Contents

- [SeqGuard: Self-Supervised Sequence Learning for Bot Traffic Detection](#seqguard-self-supervised-sequence-learning-for-bot-traffic-detection)
  - [Table of Contents](#table-of-contents)
  - [Problem Statement](#problem-statement)
  - [Approach](#approach)
  - [Architecture](#architecture)
  - [Dataset \& Preprocessing](#dataset--preprocessing)
  - [Results](#results)
  - [Repository Structure](#repository-structure)
  - [Getting Started](#getting-started)
  - [Project Status](#project-status)
  - [System Design Notes](#system-design-notes)
  - [Limitations \& Future Work](#limitations--future-work)
  - [References](#references)
  - [Author](#author)

---

## Problem Statement

Invalid traffic (IVT) — bots, scripted clickers, and non-human engagement — is one of
the hardest problems in ad tech because:

1. **Labels are sparse and noisy.** Confirmed fraud labels cover a tiny fraction of
   real traffic; most data is unlabeled or weakly labeled.
2. **Class imbalance is severe.** Fraudulent sessions are a small minority of overall
   traffic, so naive accuracy is a meaningless metric.
3. **Behavior is sequential, not static.** A single click or impression in isolation
   rarely reveals intent — it's the *pattern* of events over time (timing, ordering,
   repetition) that separates bots from humans.
4. **Latency constraints are strict.** Detection has to run inline with ad serving,
   not as an offline batch job.

SeqGuard tackles this by treating each user/device as a **sequence of ad events**
rather than a flat feature vector, and by using self-supervised pretraining to learn
general behavioral representations before ever touching the (sparse) fraud labels.

---

## Approach

**Step 1 — Self-supervised pretraining.**
A Transformer encoder is pretrained on unlabeled event sequences using a masked-event
prediction objective (analogous to BERT's masked-language modeling): random events in
a user's sequence are masked, and the model learns to predict them from context. This
lets the model learn what "normal" behavioral patterns look like without needing any
fraud labels at all.

**Step 2 — Supervised fine-tuning.**
The pretrained encoder is attached to a fusion classification head that combines the
learned sequence embedding with tabular features (device type, IP-derived features,
site/app category). The combined representation is fine-tuned on the labeled
click/fraud signal, with a class-weighted loss to handle imbalance.

**Step 3 — Ablation.**
A supervised-only baseline (same architecture, no pretraining) is trained from
scratch on the same labeled data, to isolate and measure the actual contribution of
self-supervised pretraining.

**Step 4 — Real-time serving.**
The fine-tuned model is wrapped in a FastAPI scoring service sitting behind a Kafka
consumer that simulates a live ad-event stream, with Prometheus/Grafana tracking
inference latency and flagging rate — so the project reflects production constraints,
not just offline notebook metrics.

---

## Architecture

**Status: Design finalized, implementation in progress**

> The high-level design (Transformer encoder → masked-event pretraining →
> fusion classifier → real-time serving) is fixed. Detailed architecture diagrams
> and component descriptions will be added here as each module is implemented
> and tested.

**Model components:**

| Component | Description |
|---|---|
| `SequenceEncoder` | Transformer encoder (2 layers, 4 heads, 128-dim embeddings) over tokenized event sequences with positional encoding |
| `MaskedEventHead` | Linear projection over encoder output, used only during pretraining to predict masked tokens |
| `FusionClassifier` | Combines sequence embedding with tabular feature embeddings via concatenation + MLP, outputs bot/human probability |

---

## Dataset & Preprocessing

**Status: In Progress**

- [x] Dataset identified and downloaded (Avazu CTR Prediction, Kaggle)
- [x] Sequence construction pipeline (`prepare_data.py`) — in development
- [ ] Tokenization and vocabulary building
- [ ] Sequence padding/truncation strategy

> Full preprocessing methodology will be documented here once the pipeline is
> validated end-to-end.

---

## Results

**Status: Pending**

> Results will be published here once training and evaluation are complete,
> including a baseline comparison (pretrained vs. supervised-only) to isolate
> the impact of self-supervised pretraining.

---

## Repository Structure

```
seqguard/
├── data/                # preprocessing scripts (raw/processed data not committed)
├── src/                 # model, dataset, training, evaluation code
├── streaming/           # Kafka producer/consumer for live-traffic simulation
├── serving/             # FastAPI real-time scoring service
├── monitoring/          # Prometheus + Grafana configs
├── docker/              # Dockerfiles for training and serving images
├── tests/               # unit tests
├── requirements.txt
└── docker-compose.yml
```

---

## Getting Started

> Setup instructions will be finalized once the preprocessing and training
> pipeline are complete. Currently in progress — see [Timeline](#project-status)
> below for what's done so far.

---

## Project Status

| Date | Milestone |
|---|---|
| 2026-07-23 | Repo structure created, dataset downloaded |
| 2026-07-23 | Sequence preprocessing pipeline  |
| 2026-07-24 | Self-supervised pretraining |
| 2026-07-24 | Masked sequence dataset implemented and tested on real data |
| (in progress) | Fine-tuning + baseline ablation |
| — | Real-time serving + monitoring |

> This table is updated as each stage is completed.

---

## System Design Notes

- **Class imbalance:** handled via weighted binary cross-entropy
  (`pos_weight = num_negative / num_positive`) rather than naive resampling, to avoid
  distorting the sequence distribution the encoder was pretrained on.
- **Latency:** the encoder is intentionally kept small (2 layers) to meet sub-50ms
  inference targets on CPU; this is a deliberate accuracy/latency trade-off rather
  than an oversight.
- **Streaming design:** the Kafka producer/consumer separation simulates the
  decoupling between ad-event ingestion and scoring that exists in real ad-serving
  systems, so the project reflects deployment constraints rather than a purely
  offline experiment.

---

## Limitations & Future Work

- Click behavior is used as a **weak proxy label**, not a confirmed fraud label —
  results should be read as a proof of methodology, not a production fraud-detection
  benchmark.
- No multi-modal signals beyond device/site/app metadata (e.g. no mouse-movement or
  timing-jitter features, which real IVT systems often use).
- Next steps: incorporate a graph-based view of device/IP co-occurrence, and
  evaluate on a second public dataset (e.g. TalkingData AdTracking) to check that
  the pretraining gains generalize across data sources.

---

## References

## Author

**[Meenakshi Sundaram N]**