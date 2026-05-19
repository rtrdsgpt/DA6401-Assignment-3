# DA6401 - Assignment 3: Implementing a Transformer for Machine Translation

**Name:** Aritra Dasgupta

**Roll Number:** MA25M005

**Course:** DA6401 - Introduction to Deep Learning

---

## W&B Report

[View Full Experiment Report on Weights & Biases]([https://api.wandb.ai/links/ma25m005-iit-madras/neylhr17])

LINK : [https://api.wandb.ai/links/ma25m005-iit-madras/neylhr17)]

The report covers all five required experiments:
- Noam Scheduler vs Fixed Learning Rate
- Ablation on the Scaling Factor (1/sqrt(dk))
- Attention Rollout and Head Specialization
- Sinusoidal Positional Encoding vs Learned Embeddings
- Label Smoothing (eps=0.1 vs eps=0.0)

---

## Overview

This assignment implements the landmark architecture from the paper **"Attention Is All You Need"** (Vaswani et al., 2017) from scratch using PyTorch. The goal is to build a Neural Machine Translation system that translates text from German to English using the Multi30k dataset.

- **Base Paper:** [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- **Dataset:** [bentrevett/multi30k](https://huggingface.co/datasets/bentrevett/multi30k) — 29,000 training pairs, 1,014 validation pairs, 1,000 test pairs

---

## Project Structure

```
assignment3/
├── model.py           # Transformer architecture: attention, encoder, decoder, full model
├── lr_scheduler.py    # Noam learning rate scheduler
├── dataset.py         # Multi30k dataset loading, spacy tokenization, vocabulary
├── train.py           # Training loop, greedy decoding, BLEU evaluation, checkpointing
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

---

## Implementation Summary

### model.py

| Component | Description |
|---|---|
| `scaled_dot_product_attention` | Attention(Q,K,V) = softmax(QKᵀ / sqrt(dk)) · V with optional masking |
| `make_src_mask` | Padding mask for encoder, shape [batch, 1, 1, src_len] |
| `make_tgt_mask` | Combined padding + causal look-ahead mask for decoder, shape [batch, 1, tgt_len, tgt_len] |
| `MultiHeadAttention` | h parallel attention heads with W_Q, W_K, W_V, W_O projections; no use of nn.MultiheadAttention |
| `PositionalEncoding` | Sinusoidal PE registered as a buffer (non-trainable); PE[pos, 2i] = sin, PE[pos, 2i+1] = cos |
| `PositionwiseFeedForward` | Two-layer FFN with ReLU: FFN(x) = max(0, xW1+b1)W2+b2 |
| `EncoderLayer` | Self-attention + Add & Norm + FFN + Add & Norm (Post-LayerNorm) |
| `DecoderLayer` | Masked self-attention + cross-attention + FFN, all with Add & Norm |
| `Encoder` / `Decoder` | Stack of N identical layers with final LayerNorm |
| `Transformer` | Full model with src/tgt embeddings, positional encoding, encoder, decoder, linear projection |

**LayerNorm Choice:** Post-LayerNorm is used, matching the original paper. The residual is added first and then normalised. This choice is justified in the W&B report.

### lr_scheduler.py

Implements the Noam schedule exactly:

```
lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
```

- Linear warmup for `warmup_steps` steps
- Inverse square root decay afterwards
- Peak LR occurs near step `warmup_steps`

### dataset.py

- Loads Multi30k from HuggingFace via the `datasets` library
- Tokenises German (de_core_news_sm) and English (en_core_web_sm) using spacy
- Builds vocabularies with special tokens: `<unk>`, `<pad>`, `<sos>`, `<eos>`
- Returns padded torch tensors via a custom collate function

### train.py

| Function | Description |
|---|---|
| `LabelSmoothingLoss` | Smoothed cross-entropy with eps=0.1; pad positions receive 0 probability |
| `run_epoch` | One pass of training or evaluation with W&B logging |
| `greedy_decode` | Token-by-token autoregressive decoding; stops at `<eos>` or max_len |
| `evaluate_bleu` | Corpus-level BLEU score over the test set |
| `save_checkpoint` | Saves model, optimizer, scheduler state and model config |
| `load_checkpoint` | Restores all state from disk; returns saved epoch |
| `run_training_experiment` | Full experiment entry point with W&B init and all ablations |

---

## Hyperparameters

| Parameter | Value |
|---|---|
| d_model | 256 |
| N (layers) | 3 |
| num_heads | 8 |
| d_ff | 512 |
| dropout | 0.1 |
| warmup_steps | 4000 |
| label smoothing | 0.1 |
| optimizer | Adam (beta1=0.9, beta2=0.98, eps=1e-9) |
| batch_size | 128 |
| epochs | 20 |

A smaller model (d_model=256, N=3) is used to fit within Kaggle T4 GPU memory and training time constraints while still achieving reasonable BLEU scores on Multi30k.

---

## How to Run

### 1. Install Dependencies

```bash
pip install torch numpy matplotlib scikit-learn wandb datasets tqdm evaluate
pip install spacy
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

### 2. Run Training

```bash
python train.py
```

This will:
- Load and preprocess the Multi30k dataset
- Initialise W&B logging
- Train the Transformer
- Save the best checkpoint to `best_checkpoint.pt`
- Evaluate BLEU on the test set and log to W&B


---

## Experiments

### 2.1 Noam Scheduler vs Fixed LR

The Noam scheduler prevents early divergence by warming up the learning rate gradually. Without warmup, the large random initial weights in the attention projections cause extremely large gradient updates in the first few steps, destabilising training. The W&B report overlays training loss and validation accuracy for both conditions.

### 2.2 Scaling Factor Ablation

Removing 1/sqrt(dk) allows dot products to grow large (O(dk)) for large dk, pushing softmax into saturation regions with near-zero gradients. The report logs gradient norms of Q and K weights during the first 1000 steps for both variants.

### 2.3 Attention Rollout and Head Specialization

Attention weights from the last encoder layer are visualised as per-head heatmaps. Specific heads show distinct behaviours: some attend locally (next token), others capture long-range syntactic dependencies. Head redundancy analysis is included in the report.

### 2.4 Sinusoidal PE vs Learned Embeddings

Sinusoidal encoding (using `register_buffer`) is compared against `nn.Embedding`-based learned positional embeddings. Validation BLEU is tracked for both. The report discusses how sinusoidal encodings allow extrapolation to unseen lengths via their deterministic frequency structure, unlike learned embeddings which have no inductive bias beyond the training length.

### 2.5 Label Smoothing

Training with eps=0.1 vs eps=0.0 (hard cross-entropy). Prediction confidence (softmax probability of the correct token) is logged to W&B. Label smoothing acts as a regulariser by preventing the model from assigning all probability mass to the argmax token, which reduces overconfidence and improves generalisation even at the cost of slightly higher training perplexity.


