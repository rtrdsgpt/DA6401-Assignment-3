"""
model.py — Transformer Architecture
DA6401 Assignment 3: "Attention Is All You Need"
Reference: https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘

"""

import math
import copy
import os
import sys
import subprocess
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL CONSTANTS — imported by train.py and dataset.py
# ══════════════════════════════════════════════════════════════════════
#: Special tokens are placed at fixed indices so every module agrees.
SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]
UNK_IDX = 0   # index for out-of-vocabulary tokens
PAD_IDX = 1   # index used for padding sequences to equal length
SOS_IDX = 2   # start-of-sequence marker fed to the decoder
EOS_IDX = 3   # end-of-sequence marker; generation halts here


# ══════════════════════════════════════════════════════════════════════
#  STANDALONE ATTENTION FUNCTION
#   Exposed at module level so the autograder can import and test it
#   independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention (Section 3.2.1 of the paper).

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, broadcastable to (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT.

    Returns:
        attended_output  : shape (..., seq_q, d_v)
        attention_weights: shape (..., seq_q, seq_k)
    """
    head_dim = Q.size(-1)
    raw_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)

    if mask is not None:
        raw_scores = raw_scores.masked_fill(mask, float("-inf"))

    attention_weights = F.softmax(raw_scores, dim=-1)
    attended_output = torch.matmul(attention_weights, V)

    return attended_output, attention_weights


# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = PAD_IDX,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → PAD token (masked out in attention)
        False → real token
    """
    padding_indicator = (src == pad_idx)                     # [batch, src_len]
    return padding_indicator.unsqueeze(1).unsqueeze(2)       # [batch, 1, 1, src_len]


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = PAD_IDX,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → masked out (PAD or future token)
    """
    num_samples, seq_length = tgt.shape

    padding_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)   # [batch, 1, 1, tgt_len]

    lookahead_mask = torch.triu(
        torch.ones(
            (seq_length, seq_length),
            device=tgt.device,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    return padding_mask | lookahead_mask  # [batch, 1, tgt_len, tgt_len]


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as described in Section 3.2.2 of the paper.

    Internal attribute names (W_q, W_k, W_v, W_o) 

    Args:
        d_model   (int)  : Total model dimensionality.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention output.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        # naming: W_q / W_k / W_v / W_o
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Multi-Head Attention for a batch of sequences.

        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]

        Returns:
            output : shape [batch, seq_q, d_model]
        """
        q_heads = self._split_heads(self.W_q(query))  # [B, h, seq_q, d_k]
        k_heads = self._split_heads(self.W_k(key))    # [B, h, seq_k, d_k]
        v_heads = self._split_heads(self.W_v(value))  # [B, h, seq_k, d_k]

        head_output, _attn_weights = scaled_dot_product_attention(
            q_heads, k_heads, v_heads, mask
        )

        head_output = self.dropout(head_output)
        concat_heads = self._combine_heads(head_output)  # [B, seq_q, d_model]

        return self.W_o(concat_heads)


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as described in Section 3.5 of the paper.

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        sinusoidal_table = torch.zeros(max_len, d_model)
        pos_indices = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        freq_scaling = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        sinusoidal_table[:, 0::2] = torch.sin(pos_indices * freq_scaling)
        sinusoidal_table[:, 1::2] = torch.cos(pos_indices * freq_scaling)

        self.register_buffer("pe", sinusoidal_table.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to token embeddings.

        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]
        Returns:
            Tensor of same shape, = x + PE[:, :seq_len, :]
        """
        encoded_input = x + self.pe[:, : x.size(1), :].to(dtype=x.dtype, device=x.device)
        return self.dropout(encoded_input)


# ══════════════════════════════════════════════════════════════════════
#  POSITION-WISE FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, Section 3.3 of the paper.

    Internal attribute names (linear1, linear2) 
    
    Args:
        d_model (int)  : Input / output dimensionality.
        d_ff    (int)  : Inner-layer dimensionality.
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # linear1 / linear2
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer.

    Uses Pre-LayerNorm (norm applied before each sublayer), matching
    Internal names (self_attn, norm1, norm2)
    

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(
        self,
        d_model:   int,
        num_heads: int,
        d_ff:      int,
        dropout:   float = 0.1,
    ) -> None:
        super().__init__()
        
        self.self_attn   = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1       = nn.LayerNorm(d_model)
        self.norm2       = nn.LayerNorm(d_model)
        self.dropout     = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Pass a batch through one encoder layer (Pre-LN).

        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        # Pre-LN self-attention sub-layer
        x_norm = self.norm1(x)
        x = x + self.dropout(self.self_attn(x_norm, x_norm, x_norm, src_mask))

        # Pre-LN feed-forward sub-layer
        x = x + self.dropout(self.feed_forward(self.norm2(x)))

        return x


# ══════════════════════════════════════════════════════════════════════
#  DECODER LAYER
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer (Pre-LN).

    Internal names (self_attn, cross_attn, norm1, norm2, norm3,
    feed_forward) .

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(
        self,
        d_model:   int,
        num_heads: int,
        d_ff:      int,
        dropout:   float = 0.1,
    ) -> None:
        super().__init__()
        
        self.self_attn    = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn   = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1        = nn.LayerNorm(d_model)
        self.norm2        = nn.LayerNorm(d_model)
        self.norm3        = nn.LayerNorm(d_model)
        self.dropout      = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pass a batch through one decoder layer (Pre-LN).

        Args:
            x        : Decoder input, shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        # Sub-layer 1: masked self-attention (Pre-LN)
        x_norm = self.norm1(x)
        x = x + self.dropout(self.self_attn(x_norm, x_norm, x_norm, tgt_mask))

        # Sub-layer 2: cross-attention (Pre-LN)
        x = x + self.dropout(self.cross_attn(self.norm2(x), memory, memory, src_mask))

        # Sub-layer 3: feed-forward (Pre-LN)
        x = x + self.dropout(self.feed_forward(self.norm3(x)))

        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """
    Stack of N identical EncoderLayer modules with a final LayerNorm.

    Attribute name 'layers' 
    
    Args:
        layer (EncoderLayer): Template layer.
        N     (int)         : Number of layers.
    """

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for enc_layer in self.layers:
            x = enc_layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """
    Stack of N identical DecoderLayer modules with a final LayerNorm.

    Attribute name 'layers' 
    
    Args:
        layer (DecoderLayer): Template layer.
        N     (int)         : Number of layers.
    """

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for dec_layer in self.layers:
            x = dec_layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#  FULL TRANSFORMER
#  NOTE: src_embed / tgt_embed / positional_encoding / generator
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for Neural Machine Translation.

    I

    Also handles checkpoint loading from Google Drive via gdown when
    a checkpoint_path is provided.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 256).
        N              (int)  : Number of encoder/decoder layers (default 3).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 512).
        dropout        (float): Dropout probability (default 0.1).
        checkpoint_path(str)  : If provided, load weights from this path.
    """

    # Google Drive file ID for the pre-trained best checkpoint.
    _GDRIVE_CHECKPOINT_ID = "1FmnxUcTW6PrlsHlCWEBvOb9EY8NmDigu"

    def __init__(
        self,
        src_vocab_size: int   = 10_000,
        tgt_vocab_size: int   = 10_000,
        d_model:        int   = 256,
        N:              int   = 3,
        num_heads:      int   = 8,
        d_ff:           int   = 512,
        dropout:        float = 0.1,
        checkpoint_path: str  = None,
    ) -> None:
        super().__init__()

        # ── Step 1: Build vocabulary from the Multi30k training set ────
        self._build_vocabularies(src_vocab_size, tgt_vocab_size)

        actual_src_size = len(self.src_token_to_idx)
        actual_tgt_size = len(self.tgt_token_to_idx)

        # ── Step 2: Construct the model architecture ───────────────────
        self.d_model = d_model

        # src_embed / tgt_embed
        self.src_embed = nn.Embedding(actual_src_size, d_model)
        self.tgt_embed = nn.Embedding(actual_tgt_size, d_model)

        # single shared positional_encoding for both src & tgt
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        # Encoder and decoder stacks
        encoder_prototype = EncoderLayer(d_model, num_heads, d_ff, dropout)
        decoder_prototype = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder      = Encoder(encoder_prototype, N)
        self.decoder      = Decoder(decoder_prototype, N)

        # naming: generator (instead of vocab_projection)
        self.generator = nn.Linear(d_model, actual_tgt_size)

        # Xavier uniform initialisation
        for trainable_param in self.parameters():
            if trainable_param.dim() > 1:
                nn.init.xavier_uniform_(trainable_param)

        # ── Step 3: Load the pre-trained checkpoint ──
        _ckpt_local = checkpoint_path if checkpoint_path is not None else "classifier.pt"
        self._load_pretrained_checkpoint(_ckpt_local)

    # ── VOCABULARY CONSTRUCTION (private) ──────────────────────────────

    def _build_vocabularies(
        self,
        default_src_size: int,
        default_tgt_size: int,
    ) -> None:
        """
        Load Multi30k training data, tokenise with spaCy, and build
        word-to-index mappings for both German (src) and English (tgt).

        First tries to load vocab from an existing checkpoint (much faster).
        Falls back to building from the dataset if no checkpoint exists.
        """
        # -- Initialise with the four mandatory special tokens --
        self.src_token_to_idx = {
            "<unk>": UNK_IDX,
            "<pad>": PAD_IDX,
            "<sos>": SOS_IDX,
            "<eos>": EOS_IDX,
        }
        self.tgt_token_to_idx = {
            "<unk>": UNK_IDX,
            "<pad>": PAD_IDX,
            "<sos>": SOS_IDX,
            "<eos>": EOS_IDX,
        }
        self.tgt_idx_to_token = {
            UNK_IDX: "<unk>",
            PAD_IDX: "<pad>",
            SOS_IDX: "<sos>",
            EOS_IDX: "<eos>",
        }

        # -- Try to load vocab from checkpoint first (fast path) --
        ckpt_path = "classifier.pt"
        if os.path.exists(ckpt_path):
            try:
                state = torch.load(ckpt_path, map_location="cpu")
                if isinstance(state, dict) and "src_vocab" in state and "tgt_vocab" in state:
                    sv = state["src_vocab"]
                    tv = state["tgt_vocab"]
                    
                    self.src_token_to_idx = sv["stoi"]
                    self.tgt_token_to_idx = tv["stoi"]
                    self.tgt_idx_to_token = {i: t for t, i in tv["stoi"].items()}
                    print(f"[Transformer] Vocab loaded from checkpoint: "
                          f"src={len(self.src_token_to_idx)} tgt={len(self.tgt_token_to_idx)}")
                    # Load spaCy for infer()
                    self.german_tokenizer = self._load_spacy_model("de_core_news_sm", "de")
                    return
            except Exception as e:
                print(f"[Transformer] Could not load vocab from checkpoint: {e}")

        # -- Fall back: build from dataset --
        german_nlp  = self._load_spacy_model("de_core_news_sm", "de")
        english_nlp = self._load_spacy_model("en_core_web_sm",  "en")
        self.german_tokenizer = german_nlp

        try:
            from datasets import load_dataset
            from collections import Counter

            raw_training_data = load_dataset("bentrevett/multi30k")["train"]

            german_freqs  = Counter()
            english_freqs = Counter()

            for sentence_pair in raw_training_data:
                german_tokens  = [tok.text.lower()
                                  for tok in german_nlp.tokenizer(sentence_pair["de"])]
                english_tokens = [tok.text.lower()
                                  for tok in english_nlp.tokenizer(sentence_pair["en"])]
                german_freqs.update(german_tokens)
                english_freqs.update(english_tokens)

            for sp in SPECIAL_TOKENS:
                german_freqs.pop(sp, None)
                english_freqs.pop(sp, None)

            for word, freq in sorted(german_freqs.items(), key=lambda x: (-x[1], x[0])):
                if freq >= 2 and word not in self.src_token_to_idx:
                    self.src_token_to_idx[word] = len(self.src_token_to_idx)

            for word, freq in sorted(english_freqs.items(), key=lambda x: (-x[1], x[0])):
                if freq >= 2 and word not in self.tgt_token_to_idx:
                    new_idx = len(self.tgt_token_to_idx)
                    self.tgt_token_to_idx[word]  = new_idx
                    self.tgt_idx_to_token[new_idx] = word

            print(f"[Transformer] Vocab built from dataset: src={len(self.src_token_to_idx)} "
                  f"tgt={len(self.tgt_token_to_idx)} tokens.")

        except Exception as vocab_error:
            print(f"[Transformer] Vocabulary build warning: {vocab_error}")

    @staticmethod
    def _load_spacy_model(model_name: str, lang_code: str):
        import spacy
        try:
            return spacy.load(model_name)
        except OSError:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "spacy", "download", model_name]
                )
                return spacy.load(model_name)
            except Exception:
                return spacy.blank(lang_code)

    # ── CHECKPOINT LOADING (private) ───────────────────────────────────

    def _load_pretrained_checkpoint(self, output_path: str) -> None:
        """
        Load pre-trained checkpoint weights into this model.

        saved with save_checkpoint() which stores:
          {
            "epoch": int,
            "model_state_dict": { ... },   ← keys match this class's attributes
            "optimizer_state_dict": { ... },
            "scheduler_state_dict": { ... },
            "model_config": { ... },
            "src_vocab": {"stoi": ..., "itos": ...},
            "tgt_vocab": {"stoi": ..., "itos": ...},
          }
        

        Falls back to GDrive download if the file doesn't exist locally.
        """
        # ── Download from GDrive if not on disk ─────────────────────────
        if not os.path.exists(output_path):
            try:
                import gdown
                print(f"[Transformer] Downloading checkpoint ({self._GDRIVE_CHECKPOINT_ID}) ...")
                gdown.download(
                    id=self._GDRIVE_CHECKPOINT_ID,
                    output=output_path,
                    quiet=False,
                )
            except Exception as dl_err:
                print(f"[Transformer] Checkpoint download failed: {dl_err}")
                return

        if not os.path.exists(output_path):
            print("[Transformer] Checkpoint file not found after download attempt.")
            return

        # ── Load & unwrap ────────────────────────────────────────────────
        try:
            saved_data = torch.load(output_path, map_location="cpu")
        except Exception as e:
            print(f"[Transformer] Could not open checkpoint: {e}")
            return

        if isinstance(saved_data, dict) and "model_state_dict" in saved_data:
            raw_state_dict = saved_data["model_state_dict"]
        elif isinstance(saved_data, dict):
            raw_state_dict = saved_data
        else:
            try:
                raw_state_dict = saved_data.state_dict()
            except Exception:
                print("[Transformer] Unrecognised checkpoint format.")
                return

        # ── Direct load  ──
        current_sd = self.state_dict()
        missing_in_ckpt  = [k for k in current_sd if k not in raw_state_dict]
        extra_in_ckpt    = [k for k in raw_state_dict if k not in current_sd]
        shape_mismatches = [
            k for k in current_sd
            if k in raw_state_dict and raw_state_dict[k].shape != current_sd[k].shape
        ]

        if not missing_in_ckpt and not shape_mismatches:
            # Perfect match — load directly
            self.load_state_dict(raw_state_dict, strict=False)
            print(f"[Transformer] Checkpoint loaded successfully "
                  f"({len(raw_state_dict)} tensors).")
        else:
            # Some keys need patching (e.g. vocab size changed)
            patched_sd = {}
            loaded, missing = [], []
            for model_key, current_val in current_sd.items():
                if model_key in raw_state_dict:
                    ckpt_val = raw_state_dict[model_key]
                    if ckpt_val.shape == current_val.shape:
                        patched_sd[model_key] = ckpt_val
                        loaded.append(model_key)
                    else:
                        buf = current_val.clone()
                        if ckpt_val.dim() == 2 and current_val.dim() == 2:
                            r = min(ckpt_val.size(0), current_val.size(0))
                            c = min(ckpt_val.size(1), current_val.size(1))
                            buf[:r, :c] = ckpt_val[:r, :c]
                        elif ckpt_val.dim() == 1 and current_val.dim() == 1:
                            n = min(ckpt_val.size(0), current_val.size(0))
                            buf[:n] = ckpt_val[:n]
                        patched_sd[model_key] = buf
                        loaded.append(model_key + " [shape-patched]")
                else:
                    patched_sd[model_key] = current_val
                    missing.append(model_key)

            self.load_state_dict(patched_sd, strict=False)
            print(f"[Transformer] Checkpoint loaded: {len(loaded)} tensors restored, "
                  f"{len(missing)} kept at random init.")
            if missing:
                print(f"[Transformer] Keys not in checkpoint: {missing[:8]}"
                      + (" ..." if len(missing) > 8 else ""))
            if extra_in_ckpt:
                print(f"[Transformer] Extra checkpoint keys (ignored): {extra_in_ckpt[:8]}"
                      + (" ..." if len(extra_in_ckpt) > 8 else ""))

    # ── AUTOGRADER HOOKS — keep these signatures exactly ───────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack on the source sequence.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]
        Returns:
            encoder_memory : shape [batch, src_len, d_model]
        """
        # Scale embeddings by √d_model, add positional encoding
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.generator(x)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass for training (teacher forcing).

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        enc_memory = self.encode(src, src_mask)
        return self.decode(enc_memory, src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str) -> str:
        """
        Translate a raw German sentence to English using greedy decoding.

        Args:
            src_sentence : Raw German text to translate.
        Returns:
            Translated English string.
        """
        self.eval()
        inference_device = next(self.parameters()).device

        raw_tokens = [
            tok.text.lower()
            for tok in self.german_tokenizer.tokenizer(src_sentence)
        ]

        unk_token_idx = self.src_token_to_idx.get("<unk>", UNK_IDX)
        src_indices   = (
            [SOS_IDX]
            + [self.src_token_to_idx.get(tok, unk_token_idx) for tok in raw_tokens]
            + [EOS_IDX]
        )
        src_tensor  = torch.tensor(
            src_indices, dtype=torch.long, device=inference_device
        ).unsqueeze(0)
        source_mask = make_src_mask(src_tensor, PAD_IDX).to(inference_device)

        max_decode_steps = int(1.5 * len(src_indices)) + 5

        with torch.no_grad():
            encoder_memory   = self.encode(src_tensor, source_mask)
            generated_tokens = torch.tensor(
                [[SOS_IDX]], dtype=torch.long, device=inference_device
            )

            for _ in range(max_decode_steps):
                step_tgt_mask = make_tgt_mask(
                    generated_tokens, PAD_IDX
                ).to(inference_device)
                step_logits   = self.decode(
                    encoder_memory, source_mask, generated_tokens, step_tgt_mask
                )
                next_token_id = step_logits[:, -1, :].argmax(dim=-1).item()
                next_token    = torch.tensor(
                    [[next_token_id]], dtype=torch.long, device=inference_device
                )
                generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

                if next_token_id == EOS_IDX:
                    break

        all_generated_ids = generated_tokens.squeeze(0).tolist()
        skip_ids          = {SOS_IDX, EOS_IDX, PAD_IDX}
        output_words      = [
            self.tgt_idx_to_token.get(token_id, str(token_id))
            for token_id in all_generated_ids
            if token_id not in skip_ids
        ]

        return " ".join(output_words)
