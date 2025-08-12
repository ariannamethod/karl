"""Model and training helpers for the GENESIS orchestrator."""

import argparse
import logging
import math
import shutil
import subprocess  # noqa: F401
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

try:  # torch is optional for tests
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
except Exception:  # pragma: no cover - fallback when torch missing
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

from .orchestrator import model_hyperparams

# ----------------------------------------------------------------------------
# GPT model definition (adapted from model.py)
if torch is not None:

    class LayerNorm(nn.Module):
        """LayerNorm with optional bias."""

        def __init__(self, ndim, bias):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(ndim))
            self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

        def forward(self, input):
            return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

    class CausalSelfAttention(nn.Module):
        def __init__(self, config):
            super().__init__()
            assert config.n_embd % config.n_head == 0
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.attn_dropout = nn.Dropout(config.dropout)
            self.resid_dropout = nn.Dropout(config.dropout)
            self.n_head = config.n_head
            self.n_embd = config.n_embd
            self.dropout = config.dropout
            self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
            if not self.flash:
                self.register_buffer(
                    "bias",
                    torch.tril(torch.ones(config.block_size, config.block_size))
                    .view(1, 1, config.block_size, config.block_size),
                )

        def forward(self, x):
            B, T, C = x.size()
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            if self.flash:
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
                )
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = self.resid_dropout(self.c_proj(y))
            return y

    class MLP(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.gelu = nn.GELU()
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x):
            x = self.c_fc(x)
            x = self.gelu(x)
            x = self.c_proj(x)
            x = self.dropout(x)
            return x

    class Block(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
            self.attn = CausalSelfAttention(config)
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
            self.mlp = MLP(config)

        def forward(self, x):
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x

    @dataclass
    class GPTConfig:
        block_size: int = 1024
        vocab_size: int = 50304
        n_layer: int = 12
        n_head: int = 12
        n_embd: int = 768
        dropout: float = 0.0
        bias: bool = True

    class GPT(nn.Module):
        def __init__(self, config):
            super().__init__()
            assert config.vocab_size is not None
            assert config.block_size is not None
            self.config = config
            self.transformer = nn.ModuleDict(
                dict(
                    wte=nn.Embedding(config.vocab_size, config.n_embd),
                    wpe=nn.Embedding(config.block_size, config.n_embd),
                    drop=nn.Dropout(config.dropout),
                    h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                    ln_f=LayerNorm(config.n_embd, bias=config.bias),
                )
            )
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.transformer.wte.weight = self.lm_head.weight
            self.apply(self._init_weights)
            for pn, p in self.named_parameters():
                if pn.endswith("c_proj.weight"):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        def get_num_params(self, non_embedding=True):
            n_params = sum(p.numel() for p in self.parameters())
            if non_embedding:
                n_params -= self.transformer.wpe.weight.numel()
            return n_params

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(self, idx, targets=None):
            device = idx.device
            b, t = idx.size()
            assert t <= self.config.block_size
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            tok_emb = self.transformer.wte(idx)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
            for block in self.transformer.h:
                x = block(x)
            x = self.transformer.ln_f(x)
            if targets is not None:
                logits = self.lm_head(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            else:
                logits = self.lm_head(x[:, [-1], :])
                loss = None
            return logits, loss
else:  # torch not available; provide placeholders

    @dataclass
    class GPTConfig:
        block_size: int = 0
        vocab_size: int = 0
        n_layer: int = 0
        n_head: int = 0
        n_embd: int = 0
        dropout: float = 0.0
        bias: bool = True

    class GPT:  # pragma: no cover - minimal placeholder
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required to use GPT model")


# ----------------------------------------------------------------------------
# Dataset preparation and training wrappers

def prepare_char_dataset(text: str, dest: Path) -> None:
    import pickle
    import numpy as np

    chars = sorted(set(text))
    if not text or not chars:
        raise ValueError("text must be non-empty and contain at least one unique character")
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    n = len(text)
    train_data = text[: int(n * 0.9)]
    val_data = text[int(n * 0.9):]
    train_ids = np.array([stoi[c] for c in train_data], dtype=np.uint16)
    val_ids = np.array([stoi[c] for c in val_data], dtype=np.uint16)
    dest.mkdir(parents=True, exist_ok=True)
    train_ids.tofile(dest / "train.bin")
    val_ids.tofile(dest / "val.bin")
    meta = {"vocab_size": len(chars), "itos": itos, "stoi": stoi}
    with open(dest / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)


def train_loop(args: argparse.Namespace) -> None:
    """Simplified training loop writing a dummy checkpoint."""
    logging.info("Starting training with args: %s", args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # create a dummy checkpoint so orchestration code can proceed
    (out_dir / "ckpt.pt").write_text("dummy")


def train_model(dataset_dir: Path, out_dir: Path) -> None:
    """Run the lightweight training script."""
    try:
        device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    hyperparams = dict(model_hyperparams)
    if device == "cpu":
        hyperparams.update({
            "block_size": 32,
            "batch_size": 4,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 32,
        })
    args = argparse.Namespace(
        dataset=str(dataset_dir),
        device=device,
        compile=False,
        eval_iters=1,
        log_interval=1,
        out_dir=str(out_dir),
        **hyperparams,
    )
    train_loop(args)
    ckpt = out_dir / "ckpt.pt"
    if ckpt.exists():
        shutil.copy(ckpt, out_dir / "model.pth")


def main() -> None:  # pragma: no cover - only used when run as a script
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset", required=False)
    parser.add_argument("--device", required=False)
    args, _ = parser.parse_known_args()
    train_loop(args)


if __name__ == "__main__":  # pragma: no cover
    main()
