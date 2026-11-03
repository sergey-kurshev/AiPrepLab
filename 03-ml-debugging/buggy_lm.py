"""A small transformer language model with EIGHT planted bugs.

It runs. It trains. The loss even goes down. Every bug in here is drawn from the
table in README.md, and every one of them produces valid tensors of plausible
magnitude — which is why reading the file is the wrong way to find them.

Your job: run the diagnostics in `exercise.ipynb`, let them tell you where to
look, and fix the bugs in THIS file. Re-run the checks after each single fix.

Do not change the architecture, the hyperparameters, or the task. Every bug is a
one- or two-line correction.
"""

from __future__ import annotations

import math
import sys
import pathlib

_here = pathlib.Path(__file__).resolve().parent
for _c in (_here, *_here.parents):
    if (_c / "tools").is_dir():
        sys.path.insert(0, str(_c / "tools"))
        break

import torch
import torch.nn as nn
import torch.nn.functional as F
from checks import check, report, close

VOCAB = 16
SEQ = 8          # the copy task repeats a prefix of this length


# ------------------------------------------------------------------ data
def make_data(n_seq: int, gen: torch.Generator) -> torch.Tensor:
    """Sequences [x_1..x_n, x_1..x_n]: the second half is predictable, the first is not."""
    half = torch.randint(1, VOCAB, (n_seq, SEQ), generator=gen)
    return torch.cat([half, half], dim=1)


def get_batch(data: torch.Tensor, batch: int, gen: torch.Generator):
    """Return (inputs, targets) for next-token prediction."""
    idx = torch.randint(0, data.shape[0], (batch,), generator=gen)
    seq = data[idx]
    x = seq[:, :-1]
    y = seq[:, :-1]
    return x, y


# ------------------------------------------------------------------ model
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.H, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.register_buffer("allowed", torch.ones(max_len, max_len, dtype=torch.bool).tril())
        self.last_weights: torch.Tensor | None = None

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, 3, self.H, T, self.dh)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]              # (B, H, T, dh)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        p = scores.softmax(dim=-2)
        p = p * self.allowed[:T, :T]
        self.last_weights = p.detach()
        out = self.drop(p) @ v
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, max_len, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, vocab=VOCAB, d_model=64, n_heads=4, n_layers=2, max_len=2 * SEQ, dropout=0.1):
        super().__init__()
        self.wte = nn.Embedding(vocab, d_model)
        self.wpe = nn.Embedding(max_len, d_model)
        self.blocks = [Block(d_model, n_heads, max_len, dropout) for _ in range(n_layers)]
        self.lnf = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        # GPT-2's initialisation: N(0, 0.02). PyTorch's Linear default is U(±1/sqrt(fan_in)),
        # which here gives logits spread widely enough that the initial loss overshoots
        # ln(V) by ~0.15 — small, but it is exactly the rung-2 signal, so do it properly.
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.lnf(x))
        if targets is None:
            return logits, None
        probs = F.softmax(logits, dim=-1)
        loss = F.cross_entropy(probs.reshape(-1, probs.shape[-1]), targets.reshape(-1))
        return logits, loss


# ------------------------------------------------------------------ training
def train(model, data, steps=600, batch=64, lr=3e-3, gen=None, log_every=0):
    gen = gen or torch.Generator().manual_seed(0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    loss = torch.tensor(float("nan"))
    for step in range(steps):
        x, y = get_batch(data, batch, gen)
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        if log_every and step % log_every == 0:
            print(f"  step {step:>4}  loss {loss.item():.4f}")
    return float(loss.item())


def evaluate(model, data, batch=256, gen=None):
    gen = gen or torch.Generator().manual_seed(123)
    x, y = get_batch(data, batch, gen)
    logits, loss = model(x, y)
    pred = logits.argmax(-1)
    # Score only the second half, where the answer is determined by the prefix.
    acc = float((pred[:, SEQ - 1:] == y[:, SEQ - 1:]).float().mean())
    return float(loss), acc


# ------------------------------------------------------------------ diagnostics
def missing_gradients(model, data, gen=None):
    """Rung 3: parameters with no gradient, or a gradient of exactly zero."""
    gen = gen or torch.Generator().manual_seed(0)
    model.zero_grad(set_to_none=True)
    x, y = get_batch(data, 8, gen)
    _, loss = model(x, y)
    loss.backward()
    bad = []
    for name, p in model.named_parameters():
        if p.grad is None:
            bad.append((name, "None"))
        elif float(p.grad.norm()) == 0.0:
            bad.append((name, "zero"))
    return bad


def can_overfit_one_batch(model_fn, data, steps=400, lr=3e-3):
    """Rung 1: the single most informative test in machine learning."""
    gen = torch.Generator().manual_seed(0)
    model = model_fn()
    for m in model.modules():                    # regularisation off
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    x, y = get_batch(data, 8, gen)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss = torch.tensor(float("nan"))
    for _ in range(steps):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return float(loss.detach())


def attention_weights_sum_to_one(model, data, gen=None) -> float:
    gen = gen or torch.Generator().manual_seed(0)
    x, _ = get_batch(data, 4, gen)
    model(x)
    worst = 0.0
    for blk in model.blocks:
        w = blk.attn.last_weights
        worst = max(worst, float((w.sum(-1) - 1).abs().max()))
    return worst


def causality_violation(model, data, gen=None) -> float:
    """Perturb the last input token; nothing before it may change."""
    gen = gen or torch.Generator().manual_seed(0)
    model.eval()
    with torch.no_grad():
        x, _ = get_batch(data, 4, gen)
        a, _ = model(x)
        x2 = x.clone()
        x2[:, -1] = (x2[:, -1] + 1) % VOCAB
        b, _ = model(x2)
    model.train()
    return float((a[:, :-1] - b[:, :-1]).abs().max())


