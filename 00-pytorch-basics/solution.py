"""Reference solutions for exercise 00.

Run directly to verify everything: `python solution.py`.

These are *a* correct answer, not *the* correct answer. Where a choice was
interesting, the comment says what the alternative was and why this one won.
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


# ---------------------------------------------------------------- step 1
def shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    # `data_ptr()` is the address of element 0, which differs for `x` and `x[1:]`
    # even though they alias. The storage pointer is the question actually being asked.
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


# ---------------------------------------------------------------- step 2
def add_key_bias(scores: torch.Tensor, key_bias: torch.Tensor) -> torch.Tensor:
    # A (n_keys,) tensor already aligns with the last axis under the right-alignment
    # rule, so plain addition is correct. Writing it as an explicit `(1, n_keys)`
    # documents the intent and would fail loudly if `key_bias` were 2-D by mistake.
    assert key_bias.dim() == 1 and key_bias.shape[0] == scores.shape[-1]
    return scores + key_bias  # equivalently: scores + key_bias.reshape(1, -1)


# The bug: `key_bias.reshape(4, 1)` aligns the bias with the *query* axis, adding a
# constant to each row instead of each column — a per-query bias. It is a legal
# broadcast, so nothing raises; only the loss curve knows.


def causal_mask(T: int, device=None, dtype=torch.float32) -> torch.Tensor:
    # Additive because it composes with the logits *before* the softmax: masking after
    # the softmax would require renormalising, and would leak probability mass in the
    # meantime. -inf rather than -1e9 because exp(-inf) == 0 exactly, giving a hard zero
    # regardless of dtype; -1e9 underflows to zero in fp32 but is a finite -65504-clipped
    # value in fp16, where it can still contribute. The cost of -inf is the all-masked-row
    # NaN, handled in step 5.
    allowed = torch.ones(T, T, device=device, dtype=torch.bool).tril()
    return torch.zeros(T, T, device=device, dtype=dtype).masked_fill(~allowed, float("-inf"))


# ---------------------------------------------------------------- step 3
def scores_einsum(q, k):
    return torch.einsum("bhqd,bhkd->bhqk", q, k)


def weighted_values_einsum(w, v):
    return torch.einsum("bhqk,bhkd->bhqd", w, v)


def head_norms_einsum(q):
    # Repeating `d` on the left contracts it (sum of squares); repeating `t` and not
    # naming it on the right sums over positions, so divide by T to get the mean.
    return torch.einsum("bhtd,bhtd->bh", q, q) / q.shape[-2]


# ---------------------------------------------------------------- step 4
def softmax_jacobian(z: torch.Tensor) -> torch.Tensor:
    s = torch.softmax(z, dim=-1)
    return torch.diag(s) - torch.outer(s, s)


def softmax_vjp(z: torch.Tensor, y_bar: torch.Tensor) -> torch.Tensor:
    # ȳᵀ(diag(σ) - σσᵀ) = σ ⊙ ȳ - σ (σ·ȳ) = σ ⊙ (ȳ - ⟨ȳ, σ⟩).
    # O(n) and O(1) extra memory; the explicit Jacobian is O(n²) in both.
    s = torch.softmax(z, dim=-1)
    return s * (y_bar - (y_bar * s).sum(-1, keepdim=True))


# ---------------------------------------------------------------- step 5
def stable_softmax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    m = z.amax(dim=dim, keepdim=True)
    # A fully-masked row has m = -inf, and (-inf) - (-inf) = nan. Clamping the max to be
    # finite turns that row into exp(-inf - 0) = 0 everywhere, so the row sums to 0 and
    # the division yields nan — so we also guard the denominator. torch's own softmax
    # returns a uniform distribution for such a row; we match it, which keeps gradients
    # finite. The alternative (propagate nan) is defensible as a loud failure, but in a
    # training run it destroys the whole batch, which is a worse failure mode.
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    e = (z - m).exp()
    denom = e.sum(dim=dim, keepdim=True)
    uniform = torch.full_like(z, 1.0 / z.shape[dim])
    return torch.where(denom > 0, e / denom.clamp_min(torch.finfo(z.dtype).tiny), uniform)


# ---------------------------------------------------------------- step 6
def attention(q, k, v, causal: bool = False):
    d_k = q.shape[-1]
    # Scale the scores rather than Q: mathematically identical (the scale is linear and
    # commutes out of the dot product), but scaling Q is one 1/sqrt(d) multiply over
    # B*H*Tq*D elements instead of B*H*Tq*Tk — cheaper whenever Tk > D, which is usual.
    # Fused kernels do exactly that. We keep it on the scores because it reads closer to
    # the paper, and because folding it into Q changes what `q.grad` means.
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    if causal:
        T_q, T_k = scores.shape[-2], scores.shape[-1]
        scores = scores + causal_mask(max(T_q, T_k), scores.device, scores.dtype)[-T_q:, :T_k]
    w = stable_softmax(scores, dim=-1)
    return w @ v, w


def attention_entropy_and_grad(D: int, scale: bool, T: int = 64) -> tuple[float, float]:
    g = torch.Generator().manual_seed(D)
    q = torch.randn(1, 1, T, D, generator=g, requires_grad=True)
    k = torch.randn(1, 1, T, D, generator=g)
    scores = q @ k.transpose(-2, -1)
    if scale:
        scores = scores / math.sqrt(D)
    p = torch.softmax(scores, dim=-1)
    entropy = -(p * (p + 1e-30).log()).sum(-1).mean()
    p.sum().backward()
    return float(entropy), float(q.grad.abs().max())


# ---------------------------------------------------------------- step 7
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int = 512, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        # One fused projection: a single GEMM of 3x the width beats three of width 1x,
        # because the cost here is memory traffic on `x`, which is read once instead of
        # three times. Every production implementation does this.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        # register_buffer: moves with .to(device), saved in state_dict, not optimised.
        self.register_buffer("mask", causal_mask(max_len), persistent=False)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)                                  # (B, T, 3C)
        # Split heads: view to (B, T, 3, H, Dh) then permute. Doing `view(B, H, T, Dh)`
        # directly would be silently wrong — it would interleave positions and heads,
        # because the head axis is minor to the position axis in memory.
        qkv = qkv.view(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                    # each (B, H, T, Dh)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if causal:
            scores = scores + self.mask[:T, :T]
        w = self.dropout(stable_softmax(scores, dim=-1))
        out = w @ v                                         # (B, H, T, Dh)
        # Merge heads: transpose back *then* reshape. `.contiguous()` is required because
        # the transpose leaves a non-viewable layout; `.reshape` would copy anyway, but
        # saying it explicitly documents that a copy happens here.
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


# ---------------------------------------------------------------- self-test
def main() -> int:
    torch.manual_seed(0)

    base = torch.arange(12).reshape(3, 4)
    check("1a slice aliases", shares_storage(base, base[1:]))
    check("1a clone does not", not shares_storage(base, base.clone()))

    col = torch.arange(3).reshape(3, 1)
    check("1b expand allocates nothing",
          col.expand(3, 4).untyped_storage().size() == col.untyped_storage().size())
    check("1b repeat allocates",
          col.repeat(1, 4).untyped_storage().size() > col.untyped_storage().size())

    kb = torch.tensor([0.0, 10.0, 20.0, 30.0])
    check("2a column bias", close(add_key_bias(torch.zeros(4, 4), kb)[1], kb))
    m = causal_mask(5)
    check("2b mask", close(m.diagonal(), torch.zeros(5)) and bool(torch.isinf(m[0, 1:]).all()))

    B, H, T, D = 2, 3, 5, 7
    q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
    check("3 scores", close(scores_einsum(q, k), q @ k.transpose(-2, -1)))
    w = torch.softmax(q @ k.transpose(-2, -1), -1)
    check("3 weighted values", close(weighted_values_einsum(w, v), w @ v))
    check("3 head norms", close(head_norms_einsum(q), (q ** 2).sum(-1).mean(-1)))

    z = torch.randn(6, dtype=torch.double)
    J = torch.autograd.functional.jacobian(lambda t: torch.softmax(t, -1), z)
    check("4a jacobian", close(softmax_jacobian(z), J, atol=1e-10))
    yb = torch.randn(6, dtype=torch.double)
    zl = z.clone().requires_grad_(True)
    torch.softmax(zl, -1).backward(yb)
    check("4b vjp", close(softmax_vjp(z, yb), zl.grad, atol=1e-10))

    for name, t in {
        "ordinary": torch.randn(4, 8),
        "large positive": torch.tensor([[1000.0, 1001.0, 1002.0]]),
        "large negative": torch.tensor([[-1000.0, -1001.0, -1002.0]]),
        "masked row": torch.tensor([[0.0, float("-inf"), float("-inf")]]),
    }.items():
        got = stable_softmax(t)
        check(f"5 {name}", close(got, torch.softmax(t, -1)) and bool(torch.isfinite(got).all()))

    B, H, T, D = 2, 4, 16, 32
    q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
    out, w = attention(q, k, v)
    check("6 matches sdpa", close(out, F.scaled_dot_product_attention(q, k, v), atol=1e-5))
    out_c, w_c = attention(q, k, v, causal=True)
    check("6 causal matches sdpa",
          close(out_c, F.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-5))
    check("6 causal is triangular", bool((w_c.triu(1) == 0).all()))

    e_s, _ = attention_entropy_and_grad(512, scale=True)
    e_r, _ = attention_entropy_and_grad(512, scale=False)
    check("6b scaled entropy high", e_s > 3.0, f"{e_s:.2f} nats")
    check("6b unscaled entropy collapsed", e_r < 1.0, f"{e_r:.2f} nats")

    mha = MultiHeadAttention(64, 8, dropout=0.1)
    x = torch.randn(2, 10, 64)
    check("7 shape", mha(x).shape == x.shape)
    check("7 mask is a buffer", "mask" in dict(mha.named_buffers()))
    check("7 mask is not a parameter", "mask" not in dict(mha.named_parameters()))
    mha.eval()
    with torch.no_grad():
        check("7 eval deterministic", close(mha(x), mha(x)))
        x2 = x.clone()
        x2[:, -1] = torch.randn(2, 64)
        check("7 causality holds", close(mha(x)[:, :-1], mha(x2)[:, :-1], atol=1e-6))
    mha.train()
    check("7 dropout active in train", not close(mha(x), mha(x)))

    return 0 if report() else 1


if __name__ == "__main__":
    raise SystemExit(main())
