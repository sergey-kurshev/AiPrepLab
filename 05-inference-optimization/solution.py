"""Reference solutions for exercise 05 — inference optimisation.  `python solution.py`."""

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

GB = 1024 ** 3


# ============================================================ 1. the roofline
def arithmetic_intensity(n_params: float, batch: int, bytes_per_param: int = 2) -> float:
    """FLOPs per byte for one decode step.

    The weights are read once regardless of batch size, so intensity scales with
    the batch: that one fact is why serving systems fight to keep batches full.
    """
    flops = 2.0 * n_params * batch
    bytes_moved = bytes_per_param * n_params
    return flops / bytes_moved


def min_tpot_seconds(n_params: float, bandwidth_bytes_per_s: float, bytes_per_param: int = 2) -> float:
    """Lower bound on time per output token: you must read every weight at least once."""
    return n_params * bytes_per_param / bandwidth_bytes_per_s


# ============================================================ 2. KV cache
class CachedAttention(nn.Module):
    """Single-head causal attention with an optional KV cache.

    The cache is an algebraic identity, not an approximation: because the mask is
    causal, appending a token cannot change any earlier key or value. The test
    below asserts exact equality, and it should hold to floating-point noise only.
    """

    def __init__(self, d_model: int, n_heads: int = 1):
        super().__init__()
        self.H, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def _split(self, t):
        B, T, C = t.shape
        return t.view(B, T, self.H, self.dh).transpose(1, 2)

    def forward(self, x, cache: dict | None = None):
        B, T, C = x.shape
        q, k, v = (self._split(t) for t in self.qkv(x).chunk(3, dim=-1))
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        # With a cache, the query length is 1 and the key length is the full prefix;
        # `is_causal=True` would mask the wrong triangle, so pass an explicit mask.
        Tq, Tk = q.shape[2], k.shape[2]
        mask = torch.ones(Tq, Tk, dtype=torch.bool).tril(diagonal=Tk - Tq)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(o.transpose(1, 2).reshape(B, Tq, C))


def generate_no_cache(model, x, n_new: int):
    """Recompute everything at each step: O(T^2) total work."""
    for _ in range(n_new):
        out = model(x)
        x = torch.cat([x, out[:, -1:]], dim=1)
    return x


def generate_with_cache(model, x, n_new: int):
    """Prefill once, then one token at a time: O(T) total work."""
    cache: dict = {}
    out = model(x, cache)
    nxt = out[:, -1:]
    xs = [x, nxt]
    for _ in range(n_new - 1):
        nxt = model(nxt, cache)
        xs.append(nxt)
    return torch.cat(xs, dim=1)


def kv_cache_bytes(layers: int, seq: int, batch: int, n_kv_heads: int,
                   d_head: int, bytes_per_elem: int = 2) -> float:
    """2 (K and V) x L x s x b x n_kv x d_head x bytes."""
    return 2 * layers * seq * batch * n_kv_heads * d_head * bytes_per_elem


# ============================================================ 3. quantisation
def quantize_symmetric(x: torch.Tensor, bits: int = 8, dim: int | None = None):
    """Symmetric integer quantisation. `dim=None` is per-tensor; `dim=0` is per-channel.

    Symmetric means no zero point: q = round(x/delta), which keeps the integer
    arithmetic simple at the cost of wasting half the range on a skewed tensor.
    """
    qmax = 2 ** (bits - 1) - 1
    if dim is None:
        scale = x.abs().max() / qmax
    else:
        keep = [i for i in range(x.dim()) if i != dim]
        scale = x.abs().amax(dim=keep, keepdim=True) / qmax
    scale = scale.clamp_min(1e-12)
    q = (x / scale).round().clamp(-qmax - 1, qmax)
    return q, scale


def dequantize(q, scale):
    return q * scale


def quantize_group(x: torch.Tensor, bits: int = 8, group: int = 128):
    """Per-group quantisation along the last axis — what GPTQ/AWQ actually ship."""
    *lead, n = x.shape
    assert n % group == 0, "group size must divide the last dimension"
    xg = x.reshape(*lead, n // group, group)
    qmax = 2 ** (bits - 1) - 1
    scale = (xg.abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-12)
    q = (xg / scale).round().clamp(-qmax - 1, qmax)
    return q, scale, (n // group, group)


def dequantize_group(q, scale, shape_info, orig_shape):
    return (q * scale).reshape(orig_shape)


def relative_error(x, xhat) -> float:
    return float((x - xhat).detach().norm() / x.detach().norm())


def worst_channel_error(x, xhat, dim: int = 1) -> float:
    """The metric that actually exposes a bad quantiser.

    Frobenius relative error is dominated by the largest-magnitude channels, which a
    per-tensor scale handles perfectly well — so it understates the damage. What breaks
    a model is the *small* channels being flattened to two or three levels, and only a
    per-channel view shows that.
    """
    keep = [i for i in range(x.dim()) if i != dim]
    num = (x - xhat).detach().pow(2).sum(dim=keep).sqrt()
    den = x.detach().pow(2).sum(dim=keep).sqrt().clamp_min(1e-12)
    return float((num / den).max())


# ============================================================ 4. speculative decoding
def speculative_step(p: torch.Tensor, q: torch.Tensor, draft_token: int, u: float):
    """One accept/reject decision of the Leviathan et al. sampler.

    Accept x ~ q with probability min(1, p(x)/q(x)); on rejection sample from the
    normalised residual max(0, p - q). The result is distributed EXACTLY as p —
    this is a rejection sampler, not an approximation, which is why speculative
    decoding is a pure latency win rather than a quality trade.
    """
    if u < min(1.0, float(p[draft_token] / q[draft_token])):
        return draft_token, True
    residual = (p - q).clamp_min(0)
    total = residual.sum()
    residual = residual / total if float(total) > 0 else torch.ones_like(p) / p.numel()
    return int(torch.multinomial(residual, 1)), False


def speculative_sample(p: torch.Tensor, q: torch.Tensor, gen: torch.Generator) -> int:
    draft = int(torch.multinomial(q, 1, generator=gen))
    u = float(torch.rand(1, generator=gen))
    token, _ = speculative_step(p, q, draft, u)
    return token


def expected_tokens(alpha: float, gamma: int) -> float:
    """E[tokens per verification step] = (1 - a^(g+1)) / (1 - a)."""
    if alpha >= 1.0:
        return gamma + 1.0
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


def acceptance_rate(p: torch.Tensor, q: torch.Tensor) -> float:
    """E_{x~q}[min(1, p/q)] = sum_x min(p, q) — one minus the total variation distance."""
    return float(torch.minimum(p, q).sum())


# ============================================================ self-test
def main() -> int:
    torch.manual_seed(0)

    # ---- 1. roofline
    h100_flops, h100_bw = 990e12, 3.35e12
    ridge = h100_flops / h100_bw
    ai1 = arithmetic_intensity(70e9, batch=1)
    check("1 single-token decode has intensity ~1", close(ai1, 1.0, rtol=1e-9), f"{ai1:.2f} FLOP/byte")
    check("1 that is far below the H100 ridge point", ai1 < ridge / 100,
          f"ridge is {ridge:.0f} FLOP/byte — we use ~{100 * ai1 / ridge:.2f}% of the compute")
    check("1 batching raises intensity linearly",
          close(arithmetic_intensity(70e9, 256), 256.0, rtol=1e-9))
    tpot = min_tpot_seconds(70e9, h100_bw)
    check("1 a 70B bf16 model cannot exceed ~24 tok/s on one H100",
          0.030 < tpot < 0.050, f"{tpot * 1e3:.1f} ms/token = {1 / tpot:.0f} tok/s, bandwidth-bound")
    check("1 int4 weights would be 4x faster", close(min_tpot_seconds(70e9, h100_bw, 0.5), tpot / 4))

    # ---- 2. KV cache
    torch.manual_seed(0)
    model = CachedAttention(32, n_heads=4).eval().double()
    x = torch.randn(2, 5, 32, dtype=torch.double)
    with torch.no_grad():
        a = generate_no_cache(model, x, 6)
        b = generate_with_cache(model, x, 6)
    check("2 the cache is exact, not approximate", close(a, b, atol=1e-10),
          f"max difference {float((a - b).abs().max()):.2e}")

    # Work done, counted rather than timed: O(T^2) vs O(T).
    import time
    big = CachedAttention(256, 8).eval()
    xb = torch.randn(1, 64, 256)
    with torch.no_grad():
        t0 = time.perf_counter(); generate_no_cache(big, xb, 96); t_no = time.perf_counter() - t0
        t0 = time.perf_counter(); generate_with_cache(big, xb, 96); t_yes = time.perf_counter() - t0
    check("2 the cache is faster in wall clock", t_yes < t_no, f"{t_no * 1e3:.0f} ms -> {t_yes * 1e3:.0f} ms")

    # Llama-2-70B geometry: 80 layers, 64 query heads, d_head 128, GQA with 8 KV heads.
    mha = kv_cache_bytes(80, 4096, 1, 64, 128)
    gqa = kv_cache_bytes(80, 4096, 1, 8, 128)
    mqa = kv_cache_bytes(80, 4096, 1, 1, 128)
    check("2 MHA cache for one 4k sequence is ~10 GB", 9 * GB < mha < 12 * GB, f"{mha / GB:.1f} GB")
    check("2 GQA(8) cuts it 8x", close(mha / gqa, 8.0, rtol=1e-9), f"{gqa / GB:.2f} GB")
    check("2 MQA cuts it 64x", close(mha / mqa, 64.0, rtol=1e-9), f"{mqa / GB:.2f} GB")
    weights = 70e9 * 2
    check("2 at batch 32 the MHA cache exceeds the weights",
          kv_cache_bytes(80, 4096, 32, 64, 128) > weights,
          f"{kv_cache_bytes(80, 4096, 32, 64, 128) / GB:.0f} GB of cache vs {weights / GB:.0f} GB of weights")

    # ---- 3. quantisation
    torch.manual_seed(0)
    W = torch.randn(256, 512) * torch.linspace(0.1, 10.0, 512)   # heterogeneous channel scales
    q_t, s_t = quantize_symmetric(W, 8, dim=None)
    q_c, s_c = quantize_symmetric(W, 8, dim=1)
    e_t = relative_error(W, dequantize(q_t, s_t))
    e_c = relative_error(W, dequantize(q_c, s_c))
    w_t = worst_channel_error(W, dequantize(q_t, s_t))
    w_c = worst_channel_error(W, dequantize(q_c, s_c))
    check("3 per-channel beats per-tensor on the Frobenius metric", e_c < e_t,
          f"per-tensor {e_t:.2e}, per-channel {e_c:.2e} — only ~2x, which is misleading")
    check("3 ...and dominates on the metric that matters", w_c < w_t / 20,
          f"worst channel: per-tensor {w_t:.2e}, per-channel {w_c:.2e}")

    qg, sg, info = quantize_group(W, 8, group=128)
    e_g = relative_error(W, dequantize_group(qg, sg, info, W.shape))
    check("3 per-group is at least as good as per-tensor", e_g < e_t, f"per-group {e_g:.2e}")

    # An outlier: one activation channel 100x the rest, which is what really happens
    # in transformers above ~6.7B parameters.
    act = torch.randn(64, 512)
    act[:, 17] *= 100.0
    e_out_t = relative_error(act, dequantize(*quantize_symmetric(act, 8, dim=None)))
    e_out_c = relative_error(act, dequantize(*quantize_symmetric(act, 8, dim=1)))
    clean = torch.randn(64, 512)
    e_clean = relative_error(clean, dequantize(*quantize_symmetric(clean, 8, dim=None)))
    check("3 one outlier wrecks per-tensor quantisation", e_out_t > 10 * e_clean,
          f"{e_clean:.2e} clean -> {e_out_t:.2e} with an outlier")
    check("3 per-channel isolates the outlier", e_out_c < e_out_t / 5, f"{e_out_c:.2e}")

    # Bit width vs error: each extra bit should roughly halve it.
    errs = {b: relative_error(W, dequantize(*quantize_symmetric(W, b, dim=1))) for b in (4, 6, 8, 12)}
    check("3 each extra bit halves the error",
          all(close(errs[b] / errs[b + 2], 4.0, rtol=0.25) for b in (4, 6)),
          ", ".join(f"{b}-bit {e:.2e}" for b, e in errs.items()))

    # End to end on a linear layer: does W8 actually preserve the output?
    lin = nn.Linear(512, 256, bias=False)
    xq = torch.randn(32, 512)
    ref = lin(xq)
    qw, sw = quantize_symmetric(lin.weight, 8, dim=0)
    got = xq @ dequantize(qw, sw).T
    check("3 W8A16 preserves the layer output to <1%", relative_error(ref, got) < 0.01,
          f"relative error {relative_error(ref, got):.2e}")

    # ---- 4. speculative decoding
    gen = torch.Generator().manual_seed(0)
    V = 8
    p = torch.softmax(torch.randn(V, generator=gen) * 1.5, 0)
    q = torch.softmax(torch.randn(V, generator=gen) * 1.5, 0)

    n = 400_000
    counts = torch.zeros(V)
    for _ in range(n):
        counts[speculative_sample(p, q, gen)] += 1
    emp = counts / n
    check("4 speculative sampling reproduces the TARGET distribution exactly",
          float((emp - p).abs().max()) < 0.005,
          f"max deviation {float((emp - p).abs().max()):.4f} over {n} samples")
    check("4 ...and not the draft distribution", float((emp - q).abs().max()) > 0.02,
          f"deviation from q is {float((emp - q).abs().max()):.4f}")

    # Acceptance rate: measured vs sum(min(p, q)).
    acc = 0
    for _ in range(100_000):
        d = int(torch.multinomial(q, 1, generator=gen))
        u = float(torch.rand(1, generator=gen))
        acc += speculative_step(p, q, d, u)[1]
    alpha_emp = acc / 100_000
    check("4 acceptance rate is sum(min(p,q)) = 1 - TV(p,q)",
          close(alpha_emp, acceptance_rate(p, q), atol=0.01),
          f"measured {alpha_emp:.3f}, predicted {acceptance_rate(p, q):.3f}")

    check("4 identical models accept everything", close(acceptance_rate(p, p), 1.0, atol=1e-6))
    check("4 expected tokens formula, alpha=0.8 gamma=4",
          close(expected_tokens(0.8, 4), 3.3616, atol=1e-3), f"{expected_tokens(0.8, 4):.4f}")
    check("4 a perfect draft gives gamma+1 tokens", close(expected_tokens(1.0, 4), 5.0))
    check("4 a useless draft gives 1", close(expected_tokens(0.0, 4), 1.0))
    check("4 longer drafts have diminishing returns",
          expected_tokens(0.7, 8) - expected_tokens(0.7, 4) < 0.5,
          f"gamma 4 -> {expected_tokens(0.7, 4):.2f}, gamma 8 -> {expected_tokens(0.7, 8):.2f}")

    return 0 if report() else 1


if __name__ == "__main__":
    raise SystemExit(main())
