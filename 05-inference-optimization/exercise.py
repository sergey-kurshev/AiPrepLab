# %% [markdown]
# # Exercise 05 — Inference Optimisation
#
# Four parts, in the order you should attack a real latency problem:
#
# 1. **Find the bottleneck** — arithmetic intensity, and the bandwidth bound nobody can beat.
# 2. **KV cache** — implement it, prove it is exact, measure what it buys, and compute what it costs.
# 3. **Quantisation** — three granularities, and the outlier that breaks the naive one.
# 4. **Speculative decoding** — implement the exact sampler and verify, empirically, that a 3×
#    speedup changed nothing about the output distribution.
#
# Part 4 is the one worth the most: almost everyone can describe speculative decoding, and almost
# nobody has checked that their implementation preserves the target distribution.

# %%
import sys
import pathlib

_here = pathlib.Path.cwd()
for _c in (_here, *_here.parents):
    if (_c / "tools").is_dir():
        sys.path.insert(0, str(_c / "tools"))
        break

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from checks import check, report, close

torch.manual_seed(0)
GB = 1024 ** 3
H100_FLOPS, H100_BW = 990e12, 3.35e12     # bf16 dense peak, HBM bandwidth

# %% [markdown]
# ---
# ## Part 1 — Where does the time actually go?
#
# For a single-token decode step: every weight is read from HBM once, and each is used for one
# multiply-add. Compute the FLOPs, the bytes, and the ratio.
#
# Predict the answer before you run it. Then compare it against the H100's **ridge point**
# (peak FLOP/s ÷ peak bytes/s) and decide whether the machine is compute- or memory-bound.

# %%
def arithmetic_intensity(n_params: float, batch: int, bytes_per_param: int = 2) -> float:
    """FLOPs per byte of memory traffic for one decode step."""
    # TODO
    raise NotImplementedError


def min_tpot_seconds(n_params: float, bandwidth: float, bytes_per_param: int = 2) -> float:
    """Lower bound on time per output token, from bandwidth alone."""
    # TODO
    raise NotImplementedError


try:
    ridge = H100_FLOPS / H100_BW
    print(f"H100 ridge point: {ridge:.0f} FLOP/byte\n")
    print(f"{'batch':>7} {'intensity':>12} {'vs ridge':>10}  regime")
    for b in (1, 8, 64, 256, 1024):
        ai = arithmetic_intensity(70e9, b)
        print(f"{b:>7} {ai:>12.1f} {ai / ridge:>9.1%}  {'memory-bound' if ai < ridge else 'compute-bound'}")

    check("1a batch-1 decode has intensity ~1", close(arithmetic_intensity(70e9, 1), 1.0, rtol=1e-9))
    check("1a that is <1% of the ridge point", arithmetic_intensity(70e9, 1) < ridge / 100)
    check("1a intensity scales linearly with batch",
          close(arithmetic_intensity(70e9, 256), 256.0, rtol=1e-9))

    tpot = min_tpot_seconds(70e9, H100_BW)
    print(f"\n70B in bf16: at best {tpot * 1e3:.1f} ms/token = {1 / tpot:.0f} tok/s on one H100.")
    check("1b the bandwidth bound is ~24 tok/s", 20 < 1 / tpot < 30)
    check("1b int4 weights quadruple it", close(min_tpot_seconds(70e9, H100_BW, 0.5), tpot / 4))
except Exception as _e:
    check("part 1", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **1c. (discussion)** Three optimisations, one explanation each — and each explanation must refer
# to the number you just computed:
#
# 1. Why does quantising weights speed up decoding, given that it does not reduce the FLOP count?
# 2. Why does batching improve throughput far more than it hurts latency?
# 3. Why is prefill a completely different regime, and which metric does it govern?

# %% [markdown]
# ---
# ## Part 2 — The KV cache
#
# **2a.** Implement caching in a causal attention module. The crucial property to preserve: the
# cached path must produce **exactly** the same result as full recomputation, to floating-point
# noise. It is an algebraic identity — the causal mask guarantees that appending a token cannot
# change any earlier key or value.
#
# The trap: with a cache, the query length is 1 but the key length is the whole prefix, so the
# score matrix is `(1, T)` and **not square**. `is_causal=True` masks the wrong triangle of a
# non-square matrix. Build the mask explicitly with the right diagonal offset.

# %%
class CachedAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 1):
        super().__init__()
        self.H, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cache: dict | None = None):
        """If `cache` is given, append this step's K/V to it and attend over the whole prefix."""
        # TODO
        raise NotImplementedError


def generate_no_cache(model, x, n_new: int):
    """Recompute everything each step. O(T^2)."""
    for _ in range(n_new):
        x = torch.cat([x, model(x)[:, -1:]], dim=1)
    return x


def generate_with_cache(model, x, n_new: int):
    """Prefill once, then one token at a time. O(T)."""
    # TODO
    raise NotImplementedError


try:
    model = CachedAttention(32, n_heads=4).eval().double()
    x = torch.randn(2, 5, 32, dtype=torch.double)
    with torch.no_grad():
        a, b = generate_no_cache(model, x, 6), generate_with_cache(model, x, 6)
    check("2a the cache is EXACT, not approximate", close(a, b, atol=1e-10),
          f"max difference {float((a - b).abs().max()):.2e}")

    big = CachedAttention(256, 8).eval()
    xb = torch.randn(1, 64, 256)
    with torch.no_grad():
        t0 = time.perf_counter(); generate_no_cache(big, xb, 96); t_no = time.perf_counter() - t0
        t0 = time.perf_counter(); generate_with_cache(big, xb, 96); t_yes = time.perf_counter() - t0
    print(f"96 tokens:  no cache {t_no * 1e3:.0f} ms   cached {t_yes * 1e3:.0f} ms   "
          f"({t_no / t_yes:.1f}x)")
    check("2a caching is faster", t_yes < t_no)
except Exception as _e:
    check("2a", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **2b. What it costs.** The cache is $2 \times L \times s \times b \times n_{kv} \times d_{head}$
# elements. Compute it for Llama-2-70B geometry (80 layers, 64 query heads, $d_{head}=128$) and
# compare MHA, GQA-8 and MQA.
#
# Then answer the question that motivates the whole GQA line of work: at batch 32 and 4k context,
# which is larger — the weights or the cache?

# %%
def kv_cache_bytes(layers, seq, batch, n_kv_heads, d_head, bytes_per_elem=2) -> float:
    # TODO
    raise NotImplementedError


try:
    print(f"{'scheme':>10} {'kv heads':>9} {'1 seq @ 4k':>13} {'batch 32':>12}")
    for name, nkv in [("MHA", 64), ("GQA-8", 8), ("MQA", 1)]:
        one = kv_cache_bytes(80, 4096, 1, nkv, 128)
        many = kv_cache_bytes(80, 4096, 32, nkv, 128)
        print(f"{name:>10} {nkv:>9} {one / GB:>11.2f} GB {many / GB:>10.1f} GB")

    mha = kv_cache_bytes(80, 4096, 1, 64, 128)
    check("2b MHA costs ~10 GB for one 4k sequence", 9 * GB < mha < 12 * GB, f"{mha / GB:.1f} GB")
    check("2b GQA-8 divides it by 8", close(mha / kv_cache_bytes(80, 4096, 1, 8, 128), 8.0, rtol=1e-9))
    check("2b at batch 32 the MHA cache dwarfs the 140 GB of weights",
          kv_cache_bytes(80, 4096, 32, 64, 128) > 2 * 70e9 * 2)
except Exception as _e:
    check("2b", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **2c. (discussion)** Why can you cache K and V but not the attention *output*? And why would this
# whole technique be invalid for a bidirectional (BERT-style) model? Both answers are one sentence
# about the causal mask — the same property you tested in module 03.

# %% [markdown]
# ---
# ## Part 3 — Quantisation
#
# **3a.** Symmetric integer quantisation at three granularities: per-tensor, per-channel, per-group.
#
# $$\Delta = \frac{\max|x|}{2^{b-1}-1},\qquad q = \mathrm{round}(x/\Delta),\qquad \hat x = \Delta q.$$
#
# Then measure the error two ways, because **the metric you choose decides what you conclude**.
# Frobenius relative error is dominated by the largest channels — which a per-tensor scale handles
# perfectly well. Worst-channel relative error asks what happened to the *small* channels, which is
# what actually breaks a model.

# %%
def quantize_symmetric(x, bits=8, dim=None):
    """Return (q, scale). `dim=None` is per-tensor; an int gives one scale per index along it."""
    # TODO
    raise NotImplementedError


def dequantize(q, scale):
    return q * scale


def quantize_group(x, bits=8, group=128):
    """Per-group along the last axis — what GPTQ and AWQ actually ship. Return (q, scale, info)."""
    # TODO
    raise NotImplementedError


def dequantize_group(q, scale, info, orig_shape):
    return (q * scale).reshape(orig_shape)


def relative_error(x, xhat) -> float:
    return float((x - xhat).detach().norm() / x.detach().norm())


def worst_channel_error(x, xhat, dim: int = 1) -> float:
    """Max over channels of the per-channel relative error."""
    # TODO
    raise NotImplementedError


# A weight matrix whose columns have very different scales — which is what real ones look like.
W = torch.randn(256, 512) * torch.linspace(0.1, 10.0, 512)

try:
    qt, st = quantize_symmetric(W, 8, dim=None)
    qc, sc = quantize_symmetric(W, 8, dim=1)
    qg, sg, info = quantize_group(W, 8, group=128)

    e_t, e_c = relative_error(W, dequantize(qt, st)), relative_error(W, dequantize(qc, sc))
    w_t, w_c = worst_channel_error(W, dequantize(qt, st)), worst_channel_error(W, dequantize(qc, sc))
    e_g = relative_error(W, dequantize_group(qg, sg, info, W.shape))

    print(f"                      Frobenius     worst channel")
    print(f"per-tensor  8-bit:     {e_t:.2e}        {w_t:.2e}")
    print(f"per-channel 8-bit:     {e_c:.2e}        {w_c:.2e}")
    print(f"per-group   8-bit:     {e_g:.2e}")
    check("3a per-channel wins on Frobenius, but only by ~2x", e_c < e_t)
    check("3a per-channel wins overwhelmingly on the worst channel", w_c < w_t / 20,
          "this is the number that decides whether the model survives")
    check("3a per-group beats per-tensor", e_g < e_t)

    errs = {b: relative_error(W, dequantize(*quantize_symmetric(W, b, dim=1))) for b in (4, 6, 8, 12)}
    print("\nbits vs error:", ", ".join(f"{b}: {e:.2e}" for b, e in errs.items()))
    check("3a each extra bit halves the error",
          all(close(errs[b] / errs[b + 2], 4.0, rtol=0.25) for b in (4, 6)))
except Exception as _e:
    check("3a", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **3b. The outlier.** Above roughly 6.7B parameters, transformer activations develop systematic
# outlier channels 10–100× the rest — in the *same* channels across tokens. Plant one and watch
# what a per-tensor scale does.

# %%
try:
    clean = torch.randn(64, 512)
    act = torch.randn(64, 512)
    act[:, 17] *= 100.0           # one outlier channel, as observed in real models

    e_clean = relative_error(clean, dequantize(*quantize_symmetric(clean, 8, dim=None)))
    e_out_t = relative_error(act, dequantize(*quantize_symmetric(act, 8, dim=None)))
    e_out_c = relative_error(act, dequantize(*quantize_symmetric(act, 8, dim=1)))
    print(f"per-tensor on clean activations: {e_clean:.2e}")
    print(f"per-tensor with one outlier:     {e_out_t:.2e}   ({e_out_t / e_clean:.0f}x worse)")
    print(f"per-channel with one outlier:    {e_out_c:.2e}")
    check("3b one outlier wrecks per-tensor", e_out_t > 10 * e_clean)
    check("3b per-channel isolates it", e_out_c < e_out_t / 5)
except Exception as _e:
    check("3b", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **3c.** End to end: quantise a linear layer's weights to 8 bits, leave the activations in fp32
# (this is "W8A16", the common production point), and measure the output error.

# %%
try:
    lin = nn.Linear(512, 256, bias=False)
    xq = torch.randn(32, 512)
    ref = lin(xq)
    qw, sw = quantize_symmetric(lin.weight.detach(), 8, dim=0)
    got = xq @ dequantize(qw, sw).T
    err = relative_error(ref, got)
    print(f"W8A16 relative output error: {err:.2e}")
    check("3c W8A16 preserves the output to <1%", err < 0.01)
    qw4, sw4 = quantize_symmetric(lin.weight.detach(), 4, dim=0)
    print(f"W4A16 relative output error: {relative_error(ref, xq @ dequantize(qw4, sw4).T):.2e}")
except Exception as _e:
    check("3c", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **3d. (discussion)** Why is weight-only quantisation the right first move for decoding, and when
# would you also quantise activations? What does SmoothQuant do about the outliers, and why is it
# mathematically a no-op?

# %% [markdown]
# ---
# ## Part 4 — Speculative decoding
#
# The draft model proposes $x \sim q$; the target model verifies. Accept with probability
# $\min(1, p(x)/q(x))$; on rejection, sample from the normalised residual
# $\max(0, p - q)$.
#
# **The claim to verify: the resulting samples are distributed exactly as $p$.** Not approximately
# — exactly. Implement it and check empirically over several hundred thousand draws.

# %%
def speculative_step(p, q, draft_token: int, u: float):
    """One accept/reject decision. `u` is a uniform(0,1) draw, passed in for testability.

    Returns (token, accepted).
    """
    # TODO
    raise NotImplementedError


def speculative_sample(p, q, gen) -> int:
    draft = int(torch.multinomial(q, 1, generator=gen))
    u = float(torch.rand(1, generator=gen))
    return speculative_step(p, q, draft, u)[0]


def expected_tokens(alpha: float, gamma: int) -> float:
    """Expected accepted tokens per verification step, given acceptance rate alpha."""
    # TODO
    raise NotImplementedError


def acceptance_rate(p, q) -> float:
    """Predict the acceptance rate analytically. Hint: E_{x~q}[min(1, p/q)] has a closed form."""
    # TODO
    raise NotImplementedError


try:
    gen = torch.Generator().manual_seed(0)
    V = 8
    p = torch.softmax(torch.randn(V, generator=gen) * 1.5, 0)
    q = torch.softmax(torch.randn(V, generator=gen) * 1.5, 0)

    n = 200_000
    counts = torch.zeros(V)
    for _ in range(n):
        counts[speculative_sample(p, q, gen)] += 1
    emp = counts / n
    print("target p:  ", " ".join(f"{v:.3f}" for v in p))
    print("empirical: ", " ".join(f"{v:.3f}" for v in emp))
    print("draft q:   ", " ".join(f"{v:.3f}" for v in q))
    check("4a the sampler reproduces the TARGET distribution",
          float((emp - p).abs().max()) < 0.006, f"max deviation {float((emp - p).abs().max()):.4f}")
    check("4a and is clearly NOT the draft distribution", float((emp - q).abs().max()) > 0.02)

    acc = sum(speculative_step(p, q, int(torch.multinomial(q, 1, generator=gen)),
                               float(torch.rand(1, generator=gen)))[1] for _ in range(50_000))
    check("4b the acceptance rate matches the closed form",
          close(acc / 50_000, acceptance_rate(p, q), atol=0.012),
          f"measured {acc / 50_000:.3f}, predicted {acceptance_rate(p, q):.3f}")
    check("4b an identical draft accepts everything", close(acceptance_rate(p, p), 1.0, atol=1e-6))

    print(f"\n{'alpha':>7}" + "".join(f"{g:>9}" for g in (1, 2, 4, 8, 16)))
    for a in (0.3, 0.5, 0.7, 0.9):
        print(f"{a:>7.1f}" + "".join(f"{expected_tokens(a, g):>9.2f}" for g in (1, 2, 4, 8, 16)))
    check("4c alpha=0.8, gamma=4 gives 3.36 tokens/step", close(expected_tokens(0.8, 4), 3.3616, atol=1e-3))
    check("4c a perfect draft gives gamma+1", close(expected_tokens(1.0, 4), 5.0))
    check("4c a useless draft gives 1", close(expected_tokens(0.0, 4), 1.0))
except Exception as _e:
    check("part 4", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **4d.** Read the table you just printed. Returns diminish sharply in $\gamma$ — at $\alpha=0.7$,
# going from $\gamma=4$ to $\gamma=16$ buys almost nothing, while the draft model's cost grows
# linearly. Write down the break-even condition: if the draft model costs a fraction $c$ of the
# target per token, for what $(\alpha, \gamma)$ is speculation actually worth it?
#
# **4e. (discussion)** Speculative decoding preserves the distribution *exactly*. Why is that a
# much stronger claim than "quality is preserved", and what would you have to measure if it were
# only approximately true?

# %% [markdown]
# ---
# ## Revisiting the spine
#
# Attention so far: tensor code (00) → gradient (01) → from scratch (02) → debugged (03) →
# sharded (04) → **cached and quantised** (05). Notice that the KV cache is nothing but the causal
# mask, cashed in: the property you first asserted as a test in module 03 turns out to be the
# thing that makes fast inference possible at all.
#
# **Stretch goals:**
#
# 1. Implement the full multi-token speculative loop (draft $\gamma$ tokens autoregressively,
#    verify all $\gamma$ in one target forward pass, accept the longest correct prefix) and confirm
#    the output distribution still matches. The bookkeeping is where implementations go wrong.
# 2. Implement PagedAttention's block table: allocate the KV cache in fixed 16-token blocks with an
#    indirection table, and measure the fragmentation saved versus contiguous pre-allocation.
# 3. Quantise the KV cache itself to INT8 per-token and measure the effect on generated logits.
#    Is it more or less forgiving than quantising the weights? Why?
# 4. Build the latency–throughput curve: sweep batch size, plot TPOT against tokens/second, and
#    mark the largest batch that meets a 50 ms TPOT budget.
