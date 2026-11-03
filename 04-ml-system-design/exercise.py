# %% [markdown]
# # Exercise 04 — Parallelism, Verified on One Machine
#
# You will not get a cluster in an interview, and you do not need one here. Every parallelism
# scheme in this notebook is implemented in a single process and checked against the unsharded
# computation. A sharded forward either produces the same gradients or it does not, and that
# question is answerable on a laptop.
#
# Five parts:
#
# 1. **Arithmetic** — the memory and compute budget, which is the first thing you write on a
#    whiteboard when asked "how would you train a 100B model?".
# 2. **Tensor parallelism** — column/row-parallel matrices, and head-parallel attention.
# 3. **Data parallelism** — all-reduce, gradient accumulation, and the factor-of-$n$ bug.
# 4. **Pipeline parallelism** — simulate the schedule, *derive* the bubble formula rather than
#    quoting it.
# 5. **Activation checkpointing** — measure the trade rather than assuming it.

# %%
import sys
import pathlib

_here = pathlib.Path.cwd()
for _c in (_here, *_here.parents):
    if (_c / "tools").is_dir():
        sys.path.insert(0, str(_c / "tools"))
        break

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from checks import check, report, close

torch.manual_seed(0)
GB = 1024 ** 3

# %% [markdown]
# ---
# ## Part 1 — The budget
#
# **1a.** Static training state, per parameter. Fill in the table from the article: bf16 weights,
# bf16 gradients, fp32 master weights, and the optimiser's own state.
#
# The number you are trying to arrive at is **16 bytes per parameter** for mixed-precision Adam.
# Note the result for fp32 — it is the same 16, which surprises people and is worth being able to
# explain.

# %%
def training_memory(n_params: int, precision: str = "mixed", optimizer: str = "adam") -> dict:
    """Bytes of static state, by category, plus 'total' and 'bytes_per_param'."""
    # TODO
    raise NotImplementedError


try:
    mem = training_memory(100e9)
    check("1a mixed + Adam is 16 bytes/param", mem["bytes_per_param"] == 16,
          f"{mem['total'] / 1e12:.2f} TB for a 100B model")
    check("1a fp32 + Adam is also 16", training_memory(1, "fp32")["bytes_per_param"] == 16)
    check("1a SGD with momentum is 12", training_memory(1, "mixed", "sgd_momentum")["bytes_per_param"] == 12)
    check("1a plain SGD is 8", training_memory(1, "mixed", "sgd")["bytes_per_param"] == 8)
    print(f"\n100B parameters need {mem['total'] / 1e12:.2f} TB of static state — "
          f"{mem['total'] / (80 * GB):.0f} H100s just to hold it.")
except Exception as _e:
    check("1a", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **1b. ZeRO.** Each stage shards one more category across the data-parallel group. Work out the
# per-device cost and check the limiting behaviour.

# %%
def zero_memory(n_params: int, stage: int, dp_size: int) -> float:
    """Per-device bytes under ZeRO stage 0–3. Categories: 2 weights, 2 grads, 12 optimiser."""
    # TODO
    raise NotImplementedError


try:
    n = 100e9
    z0, z1, z3 = (zero_memory(n, s, 64) for s in (0, 1, 3))
    print(f"at dp=64:  ZeRO-0 {z0 / GB:>7.0f} GB   ZeRO-1 {z1 / GB:>7.0f} GB   ZeRO-3 {z3 / GB:>7.0f} GB")
    check("1b ZeRO-0 is the full 16 bytes", close(z0, 16 * n, rtol=1e-9))
    check("1b ZeRO-3 shards everything", close(z3, z0 / 64, rtol=1e-9))
    check("1b ZeRO-1 tends to 4 bytes/param", close(zero_memory(n, 1, 10 ** 6), 4 * n, rtol=1e-3))
    check("1b ZeRO-2 tends to 2 bytes/param", close(zero_memory(n, 2, 10 ** 6), 2 * n, rtol=1e-3))
except Exception as _e:
    check("1b", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **1c. Activations.** Korthikanti et al. (2022), eq. 2, for bf16:
#
# $$s\,b\,h\,L\left(34 + 5\frac{a\,s}{h}\right)\ \text{bytes}$$
#
# Implement it, then answer the question that matters: which term dominates at long context, and
# what does that imply about where to spend your engineering effort?

# %%
def activation_memory(layers, hidden, seq, micro_batch, heads, recompute="none") -> float:
    """'none' = the full formula; 'selective' = drop the 5*a*s/h term (recompute attention);
    'full' = store only layer inputs (2*s*b*h*L)."""
    # TODO
    raise NotImplementedError


try:
    # GPT-3 175B geometry.
    L, h, a = 96, 12288, 96
    for s in (2048, 4096, 8192, 16384):
        full = activation_memory(L, h, s, 1, a)
        sel = activation_memory(L, h, s, 1, a, "selective")
        print(f"seq {s:>6}:  full {full / GB:>8.0f} GB   selective {sel / GB:>7.0f} GB   "
              f"quadratic share {1 - sel / full:>5.1%}")
    base = activation_memory(L, h, 2048, 1, a)
    check("1c 4x sequence costs more than 4x memory", activation_memory(L, h, 8192, 1, a) / base > 4.5)
    check("1c selective recomputation removes most of it at long context",
          activation_memory(L, h, 8192, 1, a, "selective") < 0.3 * activation_memory(L, h, 8192, 1, a))
except Exception as _e:
    check("1c", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **1d. Compute and Chinchilla.** $C \approx 6ND$ forward+backward, $8ND$ with full recomputation.
# Chinchilla's compute-optimal allocation is $D \approx 20N$; combine the two and solve for $N$.

# %%
def training_flops(n_params, n_tokens, recompute=False) -> float:
    # TODO
    raise NotImplementedError


def chinchilla_optimal(compute_flops: float) -> tuple[float, float]:
    """(N, D) at the 20-tokens-per-parameter ratio for a given FLOP budget."""
    # TODO
    raise NotImplementedError


try:
    c = training_flops(175e9, 300e9)
    check("1d GPT-3 cost ~3e23 FLOPs", 2.5e23 < c < 3.6e23, f"{c:.2e}")
    N, D = chinchilla_optimal(c)
    print(f"For GPT-3's budget, Chinchilla would train {N / 1e9:.0f}B params on {D / 1e9:.0f}B tokens.")
    check("1d Chinchilla prefers a smaller model on more data", N < 175e9 and D > 300e9)
    check("1d the ratio is 20:1", close(D / N, 20.0, rtol=1e-6))

    # How long on 1024 H100s at 40% MFU? (H100 bf16 dense peak ~ 990 TFLOP/s.)
    seconds = c / (1024 * 990e12 * 0.40)
    print(f"1024 H100s at 40% MFU: {seconds / 86400:.1f} days")
except Exception as _e:
    check("1d", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **1e. (discussion)** You have 512 H100s (80 GB each) and want to train a 70B model at sequence
# length 8192. Write down a configuration: tensor-parallel degree, pipeline depth, data-parallel
# width, ZeRO stage, micro-batch size, recomputation strategy. Justify every number with an
# arithmetic constraint from above, and state which one is binding.

# %% [markdown]
# ---
# ## Part 2 — Tensor parallelism
#
# Megatron's MLP split, exactly as in §3.2 of the article:
#
# - $A$ **column**-parallel: each device owns a slice of the output features, and GELU is
#   elementwise, so no communication is needed before the second matmul.
# - $B$ **row**-parallel: each device holds a slice of the contraction index, producing a *partial
#   sum*; one all-reduce finishes the job.
#
# Ask yourself before implementing: what would go wrong if you split $A$ by rows instead?

# %%
def all_reduce(partials: list[torch.Tensor]) -> torch.Tensor:
    """Simulated all-reduce: every rank ends up holding the sum."""
    # TODO
    raise NotImplementedError


def column_parallel_linear(x, W, tp: int) -> list[torch.Tensor]:
    """Return per-device output shards of X W, with W split along dim 1."""
    # TODO
    raise NotImplementedError


def row_parallel_linear(x_shards: list[torch.Tensor], W, tp: int) -> torch.Tensor:
    """X W with W split along dim 0, consuming an already-sharded input. One all-reduce."""
    # TODO
    raise NotImplementedError


def parallel_mlp(x, A, B, tp: int):
    """GELU(X A) B, tensor-parallel."""
    # TODO
    raise NotImplementedError


B_, T_, C = 2, 6, 64
x = torch.randn(B_, T_, C, dtype=torch.double, requires_grad=True)
A = torch.randn(C, 4 * C, dtype=torch.double, requires_grad=True)
Bm = torch.randn(4 * C, C, dtype=torch.double, requires_grad=True)

try:
    ref = F.gelu(x @ A) @ Bm
    for tp in (1, 2, 4, 8):
        check(f"2a MLP tp={tp} forward", close(parallel_mlp(x, A, Bm, tp), ref, atol=1e-12))

    # The check that actually matters: gradients, not just the forward.
    gref = torch.autograd.grad(ref.square().sum(), [x, A, Bm], retain_graph=True)
    gtp = torch.autograd.grad(parallel_mlp(x, A, Bm, 4).square().sum(), [x, A, Bm])
    check("2a gradients match too", all(close(a, b, atol=1e-10) for a, b in zip(gref, gtp)))

    # And the failure mode: forget the all-reduce, keep your own partial sum.
    shards = [F.gelu(h) for h in column_parallel_linear(x, A, 4)]
    partial = shards[0] @ Bm.chunk(4, dim=0)[0]
    check("2a a missing all-reduce is detectable", not close(partial, ref, atol=1e-6))
except Exception as _e:
    check("2a", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **2b. Head-parallel attention.** Heads are already independent, so the partition follows the
# mathematics: each device owns $H/\mathrm{tp}$ heads end to end, and the output projection is
# row-parallel.
#
# The layout detail that catches people: `W_qkv` is `(C, 3C)` arranged as `[Q | K | V]`. Device $r$
# needs *its head slice of each of the three*, not a contiguous third of the matrix.

# %%
def parallel_attention(x, Wqkv, Wo, n_heads: int, tp: int):
    # TODO
    raise NotImplementedError


def reference_attention(x, Wqkv, Wo, n_heads: int):
    B, T, C = x.shape
    q, k, v = (t.view(B, T, n_heads, C // n_heads).transpose(1, 2)
               for t in (x @ Wqkv).chunk(3, dim=-1))
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return o.transpose(1, 2).reshape(B, T, C) @ Wo


H = 8
Wqkv = torch.randn(C, 3 * C, dtype=torch.double, requires_grad=True)
Wo = torch.randn(C, C, dtype=torch.double, requires_grad=True)

try:
    ra = reference_attention(x, Wqkv, Wo, H)
    for tp in (1, 2, 4):
        check(f"2b attention tp={tp}", close(parallel_attention(x, Wqkv, Wo, H, tp), ra, atol=1e-10))
    g1 = torch.autograd.grad(ra.square().sum(), [x, Wqkv, Wo], retain_graph=True)
    g2 = torch.autograd.grad(parallel_attention(x, Wqkv, Wo, H, 4).square().sum(), [x, Wqkv, Wo])
    check("2b gradients match", all(close(a, b, atol=1e-10) for a, b in zip(g1, g2)))
except Exception as _e:
    check("2b", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **2c. (discussion)** Count the all-reduces per transformer layer in forward and in backward, and
# the bytes each moves for $b{=}1$, $s{=}8192$, $h{=}12288$ in bf16. Compare that against NVLink
# (~900 GB/s) and InfiniBand (~50 GB/s per direction). Which one can you afford to cross, and how
# does that answer determine the shape of the device mesh?

# %% [markdown]
# ---
# ## Part 3 — Data parallelism
#
# Split the batch, compute local gradients, combine. The only subtlety is **average, not sum** —
# each rank's loss is already a mean over its own micro-batch. Getting this wrong multiplies your
# effective learning rate by the number of ranks, which looks exactly like a tuning problem.

# %%
def data_parallel_grad(model_fn, params, x, y, dp: int):
    """Split the batch across `dp` ranks, return the combined gradient."""
    # TODO
    raise NotImplementedError


def accumulated_grad(model_fn, params, x, y, micro_batches: int):
    """Same gradient as the full batch, with 1/k the activation memory."""
    # TODO — remember to scale each micro-batch's loss.
    raise NotImplementedError


def ring_allreduce_bytes(n_bytes: float, n_devices: int) -> float:
    """Bytes sent per device by a ring all-reduce."""
    # TODO
    raise NotImplementedError


def model_fn(ps, xi, yi):
    W1, W2 = ps
    return F.mse_loss(torch.tanh(xi @ W1) @ W2, yi)


xb = torch.randn(64, 16, dtype=torch.double)
yb = torch.randn(64, 16, dtype=torch.double)
params = [torch.randn(16, 32, dtype=torch.double), torch.randn(32, 16, dtype=torch.double)]

try:
    ps = [p.detach().clone().requires_grad_(True) for p in params]
    full = torch.autograd.grad(model_fn(ps, xb, yb), ps)

    for dp in (2, 4, 8):
        g = data_parallel_grad(model_fn, params, xb, yb, dp)
        check(f"3a dp={dp} equals the full-batch gradient",
              all(close(a, b, atol=1e-12) for a, b in zip(full, g)))
    for k in (2, 4, 8):
        g = accumulated_grad(model_fn, params, xb, yb, k)
        check(f"3b accumulation k={k} equals the full-batch gradient",
              all(close(a, b, atol=1e-12) for a, b in zip(full, g)))

    check("3c ring all-reduce moves < 2x the tensor, at any n",
          ring_allreduce_bytes(1e9, 1024) < 2e9,
          f"{ring_allreduce_bytes(1e9, 1024) / 1e9:.3f} GB/device at n=1024")
    check("3c and barely grows with n",
          ring_allreduce_bytes(1e9, 1024) / ring_allreduce_bytes(1e9, 8) < 1.15)
except Exception as _e:
    check("part 3", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **3d. (discussion)** Gradient accumulation and data parallelism produce identical gradients.
# Given that, when would you choose each? Name the resource each one spends.

# %% [markdown]
# ---
# ## Part 4 — Pipeline parallelism
#
# Rather than quoting $\frac{p-1}{m+p-1}$, **derive it**: simulate the schedule and measure.
#
# One time unit = one forward or one backward of one micro-batch on one stage. Dependencies:
#
# - `F(i, j)` needs `F(i, j-1)`
# - `B(i, p-1)` needs `F(i, p-1)`
# - `B(i, j)` needs `B(i, j+1)`
#
# Each stage runs its own op list in order, one op per time unit. GPipe's order is all forwards
# then all backwards; 1F1B alternates after a warm-up of $p-1-j$ forwards on stage $j$.
#
# Measure two things: the makespan (which gives the bubble) and the **peak number of stashed
# activations** per stage. The second is why 1F1B exists — it has the *same* bubble and a much
# smaller memory footprint.

# %%
def simulate_pipeline(p: int, m: int, order: str = "gpipe"):
    """Return (makespan, bubble_fraction, peak_stashed_activations).

    bubble_fraction = (makespan - 2m) / makespan, since 2m is the work one stage
    would do with no idling.
    """
    # TODO
    raise NotImplementedError


def gpipe_bubble_formula(p: int, m: int) -> float:
    # TODO
    raise NotImplementedError


try:
    print(f"{'p':>3} {'m':>4} {'makespan':>9} {'bubble':>8} {'formula':>8} {'peak acts':>10}")
    for p, m in [(4, 4), (4, 8), (8, 8), (8, 32), (2, 16)]:
        span, bubble, peak = simulate_pipeline(p, m, "gpipe")
        print(f"{p:>3} {m:>4} {span:>9} {bubble:>8.4f} {gpipe_bubble_formula(p, m):>8.4f} {peak:>10}")
        check(f"4a GPipe p={p} m={m} matches the formula",
              close(bubble, gpipe_bubble_formula(p, m), atol=1e-9))
        check(f"4a GPipe p={p} m={m} stashes m activations", peak == m)

    print()
    for p, m in [(4, 8), (8, 32)]:
        _, bg, pg = simulate_pipeline(p, m, "gpipe")
        _, b1, p1 = simulate_pipeline(p, m, "1f1b")
        print(f"p={p} m={m}:  GPipe bubble {bg:.4f} peak {pg}   |   1F1B bubble {b1:.4f} peak {p1}")
        check(f"4b 1F1B p={p} has the same bubble", close(b1, bg, atol=1e-9))
        check(f"4b 1F1B p={p} stashes at most p", p1 <= p and p1 < pg)

    check("4c more micro-batches shrink the bubble",
          simulate_pipeline(8, 64)[1] < simulate_pipeline(8, 8)[1] / 3)
except Exception as _e:
    check("part 4", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **4d. (discussion)** Interleaved 1F1B splits each device's layers into $v$ non-contiguous chunks
# and reaches a bubble of $\frac{p-1}{v(m+p-1)}$. What does it cost, and why is $v$ not simply set
# very large? Extend your simulator to check the formula if you want the stronger version of this
# exercise.

# %% [markdown]
# ---
# ## Part 5 — Activation checkpointing
#
# The claim: recomputing the interior of each layer during backward cuts stored activations
# dramatically for roughly one extra forward pass. Measure both halves — and confirm the gradients
# are unchanged, because a memory optimisation that changes the answer is not an optimisation.
#
# `peak_activation_bytes` below walks the autograd graph counting `_saved_*` tensors, which is a
# genuinely useful debugging tool in its own right.

# %%
class DeepMLP(nn.Module):
    def __init__(self, depth=24, width=512):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])

    def forward(self, x, checkpoint: bool = False):
        for layer in self.layers:
            if checkpoint:
                # TODO: use torch.utils.checkpoint.checkpoint(..., use_reentrant=False)
                raise NotImplementedError
            else:
                x = torch.tanh(layer(x))
        return x


def peak_activation_bytes(model, x, checkpoint: bool) -> int:
    """Bytes autograd is keeping alive, by walking the graph's saved tensors."""
    y = model(x, checkpoint=checkpoint)
    loss = y.square().mean()
    seen, total = set(), 0
    stack = [loss.grad_fn]
    while stack:
        fn = stack.pop()
        if fn is None or id(fn) in seen:
            continue
        seen.add(id(fn))
        for attr in dir(fn):
            if attr.startswith("_saved_"):
                t = getattr(fn, attr, None)
                if torch.is_tensor(t) and id(t) not in seen:
                    seen.add(id(t))
                    total += t.numel() * t.element_size()
        stack.extend(nxt for nxt, _ in fn.next_functions)
    return total


try:
    import time

    torch.manual_seed(0)
    model = DeepMLP()
    xin = torch.randn(32, 512, requires_grad=True)

    plain = peak_activation_bytes(model, xin, False)
    ckpt = peak_activation_bytes(model, xin, True)
    print(f"stored activations:  plain {plain / 1e6:.2f} MB   checkpointed {ckpt / 1e6:.2f} MB")
    check("5 checkpointing cuts stored activations", ckpt < plain / 2)

    g1 = torch.autograd.grad(model(xin, False).square().mean(), xin)[0]
    g2 = torch.autograd.grad(model(xin, True).square().mean(), xin)[0]
    check("5 gradients are unchanged", close(g1, g2, atol=1e-6))

    def timed(flag):
        t0 = time.perf_counter()
        for _ in range(10):
            torch.autograd.grad(model(xin, flag).square().mean(), xin)
        return (time.perf_counter() - t0) / 10 * 1e3

    print(f"time per fwd+bwd:  plain {timed(False):.1f} ms   checkpointed {timed(True):.1f} ms")
except Exception as _e:
    check("part 5", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# ---
# ## Revisiting the spine
#
# Attention so far: tensor code (00) → derived gradient (01) → implemented from scratch (02) →
# debugged (03) → **sharded across devices** (04). Module 05 keeps the same block and changes the
# objective from throughput to latency.
#
# **Stretch goals:**
#
# 1. Extend the pipeline simulator to interleaved 1F1B and verify $\frac{p-1}{v(m+p-1)}$.
# 2. Add a communication cost to the simulator: give each stage boundary a transfer time $c$ and
#    find the $m$ at which communication, not the bubble, becomes the bottleneck.
# 3. Implement ZeRO-3 in simulation: shard the parameters across ranks, all-gather per layer in
#    forward, and confirm the resulting gradients match the unsharded model.
# 4. Write the memory calculator as a CLI that takes a model config and a device count and prints
#    a feasible parallelism configuration — that, roughly, is what you would be asked to design.
