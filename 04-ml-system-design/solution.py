"""Reference solutions for exercise 04 — parallelism, verified in simulation.

Run with `python solution.py`.

Everything runs in one process on the CPU. That is the point: a sharded computation
either produces bit-comparable gradients to the unsharded one or it does not, and
that question is answerable without a cluster. The bugs that cost real money at
scale — a missing all-reduce, a partition along the wrong axis, an off-by-one in a
pipeline schedule — are all visible here.
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

GB = 1024 ** 3


# ============================================================ 1. memory budget
def training_memory(n_params: int, precision: str = "mixed", optimizer: str = "adam") -> dict:
    """Bytes of *static* state per model, by category.

    Mixed precision does NOT reduce optimiser memory: you still need fp32 master
    weights, because a bf16 weight cannot absorb an update of relative size 1e-4.
    It buys speed and activation memory.
    """
    if precision == "mixed":
        weights, grads, master = 2, 2, 4
    elif precision == "fp32":
        weights, grads, master = 4, 4, 0
    else:
        raise ValueError(precision)
    opt = {"adam": 8, "sgd_momentum": 4, "sgd": 0}[optimizer]
    parts = {"weights": weights, "gradients": grads, "master_weights": master,
             "optimizer": opt}
    return {k: v * n_params for k, v in parts.items()} | {
        "total": sum(parts.values()) * n_params,
        "bytes_per_param": sum(parts.values()),
    }


def zero_memory(n_params: int, stage: int, dp_size: int) -> float:
    """Per-device bytes under ZeRO stage 0-3 with a data-parallel group of `dp_size`."""
    w, g, o = 2.0, 2.0, 12.0          # bf16 weights, bf16 grads, fp32 master+Adam
    if stage >= 1:
        o /= dp_size
    if stage >= 2:
        g /= dp_size
    if stage >= 3:
        w /= dp_size
    return (w + g + o) * n_params


def activation_memory(layers: int, hidden: int, seq: int, micro_batch: int,
                      heads: int, recompute: str = "none") -> float:
    """Korthikanti et al. (2022) eq. 2, in bytes, for bf16 activations.

    The 5*a*s/h term is the attention score matrix: quadratic in sequence length,
    and the reason FlashAttention exists.
    """
    per_layer = seq * micro_batch * hidden * (34 + 5 * heads * seq / hidden)
    if recompute == "full":
        # Only the layer inputs are stored; everything else is recomputed.
        return seq * micro_batch * hidden * layers * 2
    if recompute == "selective":
        # Drop the quadratic term: recompute the attention block only.
        per_layer = seq * micro_batch * hidden * 34
    return per_layer * layers


def training_flops(n_params: int, n_tokens: int, recompute: bool = False) -> float:
    """C = 6ND (8ND with full activation recomputation)."""
    return (8 if recompute else 6) * n_params * n_tokens


def chinchilla_optimal(compute_flops: float) -> tuple[float, float]:
    """(N, D) for a compute budget, at the ~20 tokens/param Chinchilla ratio."""
    # C = 6ND and D = 20N  =>  C = 120 N^2.
    n = math.sqrt(compute_flops / 120.0)
    return n, 20.0 * n


# ============================================================ 2. tensor parallel
def column_parallel_linear(x: torch.Tensor, W: torch.Tensor, tp: int):
    """Y = X W with W split by COLUMNS across `tp` devices.

    Each device produces a slice of the output features. No communication is needed
    if the next operation is elementwise (GELU) — which is exactly why Megatron
    splits the first MLP matrix this way round.
    Returns the per-device shards; the caller decides whether to concatenate.
    """
    shards = W.chunk(tp, dim=1)
    return [x @ Wi for Wi in shards]


def row_parallel_linear(x_shards: list[torch.Tensor], W: torch.Tensor, tp: int):
    """Y = X W with W split by ROWS, consuming an already-sharded input.

    Each device computes a partial sum over its slice of the contraction index, so
    the shards must be SUMMED — one all-reduce. Concatenating instead is the
    classic bug, and it produces a tensor of the wrong shape, so at least it is loud.
    """
    shards = W.chunk(tp, dim=0)
    partials = [xi @ Wi for xi, Wi in zip(x_shards, shards)]
    return all_reduce(partials)


def all_reduce(partials: list[torch.Tensor]) -> torch.Tensor:
    """Simulated all-reduce: every rank ends up with the sum."""
    return torch.stack(partials).sum(0)


def parallel_mlp(x, A, B, tp: int):
    """GELU(X A) B, tensor-parallel: column-parallel A, row-parallel B, one all-reduce."""
    h_shards = [F.gelu(h) for h in column_parallel_linear(x, A, tp)]
    return row_parallel_linear(h_shards, B, tp)


def parallel_attention(x, Wqkv, Wo, n_heads: int, tp: int):
    """Head-parallel attention. Heads are already independent, so the partition
    follows the mathematics: each device owns n_heads/tp heads end to end, and the
    output projection is row-parallel with a single all-reduce."""
    assert n_heads % tp == 0
    B, T, C = x.shape
    dh = C // n_heads
    heads_per = n_heads // tp

    # W_qkv is (C, 3C) laid out as [Q | K | V]; each device needs its slice of each.
    q_w, k_w, v_w = Wqkv.chunk(3, dim=1)
    out_shards = []
    for r in range(tp):
        sl = slice(r * heads_per * dh, (r + 1) * heads_per * dh)
        q = (x @ q_w[:, sl]).view(B, T, heads_per, dh).transpose(1, 2)
        k = (x @ k_w[:, sl]).view(B, T, heads_per, dh).transpose(1, 2)
        v = (x @ v_w[:, sl]).view(B, T, heads_per, dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out_shards.append(o.transpose(1, 2).reshape(B, T, heads_per * dh))
    return row_parallel_linear(out_shards, Wo, tp)


def reference_attention(x, Wqkv, Wo, n_heads: int):
    B, T, C = x.shape
    q, k, v = (t.view(B, T, n_heads, C // n_heads).transpose(1, 2)
               for t in (x @ Wqkv).chunk(3, dim=-1))
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return o.transpose(1, 2).reshape(B, T, C) @ Wo


# ============================================================ 3. data parallel
def data_parallel_grad(model_fn, params, x, y, dp: int):
    """Split the batch, compute local gradients, AVERAGE them.

    Average, not sum: each rank's loss is already a mean over its own micro-batch,
    so summing would scale the gradient by dp. This off-by-a-factor-of-n is the
    most common data-parallel bug and it looks exactly like a bad learning rate.
    """
    xs, ys = x.chunk(dp), y.chunk(dp)
    grads = None
    for xi, yi in zip(xs, ys):
        ps = [p.detach().clone().requires_grad_(True) for p in params]
        loss = model_fn(ps, xi, yi)
        g = torch.autograd.grad(loss, ps)
        grads = list(g) if grads is None else [a + b for a, b in zip(grads, g)]
    return [g / dp for g in grads]


def accumulated_grad(model_fn, params, x, y, micro_batches: int):
    """Gradient accumulation: identical maths, 1/k the activation memory.

    The loss of each micro-batch must be divided by k, or the effective learning
    rate is multiplied by k.
    """
    ps = [p.detach().clone().requires_grad_(True) for p in params]
    for xi, yi in zip(x.chunk(micro_batches), y.chunk(micro_batches)):
        (model_fn(ps, xi, yi) / micro_batches).backward()
    return [p.grad for p in ps]


def ring_allreduce_bytes(n_bytes: float, n_devices: int) -> float:
    """Bytes sent per device by ring all-reduce: 2(n-1)/n * size.

    Note what this does NOT contain: `n`. The volume per device is bounded by 2x the
    tensor size however large the cluster grows, which is why data parallelism scales.
    """
    return 2.0 * (n_devices - 1) / n_devices * n_bytes


# ============================================================ 4. pipeline
def simulate_pipeline(p: int, m: int, order: str = "gpipe"):
    """Discrete-time simulation of a pipeline schedule.

    One time unit = one forward or one backward of one micro-batch on one stage.
    Dependencies: F(i,j) needs F(i,j-1); B(i,p-1) needs F(i,p-1); B(i,j) needs B(i,j+1).
    Each stage executes its own op list in order, one op per time unit.

    Returns (makespan, bubble_fraction, peak_stashed_activations).
    """
    # Per-stage op order.
    queues: list[list[tuple[str, int]]] = []
    for j in range(p):
        if order == "gpipe":
            q = [("F", i) for i in range(m)] + [("B", i) for i in range(m)]
        elif order == "1f1b":
            warmup = min(p - 1 - j, m)          # forwards to run before the first backward
            q = [("F", i) for i in range(warmup)]
            steady = m - warmup
            for i in range(steady):
                q.append(("F", warmup + i))
                q.append(("B", i))
            q += [("B", i) for i in range(steady, m)]
        else:
            raise ValueError(order)
        queues.append(q)

    done: dict[tuple[str, int, int], int] = {}
    idx = [0] * p
    stashed = [0] * p
    peak = 0
    t = 0
    total_ops = p * 2 * m
    while len(done) < total_ops:
        progressed = False
        for j in range(p):
            if idx[j] >= len(queues[j]):
                continue
            kind, i = queues[j][idx[j]]
            if kind == "F":
                ready = j == 0 or done.get(("F", i, j - 1), 1 << 30) < t
            else:
                ready = (done.get(("F", i, j), 1 << 30) < t and
                         (j == p - 1 or done.get(("B", i, j + 1), 1 << 30) < t))
            if ready:
                done[(kind, i, j)] = t
                idx[j] += 1
                stashed[j] += 1 if kind == "F" else -1
                peak = max(peak, max(stashed))
                progressed = True
        if not progressed and len(done) < total_ops and t > 4 * total_ops:
            raise RuntimeError("deadlock in the schedule")
        t += 1

    makespan = t
    ideal = 2 * m                       # the work one stage would do with no bubbles
    return makespan, (makespan - ideal) / makespan, peak


def gpipe_bubble_formula(p: int, m: int) -> float:
    return (p - 1) / (m + p - 1)


# ============================================================ 5. checkpointing
class DeepMLP(nn.Module):
    """A stack deep enough that activation memory is the interesting quantity."""

    def __init__(self, depth: int = 24, width: int = 512):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])

    def forward(self, x, checkpoint: bool = False):
        for layer in self.layers:
            if checkpoint:
                # Recompute this layer's interior during backward instead of storing it.
                x = torch.utils.checkpoint.checkpoint(
                    lambda t, l=layer: torch.tanh(l(t)), x, use_reentrant=False)
            else:
                x = torch.tanh(layer(x))
        return x


def peak_activation_bytes(model: DeepMLP, x: torch.Tensor, checkpoint: bool) -> int:
    """Count the bytes autograd keeps alive, by walking the graph's saved tensors."""
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


# ============================================================ self-test
def main() -> int:
    torch.manual_seed(0)

    # ---- 1. memory arithmetic
    mem = training_memory(100e9)
    check("1 mixed precision + Adam costs 16 bytes/param", mem["bytes_per_param"] == 16,
          f"{mem['total'] / 1e12:.2f} TB for 100B parameters")
    check("1 fp32 + Adam also costs 16", training_memory(1, "fp32")["bytes_per_param"] == 16,
          "mixed precision buys speed and activations, not optimiser memory")
    check("1 plain SGD costs 8", training_memory(1, "mixed", "sgd")["bytes_per_param"] == 8)

    n = 100e9
    z0, z3 = zero_memory(n, 0, 64), zero_memory(n, 3, 64)
    check("1 ZeRO-3 shards everything", close(z3, z0 / 64, rtol=1e-9),
          f"{z0 / 1e12:.2f} TB -> {z3 / 1e9:.1f} GB per device at dp=64")
    check("1 ZeRO-1 saves 12 of the 16 bytes",
          close(zero_memory(n, 1, 1_000_000), 4.0 * n, rtol=1e-3))

    # GPT-3 175B: 96 layers, hidden 12288, 96 heads.
    act = activation_memory(96, 12288, 2048, 1, 96)
    act_sel = activation_memory(96, 12288, 2048, 1, 96, "selective")
    check("1 selective recomputation removes the quadratic term", act_sel < act * 0.75,
          f"{act / GB:.0f} GB -> {act_sel / GB:.0f} GB per micro-batch")
    long = activation_memory(96, 12288, 8192, 1, 96)
    check("1 activations grow superlinearly in sequence length", long / act > 4.5,
          f"4x the sequence costs {long / act:.1f}x the activation memory")

    check("1 GPT-3's training compute is ~3e23 FLOPs",
          2.5e23 < training_flops(175e9, 300e9) < 3.6e23,
          f"{training_flops(175e9, 300e9):.2e}")
    N, D = chinchilla_optimal(training_flops(175e9, 300e9))
    check("1 Chinchilla would have chosen a smaller model", N < 175e9,
          f"optimal for that budget: {N / 1e9:.0f}B params on {D / 1e9:.0f}B tokens")

    # ---- 2. tensor parallelism
    B, T, C, H = 2, 6, 64, 8
    x = torch.randn(B, T, C, dtype=torch.double, requires_grad=True)
    A = torch.randn(C, 4 * C, dtype=torch.double, requires_grad=True)
    Bm = torch.randn(4 * C, C, dtype=torch.double, requires_grad=True)

    ref = F.gelu(x @ A) @ Bm
    for tp in (1, 2, 4, 8):
        got = parallel_mlp(x, A, Bm, tp)
        check(f"2 MLP tp={tp} matches the unsharded forward", close(got, ref, atol=1e-12))

    # ...and the gradients, which is the check that actually matters.
    gref = torch.autograd.grad(ref.square().sum(), [x, A, Bm], retain_graph=True)
    gtp = torch.autograd.grad(parallel_mlp(x, A, Bm, 4).square().sum(), [x, A, Bm])
    check("2 MLP tp=4 matches the unsharded gradients",
          all(close(a, b, atol=1e-10) for a, b in zip(gref, gtp)))

    Wqkv = torch.randn(C, 3 * C, dtype=torch.double, requires_grad=True)
    Wo = torch.randn(C, C, dtype=torch.double, requires_grad=True)
    ra = reference_attention(x, Wqkv, Wo, H)
    for tp in (1, 2, 4):
        check(f"2 head-parallel attention tp={tp}", close(parallel_attention(x, Wqkv, Wo, H, tp), ra, atol=1e-10))
    g1 = torch.autograd.grad(ra.square().sum(), [x, Wqkv, Wo], retain_graph=True)
    g2 = torch.autograd.grad(parallel_attention(x, Wqkv, Wo, H, 4).square().sum(), [x, Wqkv, Wo])
    check("2 head-parallel gradients match", all(close(a, b, atol=1e-10) for a, b in zip(g1, g2)))

    # The bug that motivates the exercise: forget the all-reduce and keep one shard.
    wrong = [xi @ Wi for xi, Wi in zip([F.gelu(h) for h in column_parallel_linear(x, A, 4)],
                                       Bm.chunk(4, dim=0))][0]
    check("2 a missing all-reduce is detectable", not close(wrong, ref, atol=1e-6),
          "one rank's partial sum is not the answer")

    # ---- 3. data parallelism
    def model_fn(ps, xi, yi):
        W1, W2 = ps
        return F.mse_loss(torch.tanh(xi @ W1) @ W2, yi)

    xb = torch.randn(64, 16, dtype=torch.double)
    yb = torch.randn(64, 16, dtype=torch.double)
    params = [torch.randn(16, 32, dtype=torch.double), torch.randn(32, 16, dtype=torch.double)]
    ps = [p.detach().clone().requires_grad_(True) for p in params]
    full = torch.autograd.grad(model_fn(ps, xb, yb), ps)

    for dp in (2, 4, 8):
        g = data_parallel_grad(model_fn, params, xb, yb, dp)
        check(f"3 data-parallel dp={dp} equals the full-batch gradient",
              all(close(a, b, atol=1e-12) for a, b in zip(full, g)))
    for k in (2, 4):
        g = accumulated_grad(model_fn, params, xb, yb, k)
        check(f"3 gradient accumulation k={k} equals the full-batch gradient",
              all(close(a, b, atol=1e-12) for a, b in zip(full, g)))

    summed = [g * 4 for g in data_parallel_grad(model_fn, params, xb, yb, 4)]
    check("3 summing instead of averaging is wrong by exactly dp",
          not close(summed[0], full[0], atol=1e-6) and close(summed[0] / 4, full[0], atol=1e-12))

    bytes_1gb = 1e9
    check("3 ring all-reduce volume is bounded by 2x the tensor",
          ring_allreduce_bytes(bytes_1gb, 1024) < 2 * bytes_1gb,
          f"{ring_allreduce_bytes(bytes_1gb, 1024) / 1e9:.3f} GB per device at n=1024")
    check("3 ...and barely grows with n",
          ring_allreduce_bytes(bytes_1gb, 1024) / ring_allreduce_bytes(bytes_1gb, 8) < 1.15)

    # ---- 4. pipeline
    for p, m in [(4, 4), (4, 8), (8, 8), (8, 32), (2, 16)]:
        span, bubble, peak = simulate_pipeline(p, m, "gpipe")
        check(f"4 GPipe p={p} m={m} matches the bubble formula",
              close(bubble, gpipe_bubble_formula(p, m), atol=1e-9),
              f"simulated {bubble:.4f}, formula {gpipe_bubble_formula(p, m):.4f}")
        check(f"4 GPipe p={p} m={m} stashes m activations", peak == m)

    for p, m in [(4, 8), (8, 32)]:
        _, b_g, peak_g = simulate_pipeline(p, m, "gpipe")
        _, b_1, peak_1 = simulate_pipeline(p, m, "1f1b")
        check(f"4 1F1B p={p} m={m} has the same bubble", close(b_1, b_g, atol=1e-9))
        check(f"4 1F1B p={p} m={m} stashes far fewer activations", peak_1 <= p and peak_1 < peak_g,
              f"{peak_1} vs {peak_g}")

    check("4 more micro-batches shrink the bubble",
          simulate_pipeline(8, 64)[1] < simulate_pipeline(8, 8)[1] / 3,
          f"{simulate_pipeline(8, 8)[1]:.3f} -> {simulate_pipeline(8, 64)[1]:.3f}")

    # ---- 5. checkpointing
    torch.manual_seed(0)
    model = DeepMLP(depth=24, width=512)
    xin = torch.randn(32, 512, requires_grad=True)
    plain = peak_activation_bytes(model, xin, checkpoint=False)
    ckpt = peak_activation_bytes(model, xin, checkpoint=True)
    check("5 checkpointing cuts stored activations", ckpt < plain / 2,
          f"{plain / 1e6:.2f} MB -> {ckpt / 1e6:.2f} MB")

    y1 = model(xin, checkpoint=False)
    g1 = torch.autograd.grad(y1.square().mean(), xin, retain_graph=False)[0]
    y2 = model(xin, checkpoint=True)
    g2 = torch.autograd.grad(y2.square().mean(), xin)[0]
    check("5 checkpointing does not change the gradient", close(g1, g2, atol=1e-6))

    return 0 if report() else 1


if __name__ == "__main__":
    raise SystemExit(main())
