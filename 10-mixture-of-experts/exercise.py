# %% [markdown]
# # Exercise 10 — Mixture of Experts
#
# A mixture of experts is a small idea with four large consequences, and every one of them is
# measurable on a laptop. You will build the layer, break it, fix it, shard it, and then work
# out why it decodes slowly.
#
# Six parts:
#
# 1. **The budget** — count Mixtral's parameters and reproduce its published 46.7 B / 12.9 B.
# 2. **The layer** — a router, top-$k$ dispatch, and the gate multiply that trains the router.
# 3. **Collapse** — reproduce routing collapse, then reverse it with the auxiliary loss.
# 4. **Capacity** — token dropping, and expert-choice routing.
# 5. **Expert parallelism** — the all-to-all, simulated, checked for identical gradients.
# 6. **Inference** — why a bigger batch stops helping.
#
# Everything runs on a CPU. Part 3 trains for about 25 seconds; nothing else takes a second.

# %%
import sys
import pathlib

_here = pathlib.Path.cwd()
for _c in (_here, *_here.parents):
    if (_c / "tools").is_dir():
        sys.path.insert(0, str(_c / "tools"))
        break

import torch
import torch.nn as nn
import torch.nn.functional as F
from checks import check, report, close

torch.manual_seed(0)

MIXTRAL = dict(h=4096, layers=32, vocab=32000, ffn=14336, experts=8, top_k=2,
               q_heads=32, kv_heads=8)

# %% [markdown]
# ---
# ## Part 1 — The budget
#
# **1a.** The whole argument for MoE lives in the gap between two numbers: how many parameters the
# model stores, and how many it uses per token. The experts are multiplied by `experts` in the
# first and by `top_k` in the second. Everything else is counted once.
#
# Count these pieces per layer:
#
# - attention: the query and output matrices are $h \times h$; the key and value matrices are
#   $h \times n_{kv} d_{head}$, where $d_{head} = h / n_q$;
# - one expert: three matrices of $h \times \text{ffn}$, because the activation is SwiGLU;
# - the router: $h \times E$.
#
# Then add the embedding and the output head, which are $\text{vocab} \times h$ each.

# %%
def moe_params(cfg: dict) -> dict:
    """Total and per-token-active parameter counts.

    Return a dict with at least 'total', 'active' and 'ratio'.
    """
    # TODO
    raise NotImplementedError


try:
    p = moe_params(MIXTRAL)
    print(f"   attention per layer : {p['attn']/1e6:8.2f} M")
    print(f"   one expert          : {p['expert']/1e6:8.2f} M")
    print(f"   total               : {p['total']/1e9:8.2f} B   (published 46.7)")
    print(f"   active per token    : {p['active']/1e9:8.2f} B   (published 12.9)")
    check("1 total matches Mixtral's published 46.7B", close(p["total"] / 1e9, 46.7, atol=0.05),
          f"{p['total']/1e9:.2f} B")
    check("1 active matches the published 12.9B", close(p["active"] / 1e9, 12.9, atol=0.05),
          f"{p['active']/1e9:.2f} B")

    dense = moe_params({**MIXTRAL, "experts": 1, "top_k": 1})
    print(f"\n   dense model of the same shape: {dense['total']/1e9:.2f} B")
    print(f"   MoE buys {p['active']/dense['active']:.2f}x its compute "
          f"for {p['total']/dense['total']:.2f}x its memory")
    check("1 a dense model has no gap between total and active",
          close(dense["total"], dense["active"]))
    check("1 the memory cost exceeds the compute gain",
          p["total"] / dense["total"] > p["active"] / dense["active"])
except Exception as _e:
    check("1", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **Question.** Which of the two numbers sets the step time during training, and which one decides
# how many GPUs you need to serve the model? Answer in a sentence.

# %% [markdown]
# ---
# ## Part 2 — The layer
#
# **2a.** Write the forward pass. Given `x` of shape `(T, d)`, a flat list of tokens:
#
# 1. score every expert with the router, and softmax over the experts;
# 2. keep the top $k$ with `topk`, which returns both gates and indices;
# 3. renormalise the kept gates so they sum to 1;
# 4. for each expert, gather its tokens, run them through, and add the gated result back.
#
# Two traps. Several tokens land on the same expert at once, so accumulate with `index_add_`
# rather than plain indexed assignment. And an expert may receive no tokens at all in a given
# batch, so handle the empty case.
#
# The `use_gate` flag exists for 2b. When it is False, add the expert's output without multiplying
# by the gate.

# %%
class MoELayer(nn.Module):
    """A sparse feed-forward layer: a router, E experts, top-k of them per token."""

    def __init__(self, d: int, n_experts: int, top_k: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or 4 * d
        self.n_experts, self.top_k = n_experts, top_k
        # The bias starts at zero and is rarely used in practice, but it is the
        # cleanest handle for part 3's head start, and DeepSeek-V3 balances its
        # experts by adjusting exactly this vector.
        self.router = nn.Linear(d, n_experts, bias=True)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
            for _ in range(n_experts))

    def forward(self, x: torch.Tensor, use_gate: bool = True):
        """Return (output, info).

        `info` must carry at least: 'idx' (T, k), 'gate' (T, k), 'probs' (T, E),
        'f' (E,) the fraction of token-expert pairs each expert received, and
        'P' (E,) the mean router probability per expert.
        """
        # TODO
        raise NotImplementedError


def load_balance_loss(f: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Switch Transformer's auxiliary loss: E * sum_i f_i * P_i.

    Remember which of the two factors carries a gradient, and detach the other.
    """
    # TODO
    raise NotImplementedError


try:
    d, E, k, T = 32, 4, 2, 64
    layer = MoELayer(d, E, k, hidden=64)
    x = torch.randn(T, d)
    out, info = layer(x)
    print(f"   input {tuple(x.shape)} -> output {tuple(out.shape)}, {E} experts, top-{k}")
    check("2 output keeps the input's shape", out.shape == x.shape)
    check("2 each token uses exactly k experts", info["idx"].shape == (T, k))
    check("2 the kept gates sum to 1", close(info["gate"].sum(-1), torch.ones(T)))
    check("2 the load fractions sum to 1", close(info["f"].sum(), 1.0))

    # identical experts: the sparse layer must equal the dense one it imitates
    same = MoELayer(d, E, k, hidden=64)
    with torch.no_grad():
        for e in same.experts[1:]:
            for a, b in zip(e.parameters(), same.experts[0].parameters()):
                a.copy_(b)
    sparse_out, _ = same(x)
    dense_out = same.experts[0](x)
    check("2 identical experts reduce to a plain MLP", close(sparse_out, dense_out, atol=1e-5),
          f"max difference {(sparse_out - dense_out).abs().max():.2e}")
except Exception as _e:
    check("2a", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **2b. The gate multiply.** `topk` returns an index, and an index has no derivative. So the only
# way a loss can reach the router's weights is through the gate value that multiplies the expert's
# output.
#
# Run a backward pass with and without that multiply, and look at `router.weight.grad`. The second
# case does not produce a small gradient. It produces no gradient at all.

# %%
try:
    layer.zero_grad(set_to_none=True)
    layer(x)[0].square().sum().backward()
    g = layer.router.weight.grad
    with_gate = None if g is None else g.abs().sum().item()

    layer.zero_grad(set_to_none=True)
    layer(x, use_gate=False)[0].square().sum().backward()
    without = layer.router.weight.grad

    print(f"   router gradient with the gate multiply   : {with_gate:.4f}")
    print(f"   router gradient without it               : {without}")
    check("2 the gate multiply is what trains the router", with_gate and with_gate > 1e-3)
    check("2 without it the router is not in the graph at all", without is None)
except Exception as _e:
    check("2b", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# ---
# ## Part 3 — Collapse
#
# **3a.** Nothing in a task loss asks the router to use all its experts. One expert that works is
# enough, and an expert that is briefly better attracts more tokens, trains faster, and becomes
# more attractive still.
#
# Train a top-1 layer on a fixed random task for 1,500 steps, giving one expert a head start. Add
# `alpha * load_balance_loss(...)` to the task loss, so the same function covers both runs.
#
# Give the head start as a **bias** on the router, not a perturbation of its weight matrix. A bias
# favours one expert for every token, which is what makes the feedback loop run. Perturbing the
# weight rotates the router instead, and the layer will not collapse.
#
# This cell takes about 25 seconds.

# %%
def train_moe(alpha: float, head_start: float = 2.0, steps: int = 1500,
              d: int = 64, n_experts: int = 8, batch: int = 256, seed: int = 1,
              trace_every: int = 300):
    """Train a top-1 MoE; return (final load fraction, trace of [(step, busiest share)])."""
    # TODO
    raise NotImplementedError


try:
    print(f"   8 experts, top-1. A balanced layer would give each {100/8:.1f}%.\n")
    results = {}
    for alpha in (0.0, 0.01):
        f, trace = train_moe(alpha)
        results[alpha] = f
        share = " ".join(f"{v*100:5.1f}" for v in f.tolist())
        print(f"   alpha = {alpha}")
        print(f"     final share %   : {share}")
        print(f"     experts alive   : {int((f > 0.01).sum())}/8     busiest {f.max()*100:.1f}%")
        print(f"     busiest at step : " + "  ".join(f"t{t}:{m*100:.0f}%" for t, m in trace))
    check("3 without the balance loss the layer collapses onto one expert",
          results[0.0].max() > 0.9, f"busiest holds {results[0.0].max()*100:.1f}%")
    check("3 the balance loss pulls it back apart",
          results[0.01].max() < 0.5, f"busiest holds {results[0.01].max()*100:.1f}%")
    check("3 it revives experts the collapsed run had killed",
          int((results[0.01] > 0.01).sum()) > int((results[0.0] > 0.01).sum()))
except Exception as _e:
    check("3", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **Question.** The balanced run starts at 94% on one expert, the same as the collapsed run, and
# recovers. Which factor in $\alpha E \sum_i f_i P_i$ makes recovery possible for an expert that
# currently receives no tokens at all?

# %% [markdown]
# ---
# ## Part 4 — Capacity, dropping and expert choice
#
# **4a.** Buffers are allocated before the batch is routed, so each expert gets a fixed number of
# slots: `capacity_factor * T * k / E`. Tokens arriving at a full expert are dropped, meaning they
# are not computed and pass through the residual unchanged.
#
# Return a boolean mask of the same shape as `idx`, plus the fraction dropped. Serve tokens first
# come, first served.

# %%
def apply_capacity(idx: torch.Tensor, n_experts: int, capacity_factor: float):
    """Return (keep mask of idx's shape, fraction of token-expert pairs dropped)."""
    # TODO
    raise NotImplementedError


# %% [markdown]
# **4b. Expert choice.** Invert the selection: instead of each token taking its best $k$ experts,
# each expert takes its best `capacity` tokens. Balance is then exact by construction, with no
# auxiliary loss. The cost is that a token may get any number of experts, including none.
#
# Given `probs` of shape `(T, E)`, return a `(E, capacity)` tensor of token indices.

# %%
def expert_choice(probs: torch.Tensor, capacity: int):
    """Each expert's own top-`capacity` tokens, as indices of shape (E, capacity)."""
    # TODO
    raise NotImplementedError


try:
    torch.manual_seed(0)
    skewed = MoELayer(32, 8, 2, hidden=64)
    with torch.no_grad():
        skewed.router.weight[0] += 1.5                      # one popular expert
    _, sk = skewed(torch.randn(512, 32))
    print(f"   a skewed router's load: " + " ".join(f"{v*100:.0f}%" for v in sk['f'].tolist()))
    print(f"\n   {'capacity factor':>16}{'tokens dropped':>16}")
    drops = {}
    for cf in (1.0, 1.25, 1.5, 2.0):
        _, dropped = apply_capacity(sk["idx"], 8, cf)
        drops[cf] = dropped
        print(f"   {cf:>16.2f}{dropped*100:>15.1f}%")
    check("4 a bigger buffer drops fewer tokens",
          drops[1.0] > drops[1.25] > drops[2.0] or drops[2.0] == 0.0)
    check("4 capacity factor 2.0 drops almost nothing", drops[2.0] < 0.02,
          f"{drops[2.0]*100:.1f}%")

    # perfectly balanced routing: token t takes experts t and t+1, round the ring
    t_idx = torch.arange(512)
    even = torch.stack([t_idx % 8, (t_idx + 1) % 8], dim=-1)
    _, balanced = apply_capacity(even, 8, 1.0)
    check("4 perfectly balanced routing loses nothing at capacity 1.0",
          close(balanced, 0.0, atol=1e-9), f"dropped {balanced*100:.1f}%")

    chosen = expert_choice(sk["probs"], capacity=128)
    counts = torch.bincount(chosen.reshape(-1), minlength=512)
    print(f"\n   expert choice, capacity 128: every expert takes exactly {chosen.shape[1]} tokens")
    print(f"   tokens served 0 times: {(counts == 0).sum()}, "
          f"once: {(counts == 1).sum()}, more: {(counts > 1).sum()}")
    check("4 expert choice balances the experts exactly",
          all(row.numel() == 128 for row in chosen))
    check("4 but it leaves some tokens unserved and others over-served",
          (counts == 0).sum() > 0 and (counts > 1).sum() > 0)
except Exception as _e:
    check("4", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **Question.** A dropped token still produces an output. Where does it come from, and why does
# nothing crash?

# %% [markdown]
# ---
# ## Part 5 — Expert parallelism
#
# **5a.** Experts are independent functions, so the natural way to shard an MoE is to put different
# experts on different devices. Tokens then have to travel: out to the device owning their expert
# (one all-to-all), and back again with the result (a second all-to-all).
#
# Simulate it in one process. Nothing is actually distributed; the point is that the arithmetic is
# unchanged, so the gradients must match the unsharded layer exactly. Count the token-expert pairs
# that cross a device boundary while you are there.

# %%
def moe_expert_parallel(layer: MoELayer, x: torch.Tensor, n_devices: int):
    """Run `layer` as if its experts were spread over `n_devices`.

    Return (output, number of token-expert pairs moved).
    """
    # TODO
    raise NotImplementedError


try:
    torch.manual_seed(0)
    layer = MoELayer(32, 8, 2, hidden=64)
    x = torch.randn(256, 32, requires_grad=True)
    ref, _ = layer(x)
    g_ref = torch.autograd.grad(ref.square().sum(), [x] + list(layer.parameters()),
                                allow_unused=True)
    for n_dev in (1, 2, 4, 8):
        shard, moved = moe_expert_parallel(layer, x, n_dev)
        g = torch.autograd.grad(shard.square().sum(), [x] + list(layer.parameters()),
                                allow_unused=True)
        same_out = close(shard, ref, atol=1e-6)
        same_grad = all(a is None or close(a, b, atol=1e-6) for a, b in zip(g, g_ref))
        print(f"   {n_dev} device(s): {moved} token-expert pairs move, "
              f"output identical {same_out}, gradients identical {same_grad}")
        check(f"5 sharding over {n_dev} devices changes nothing", same_out and same_grad)
except Exception as _e:
    check("5", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **5b. On paper.** With 4,096 tokens per step, $k = 2$, hidden size 4,096 and 2-byte values, how
# many bytes cross the network per layer? Multiply by 32 layers. Compare your answer with the table
# in section 5 of the article.

# %% [markdown]
# ---
# ## Part 6 — Why decoding gets worse, not better
#
# **6a.** Module 05 showed that batching raises arithmetic intensity, because a batch shares one
# reading of the weights. That holds only while the tokens use the *same* weights.
#
# A given expert is missed by one token with probability $1 - k/E$, and by all $B$ tokens with that
# raised to the $B$. Write the expected number of distinct experts reached, then the intensity of
# an MoE decode step against a dense model of the same active size.
#
# Work in units of "one expert's parameters", so the units cancel and no model size is needed.

# %%
def experts_touched(batch: int, n_experts: int, top_k: int) -> float:
    """Expected number of distinct experts a balanced batch reaches."""
    # TODO
    raise NotImplementedError


def moe_intensity(batch: int, n_experts: int, top_k: int) -> tuple[float, float]:
    """(MoE, dense-equivalent) arithmetic intensity for one decode step."""
    # TODO
    raise NotImplementedError


try:
    E, k = MIXTRAL["experts"], MIXTRAL["top_k"]
    print(f"   {'batch':>6}{'experts touched':>17}{'MoE':>8}{'dense':>8}{'gap':>7}")
    for B in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        touched = experts_touched(B, E, k)
        moe, dense_i = moe_intensity(B, E, k)
        print(f"   {B:>6}{touched:>17.2f}{moe:>8.1f}{dense_i:>8.1f}{dense_i/moe:>6.1f}x")
    gap = moe_intensity(256, E, k)
    check("6 one token touches exactly k experts",
          close(experts_touched(1, E, k), k, atol=1e-9))
    check("6 a large batch touches all of them",
          close(experts_touched(4096, E, k), E, atol=1e-6))
    check("6 the intensity gap saturates at E/k",
          close(gap[1] / gap[0], E / k, atol=1e-3), f"{gap[1]/gap[0]:.3f} vs {E/k}")
except Exception as _e:
    check("6", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **Question.** The gap stops growing at $E/k$. Say in one sentence what stops it.

# %% [markdown]
# ---
# ## Revisiting the spine
#
# Attention has appeared in every module: tensor code (00), a derived gradient (01), numpy from
# scratch (02), a planted bug (03), a sharded computation (04), a cache (05), soft retrieval (06),
# a paper (07), an alignment target (08), a CUDA kernel (09).
#
# This module is the one place where attention is the part that stays **dense**. The MLP goes
# sparse, attention does not, and the all-to-all sits between them. That asymmetry is the reason
# expert parallelism composes with everything in module 04 rather than replacing it.
#
# **Stretch goals:**
#
# 1. Add a **shared expert**: one expert every token always visits, alongside the top-$k$ choice.
#    Measure whether the routed experts specialise more once common work is handled elsewhere.
# 2. Implement the **router z-loss**, $\frac{1}{T}\sum_t (\log \sum_i e^{\ell_{ti}})^2$, and check
#    that it keeps the router's logits small without changing the load.
# 3. Measure what balance costs. Sweep $\alpha$ over several values and plot final task loss
#    against the busiest expert's share. There is a knee; find it.
# 4. Implement **DeepSeek-V3's auxiliary-loss-free balancing**: drop the extra loss term and
#    instead nudge a per-expert bias up or down after each step, depending on whether that expert
#    was under- or over-subscribed. Compare its final balance with part 3's.
