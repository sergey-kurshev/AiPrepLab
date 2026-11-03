"""Reference solutions for exercise 10 — mixture of experts.  `python solution.py`."""

from __future__ import annotations

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

torch.manual_seed(0)

MIXTRAL = dict(h=4096, layers=32, vocab=32000, ffn=14336, experts=8, top_k=2,
               q_heads=32, kv_heads=8)


# ============================================================ 1. the budget
def moe_params(cfg: dict) -> dict:
    """Total and per-token-active parameter counts for an MoE transformer.

    The whole argument for MoE lives in the gap between these two numbers: the
    experts are multiplied by E in the total and by k in the active count.
    """
    h, d_head = cfg["h"], cfg["h"] // cfg["q_heads"]
    attn = 2 * h * h + 2 * h * cfg["kv_heads"] * d_head     # q, o, then k, v
    expert = 3 * h * cfg["ffn"]                             # gate, up, down
    router = h * cfg["experts"]
    embed = 2 * cfg["vocab"] * h                            # embedding and output head

    per_layer_total = attn + cfg["experts"] * expert + router
    per_layer_active = attn + cfg["top_k"] * expert + router
    total = cfg["layers"] * per_layer_total + embed
    active = cfg["layers"] * per_layer_active + embed
    return dict(attn=attn, expert=expert, router=router,
                total=total, active=active, ratio=total / active)


# ============================================================ 2. the layer
class MoELayer(nn.Module):
    """A sparse feed-forward layer: a router, E experts, top-k of them per token.

    The gate value multiplies the expert's output. That is not cosmetic. `topk`
    hands back an index, and an index has no gradient, so the multiply is the
    only path by which a loss can reach the router's weights at all.
    """

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
        """x is (T, d) -- a flat list of tokens, which is how routing sees them.

        Returns (output, info), where info carries the routing decisions that
        parts 3 and 4 need: the chosen indices, the gates, the measured load
        fraction f, and the mean router probability P.
        """
        probs = self.router(x).softmax(-1)                  # (T, E)
        gate, idx = probs.topk(self.top_k, dim=-1)          # (T, k) each
        gate = gate / gate.sum(-1, keepdim=True)            # renormalise over the k kept

        out = torch.zeros_like(x)
        for e in range(self.n_experts):
            tok, slot = (idx == e).nonzero(as_tuple=True)   # which tokens, in which slot
            if tok.numel() == 0:
                continue
            y = self.experts[e](x[tok])
            w = gate[tok, slot].unsqueeze(-1) if use_gate else 1.0
            out.index_add_(0, tok, w * y)

        f = torch.bincount(idx.reshape(-1), minlength=self.n_experts).float() / idx.numel()
        return out, dict(idx=idx, gate=gate, probs=probs, f=f, P=probs.mean(0))


def load_balance_loss(f: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Switch Transformer's auxiliary loss: E * sum_i f_i * P_i.

    f is a token count and carries no gradient; P does. Minimising the product
    therefore pushes probability away from experts that are already busy. The
    factor E makes a perfectly balanced layer score 1 for any E.
    """
    return f.numel() * (f.detach() * P).sum()


# ============================================================ 3. collapse
def train_moe(alpha: float, head_start: float = 2.0, steps: int = 1500,
              d: int = 64, n_experts: int = 8, batch: int = 256, seed: int = 1,
              trace_every: int = 300):
    """Train a top-1 MoE on a fixed random task; return its final expert load.

    `head_start` adds a bias to one expert's router logit, which is the small
    asymmetry the feedback loop needs in order to run away.
    """
    torch.manual_seed(seed)
    layer = MoELayer(d, n_experts, top_k=1)
    with torch.no_grad():
        layer.router.bias.zero_()
        layer.router.bias[0] = head_start                   # favour expert 0 for every token
    opt = torch.optim.Adam(layer.parameters(), lr=1e-3)
    W = torch.randn(d, d) * 0.3

    trace = []
    for t in range(steps):
        x = torch.randn(batch, d)
        y = torch.tanh(x @ W)
        out, info = layer(x)
        loss = (out - y).square().mean() + alpha * load_balance_loss(info["f"], info["P"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if t % trace_every == 0:
            trace.append((t, info["f"].max().item()))

    with torch.no_grad():
        _, info = layer(torch.randn(4096, d))
    return info["f"], trace


# ============================================================ 4. capacity
def apply_capacity(idx: torch.Tensor, n_experts: int, capacity_factor: float):
    """Drop tokens beyond each expert's fixed buffer; return (keep mask, dropped fraction).

    Capacity is a fixed buffer size per expert, decided before the batch arrives:
    capacity = capacity_factor * (tokens * k) / E. Tokens arriving at a full
    expert are not computed at all, and pass through the residual unchanged.
    """
    T, k = idx.shape
    capacity = int(capacity_factor * T * k / n_experts)
    keep = torch.zeros_like(idx, dtype=torch.bool)
    for e in range(n_experts):
        tok, slot = (idx == e).nonzero(as_tuple=True)
        head = tok[:capacity], slot[:capacity]              # first come, first served
        keep[head] = True
    return keep, 1.0 - keep.float().mean().item()


def expert_choice(probs: torch.Tensor, capacity: int):
    """Expert-choice routing: each expert takes its own top-`capacity` tokens.

    Flipping the argmax makes the load exactly equal by construction, so no
    auxiliary loss is needed. The price is that a token may get any number of
    experts, including none.
    """
    _, chosen = probs.t().topk(capacity, dim=-1)            # (E, capacity) token indices
    return chosen


# ============================================================ 5. expert parallelism
def moe_expert_parallel(layer: MoELayer, x: torch.Tensor, n_devices: int):
    """Run the layer as if its experts were spread over `n_devices`.

    Each device owns E / n_devices experts. Tokens are sent to the device that
    owns their expert (one all-to-all), computed there, and sent back (a second
    all-to-all). Nothing here is distributed for real; the point is that the
    arithmetic is unchanged, so the gradients must match exactly.
    """
    per_device = layer.n_experts // n_devices
    probs = layer.router(x).softmax(-1)
    gate, idx = probs.topk(layer.top_k, dim=-1)
    gate = gate / gate.sum(-1, keepdim=True)

    out = torch.zeros_like(x)
    moved = 0
    for dev in range(n_devices):
        owned = range(dev * per_device, (dev + 1) * per_device)
        for e in owned:
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            moved += tok.numel()                            # tokens crossing into this device
            y = layer.experts[e](x[tok])                    # computed "on device dev"
            out.index_add_(0, tok, gate[tok, slot].unsqueeze(-1) * y)
    return out, moved


# ============================================================ 6. inference
def experts_touched(batch: int, n_experts: int, top_k: int) -> float:
    """Expected number of distinct experts a balanced batch reaches.

    Each token picks k of E, so one expert is missed by one token with
    probability 1 - k/E, and by all of them with that raised to the batch size.
    """
    return n_experts * (1.0 - (1.0 - top_k / n_experts) ** batch)


def moe_intensity(batch: int, n_experts: int, top_k: int) -> tuple[float, float]:
    """(MoE, dense-equivalent) arithmetic intensity for one decode step.

    Both read expert weights and do the same arithmetic. The MoE reads every
    expert it touches; the dense model of equal active size reads exactly k
    experts' worth. Units of 'one expert's parameters' cancel throughout.
    """
    touched = experts_touched(batch, n_experts, top_k)
    return batch * top_k / touched, float(batch)


# ============================================================ main
def main() -> int:
    # ---------------------------------------------------------------- part 1
    print("=" * 72, "\n1. the budget\n")
    p = moe_params(MIXTRAL)
    print(f"   attention per layer : {p['attn']/1e6:8.2f} M")
    print(f"   one expert          : {p['expert']/1e6:8.2f} M")
    print(f"   total               : {p['total']/1e9:8.2f} B   (published 46.7)")
    print(f"   active per token    : {p['active']/1e9:8.2f} B   (published 12.9)")
    print(f"   ratio               : {p['ratio']:8.2f} x")

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
    report()

    # ---------------------------------------------------------------- part 2
    print("=" * 72, "\n2. the layer\n")
    d, E, k, T = 32, 4, 2, 64
    layer = MoELayer(d, E, k, hidden=64)
    x = torch.randn(T, d)
    out, info = layer(x)
    print(f"   input {tuple(x.shape)} -> output {tuple(out.shape)}, "
          f"{E} experts, top-{k}")
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

    # the gate multiply is the router's only gradient path
    layer.zero_grad(set_to_none=True)
    layer(x)[0].square().sum().backward()
    g = layer.router.weight.grad
    with_gate = None if g is None else g.abs().sum().item()
    layer.zero_grad(set_to_none=True)
    layer(x, use_gate=False)[0].square().sum().backward()
    without = layer.router.weight.grad
    print(f"\n   router gradient with the gate multiply   : {with_gate:.4f}")
    print(f"   router gradient without it               : {without}")
    check("2 the gate multiply is what trains the router", with_gate and with_gate > 1e-3)
    check("2 without it the router is not in the graph at all", without is None)
    report()

    # ---------------------------------------------------------------- part 3
    print("=" * 72, "\n3. does the router collapse?\n")
    print(f"   8 experts, top-1. A balanced layer would give each {100/8:.1f}%.\n")
    results = {}
    for alpha in (0.0, 0.01):
        f, trace = train_moe(alpha)
        results[alpha] = f
        share = " ".join(f"{v*100:5.1f}" for v in f.tolist())
        print(f"   alpha = {alpha}")
        print(f"     final share %   : {share}")
        print(f"     experts alive   : {int((f > 0.01).sum())}/8     busiest {f.max()*100:.1f}%")
        print(f"     busiest at step : " +
              "  ".join(f"t{t}:{m*100:.0f}%" for t, m in trace))
    check("3 without the balance loss the layer collapses onto one expert",
          results[0.0].max() > 0.9, f"busiest holds {results[0.0].max()*100:.1f}%")
    check("3 the balance loss pulls it back apart",
          results[0.01].max() < 0.5, f"busiest holds {results[0.01].max()*100:.1f}%")
    check("3 it revives experts the collapsed run had killed",
          int((results[0.01] > 0.01).sum()) > int((results[0.0] > 0.01).sum()))
    report()

    # ---------------------------------------------------------------- part 4
    print("=" * 72, "\n4. capacity, dropping and expert choice\n")
    torch.manual_seed(0)
    skewed = MoELayer(32, 8, 2, hidden=64)
    with torch.no_grad():
        skewed.router.weight[0] += 1.5                      # one popular expert
    _, sk = skewed(torch.randn(512, 32))
    print(f"   a skewed router's load: " +
          " ".join(f"{v*100:.0f}%" for v in sk['f'].tolist()))
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
    print(f"\n   expert choice, capacity 128: every expert takes exactly "
          f"{chosen.shape[1]} tokens")
    print(f"   tokens served 0 times: {(counts == 0).sum()}, "
          f"once: {(counts == 1).sum()}, more: {(counts > 1).sum()}")
    check("4 expert choice balances the experts exactly",
          all(row.numel() == 128 for row in chosen))
    check("4 but it leaves some tokens unserved and others over-served",
          (counts == 0).sum() > 0 and (counts > 1).sum() > 0)
    report()

    # ---------------------------------------------------------------- part 5
    print("=" * 72, "\n5. expert parallelism\n")
    torch.manual_seed(0)
    layer = MoELayer(32, 8, 2, hidden=64)
    x = torch.randn(256, 32, requires_grad=True)

    ref, _ = layer(x)
    g_ref = torch.autograd.grad(ref.square().sum(), [x] + list(layer.parameters()),
                                retain_graph=False, allow_unused=True)

    for n_dev in (1, 2, 4, 8):
        shard, moved = moe_expert_parallel(layer, x, n_dev)
        g = torch.autograd.grad(shard.square().sum(), [x] + list(layer.parameters()),
                                allow_unused=True)
        same_out = close(shard, ref, atol=1e-6)
        same_grad = all(a is None or close(a, b, atol=1e-6) for a, b in zip(g, g_ref))
        print(f"   {n_dev} device(s): {moved} token-expert pairs move, "
              f"output identical {same_out}, gradients identical {same_grad}")
        check(f"5 sharding over {n_dev} devices changes nothing", same_out and same_grad)
    print("\n   Each token-expert pair crosses the network twice: once out, once back.")
    report()

    # ---------------------------------------------------------------- part 6
    print("=" * 72, "\n6. why decoding gets worse, not better\n")
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
    print(f"\n   The gap stops at E/k = {E/k:.0f}x. An MoE decodes with {E/k:.0f} times")
    print("   less arithmetic per byte read than a dense model of equal active size.")
    return 0 if report() else 1


if __name__ == "__main__":
    raise SystemExit(main())
