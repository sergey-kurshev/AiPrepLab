"""Reference solution for exercise 03 — the corrected model, plus the bug list.

Run with `python solution.py` to see every check pass on the fixed implementation.
`diff buggy_lm.py solution.py` shows the eight edits (ignore the docstrings).

================================================================================
THE EIGHT BUGS, and the rung of the diagnostic ladder that finds each
================================================================================

1.  `self.blocks = [Block(...) for _ in range(n_layers)]`
    -> `nn.ModuleList([...])`
    A plain list is invisible to the module registry, so the blocks' parameters
    are absent from `.parameters()`, never reach the optimiser, and never move.
    The forward pass works perfectly, which is what makes it nasty.
    FOUND BY: rung 3 (every parameter gets a gradient) — the block parameters
    do not appear in `named_parameters()` at all. Also `.to(device)` would
    silently leave them on the CPU.

2.  `p = scores.softmax(dim=-2)`  ->  `dim=-1`
    Normalising over queries instead of keys. The result is still a non-negative
    tensor of the right shape; the rows just do not sum to one.
    FOUND BY: rung 5 — assert that attention weights sum to 1 along the key axis.

3.  `p = p * allowed` (masking AFTER the softmax)
    -> add an additive -inf mask to the scores BEFORE the softmax.
    Multiplicative masking after the fact leaves the rows unnormalised: the
    surviving weights sum to less than one, so the attention output is silently
    scaled down by a position-dependent factor.
    FOUND BY: rung 5 — the same sum-to-one assertion, applied with a mask.

4.  `qkv.view(B, T, 3, H, dh)` then indexing, vs the buggy
    `qkv.view(B, 3, H, T, dh)` — which interleaves positions into the head axis.
    Both run. The second one shuffles which token's features land in which head.
    FOUND BY: rung 5 with asymmetric dimensions (B != T != H != dh), plus the
    causality invariant, which breaks because positions have been permuted.

5.  Missing `opt.zero_grad()` in the training loop.
    Gradients accumulate across steps, so each update uses a stale running sum
    rather than the current gradient. The model still improves — slowly, noisily,
    and to a worse optimum — which is why this one survives code review.
    FOUND BY: rung 1 (cannot cleanly overfit one batch) and by watching the
    gradient norm grow monotonically across steps.

6.  `F.cross_entropy(F.softmax(logits, -1), targets)`
    -> `F.cross_entropy(logits, targets)`
    `cross_entropy` applies `log_softmax` internally. Feeding it probabilities
    applies the squashing twice: the effective logits are now in [0, 1], the
    achievable confidence is capped, and the gradient is scaled down by the
    softmax Jacobian a second time.
    Note what this bug does NOT do: the initial loss is still ln(V) = 2.773,
    because a near-uniform output is near-uniform after a second softmax too. So
    rung 2 misses it entirely. Its signature is a FLOOR: with every other bug
    fixed, overfitting a single batch stalls at ~1.88 and will not go lower for
    any number of steps, because squashing the logits into [0, 1] caps how
    confident the model is able to be.
    FOUND BY: rung 1 — a loss that plateaus at a specific non-zero value on eight
    examples is a modelling impossibility, not a training-length problem.

7.  `y = seq[:, :-1]` in `get_batch` -> `y = seq[:, 1:]`
    The targets are not shifted, so the model is asked to predict the token it
    was just given. It learns the identity function, loss collapses to ~0, and
    generation produces nonsense.
    FOUND BY: rung 1 in reverse — a loss that goes to zero *too fast* is as
    suspicious as one that does not move. Plus an explicit test on `get_batch`.

8.  `evaluate()` without `model.eval()` and without `torch.no_grad()`.
    Dropout stays active, so validation numbers are noisy and pessimistic and
    change between identical calls; and the graph is retained for every batch,
    which is a memory leak that looks like a model-size problem.
    FOUND BY: calling `evaluate()` twice and getting different answers.
================================================================================
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
    y = seq[:, 1:]          # FIX 7: targets are the NEXT token, not the same one
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
        # FIX 4: split the LAST axis into (3, H, dh) and then move heads forward.
        # `view(B, 3, H, T, dh)` also runs, and quietly permutes positions into heads.
        qkv = qkv.view(B, T, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                       # (B, H, T, dh)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        # FIX 3: additive -inf mask BEFORE the softmax, so the surviving weights renormalise.
        scores = scores.masked_fill(~self.allowed[:T, :T], float("-inf"))
        p = scores.softmax(dim=-1)                             # FIX 2: over keys, not queries
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
        # FIX 1: ModuleList, not list — a plain list is invisible to the registry.
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, max_len, dropout) for _ in range(n_layers)]
        )
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
        # FIX 6: cross_entropy takes LOGITS. It applies log_softmax itself.
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
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
        opt.zero_grad(set_to_none=True)        # FIX 5: clear before accumulating
        loss.backward()
        opt.step()
        if log_every and step % log_every == 0:
            print(f"  step {step:>4}  loss {loss.item():.4f}")
    return float(loss.item())


@torch.no_grad()                                # FIX 8: no graph, no memory growth
def evaluate(model, data, batch=256, gen=None):
    gen = gen or torch.Generator().manual_seed(123)
    model.eval()                                # FIX 8: dropout off, deterministic
    x, y = get_batch(data, batch, gen)
    logits, loss = model(x, y)
    pred = logits.argmax(-1)
    # Score only the second half, where the answer is determined by the prefix.
    acc = float((pred[:, SEQ - 1:] == y[:, SEQ - 1:]).float().mean())
    model.train()
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


# ------------------------------------------------------------------ self-test
def main() -> int:
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    data = make_data(4096, gen)

    x, y = get_batch(data, 4, torch.Generator().manual_seed(1))
    check("7 targets are shifted by one", bool((x[:, 1:] == y[:, :-1]).all()),
          "y[t] must be the token after x[t]")

    model = TinyLM()
    n_params = sum(p.numel() for p in model.parameters())
    block_params = sum(p.numel() for b in model.blocks for p in b.parameters())
    check("1 block parameters are registered", block_params > 0 and block_params < n_params,
          f"{block_params} of {n_params} parameters live in the blocks")

    bad = missing_gradients(model, data)
    check("3 every parameter gets a gradient", not bad, f"{bad[:3]}" if bad else "")

    with torch.no_grad():
        _, loss0 = model(*get_batch(data, 256, torch.Generator().manual_seed(2)))
    check("2 initial loss is about ln(V)", abs(float(loss0) - math.log(VOCAB)) < 0.15,
          f"{float(loss0):.3f} vs ln({VOCAB}) = {math.log(VOCAB):.3f}")

    worst = attention_weights_sum_to_one(model, data)
    check("5 attention weights sum to 1", worst < 1e-5, f"worst deviation {worst:.2e}")

    viol = causality_violation(model, data)
    check("5 causality holds", viol < 1e-6, f"max leakage {viol:.2e}")

    l1 = can_overfit_one_batch(TinyLM, data)
    check("1 can overfit a single batch", l1 < 0.05, f"final loss {l1:.5f}")

    m2 = TinyLM()
    a = evaluate(m2, data)
    b = evaluate(m2, data)
    check("8 evaluation is deterministic", close(a[0], b[0], atol=1e-9),
          f"{a[0]:.6f} vs {b[0]:.6f}")

    model = TinyLM()
    final = train(model, data, steps=600)
    val_loss, acc = evaluate(model, data)
    check("end-to-end: the model learns to copy", acc > 0.95,
          f"train loss {final:.4f}, val loss {val_loss:.4f}, accuracy {acc:.3f}")

    return 0 if report() else 1


if __name__ == "__main__":
    raise SystemExit(main())
