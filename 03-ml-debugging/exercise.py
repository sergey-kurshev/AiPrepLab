# %% [markdown]
# # Exercise 03 — Eight Bugs
#
# `buggy_lm.py` in this folder is a two-layer transformer language model trained on the copy task
# from module 02. It contains **eight planted bugs**. It runs. It trains. The loss goes down.
#
# **Do not read `buggy_lm.py` hunting for the bugs.** That is the failure mode this module exists
# to break. Run the diagnostics below, let a failing measurement tell you where to look, then open
# the file at that spot.
#
# Working rules:
#
# 1. Fix **one** bug at a time and re-run the checks. Two fixes at once and you learn nothing about
#    which mattered.
# 2. Before each fix, make sure you have a check that *fails* because of it. Afterwards you will
#    know you fixed it, rather than hoping.
# 3. Keep a log: bug, the measurement that exposed it, the fix. You will be asked to talk through
#    exactly this in an interview.
#
# Target: all eight in under an hour.

# %%
import sys
import pathlib
import importlib

_here = pathlib.Path.cwd()
for _c in (_here, *_here.parents):
    if (_c / "tools").is_dir():
        sys.path.insert(0, str(_c / "tools"))
        break
sys.path.insert(0, str(_here if (_here / "buggy_lm.py").exists() else _here / "03-ml-debugging"))

import math
import torch
import torch.nn as nn
from checks import check, report, close

import buggy_lm


def reload():
    """Re-import buggy_lm after you edit it. Run this cell after every fix."""
    importlib.reload(buggy_lm)
    return buggy_lm


lm = reload()
torch.manual_seed(0)
data = lm.make_data(4096, torch.Generator().manual_seed(0))
print("model:", sum(p.numel() for p in lm.TinyLM().parameters()), "parameters")

# %% [markdown]
# ---
# ## The diagnostic ladder
#
# These are the tools from the article, as runnable code. `buggy_lm.py` already contains some of
# them (`missing_gradients`, `can_overfit_one_batch`, `attention_weights_sum_to_one`,
# `causality_violation`); the ones below are the extra two you will need.

# %%
def unregistered_submodules(model: nn.Module) -> list[str]:
    """Rung 3, the version that catches a plain `list` of layers.

    `named_parameters()` cannot report a parameter it has never heard of, so the "does every
    parameter get a gradient" check passes vacuously. Look instead for attributes that hold
    modules without going through the registry.
    """
    registered = {id(m) for m in model.modules()}
    out = []
    for name, value in vars(model).items():
        if name.startswith("_"):
            continue
        items = value if isinstance(value, (list, tuple)) else [value]
        for i, item in enumerate(items):
            if isinstance(item, nn.Module) and id(item) not in registered:
                out.append(f"{name}[{i}]" if isinstance(value, (list, tuple)) else name)
    return out


def dropout_active_during(fn, model: nn.Module):
    """Did any Dropout run in training mode, or with autograd on, while `fn` executed?

    A forward hook is the precise way to ask this — much better than inferring it from noisy
    numbers. The same trick answers "was a graph being built?" via `torch.is_grad_enabled()`.

    Returns `(n_observed, any_in_train_mode, any_with_grad)`. Watch `n_observed`: if it is zero
    the hooks never fired, which means `model.modules()` does not contain the dropout layers —
    a different bug entirely, and one this diagnostic would otherwise hide behind a green tick.
    """
    seen = []
    handles = [m.register_forward_hook(lambda mod, i, o: seen.append((mod.training, torch.is_grad_enabled())))
               for m in model.modules() if isinstance(m, nn.Dropout)]
    try:
        fn()
    finally:
        for h in handles:
            h.remove()
    return len(seen), any(t for t, _ in seen), any(g for _, g in seen)


def gradient_report(model, data, batch=8):
    """Rung 4: per-parameter gradient norm, and the scale-invariant ratio that matters."""
    gen = torch.Generator().manual_seed(0)
    model.zero_grad(set_to_none=True)
    x, y = lm.get_batch(data, batch, gen)
    _, loss = model(x, y)
    loss.backward()
    rows = []
    for name, p in model.named_parameters():
        gn = float(p.grad.norm()) if p.grad is not None else float("nan")
        rows.append((name, tuple(p.shape), gn, gn / max(float(p.norm()), 1e-12)))
    return rows

# %% [markdown]
# ---
# ## The checks
#
# Nine assertions. Each maps to at least one of the eight bugs. Run this cell now, before changing
# anything, to see the starting state — then after every single fix.

# %%
def run_all_checks(verbose: bool = True) -> bool:
    lm = reload()
    torch.manual_seed(0)
    data = lm.make_data(4096, torch.Generator().manual_seed(0))

    # --- data
    x, y = lm.get_batch(data, 4, torch.Generator().manual_seed(1))
    check("A  targets are the NEXT token", bool((x[:, 1:] == y[:, :-1]).all()),
          "y[t] should equal x[t+1]")

    # --- registration
    model = lm.TinyLM()
    orphans = unregistered_submodules(model)
    check("B  every submodule is registered", not orphans, f"orphaned: {orphans}" if orphans else "")
    n_params = sum(p.numel() for p in model.parameters())
    check("C  the optimiser would see the whole model", n_params > 50_000,
          f"{n_params} parameters visible to .parameters()")

    # --- gradients
    bad = lm.missing_gradients(model, data)
    check("D  every registered parameter gets a nonzero gradient", not bad,
          f"{bad[:3]}" if bad else "")

    # --- the model's own invariants
    worst = lm.attention_weights_sum_to_one(model, data)
    check("E  attention weights sum to 1 over keys", worst < 1e-5, f"worst deviation {worst:.3e}")
    viol = lm.causality_violation(model, data)
    check("F  the future cannot leak into the past", viol < 1e-6, f"max leakage {viol:.3e}")

    # --- loss scale
    with torch.no_grad():
        _, loss0 = model(*lm.get_batch(data, 256, torch.Generator().manual_seed(2)))
    check("G  initial loss is ln(V)", abs(float(loss0) - math.log(lm.VOCAB)) < 0.15,
          f"{float(loss0):.3f} vs ln({lm.VOCAB}) = {math.log(lm.VOCAB):.3f}")

    # --- evaluation hygiene
    m2 = lm.TinyLM()
    n_seen, train_mode, grad_on = dropout_active_during(lambda: lm.evaluate(m2, data), m2)
    check("H  evaluate() runs with dropout off and no graph",
          n_seen > 0 and not train_mode and not grad_on,
          f"{n_seen} dropout calls observed, training={train_mode}, grad enabled={grad_on}")

    # --- the one that subsumes everything
    l1 = lm.can_overfit_one_batch(lm.TinyLM, data)
    check("I  can overfit a single batch", l1 < 0.05, f"final loss {l1:.5f}")
    return report()


run_all_checks()

# %% [markdown]
# ---
# ## Working through it
#
# Suggested order — cheapest signal first.
#
# ### 1. Look at check C
#
# How many parameters does a two-layer, 64-dimensional transformer with a 16-token vocabulary
# have? Work it out on paper: embeddings, then per block `qkv` (3·d²), `proj` (d²), two LayerNorms,
# and an MLP (2·4d²). If the number printed is far smaller than your arithmetic, some of the model
# is not registered — and `unregistered_submodules` will tell you which attribute.
#
# Note the trap: check D *passes* while this is broken. `named_parameters()` cannot report on
# parameters it has never been told about. A diagnostic that can only see what is registered
# cannot detect a registration bug.

# %%
model = lm.TinyLM()
print("parameters visible to .parameters():", sum(p.numel() for p in model.parameters()))
print("unregistered submodules:", unregistered_submodules(model))
print("blocks is a:", type(model.blocks).__name__)

# %% [markdown]
# ### 2. Look at check E
#
# Attention weights are a probability distribution over keys. Print the actual row sums and their
# pattern — the *shape* of the deviation names the bug. A row sum that varies with position points
# somewhere different from one that is uniformly wrong.

# %%
lm = reload()
m = lm.TinyLM()
x, _ = lm.get_batch(lm.make_data(64, torch.Generator().manual_seed(0)), 2,
                    torch.Generator().manual_seed(0))
m(x)
w = m.blocks[0].attn.last_weights
print("weight shape:", tuple(w.shape))
print("row sums over the last axis (should be all 1):\n", w[0, 0].sum(-1))
print("sums over the second-to-last axis:\n", w[0, 0].sum(-2))

# %% [markdown]
# ### 3. Look at check F
#
# Causality is an invariant, so a violation is unambiguous. But *two* different bugs can break it:
# a mask that does not mask, and a reshape that permutes positions. Distinguish them by looking at
# the attention pattern directly — print the weight matrix for one head. A correct causal pattern
# is lower-triangular. Which is it: mass above the diagonal, or mass scattered with no triangular
# structure at all?

# %%
import textwrap
print(textwrap.indent("\n".join(
    " ".join(f"{v:5.2f}" for v in row) for row in w[0, 0].tolist()), "  "))

# %% [markdown]
# ### 4. Use asymmetric shapes
#
# The head-reshape bug is invisible when dimensions coincide. Instantiate the attention module with
# `B`, `T`, `H` and `dh` all different and all small, and check that each output position depends
# only on the inputs it should. This is the habit to keep: **never test a tensor operation on
# square data.**

# %%
lm = reload()
B, T, H, dh = 2, 5, 3, 7
att = lm.CausalSelfAttention(H * dh, H, max_len=16, dropout=0.0).eval()
xs = torch.randn(B, T, H * dh)
with torch.no_grad():
    base = att(xs)
    for t in range(T):
        xp = xs.clone()
        xp[:, t] += 10.0
        delta = (att(xp) - base).abs().amax(dim=(0, 2))
        moved = [int(i) for i in range(T) if delta[i] > 1e-6]
        print(f"perturbing position {t} moves outputs {moved}   (should be {list(range(t, T))})")

# %% [markdown]
# ### 5. Look at check I, and the shape of the plateau
#
# Once the structural bugs are gone, `can_overfit_one_batch` is the last line of defence. Two of
# the remaining bugs show up here and nowhere else:
#
# - one makes the loss **stall at a specific non-zero value** no matter how many steps you run —
#   a value the model is mathematically incapable of going below;
# - one makes each update use a **stale** gradient, so the loss descends noisily and never settles.
#
# Plot the curve rather than reading the final number. The two look completely different.

# %%
lm = reload()


def overfit_curve(steps=400, lr=3e-3):
    gen = torch.Generator().manual_seed(0)
    data = lm.make_data(256, torch.Generator().manual_seed(0))
    model = lm.TinyLM()
    for mod in model.modules():
        if isinstance(mod, nn.Dropout):
            mod.p = 0.0
    x, y = lm.get_batch(data, 8, gen)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses, gnorms = [], []
    for _ in range(steps):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)))
        opt.step()
        losses.append(float(loss.detach()))
    return losses, gnorms


losses, gnorms = overfit_curve()
for i in range(0, len(losses), 50):
    bar = "#" * int(40 * losses[i] / max(losses))
    print(f"step {i:>4}  loss {losses[i]:7.4f}  |grad| {gnorms[i]:8.3f}  {bar}")
print(f"\nfloor: {min(losses):.4f}   (a correct model reaches < 0.05 on eight examples)")

# %% [markdown]
# A growing gradient norm across steps is the fingerprint of the accumulation bug: nothing is
# clearing `.grad`, so what you are plotting is a running sum.
#
# ### 6. Check H, and evaluation hygiene
#
# The hook-based diagnostic tells you both halves at once: whether dropout was in training mode,
# and whether a graph was being built. Neither should be true inside an evaluation function.

# %%
lm = reload()
m = lm.TinyLM()
d = lm.make_data(256, torch.Generator().manual_seed(0))
n_seen, train_mode, grad_on = dropout_active_during(lambda: lm.evaluate(m, d), m)
print(f"during evaluate(): {n_seen} dropout calls, in-train-mode={train_mode}, grad-enabled={grad_on}")
if n_seen == 0:
    print("  ...zero dropout calls observed: the hooks never fired, which is itself a finding.")

# %% [markdown]
# ---
# ## Finish line
#
# When all nine checks pass, train the fixed model end to end. A correct implementation reaches
# essentially 100% accuracy on the second half of the copy task, and a validation loss close to
# $\tfrac{1}{2}\ln(15) \approx 1.35$ — the first half of each sequence is genuinely unpredictable,
# so that is the floor, not a failure.

# %%
lm = reload()
if run_all_checks():
    torch.manual_seed(0)
    data = lm.make_data(4096, torch.Generator().manual_seed(0))
    model = lm.TinyLM()
    final = lm.train(model, data, steps=600, log_every=150)
    val, acc = lm.evaluate(model, data)
    print(f"\ntrain {final:.4f}   val {val:.4f}   second-half accuracy {acc:.3f}")
    check("the model learns the task", acc > 0.95)
    report()
else:
    print("\nFix the failing checks above first.")

# %% [markdown]
# ---
# ## Write it up
#
# For each of the eight bugs, in two lines: what it was, which measurement exposed it, and why it
# did not raise an exception. Then compare with the annotated list at the top of `solution.py` —
# and more importantly, compare your *route*. Did you find each one by measurement, or by reading?
#
# **Stretch goals:**
#
# 1. Plant a ninth bug of your own for someone else, from the table in `README.md`, and write the
#    one check that catches it. Designing the check is harder than finding the bug.
# 2. Make the model NaN on purpose (raise the learning rate, or switch to float16 without scaling)
#    and practise the NaN protocol: `torch.autograd.set_detect_anomaly(True)` to name the op, then
#    bisect to the tensor.
# 3. Add a `assert_shape(x, "B T C", B=B, T=T, C=C)` helper and thread it through `buggy_lm.py`.
#    How many of the eight would it have caught on its own?
