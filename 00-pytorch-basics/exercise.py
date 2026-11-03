# %% [markdown]
# # Exercise 00 — PyTorch as a Numerical Language
#
# Seven graded steps, ending in a scaled dot-product attention implementation that you will
# reuse for the rest of the course. Read `README.md` first.
#
# **How to work through this:** each `TODO` raises `NotImplementedError`. Replace it, then run
# the check cell below it. `solution.py` in this folder has one correct answer to every step —
# use it after you have made an honest attempt, and compare approaches rather than just outcomes.
#
# **Rule for the whole notebook:** no Python `for` loop may iterate over a tensor dimension
# unless a step explicitly asks for one.

# %%
import sys
import pathlib

# Make `tools/` importable regardless of where the kernel was started.
_here = pathlib.Path.cwd()
for _c in (_here, *_here.parents):
    if (_c / "tools").is_dir():
        sys.path.insert(0, str(_c / "tools"))
        break

import math
import torch
import torch.nn.functional as F
from checks import check, report, close

torch.manual_seed(0)
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())

# %% [markdown]
# ---
# ## Step 1 — Strides, views and aliasing
#
# Before running anything, write down your prediction for each `print` below. The point is not
# to learn the answers; it is to discover which of your beliefs about tensors were guesses.

# %%
x = torch.arange(12).reshape(3, 4)

print("shape   ", x.shape)
print("stride  ", x.stride())
print("T stride", x.t().stride())
print("contig? ", x.t().is_contiguous())
print("same storage?", x.data_ptr() == x.t().data_ptr())

# %% [markdown]
# **1a.** Write a function that reports whether two tensors share storage. Note that
# `data_ptr()` compares the address of element `[0, 0, ...]`, which is *not* the same question:
# `x[1:]` shares storage with `x` but has a different `data_ptr`. Use the untyped storage.

# %%
def shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    """True if `a` and `b` are views onto the same underlying memory."""
    # TODO: compare the storages, not the element pointers.
    # Hint: `Tensor.untyped_storage().data_ptr()`
    raise NotImplementedError


base = torch.arange(12).reshape(3, 4)
try:
    check("slice aliases", shares_storage(base, base[1:]))
    check("transpose aliases", shares_storage(base, base.t()))
    check("expand aliases", shares_storage(base[:, :1], base[:, :1].expand(3, 4)))
    check("clone does not", not shares_storage(base, base.clone()))
    check("contiguous() on a transpose copies", not shares_storage(base, base.t().contiguous()))
except Exception as _e:
    check("step 1a", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **1b.** `expand` vs `repeat`. Both produce a `(3, 4)` tensor from a `(3, 1)` one. Show that
# only one of them allocates, by comparing the number of elements in the *storage*.

# %%
col = torch.arange(3).reshape(3, 1)

expanded = None  # TODO: expand `col` to (3, 4)
repeated = None  # TODO: repeat `col` to (3, 4)

check("expand has the right shape", expanded is not None and expanded.shape == (3, 4))
check("repeat has the right shape", repeated is not None and repeated.shape == (3, 4))
check("expand allocates nothing", expanded is not None and expanded.untyped_storage().size() == col.untyped_storage().size())
check("repeat allocates", repeated is not None and repeated.untyped_storage().size() > col.untyped_storage().size())
check("expand has a zero stride", expanded is not None and 0 in expanded.stride())
report()

# %% [markdown]
# ---
# ## Step 2 — Broadcasting, and the bug that does not raise
#
# The cell below is the single most common shape bug in attention code. `scores` is a
# `(queries, keys)` matrix and we want to add a *per-key* bias — one number per column. The
# author wrote it in the way that felt natural, and it silently did something else.

# %%
scores = torch.zeros(4, 4)
key_bias = torch.tensor([0.0, 10.0, 20.0, 30.0])

wrong = scores + key_bias.reshape(4, 1)   # the bug
print(wrong)

# %% [markdown]
# **2a.** Produce the intended result, and state in one sentence what the buggy line actually
# computed. Then write an assertion that would have caught it.

# %%
def add_key_bias(scores: torch.Tensor, key_bias: torch.Tensor) -> torch.Tensor:
    """Add a per-key (per-column) bias to a (..., n_queries, n_keys) score matrix.

    Must work for an arbitrary number of leading batch dimensions.
    """
    # TODO
    raise NotImplementedError


out = None
try:
    out = add_key_bias(scores, key_bias)
except Exception:
    pass

check("column 1 is all 10", out is not None and close(out[:, 1], torch.full((4,), 10.0)))
check("row 1 is [0,10,20,30]", out is not None and close(out[1], key_bias))
check("works with batch dims", out is not None and add_key_bias(torch.zeros(2, 3, 4, 4), key_bias).shape == (2, 3, 4, 4))
report()

# %% [markdown]
# **2b.** A causal mask. For a `(T, T)` score matrix, position `i` may attend to positions
# `j <= i`. Build the additive mask: `0` where attention is allowed, `-inf` where it is not.
#
# Two questions to answer before you write it: why additive rather than multiplicative, and why
# `-inf` rather than a large negative number like `-1e9`?

# %%
def causal_mask(T: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """(T, T) additive mask: 0 on and below the diagonal, -inf above."""
    # TODO
    raise NotImplementedError


m = None
try:
    m = causal_mask(5)
except Exception:
    pass

check("shape", m is not None and m.shape == (5, 5))
check("diagonal is allowed", m is not None and close(m.diagonal(), torch.zeros(5)))
check("strictly upper triangle is -inf", m is not None and bool(torch.isinf(m[0, 1:]).all()) and bool((m[0, 1:] < 0).all()))
check("lower triangle is zero", m is not None and close(m[4, :5], torch.zeros(5)))
report()

# %% [markdown]
# ---
# ## Step 3 — `einsum` as executable index notation
#
# Rewrite three operations as a single `einsum` each, using the index letters
# `b` (batch), `h` (head), `q`/`k` (query/key position), `d` (head dimension).

# %%
B, H, T, D = 2, 3, 5, 7
q = torch.randn(B, H, T, D)
k = torch.randn(B, H, T, D)
v = torch.randn(B, H, T, D)


def scores_einsum(q, k):
    """(B,H,Tq,D) x (B,H,Tk,D) -> (B,H,Tq,Tk), the Q Kᵀ contraction."""
    # TODO
    raise NotImplementedError


def weighted_values_einsum(w, v):
    """(B,H,Tq,Tk) x (B,H,Tk,D) -> (B,H,Tq,D)."""
    # TODO
    raise NotImplementedError


def head_norms_einsum(q):
    """(B,H,T,D) -> (B,H): the mean squared norm of the query vectors in each head.

    One einsum plus one division; no `.pow`, no `.sum` calls.
    """
    # TODO
    raise NotImplementedError


ref_scores = q @ k.transpose(-2, -1)
w_probs = torch.softmax(ref_scores, dim=-1)
ref_wv = w_probs @ v
ref_norms = (q ** 2).sum(-1).mean(-1)

for name, fn, args, ref in [
    ("scores", scores_einsum, (q, k), ref_scores),
    ("weighted values", weighted_values_einsum, (w_probs, v), ref_wv),
    ("head norms", head_norms_einsum, (q,), ref_norms),
]:
    try:
        check(name, close(fn(*args), ref))
    except Exception as _e:
        check(name, False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# ---
# ## Step 4 — Autograd is a vector–Jacobian product
#
# The claim from the article: `backward` never forms a Jacobian, it applies one from the left.
# Verify it two ways.
#
# **4a.** Compute the full Jacobian of `softmax` at a point, by hand, from the closed form
#
# $$\frac{\partial \sigma_i}{\partial z_j} = \sigma_i(\delta_{ij} - \sigma_j)
#   \quad\Longleftrightarrow\quad J = \mathrm{diag}(\sigma) - \sigma\sigma^{\top}$$
#
# and check it against `torch.autograd.functional.jacobian`.

# %%
z = torch.randn(6, dtype=torch.double)


def softmax_jacobian(z: torch.Tensor) -> torch.Tensor:
    """(n,) -> (n, n) Jacobian of softmax, from the closed form. No autograd."""
    # TODO
    raise NotImplementedError


try:
    J_manual = softmax_jacobian(z)
    J_auto = torch.autograd.functional.jacobian(lambda t: torch.softmax(t, dim=-1), z)
    check("Jacobian matches autograd", close(J_manual, J_auto, atol=1e-10))
    check("rows sum to zero", close(J_manual.sum(-1), torch.zeros(6), atol=1e-10),
          "because softmax outputs are constrained to sum to 1")
    check("Jacobian is symmetric", close(J_manual, J_manual.T, atol=1e-10))
except Exception as _e:
    check("Jacobian", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **4b.** Now the VJP. Given a cotangent $\bar{y}$, reverse-mode returns
# $\bar{z}^{\top} = \bar{y}^{\top} J$. For softmax this simplifies to
#
# $$\bar{z} = \sigma \odot (\bar{y} - \langle \bar{y}, \sigma\rangle)$$
#
# which costs $O(n)$ rather than the $O(n^2)$ of forming $J$. Implement it and confirm it agrees
# with both the explicit matrix product and with `.backward()`.

# %%
def softmax_vjp(z: torch.Tensor, y_bar: torch.Tensor) -> torch.Tensor:
    """Reverse-mode gradient of softmax, in O(n) and without building the Jacobian."""
    # TODO
    raise NotImplementedError


y_bar = torch.randn(6, dtype=torch.double)
z_leaf = z.clone().requires_grad_(True)
torch.softmax(z_leaf, dim=-1).backward(y_bar)

try:
    g = softmax_vjp(z, y_bar)
    check("VJP matches autograd", close(g, z_leaf.grad, atol=1e-10))
    check("VJP matches ȳᵀJ", close(g, softmax_jacobian(z).T @ y_bar, atol=1e-10))
except Exception as _e:
    check("VJP", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **4c.** Three one-line predictions. Run the cell and explain each result *before* you look it
# up. If any surprised you, that is the part of autograd you did not actually know.

# %%
a = torch.tensor([2.0], requires_grad=True)
(a * 3).sum().backward()
(a * 3).sum().backward()
print("after two backwards, a.grad =", a.grad, "  <- why is this not 3?")

b = torch.tensor([2.0], requires_grad=True)
c = b * 2
d = c.detach() * 5
try:
    d.sum().backward()
    print("b.grad after detach:", b.grad)
except RuntimeError as e:
    print("backward through a detached tensor raises:", e)
print("   <- `detach` did not just zero the gradient; it removed `d` from the graph entirely.")

e = torch.tensor([2.0], requires_grad=True)
with torch.no_grad():
    f_ = e * 2
print("f_.requires_grad:", f_.requires_grad, "  <- and what would .backward() do?")

# %% [markdown]
# ---
# ## Step 5 — A softmax that does not overflow
#
# Implement softmax from primitives (`exp`, `sum`, `max`, arithmetic). No `torch.softmax`, no
# `logsumexp`. It must survive the adversarial inputs below.
#
# Note the second one: a row that is entirely `-inf` arises in real code from a fully masked
# query (a padding position under a causal mask). Naive code returns `nan` there and poisons the
# whole batch. Decide what your implementation should do, and defend it.

# %%
def stable_softmax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax over `dim`."""
    # TODO
    raise NotImplementedError


tests = {
    "ordinary": torch.randn(4, 8),
    "large positive": torch.tensor([[1000.0, 1001.0, 1002.0]]),
    "large negative": torch.tensor([[-1000.0, -1001.0, -1002.0]]),
    "masked row": torch.tensor([[0.0, float("-inf"), float("-inf")]]),
}

for name, t in tests.items():
    try:
        got = stable_softmax(t)
        ref = torch.softmax(t, dim=-1)
        check(f"{name}: matches torch", close(got, ref))
        check(f"{name}: finite", bool(torch.isfinite(got).all()))
        check(f"{name}: rows sum to 1", close(got.sum(-1), torch.ones(t.shape[0])))
    except Exception as _e:
        check(name, False, f"{type(_e).__name__}: {_e}")

# The naive version, for contrast — run it and see which cases it loses.
naive = lambda t: t.exp() / t.exp().sum(-1, keepdim=True)
print("\nnaive on large positive:", naive(tests["large positive"]))
report()

# %% [markdown]
# ---
# ## Step 6 — Scaled dot-product attention
#
# The spine of the course. Implement
#
# $$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}} + M\right)V$$
#
# for inputs shaped `(B, H, T, D)`, with an optional causal mask. Use your `stable_softmax`.
#
# Two things to get right and be ready to justify: **where** the $\sqrt{d_k}$ goes (scale the
# scores, not $Q$ or the probabilities — though scaling $Q$ is defensible, and cheaper: say why),
# and the fact that the mask is applied **before** the softmax, not after.

# %%
def attention(q, k, v, causal: bool = False):
    """(B,H,Tq,D), (B,H,Tk,D), (B,H,Tk,Dv) -> ((B,H,Tq,Dv) output, (B,H,Tq,Tk) weights)."""
    # TODO
    raise NotImplementedError


B, H, T, D = 2, 4, 16, 32
q = torch.randn(B, H, T, D)
k = torch.randn(B, H, T, D)
v = torch.randn(B, H, T, D)

try:
    out, w = attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    check("shape", out.shape == (B, H, T, D))
    check("matches F.scaled_dot_product_attention", close(out, ref, atol=1e-5))
    check("weights are a distribution", close(w.sum(-1), torch.ones(B, H, T)))

    out_c, w_c = attention(q, k, v, causal=True)
    ref_c = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    check("causal matches torch", close(out_c, ref_c, atol=1e-5))
    check("causal weights are lower-triangular", bool((w_c.triu(1) == 0).all()))
    check("first query attends only to itself", close(w_c[0, 0, 0, 0], torch.tensor(1.0)))
except Exception as _e:
    check("attention", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **6b.** Why does the scale matter? With $q, k$ having i.i.d. unit-variance entries,
# $\mathrm{Var}[q \cdot k] = d_k$, so without the $1/\sqrt{d_k}$ the logits grow with dimension
# and the softmax saturates. Measure the consequence: for increasing `D`, report the entropy of
# the attention distribution and the largest gradient flowing back into `q`, with and without
# the scale.
#
# You should see the unscaled version collapse towards a one-hot distribution (entropy → 0) and
# its gradients vanish — this is the actual argument in §3.2.1 of Vaswani et al., made concrete.

# %%
def attention_entropy_and_grad(D: int, scale: bool, T: int = 64) -> tuple[float, float]:
    """Return (mean entropy in nats, max |grad| into q) for one random attention layer."""
    # TODO: build q, k with requires_grad; compute scores with or without 1/sqrt(D);
    #       softmax; entropy = -(p * log p).sum(-1); backward on the output sum.
    raise NotImplementedError


try:
    print(f"{'D':>6} {'entropy (scaled)':>18} {'entropy (raw)':>16} {'maxgrad scaled':>16} {'maxgrad raw':>14}")
    for D in (8, 32, 128, 512):
        e_s, g_s = attention_entropy_and_grad(D, scale=True)
        e_r, g_r = attention_entropy_and_grad(D, scale=False)
        print(f"{D:>6} {e_s:>18.3f} {e_r:>16.3f} {g_s:>16.3e} {g_r:>14.3e}")
    check("scaled entropy stays high at D=512", attention_entropy_and_grad(512, True)[0] > 3.0)
    check("unscaled entropy collapses at D=512", attention_entropy_and_grad(512, False)[0] < 1.0)
except Exception as _e:
    check("entropy study", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# ---
# ## Step 7 — Wrap it in an `nn.Module`, and time it
#
# Now the engineering. Build a multi-head attention module: project the input to Q, K, V with a
# single fused linear layer, reshape into heads, attend, concatenate, project out.
#
# Requirements that are each a planted-bug category from module 03:
# - the causal mask must be a **buffer**, not an attribute and not a parameter;
# - the head split must be `(B, T, H*D) -> (B, H, T, D)` with the *correct* permutation — a
#   `reshape` in the wrong order will run and produce garbage;
# - dropout must respect `.train()` / `.eval()`.

# %%
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int = 512, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        # TODO: qkv projection (one nn.Linear producing 3*d_model), output projection,
        #       dropout module, and a registered causal-mask buffer.
        raise NotImplementedError

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """(B, T, d_model) -> (B, T, d_model)."""
        # TODO
        raise NotImplementedError


try:
    mha = MultiHeadAttention(64, 8, dropout=0.1)
    x = torch.randn(2, 10, 64)
    y = mha(x)
    check("output shape", y.shape == x.shape)

    names = dict(mha.named_buffers())
    check("mask is a buffer", any(v.dim() >= 2 for v in names.values()), f"buffers: {list(names)}")
    check("mask is not a parameter", all("mask" not in n for n, _ in mha.named_parameters()))

    mha.eval()
    with torch.no_grad():
        check("eval is deterministic", close(mha(x), mha(x)))
    mha.train()
    check("train is stochastic with dropout", not close(mha(x), mha(x)))

    # Causality: changing a future token must not change the present one.
    mha.eval()
    with torch.no_grad():
        x2 = x.clone()
        x2[:, -1] = torch.randn(2, 64)
        check("causal: past outputs unchanged", close(mha(x)[:, :-1], mha(x2)[:, :-1], atol=1e-6))
except Exception as _e:
    check("MultiHeadAttention", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **7b.** Benchmark it against `F.scaled_dot_product_attention` (which dispatches to a fused
# kernel). Note the timing discipline: warm up first, and synchronise on CUDA — otherwise you
# are timing the launch queue, not the work. This harness returns to you in module 05.

# %%
import time


def bench(fn, *args, n_warmup: int = 5, n_iter: int = 20) -> float:
    """Median wall-clock milliseconds per call."""
    for _ in range(n_warmup):
        fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn(*args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


dev = "cuda" if torch.cuda.is_available() else "cpu"
qq = torch.randn(8, 8, 512, 64, device=dev)

try:
    naive_ms = bench(lambda: attention(qq, qq, qq, causal=True))
    fused_ms = bench(lambda: F.scaled_dot_product_attention(qq, qq, qq, is_causal=True))
    print(f"device={dev}  naive {naive_ms:.2f} ms   fused {fused_ms:.2f} ms   speedup {naive_ms/fused_ms:.1f}x")
    print("\nWhere does the difference come from? It is not arithmetic — both do the same FLOPs.")
    print("Module 07 reads the paper that explains it; module 09 makes you write the kernel.")
except NotImplementedError:
    print("implement `attention` first")

# %% [markdown]
# ---
# ## Where this goes next
#
# Keep your `attention` and `MultiHeadAttention`. Specifically:
#
# | Module | What it does to this code |
# |---|---|
# | 01 | derives its gradient by hand and asks why softmax is the right normaliser at all |
# | 02 | rebuilds it in numpy with a hand-written backward pass, inside a full transformer block |
# | 03 | hands you four broken variants of it and a training loop that will not learn |
# | 04 | shards it across simulated devices and computes the memory budget |
# | 05 | adds a KV cache and quantises it to INT8 |
# | 07 | reproduces FlashAttention's tiled, online-softmax formulation |
# | 09 | writes its matmul as a CUDA kernel and races it against cuBLAS |
#
# **Stretch goals if you finished early:**
#
# 1. Implement `attention` with an explicit `torch.autograd.Function` so you own the backward
#    pass, and validate it with `torch.autograd.gradcheck` in double precision.
# 2. Measure peak memory (`torch.cuda.max_memory_allocated`) as a function of sequence length for
#    your implementation. Confirm the $O(T^2)$ term, then predict where a 40 GB card runs out.
# 3. Re-express the whole module in `einops` and decide, honestly, whether it reads better.
