# %% [markdown]
# # Exercise 02 — A Transformer in numpy, Forward and Backward
#
# No autograd. You write every gradient, and finite differences keep you honest.
#
# The payoff at the end is a one-block language model, built entirely from what you wrote here,
# that learns a task it can only solve by attending to the right position — which is the smallest
# convincing demonstration that your implementation is actually a transformer.
#
# **Time-box each step.** By the end of this module, `MultiHeadAttention.forward` from a blank
# file should take you under fifteen minutes.

# %%
import sys
import pathlib

_here = pathlib.Path.cwd()
for _c in (_here, *_here.parents):
    if (_c / "tools").is_dir():
        sys.path.insert(0, str(_c / "tools"))
        break

import math
import numpy as np
from checks import check, report, close

rng = np.random.default_rng(0)
np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ---
# ## Step 0 — The tool that makes the rest possible
#
# Before any layer, build the gradient checker. Central differences,
# $\partial f/\partial x_i \approx (f(x+\epsilon e_i) - f(x-\epsilon e_i))/2\epsilon$, accurate to
# $O(\epsilon^2)$. Every subsequent step is validated with it.
#
# Two things to get right: iterate over *every* element (use `np.nditer` with `multi_index`), and
# restore the perturbed entry afterwards.

# %%
def numerical_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of scalar f at x. Perturbs x in place, then restores it."""
    # TODO
    raise NotImplementedError


try:
    x0 = rng.normal(size=(3, 4))
    g = numerical_grad(lambda t: float((t ** 3).sum()), x0.copy())
    check("0 matches the analytic gradient of sum(x³)", close(g, 3 * x0 ** 2, atol=1e-6))
    check("0 leaves the input untouched", close(numerical_grad(lambda t: float(t.sum()), x0), np.ones_like(x0), atol=1e-6))
except Exception as _e:
    check("step 0", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **A note on $\epsilon$.** Truncation error falls as $\epsilon^2$ but cancellation error grows as
# $1/\epsilon$, so the total error is minimised around $\epsilon \approx \sqrt[3]{u} \approx 10^{-5}$
# for float64. In float32 ($u \approx 10^{-7}$) the best achievable accuracy is roughly $10^{-3}$ —
# which is why gradient checking is done in double precision, and why a float32 "gradcheck failure"
# is usually a false alarm. Convince yourself by sweeping $\epsilon$:

# %%
try:
    f = lambda t: float(np.sin(t).sum())
    pt = rng.normal(size=(2, 2))
    exact = np.cos(pt)
    for e in (1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11):
        err = np.abs(numerical_grad(f, pt.copy(), eps=e) - exact).max()
        print(f"eps={e:>7.0e}   max error {err:.3e}")
except Exception:
    pass

# %% [markdown]
# ---
# ## The layer protocol
#
# Every layer below implements:
#
# - `forward(x)` — returns the output and stashes whatever `backward` will need;
# - `backward(dy)` — returns `dx` and writes parameter gradients into `self.grads`.
#
# That is all a framework does, minus the graph bookkeeping. `_grad_ok` checks a layer against
# finite differences; it is given to you.

# %%
def _grad_ok(layer, x, dy, param_name=None, param=None, atol=1e-6):
    """Finite-difference check of one layer's backward pass, for the input or one parameter."""
    layer.forward(x)
    dx = layer.backward(dy)
    if param_name is None:
        num = numerical_grad(lambda t: float((layer.forward(t) * dy).sum()), x.copy())
        return close(dx, num, atol=atol)

    def f(p):
        orig = param.copy()
        param[...] = p
        out = float((layer.forward(x) * dy).sum())
        param[...] = orig
        return out

    num = numerical_grad(f, param.copy())
    layer.forward(x)
    layer.backward(dy)
    return close(layer.grads[param_name], num, atol=atol)


x = rng.normal(size=(2, 5, 8))
dy8 = rng.normal(size=(2, 5, 8))

# %% [markdown]
# ---
# ## Step 1 — Linear
#
# The two matmul VJPs are the workhorses of everything that follows:
# $C = AB \Rightarrow \bar A = \bar C B^{\top},\; \bar B = A^{\top}\bar C$.
#
# The subtlety is the leading axes. `x` is `(B, T, d_in)` but `W` is `(d_in, d_out)` — the weight is
# *shared* across batch and time, so its gradient must be **summed** over both. Getting this wrong
# gives you a `(B, T, d_in, d_out)` gradient, or a silently averaged one.

# %%
class Linear:
    def __init__(self, d_in, d_out, bias=True, scale=None):
        s = scale if scale is not None else 1.0 / math.sqrt(d_in)
        self.W = rng.normal(0, s, (d_in, d_out))
        self.b = np.zeros(d_out) if bias else None
        self.grads = {}

    def params(self):
        return {"W": self.W} | ({"b": self.b} if self.b is not None else {})

    def forward(self, x):
        # TODO
        raise NotImplementedError

    def backward(self, dy):
        # TODO: set self.grads["W"], self.grads["b"], return dx
        raise NotImplementedError


try:
    lin = Linear(8, 8)
    check("1 dx", _grad_ok(lin, x, dy8))
    check("1 dW", _grad_ok(lin, x, dy8, "W", lin.W))
    check("1 db", _grad_ok(lin, x, dy8, "b", lin.b))
    check("1 dW has the shape of W", lin.grads["W"].shape == lin.W.shape)
except Exception as _e:
    check("step 1", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# ---
# ## Step 2 — LayerNorm, the interesting one
#
# $\mathrm{LN}(x) = \gamma\odot\hat x + \beta$ with $\hat x = (x-\mu)/\sqrt{\sigma^2+\epsilon}$,
# over the **feature** axis. Both $\mu$ and $\sigma^2$ depend on every $x_i$, which is what makes
# the backward pass more than an elementwise multiply:
#
# $$\bar x = \frac{1}{\sqrt{\sigma^2+\epsilon}}\Big(\bar{\hat x} - \mathrm{mean}(\bar{\hat x}) - \hat x\,\mathrm{mean}(\bar{\hat x}\odot\hat x)\Big),\qquad \bar{\hat x} = \bar y\odot\gamma.$$
#
# Derive it before you implement it. Then say, in one sentence, what the two subtracted terms *are*
# — the answer is about projection, and it is the thing to say in an interview.

# %%
class LayerNorm:
    def __init__(self, d, eps=1e-5):
        self.g, self.b, self.eps = np.ones(d), np.zeros(d), eps
        self.grads = {}

    def params(self):
        return {"g": self.g, "b": self.b}

    def forward(self, x):
        # TODO: stash inv = 1/sqrt(var+eps) and xhat; you need both in backward.
        raise NotImplementedError

    def backward(self, dy):
        # TODO
        raise NotImplementedError


try:
    ln = LayerNorm(8)
    ln.g[:] = rng.normal(1, 0.2, 8)
    ln.b[:] = rng.normal(0, 0.2, 8)
    check("2 dx", _grad_ok(ln, x, dy8))
    check("2 dg", _grad_ok(ln, x, dy8, "g", ln.g))
    check("2 db", _grad_ok(ln, x, dy8, "b", ln.b))

    out = ln.forward(x)
    xh = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
    check("2 normalises per token", close(xh.mean(-1), np.zeros((2, 5)), atol=1e-12))
    # Invariance is exact only as eps -> 0. The residual you see is eps leaking through.
    dev = float(np.abs(ln.forward(3.0 * x + 7.0) - out).max())
    check("2 invariant to shift and scale (up to eps)", dev < 1e-4, f"max deviation {dev:.2e}")
except Exception as _e:
    check("step 2", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **2b.** Why does normalisation help *optimisation*? Connect it to module 01: LayerNorm makes
# the loss invariant to the scale of the incoming weights, so the gradient with respect to those
# weights is orthogonal to the weight vector itself. Verify this numerically: show that
# $\langle \bar W, W\rangle \approx 0$ for a linear layer feeding directly into a LayerNorm.

# %%
try:
    W = rng.normal(size=(8, 8))
    lin2, ln2 = Linear(8, 8, bias=False), LayerNorm(8)
    lin2.W = W
    dy_ = rng.normal(size=(2, 5, 8))
    h = lin2.forward(x)
    ln2.forward(h)
    lin2.backward(ln2.backward(dy_))
    cos = float((lin2.grads["W"] * W).sum() / (np.linalg.norm(lin2.grads["W"]) * np.linalg.norm(W)))
    check("2b grad is orthogonal to the weights", abs(cos) < 1e-5, f"cosine = {cos:.2e} (exactly 0 only as eps -> 0)")
except Exception as _e:
    check("step 2b", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# ---
# ## Step 3 — GELU and softmax
#
# GELU, tanh approximation (what most kernels actually compute):
# $\;0.5x\big(1+\tanh[\sqrt{2/\pi}(x + 0.044715x^3)]\big)$.
#
# Softmax you have already written twice; here you need its VJP again, this time expressed in
# terms of the *output* `p` rather than the input, because that is what a backward pass has to hand.

# %%
class GELU:
    C = math.sqrt(2.0 / math.pi)
    grads = {}

    def params(self):
        return {}

    def forward(self, x):
        # TODO
        raise NotImplementedError

    def backward(self, dy):
        # TODO — chain through the tanh; d/dx tanh(u) = (1 - tanh²u) u'.
        raise NotImplementedError


def softmax(z, axis=-1):
    # TODO: the stable version (module 00, step 5).
    raise NotImplementedError


def softmax_backward(p, dp, axis=-1):
    """VJP of softmax expressed in terms of its output p."""
    # TODO
    raise NotImplementedError


try:
    check("3 GELU dx", _grad_ok(GELU(), x, dy8))
    check("3 GELU(0) = 0", close(GELU().forward(np.zeros(1)), np.zeros(1)))
    check("3 GELU is close to ReLU far from 0",
          close(GELU().forward(np.array([5.0, -5.0])), np.array([5.0, 0.0]), atol=1e-3))

    z0 = rng.normal(size=(4, 6))
    dp = rng.normal(size=(4, 6))
    check("3 softmax rows sum to 1", close(softmax(z0).sum(-1), np.ones(4)))
    check("3 softmax VJP",
          close(softmax_backward(softmax(z0), dp),
                numerical_grad(lambda t: float((softmax(t) * dp).sum()), z0.copy()), atol=1e-6))
except Exception as _e:
    check("step 3", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# ---
# ## Step 4 — Multi-head attention
#
# **Write the shape table first.** Fill this in before writing a line of code:
#
# | tensor | shape |
# |---|---|
# | `x` | (B, T, C) |
# | `qkv` | ? |
# | `q, k, v` after the head split | ? |
# | `scores` | ? |
# | `p` | ? |
# | `o` before merge | ? |
# | output | ? |
#
# The head split is `(B, T, 3C) -> (B, T, 3, H, dh) -> permute`. Reshaping straight to `(B, H, T, dh)`
# runs and is wrong: it interleaves positions with heads. The causality test at the bottom catches it.

# %%
class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.H, self.dh = n_heads, d_model // n_heads
        self.qkv = Linear(d_model, 3 * d_model, bias=False)
        self.proj = Linear(d_model, d_model, bias=False)

    def params(self):
        return {"qkv.W": self.qkv.W, "proj.W": self.proj.W}

    @property
    def grads(self):
        return {"qkv.W": self.qkv.grads["W"], "proj.W": self.proj.grads["W"]}

    def forward(self, x, causal=True):
        # TODO
        raise NotImplementedError

    def backward(self, dy):
        # TODO: mirror the forward exactly, in reverse. Recall from module 01:
        #   dV = Pᵀ dO,  dP = dO Vᵀ,  dS = softmax_backward(P, dP)/sqrt(dh),
        #   dQ = dS K,   dK = dSᵀ Q.
        raise NotImplementedError


try:
    attn = MultiHeadAttention(8, 2)
    check("4 dx", _grad_ok(attn, x, dy8))
    y1 = attn.forward(x)
    x2 = x.copy()
    x2[:, -1] += 1.0
    check("4 causality: the future cannot reach the past",
          close(attn.forward(x2)[:, :-1], y1[:, :-1], atol=1e-12))
    attn.forward(x); attn.backward(dy8)
    check("4 dW_qkv shape", attn.qkv.grads["W"].shape == (8, 24))

    def obj(p):
        orig = attn.qkv.W.copy()
        attn.qkv.W[...] = p
        out = float((attn.forward(x) * dy8).sum())
        attn.qkv.W[...] = orig
        return out

    attn.forward(x); attn.backward(dy8)
    check("4 dW_qkv", close(attn.qkv.grads["W"], numerical_grad(obj, attn.qkv.W.copy()), atol=1e-6))

    # One head with H=1 must equal plain single-head attention — a cheap sanity oracle.
    a1 = MultiHeadAttention(8, 1)
    check("4 H=1 is a valid special case", a1.forward(x).shape == x.shape)
except Exception as _e:
    check("step 4", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# ---
# ## Step 5 — The block
#
# Pre-norm: $h = x + \mathrm{MHA}(\mathrm{LN}_1(x))$, $y = h + \mathrm{MLP}(\mathrm{LN}_2(h))$.
#
# The bug to avoid in `backward`: **a residual add sends the gradient down both paths and sums
# them.** Miss one branch and the model still trains, just worse — which is exactly the kind of
# thing module 03 is about.

# %%
class TransformerBlock:
    def __init__(self, d_model, n_heads, d_ff=None, n_layers=1):
        d_ff = d_ff or 4 * d_model
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.fc1 = Linear(d_model, d_ff)
        self.act = GELU()
        # 1/sqrt(2L) on the residual-writing projection keeps the stream variance bounded in depth.
        self.fc2 = Linear(d_ff, d_model, scale=1.0 / (math.sqrt(d_ff) * math.sqrt(2 * n_layers)))

    def params(self):
        out = {}
        for n, m in [("ln1", self.ln1), ("attn", self.attn), ("ln2", self.ln2),
                     ("fc1", self.fc1), ("fc2", self.fc2)]:
            out |= {f"{n}.{k}": v for k, v in m.params().items()}
        return out

    def grad_dict(self):
        out = {}
        for n, m in [("ln1", self.ln1), ("attn", self.attn), ("ln2", self.ln2),
                     ("fc1", self.fc1), ("fc2", self.fc2)]:
            out |= {f"{n}.{k}": v for k, v in m.grads.items()}
        return out

    def forward(self, x, causal=True):
        # TODO
        raise NotImplementedError

    def backward(self, dy):
        # TODO
        raise NotImplementedError


try:
    blk = TransformerBlock(8, 2)
    check("5 dx", _grad_ok(blk, x, dy8, atol=1e-5))
    check("5 output shape", blk.forward(x).shape == x.shape)
    # At initialisation the residual should dominate: y should be close to x.
    rel = float(np.abs(blk.forward(x) - x).mean() / np.abs(x).mean())
    check("5 the residual path dominates at init", rel < 1.0, f"relative change {rel:.3f}")
except Exception as _e:
    check("step 5", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **5b. The oracle.** Build the same block in PyTorch with the *same* weights and compare. This is
# the test you should reach for whenever you hand-write a backward pass: an independent
# implementation beats any amount of re-reading your own code.
#
# (`solution.py::_torch_agreement` does this in full if you want to see one way to wire it up.)

# %% [markdown]
# ---
# ## Step 6 — Does it learn?
#
# Assemble a one-block language model and train it on the **copy task**: sequences
# $[x_1 \dots x_n,\, x_1 \dots x_n]$. Predicting the second half is impossible from local context
# and trivial if the model learns to attend exactly $n$ positions back. Loss on the second half
# going to ~0 is therefore evidence that attention is doing its job — not just that the MLP
# memorised something.
#
# You need three more pieces: an embedding (backward is a scatter-add — `np.add.at`), the
# cross-entropy gradient $(p - y)/N$, and Adam.

# %%
# A helper for the gradient check below: evaluate the loss with one parameter replaced.
def _lm_obj(model, param, p, seq):
    orig = param.copy()
    param[...] = p
    loss, _ = model.forward(seq[:, :-1], seq[:, 1:])
    param[...] = orig
    return loss


# %%
class TinyLM:
    def __init__(self, vocab, d_model, n_heads, max_len):
        self.wte = rng.normal(0, 0.02, (vocab, d_model))
        self.wpe = rng.normal(0, 0.02, (max_len, d_model))
        self.block = TransformerBlock(d_model, n_heads)
        self.lnf = LayerNorm(d_model)
        self.head = Linear(d_model, vocab, bias=False)

    def params(self):
        return {"wte": self.wte, "wpe": self.wpe, "lnf.g": self.lnf.g, "lnf.b": self.lnf.b,
                "head.W": self.head.W} | {f"block.{k}": v for k, v in self.block.params().items()}

    def grad_dict(self):
        return {"wte": self.dwte, "wpe": self.dwpe, "lnf.g": self.lnf.grads["g"],
                "lnf.b": self.lnf.grads["b"], "head.W": self.head.grads["W"]} | \
               {f"block.{k}": v for k, v in self.block.grad_dict().items()}

    def forward(self, idx, targets):
        """idx, targets: (B, T) int arrays. Returns (mean cross-entropy, logits)."""
        # TODO
        raise NotImplementedError

    def backward(self):
        """Backprop from the loss. Remember: dlogits = (p - onehot)/N."""
        # TODO
        raise NotImplementedError


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.p, self.lr, (self.b1, self.b2), self.eps = params, lr, betas, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        # TODO: bias-corrected moments, then p -= lr * mhat / (sqrt(vhat) + eps).
        # Update the arrays IN PLACE (`p -= ...`), or the model will keep its old ones.
        raise NotImplementedError


def make_copy_batch(batch, n, vocab, gen):
    half = gen.integers(1, vocab, size=(batch, n))
    return np.concatenate([half, half], axis=1)


try:
    n, vocab = 8, 12
    gen = np.random.default_rng(1)
    model = TinyLM(vocab, 64, 4, max_len=2 * n)
    opt = Adam(model.params(), lr=3e-3)

    # Sanity first: one gradient of the full model against finite differences, on the
    # smallest parameter. If this fails, do not bother training.
    seq = make_copy_batch(4, n, vocab, gen)
    loss, _ = model.forward(seq[:, :-1], seq[:, 1:])
    model.backward()
    g_num = numerical_grad(lambda p: _lm_obj(model, model.lnf.g, p, seq), model.lnf.g.copy())
    check("6 full-model gradient is right", close(model.lnf.grads["g"], g_num, atol=1e-5))

    for step in range(400):
        seq = make_copy_batch(64, n, vocab, gen)
        loss, _ = model.forward(seq[:, :-1], seq[:, 1:])
        model.backward()
        opt.step(model.grad_dict())
        if step % 100 == 0:
            print(f"step {step:>4}  loss {loss:.4f}")

    seq = make_copy_batch(256, n, vocab, gen)
    _, logits = model.forward(seq[:, :-1], seq[:, 1:])
    acc = float((logits.argmax(-1)[:, n - 1:] == seq[:, 1:][:, n - 1:]).mean())
    print(f"final loss {loss:.4f}   second-half accuracy {acc:.3f}")
    check("6 the model learns to copy", acc > 0.95)
    check("6 chance would be much worse", acc > 5.0 / vocab)
except Exception as _e:
    check("step 6", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **6b.** Inspect what it learned. Pull `model.block.attn.p` after a forward pass and look at the
# attention pattern for a second-half query. You should see a sharp diagonal band offset by
# exactly $n$. That picture is the entire point of the exercise.

# %% [markdown]
# ---
# ## Step 7 — The classics
#
# **7a. k-means with k-means++ seeding.** Traps: vectorise the distances with
# $\|x-c\|^2 = \|x\|^2 - 2x\cdot c + \|c\|^2$; handle empty clusters explicitly; seed with
# probability proportional to $D(x)^2$.
#
# **7b. AUC in $O(n\log n)$.** Do *not* write the double loop. Use the Mann–Whitney identity
# $\mathrm{AUC} = (\sum_{i\in\text{pos}} r_i - n_p(n_p+1)/2)/(n_pn_n)$ with average ranks for ties.

# %%
def kmeans_pp_init(X, k, gen):
    """D² seeding."""
    # TODO
    raise NotImplementedError


def kmeans(X, k, n_iter=100, gen=None, seed_pp=True):
    """Return (centres, labels, inertia)."""
    # TODO
    raise NotImplementedError


def auc(scores, labels):
    """O(n log n) AUC via ranks, ties averaged."""
    # TODO
    raise NotImplementedError


def auc_quadratic(scores, labels):
    """The O(n²) definition, as an oracle. Given to you."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


try:
    gen = np.random.default_rng(1)
    true_c = np.array([[0.0, 0.0], [6.0, 6.0], [0.0, 6.0]])
    X = np.repeat(true_c, 120, axis=0) + gen.normal(0, 0.45, (360, 2))
    C, labels, inertia = kmeans(X, 3, gen=gen)
    # Compare as sets — k-means has no canonical cluster ordering.
    err = float(np.sqrt(((true_c[:, None, :] - C[None]) ** 2).sum(-1)).min(1).max())
    check("7a recovers the centres", err < 0.25, f"worst centre error {err:.3f}")

    pp = [kmeans(X, 3, gen=np.random.default_rng(s), seed_pp=True)[2] for s in range(20)]
    rnd = [kmeans(X, 3, gen=np.random.default_rng(s), seed_pp=False)[2] for s in range(20)]
    print(f"mean inertia:  k-means++ {np.mean(pp):.1f}   random init {np.mean(rnd):.1f}")
    check("7a k-means++ is no worse", np.mean(pp) <= np.mean(rnd) + 1e-9)

    s = gen.normal(size=400)
    lab = (gen.random(400) < 1 / (1 + np.exp(-s))).astype(int)
    check("7b matches the definition", close(auc(s, lab), auc_quadratic(s, lab), atol=1e-12))
    tied = np.round(gen.normal(size=200), 1)
    lab2 = (gen.random(200) < 0.5).astype(int)
    check("7b handles ties", close(auc(tied, lab2), auc_quadratic(tied, lab2), atol=1e-12))
    check("7b perfect ranker", close(auc(np.arange(10.0), np.r_[np.zeros(5), np.ones(5)]), 1.0))
    check("7b reversed ranker", close(auc(np.arange(10.0), np.r_[np.ones(5), np.zeros(5)]), 0.0))
except Exception as _e:
    check("step 7", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **7c. (discussion)** When is AUC the wrong metric, and what would you use instead? Be specific
# about the denominator that causes the problem.

# %% [markdown]
# ---
# ## Revisiting the spine
#
# Attention so far: tensor code (00) → derived gradient (01) → **complete implementation you own,
# forward and backward** (02). Next, module 03 hands you this architecture already broken, and the
# fact that you have written every line of it is exactly what makes the bugs findable.
#
# **Stretch goals:**
#
# 1. Replace LayerNorm with RMSNorm (drop the mean and $\beta$). Derive the new backward pass,
#    check it, and measure the wall-clock difference.
# 2. Add RoPE: rotate `q` and `k` in 2-D subspaces by an angle proportional to position. Verify
#    the key property — that $q_m \cdot k_n$ depends only on $m-n$ — numerically.
# 3. Implement gradient checkpointing for the block: discard activations in forward, recompute in
#    backward. Measure the memory saved and the time cost. You will meet this again in module 04.
# 4. Implement a 2-D convolution forward and backward with `im2col`, and check it against
#    `torch.nn.functional.conv2d`.
