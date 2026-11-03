"""Reference solutions for exercise 02: a transformer in numpy, forward and backward.

Run with `python solution.py`. Everything here is plain numpy — PyTorch appears only
in the self-test, as an oracle.

The layer protocol is deliberately minimal: `forward(x)` stashes what backward needs
on the instance, `backward(dy)` returns `dx` and accumulates parameter gradients in
`self.grads`. That is the whole of what a deep-learning framework does, minus the
graph bookkeeping.
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

import numpy as np
from checks import check, report, close

rng = np.random.default_rng(0)


# ===================================================================== helpers
def numerical_grad(f, x, eps: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of a scalar function. O(eps^2) accurate.

    This is `torch.autograd.gradcheck` in six lines. Use float64 or the
    subtraction loses more precision than the truncation error you are bounding.
    """
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        orig = x[i]
        x[i] = orig + eps
        fp = f(x)
        x[i] = orig - eps
        fm = f(x)
        x[i] = orig
        g[i] = (fp - fm) / (2 * eps)
        it.iternext()
    return g


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def softmax_backward(p: np.ndarray, dp: np.ndarray, axis: int = -1) -> np.ndarray:
    """VJP of softmax given its *output* p — see module 01 §3.2."""
    return p * (dp - (dp * p).sum(axis=axis, keepdims=True))


# ===================================================================== layers
class Linear:
    def __init__(self, d_in: int, d_out: int, bias: bool = True, scale: float | None = None):
        # Xavier/Glorot: var = 1/d_in keeps the forward activation variance stable.
        s = scale if scale is not None else 1.0 / math.sqrt(d_in)
        self.W = rng.normal(0, s, (d_in, d_out))
        self.b = np.zeros(d_out) if bias else None
        self.grads: dict[str, np.ndarray] = {}

    def params(self):
        return {"W": self.W} | ({"b": self.b} if self.b is not None else {})

    def forward(self, x):
        self.x = x
        y = x @ self.W
        return y + self.b if self.b is not None else y

    def backward(self, dy):
        # The two matmul VJPs, which between them differentiate most of deep learning:
        #   C = A B  =>  dA = dC Bᵀ,  dB = Aᵀ dC.
        # The reshape to 2-D sums the gradient over every leading (batch, time) axis,
        # which is exactly what "the parameter is shared across positions" means.
        x2 = self.x.reshape(-1, self.x.shape[-1])
        dy2 = dy.reshape(-1, dy.shape[-1])
        self.grads["W"] = x2.T @ dy2
        if self.b is not None:
            self.grads["b"] = dy2.sum(0)
        return dy @ self.W.T


class LayerNorm:
    def __init__(self, d: int, eps: float = 1e-5):
        self.g = np.ones(d)
        self.b = np.zeros(d)
        self.eps = eps
        self.grads: dict[str, np.ndarray] = {}

    def params(self):
        return {"g": self.g, "b": self.b}

    def forward(self, x):
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        self.inv = 1.0 / np.sqrt(var + self.eps)
        self.xhat = (x - mu) * self.inv
        return self.g * self.xhat + self.b

    def backward(self, dy):
        d = dy.shape[-1]
        self.grads["g"] = (dy * self.xhat).reshape(-1, d).sum(0)
        self.grads["b"] = dy.reshape(-1, d).sum(0)
        dxhat = dy * self.g
        # dx = inv * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat)).
        # Read it as a projection: the component of the gradient along 1 and along xhat
        # is removed, because the output is invariant to shifting and scaling the input
        # in exactly those two directions.
        return self.inv * (
            dxhat
            - dxhat.mean(-1, keepdims=True)
            - self.xhat * (dxhat * self.xhat).mean(-1, keepdims=True)
        )


class GELU:
    """The tanh approximation, matching torch's `F.gelu(x, approximate='tanh')`.

    Exact GELU is x·Φ(x); numpy has no erf, and the approximation is what most
    production kernels compute anyway.
    """

    C = math.sqrt(2.0 / math.pi)

    def forward(self, x):
        self.x = x
        self.inner = self.C * (x + 0.044715 * x ** 3)
        self.t = np.tanh(self.inner)
        return 0.5 * x * (1.0 + self.t)

    def backward(self, dy):
        x, t = self.x, self.t
        dinner = self.C * (1.0 + 3 * 0.044715 * x ** 2)
        dx = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * dinner
        return dy * dx

    def params(self):
        return {}

    grads: dict = {}


class MultiHeadAttention:
    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0
        self.H, self.dh = n_heads, d_model // n_heads
        self.qkv = Linear(d_model, 3 * d_model, bias=False)
        self.proj = Linear(d_model, d_model, bias=False)

    def params(self):
        return {"qkv.W": self.qkv.W, "proj.W": self.proj.W}

    @property
    def grads(self):
        return {"qkv.W": self.qkv.grads["W"], "proj.W": self.proj.grads["W"]}

    def _split(self, t, B, T):
        return t.reshape(B, T, self.H, self.dh).transpose(0, 2, 1, 3)   # (B,H,T,dh)

    def _merge(self, t, B, T):
        return t.transpose(0, 2, 1, 3).reshape(B, T, self.H * self.dh)

    def forward(self, x, causal: bool = True):
        B, T, C = x.shape
        self.B, self.T, self.causal = B, T, causal
        qkv = self.qkv.forward(x)
        q, k, v = (self._split(qkv[..., i * C:(i + 1) * C], B, T) for i in range(3))
        self.q, self.k, self.v = q, k, v
        s = q @ k.transpose(0, 1, 3, 2) / math.sqrt(self.dh)
        if causal:
            s = np.where(np.tril(np.ones((T, T), dtype=bool)), s, -np.inf)
        self.p = softmax(s, axis=-1)
        o = self.p @ v
        self.o_merged = self._merge(o, B, T)
        return self.proj.forward(self.o_merged)

    def backward(self, dy):
        B, T, C = self.B, self.T, self.q.shape[-1] * self.H
        do_merged = self.proj.backward(dy)
        do = self._split(do_merged, B, T)                      # (B,H,T,dh)
        dv = self.p.transpose(0, 1, 3, 2) @ do
        dp = do @ self.v.transpose(0, 1, 3, 2)
        ds = softmax_backward(self.p, dp) / math.sqrt(self.dh)
        dq = ds @ self.k
        dk = ds.transpose(0, 1, 3, 2) @ self.q
        dqkv = np.concatenate([self._merge(t, B, T) for t in (dq, dk, dv)], axis=-1)
        return self.qkv.backward(dqkv)


class TransformerBlock:
    def __init__(self, d_model: int, n_heads: int, d_ff: int | None = None, n_layers: int = 1):
        d_ff = d_ff or 4 * d_model
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.fc1 = Linear(d_model, d_ff)
        self.act = GELU()
        # Scale the residual-writing projections by 1/sqrt(2L): with L blocks each adding
        # to the stream, this keeps the stream's variance from growing with depth.
        self.fc2 = Linear(d_ff, d_model, scale=1.0 / (math.sqrt(d_ff) * math.sqrt(2 * n_layers)))

    def params(self):
        out = {}
        for name, mod in [("ln1", self.ln1), ("attn", self.attn), ("ln2", self.ln2),
                          ("fc1", self.fc1), ("fc2", self.fc2)]:
            out |= {f"{name}.{k}": v for k, v in mod.params().items()}
        return out

    def grad_dict(self):
        out = {}
        for name, mod in [("ln1", self.ln1), ("attn", self.attn), ("ln2", self.ln2),
                          ("fc1", self.fc1), ("fc2", self.fc2)]:
            out |= {f"{name}.{k}": v for k, v in mod.grads.items()}
        return out

    def forward(self, x, causal: bool = True):
        h = x + self.attn.forward(self.ln1.forward(x), causal)
        self.h = h
        y = h + self.fc2.forward(self.act.forward(self.fc1.forward(self.ln2.forward(h))))
        return y

    def backward(self, dy):
        # A residual add sends the gradient down BOTH paths and sums them. Forgetting
        # one of the two branches is the most common hand-written-backward bug.
        dh = dy + self.ln2.backward(
            self.fc1.backward(self.act.backward(self.fc2.backward(dy)))
        )
        dx = dh + self.ln1.backward(self.attn.backward(dh))
        return dx


# ===================================================================== a tiny LM
class TinyLM:
    """Embedding -> [block] -> final LN -> unembed, with cross-entropy."""

    def __init__(self, vocab: int, d_model: int, n_heads: int, max_len: int):
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
        B, T = idx.shape
        self.idx = idx
        x = self.wte[idx] + self.wpe[:T]
        y = self.lnf.forward(self.block.forward(x))
        logits = self.head.forward(y)
        self.probs = softmax(logits, axis=-1)
        n = B * T
        flat = self.probs.reshape(n, -1)
        loss = -np.log(flat[np.arange(n), targets.reshape(-1)] + 1e-12).mean()
        self.targets = targets
        return loss, logits

    def backward(self):
        B, T = self.idx.shape
        n = B * T
        # d(cross-entropy)/d(logits) = (p - onehot)/n — the identity from module 01 §4.2.
        dlogits = self.probs.copy()
        dlogits.reshape(n, -1)[np.arange(n), self.targets.reshape(-1)] -= 1.0
        dlogits /= n
        dx = self.block.backward(self.lnf.backward(self.head.backward(dlogits)))
        self.dwpe = np.zeros_like(self.wpe)
        self.dwpe[:T] = dx.sum(0)
        self.dwte = np.zeros_like(self.wte)
        np.add.at(self.dwte, self.idx, dx)      # scatter-add: a token may appear many times
        return dx


class Adam:
    def __init__(self, params: dict, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.p, self.lr, (self.b1, self.b2), self.eps = params, lr, betas, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict):
        self.t += 1
        for k, p in self.p.items():
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)   # in-place: p aliases the model


def make_copy_batch(batch: int, n: int, vocab: int, gen):
    """Sequences [x_1..x_n, x_1..x_n]; predicting the second half requires attending back n."""
    half = gen.integers(1, vocab, size=(batch, n))
    return np.concatenate([half, half], axis=1)


# ===================================================================== classics
def kmeans_pp_init(X: np.ndarray, k: int, gen) -> np.ndarray:
    """D² seeding: O(log k) expected approximation, versus unbounded for random init."""
    centres = [X[gen.integers(len(X))]]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None, :] - np.array(centres)[None]) ** 2).sum(-1), axis=1)
        probs = d2 / d2.sum()
        centres.append(X[gen.choice(len(X), p=probs)])
    return np.array(centres)


def kmeans(X: np.ndarray, k: int, n_iter: int = 100, gen=None, seed_pp: bool = True):
    gen = gen or np.random.default_rng(0)
    C = kmeans_pp_init(X, k, gen) if seed_pp else X[gen.choice(len(X), k, replace=False)]
    prev = None
    for _ in range(n_iter):
        # ||x-c||² = ||x||² - 2x·c + ||c||²; the ||x||² term is constant per row and could
        # be dropped, but keeping it makes the returned inertia correct.
        d2 = (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None, :]
        labels = d2.argmin(1)
        for j in range(k):
            m = labels == j
            if m.any():
                C[j] = X[m].mean(0)
            else:
                C[j] = X[d2[np.arange(len(X)), labels].argmax()]   # re-seed the empty cluster
        inertia = d2[np.arange(len(X)), labels].sum()
        if prev is not None and abs(prev - inertia) < 1e-12:
            break
        prev = inertia
    return C, labels, float(inertia)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U form: O(n log n), with average ranks for ties."""
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0      # average rank, 1-based
        i = j + 1
    r = np.empty_like(ranks)
    r[order] = ranks
    n_p = int(labels.sum())
    n_n = len(labels) - n_p
    if n_p == 0 or n_n == 0:
        return float("nan")
    return float((r[labels == 1].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def auc_quadratic(scores, labels):
    """The O(n²) definition, as an oracle: P(score_pos > score_neg), ties counting ½."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


# ===================================================================== self-test
def _grad_ok(layer, x, dy, param_name=None, param=None):
    """Finite-difference check of one layer's backward pass."""
    y = layer.forward(x)
    dx = layer.backward(dy)
    if param_name is None:
        num = numerical_grad(lambda t: float((layer.forward(t) * dy).sum()), x.copy())
        return close(dx, num, atol=1e-6)

    def f(p):
        orig = param.copy()
        param[...] = p
        out = float((layer.forward(x) * dy).sum())
        param[...] = orig
        return out

    num = numerical_grad(f, param.copy())
    layer.forward(x)
    layer.backward(dy)
    return close(layer.grads[param_name], num, atol=1e-6)


def main() -> int:
    global rng
    rng = np.random.default_rng(0)

    x = rng.normal(size=(2, 5, 8))
    dy8 = rng.normal(size=(2, 5, 8))

    lin = Linear(8, 8)
    check("1 Linear dx", _grad_ok(lin, x, dy8))
    check("1 Linear dW", _grad_ok(lin, x, dy8, "W", lin.W))
    check("1 Linear db", _grad_ok(lin, x, dy8, "b", lin.b))

    ln = LayerNorm(8)
    ln.g[:] = rng.normal(1, 0.2, 8)
    ln.b[:] = rng.normal(0, 0.2, 8)
    check("2 LayerNorm dx", _grad_ok(ln, x, dy8))
    check("2 LayerNorm dg", _grad_ok(ln, x, dy8, "g", ln.g))
    check("2 LayerNorm db", _grad_ok(ln, x, dy8, "b", ln.b))
    out = ln.forward(x)
    xh = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
    check("2 normalises per token", close(xh.mean(-1), np.zeros((2, 5)), atol=1e-12))
    # Invariance is exact only in the eps -> 0 limit: eps is added to the variance, so it
    # bites differently once the input is rescaled. At eps=1e-5 the discrepancy is ~1e-6,
    # which is worth seeing — it is why a "LayerNorm is scale-invariant" argument is about
    # the mathematics, not about the floating-point implementation.
    check("2 is shift/scale invariant (up to eps)",
          close(ln.forward(3.0 * x + 7.0), out, atol=1e-4),
          f"max deviation {np.abs(ln.forward(3.0 * x + 7.0) - out).max():.2e}")

    g = GELU()
    check("3 GELU dx", _grad_ok(g, x, dy8))
    p = softmax(rng.normal(size=(4, 6)))
    dp = rng.normal(size=(4, 6))
    num = numerical_grad(lambda t: float((softmax(t) * dp).sum()), rng.normal(size=(4, 6)))
    z0 = rng.normal(size=(4, 6))
    check("3 softmax VJP", close(softmax_backward(softmax(z0), dp),
                                 numerical_grad(lambda t: float((softmax(t) * dp).sum()), z0.copy()),
                                 atol=1e-6))

    attn = MultiHeadAttention(8, 2)
    check("4 MHA dx", _grad_ok(attn, x, dy8))
    check("4 MHA dW_qkv", close(
        (attn.forward(x), attn.backward(dy8), attn.qkv.grads["W"])[2],
        numerical_grad(lambda p: _obj(attn, attn.qkv.W, p, x, dy8), attn.qkv.W.copy()), atol=1e-6))
    y1 = attn.forward(x)
    x2 = x.copy()
    x2[:, -1] += 1.0
    check("4 causality", close(attn.forward(x2)[:, :-1], y1[:, :-1], atol=1e-12))

    blk = TransformerBlock(8, 2)
    check("5 block dx", _grad_ok(blk, x, dy8))

    # The oracle: same weights in PyTorch, compare forward and every parameter gradient.
    try:
        import torch

        check("5 matches torch", _torch_agreement(blk, x, dy8))
    except ImportError:
        print("   (torch not installed - skipping the oracle comparison)")

    # 6: does it learn? The copy task is unlearnable without attending back n positions.
    loss0, acc = train_copy_task(steps=400)
    check("6 tiny LM learns the copy task", acc > 0.95, f"second-half accuracy {acc:.3f}, loss {loss0:.4f}")

    # 7: the classics.
    gen = np.random.default_rng(1)
    true_c = np.array([[0.0, 0.0], [6.0, 6.0], [0.0, 6.0]])
    X = np.repeat(true_c, 120, axis=0) + gen.normal(0, 0.45, (360, 2))
    C, labels, inertia = kmeans(X, 3, gen=gen)
    # Compare as *sets*: k-means has no canonical labelling, so match each true centre to
    # its nearest recovered one rather than sorting rows (which is not a stable key).
    err = np.sqrt(((true_c[:, None, :] - C[None]) ** 2).sum(-1)).min(1).max()
    check("7 k-means finds the clusters", err < 0.25,
          f"worst centre error {err:.3f}, inertia {inertia:.1f}")
    pp = [kmeans(X, 3, gen=np.random.default_rng(s), seed_pp=True)[2] for s in range(20)]
    rnd = [kmeans(X, 3, gen=np.random.default_rng(s), seed_pp=False)[2] for s in range(20)]
    check("7 k-means++ is no worse on average", np.mean(pp) <= np.mean(rnd) + 1e-9,
          f"++ {np.mean(pp):.1f} vs random {np.mean(rnd):.1f}")

    s = gen.normal(size=400)
    lab = (gen.random(400) < 1 / (1 + np.exp(-s))).astype(int)
    check("7 AUC matches the O(n²) definition", close(auc(s, lab), auc_quadratic(s, lab), atol=1e-12))
    tied = np.round(gen.normal(size=200), 1)
    lab2 = (gen.random(200) < 0.5).astype(int)
    check("7 AUC handles ties", close(auc(tied, lab2), auc_quadratic(tied, lab2), atol=1e-12))
    check("7 AUC of a perfect ranker is 1", close(auc(np.arange(10.0), np.r_[np.zeros(5), np.ones(5)]), 1.0))
    check("7 AUC of a reversed ranker is 0", close(auc(np.arange(10.0), np.r_[np.ones(5), np.zeros(5)]), 0.0))

    return 0 if report() else 1


def _obj(layer, param, p, x, dy):
    orig = param.copy()
    param[...] = p
    out = float((layer.forward(x) * dy).sum())
    param[...] = orig
    return out


def _torch_agreement(blk: TransformerBlock, x: np.ndarray, dy: np.ndarray) -> bool:
    """Rebuild the same block in PyTorch, share the weights, compare gradients."""
    import torch
    import torch.nn.functional as F

    d_model = x.shape[-1]
    H = blk.attn.H
    t = lambda a: torch.tensor(a, dtype=torch.float64, requires_grad=True)

    xt = t(x)
    g1, b1 = t(blk.ln1.g), t(blk.ln1.b)
    Wqkv, Wo = t(blk.attn.qkv.W), t(blk.attn.proj.W)
    g2, b2 = t(blk.ln2.g), t(blk.ln2.b)
    W1, bb1 = t(blk.fc1.W), t(blk.fc1.b)
    W2, bb2 = t(blk.fc2.W), t(blk.fc2.b)

    B, T, C = xt.shape
    h_in = F.layer_norm(xt, (C,), g1, b1, eps=1e-5)
    qkv = h_in @ Wqkv
    q, k, v = (z.view(B, T, H, C // H).transpose(1, 2) for z in qkv.split(C, dim=-1))
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    o = o.transpose(1, 2).reshape(B, T, C) @ Wo
    h = xt + o
    m = F.layer_norm(h, (C,), g2, b2, eps=1e-5)
    y = h + (F.gelu(m @ W1 + bb1, approximate="tanh") @ W2 + bb2)

    y_np = blk.forward(x)
    if not close(y_np, y.detach().numpy(), atol=1e-9):
        return False
    y.backward(torch.tensor(dy, dtype=torch.float64))
    dx = blk.backward(dy)

    pairs = [
        (dx, xt.grad), (blk.ln1.grads["g"], g1.grad), (blk.ln1.grads["b"], b1.grad),
        (blk.attn.qkv.grads["W"], Wqkv.grad), (blk.attn.proj.grads["W"], Wo.grad),
        (blk.ln2.grads["g"], g2.grad), (blk.ln2.grads["b"], b2.grad),
        (blk.fc1.grads["W"], W1.grad), (blk.fc1.grads["b"], bb1.grad),
        (blk.fc2.grads["W"], W2.grad), (blk.fc2.grads["b"], bb2.grad),
    ]
    return all(close(a, b.numpy(), atol=1e-8) for a, b in pairs)


def train_copy_task(steps: int = 400, n: int = 8, vocab: int = 12, d_model: int = 64,
                    n_heads: int = 4, batch: int = 64, lr: float = 3e-3):
    global rng
    rng = np.random.default_rng(0)
    gen = np.random.default_rng(1)
    model = TinyLM(vocab, d_model, n_heads, max_len=2 * n)
    opt = Adam(model.params(), lr=lr)
    loss = float("nan")
    for _ in range(steps):
        seq = make_copy_batch(batch, n, vocab, gen)
        loss, _ = model.forward(seq[:, :-1], seq[:, 1:])
        model.backward()
        opt.step(model.grad_dict())

    seq = make_copy_batch(256, n, vocab, gen)
    _, logits = model.forward(seq[:, :-1], seq[:, 1:])
    pred = logits.argmax(-1)
    # Only score the second half, where the answer is determined by the prefix.
    acc = float((pred[:, n - 1:] == seq[:, 1:][:, n - 1:]).mean())
    return loss, acc


if __name__ == "__main__":
    raise SystemExit(main())
