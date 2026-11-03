# %% [markdown]
# # Exercise 01 — Theoretical Foundations, Made Numerical
#
# Seven steps. Each one turns a statement from `README.md` into something you can measure, on the
# principle that a theorem you have verified numerically is a theorem you can defend at a
# whiteboard.
#
# Work in `float64` throughout unless told otherwise — several of these checks are tight enough
# that float32 rounding would muddy the result, and knowing *when* precision matters is itself
# part of the lesson.

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
from checks import check, report, close

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
print("torch", torch.__version__, "| default dtype", torch.get_default_dtype())

# %% [markdown]
# ---
# ## Step 1 — Eckart–Young, verified
#
# The claim: truncating the SVD at rank $r$ is the *optimal* rank-$r$ approximation, and the
# squared Frobenius error is exactly $\sum_{i>r}\sigma_i^2$.
#
# Implement the truncation, confirm the error formula, then try to beat it by random search. You
# will not, and the failure is the point.

# %%
def truncated_svd(A: torch.Tensor, r: int) -> torch.Tensor:
    """Best rank-`r` approximation of A in Frobenius norm."""
    # TODO: torch.linalg.svd(A, full_matrices=False), keep the top r components, recombine.
    raise NotImplementedError


def predicted_error(A: torch.Tensor, r: int) -> float:
    """The Eckart–Young prediction for ||A - A_r||_F^2, from the spectrum alone."""
    # TODO
    raise NotImplementedError


A = torch.randn(40, 30)
try:
    for r in (1, 5, 15, 30):
        Ar = truncated_svd(A, r)
        actual = float((A - Ar).pow(2).sum())
        check(f"r={r}: error matches spectrum tail", close(actual, predicted_error(A, r), atol=1e-8),
              f"{actual:.4f}")
        check(f"r={r}: rank is correct", int(torch.linalg.matrix_rank(Ar)) == min(r, 30))

    # Can random perturbation do better? (Eckart–Young says no.)
    r = 5
    best = float((A - truncated_svd(A, r)).pow(2).sum())
    beaten = False
    for _ in range(2000):
        B = torch.randn(40, r) @ torch.randn(r, 30) * 0.1 + truncated_svd(A, r)
        # re-truncate to stay on the rank-r manifold
        B = truncated_svd(B, r)
        if float((A - B).pow(2).sum()) < best - 1e-9:
            beaten = True
    check("random search cannot beat the SVD", not beaten)
except Exception as _e:
    check("step 1", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# ---
# ## Step 2 — An attention head is a low-rank bilinear form
#
# With $Q = XW_Q$ and $K = XW_K$, the scores are $X(W_QW_K^{\top})X^{\top}$. Two claims to verify:
#
# **2a.** $\mathrm{rank}(W_QW_K^{\top}) \le d_{\text{head}}$, so a head cannot express an arbitrary
# token-pair interaction.
#
# **2b.** $W_Q$ and $W_K$ are individually meaningless: for any invertible $R$, the pair
# $(W_QR,\; W_KR^{-\top})$ produces identical scores. Construct the reparametrisation and show the
# scores are unchanged while the individual matrices are not.

# %%
d_model, d_head, T = 64, 8, 12
X = torch.randn(T, d_model)
W_Q = torch.randn(d_model, d_head) / math.sqrt(d_model)
W_K = torch.randn(d_model, d_head) / math.sqrt(d_model)


def qk_scores(X, W_Q, W_K):
    """Un-scaled attention scores X W_Q W_Kᵀ Xᵀ, computed from the projections."""
    # TODO
    raise NotImplementedError


def gauge_transform(W_Q, W_K, R):
    """Return (W_Q', W_K') that leave every score unchanged, given invertible R."""
    # TODO
    raise NotImplementedError


try:
    W = W_Q @ W_K.T
    check("2a rank <= d_head", int(torch.linalg.matrix_rank(W)) <= d_head, f"rank {int(torch.linalg.matrix_rank(W))}")
    check("2a scores agree with the bilinear form", close(qk_scores(X, W_Q, W_K), X @ W @ X.T))

    R = torch.randn(d_head, d_head)
    R = R + d_head * torch.eye(d_head)          # keep it well away from singular
    Wq2, Wk2 = gauge_transform(W_Q, W_K, R)
    check("2b scores are invariant", close(qk_scores(X, Wq2, Wk2), qk_scores(X, W_Q, W_K), atol=1e-8))
    check("2b but the factors changed", not close(Wq2, W_Q))
    check("2b and the product did not", close(Wq2 @ Wk2.T, W, atol=1e-8))
except Exception as _e:
    check("step 2", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **2c. (discussion, no code)** Given 2b, what is the *smallest* set of quantities that fully
# determines a single attention head's behaviour? Write your answer down before reading the
# Transformer Circuits reference — this is the "QK circuit / OV circuit" observation, and it is a
# strong thing to be able to state unprompted in an interview.

# %% [markdown]
# ---
# ## Step 3 — Conditioning is what a plateau is made of
#
# Run gradient descent on $f(x) = \tfrac12 x^{\top}Ax$ with $A = \mathrm{diag}(\lambda)$ and the
# optimal fixed step size $\eta = 2/(\lambda_{\max} + \lambda_{\min})$, and measure the empirical
# per-step contraction factor. Theory says it converges to $(\kappa-1)/(\kappa+1)$.

# %%
def gd_on_quadratic(kappa: float, n_steps: int = 300, dim: int = 2):
    """Return (empirical contraction factor, theoretical factor) for a quadratic of condition κ.

    Use the *worst-case* spectrum: `dim=2` with eigenvalues {1, κ}. The bound is attained
    there. With a dense spectrum the mid-range modes decay much faster and dominate the
    measurement for thousands of steps, so the observed rate looks better than the bound —
    worth seeing once, which is what the `dim=50` variant below is for.
    """
    # TODO:
    #   lam = torch.logspace(0, log10(kappa), dim); x0 = ones; eta = 2/(lmax + lmin);
    #   step: x <- x - eta * (lam * x);
    #   record ||x_new|| / ||x||, then RENORMALISE x (otherwise it underflows at large κ);
    #   return the geometric mean of the last quarter of the recorded ratios.
    raise NotImplementedError


try:
    print(f"{'kappa':>10} {'empirical':>14} {'theory':>14} {'steps per digit':>18}")
    for kappa in (2.0, 10.0, 100.0, 1e4):
        emp, theo = gd_on_quadratic(kappa)
        steps = math.log(10) / -math.log(theo) if theo < 1 else float("inf")
        print(f"{kappa:>10.0f} {emp:>14.8f} {theo:>14.8f} {steps:>18.1f}")
        check(f"κ={kappa:.0f} matches theory", close(emp, theo, atol=1e-9))

    # And the caveat, made visible: a dense spectrum converges faster than worst case.
    dense, theo = gd_on_quadratic(1e4, dim=50)
    print()
    print(f"dense spectrum at κ=1e4: {dense:.6f} vs worst-case bound {theo:.6f}")
    check("dense spectrum beats the worst case", dense < theo)
except Exception as _e:
    check("step 3", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **3b.** Now the point of adaptive optimisers. Same problem, but divide each coordinate's
# gradient by $\sqrt{\lambda_i}$ — a crude stand-in for Adam's $1/\sqrt{v_t}$.
#
# Predict the answer before running it: if the effective spectrum becomes $\sqrt{\lambda}$, what
# is the new condition number, and what rate should you see?
#
# Then answer in a sentence each: how do (i) Adam, (ii) LayerNorm, (iii) He/Xavier initialisation
# each attack this same quantity?

# %%
def gd_preconditioned(kappa: float, n_steps: int = 300, dim: int = 2):
    """Return (empirical, theoretical) contraction with a diagonal 1/sqrt(λ) preconditioner."""
    # TODO — the theoretical value is *not* (κ-1)/(κ+1). Work out what it is.
    raise NotImplementedError


try:
    for kappa in (100.0, 1e4):
        plain, _ = gd_on_quadratic(kappa)
        emp, theo = gd_preconditioned(kappa)
        print(f"κ={kappa:>8.0f}  plain {plain:.8f}   preconditioned {emp:.8f}  (theory {theo:.8f})")
        check(f"κ={kappa:.0f}: rate is (√κ-1)/(√κ+1)", close(emp, theo, atol=1e-9))
        check(f"κ={kappa:.0f}: preconditioning helps", emp < plain)
except Exception as _e:
    check("step 3b", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# ---
# ## Step 4 — Softmax falls out of maximum entropy
#
# Claim: among all distributions $p$ on $n$ outcomes with a prescribed mean score
# $\mathbb{E}_p[z] = c$, the entropy-maximising one is $p \propto e^{\beta z}$ for the unique
# $\beta$ matching the constraint.
#
# **4a.** Find that $\beta$ by bisection. **4b.** Confirm no other feasible distribution has higher
# entropy, by projected gradient ascent on the entropy starting from random feasible points.

# %%
z = torch.tensor([-2.0, -0.5, 0.0, 1.0, 3.0])


def beta_for_mean(z: torch.Tensor, c: float, lo: float = -50.0, hi: float = 50.0) -> float:
    """Find β such that E[z] under softmax(β z) equals c. E[z] is increasing in β."""
    # TODO: bisection.
    raise NotImplementedError


def max_entropy_distribution(z: torch.Tensor, c: float) -> torch.Tensor:
    """The maximum-entropy distribution with E_p[z] = c."""
    # TODO
    raise NotImplementedError


def feasible_directions(z: torch.Tensor) -> torch.Tensor:
    """Basis for {d : sum(d) = 0 and z.d = 0} — the tangent space of the feasible set.

    Hint: it is the null space of the 2 x n matrix whose rows are `ones` and `z`.
    `torch.linalg.svd(M, full_matrices=True)` hands it to you in the trailing rows of Vh.
    """
    # TODO
    raise NotImplementedError


def entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * (p + 1e-300).log()).sum()


target_c = 1.0
try:
    p_star = max_entropy_distribution(z, target_c)
    check("4a satisfies the constraint", close(float((p_star * z).sum()), target_c, atol=1e-8))
    check("4a is a distribution", close(float(p_star.sum()), 1.0, atol=1e-12))
    check("4a has softmax form", close(p_star, torch.softmax(beta_for_mean(z, target_c) * z, 0), atol=1e-10))

    # 4b: can anything else feasible beat it? Rather than optimising with a penalty (which
    # never lands exactly on the constraint set), move along the *tangent space* of the
    # constraints — directions d with Σd = 0 and z·d = 0 — so feasibility is exact.
    N = feasible_directions(z)
    H_star = float(entropy(p_star))
    best_other, n_tested = -math.inf, 0
    for _ in range(500):
        d = torch.randn(N.shape[0]) @ N
        d = d / d.norm()
        for t in (0.02, 0.05, 0.1):
            p = p_star + t * d
            if float(p.min()) <= 0:
                continue          # left the simplex
            n_tested += 1
            assert abs(float(p.sum()) - 1.0) < 1e-12
            assert abs(float((p * z).sum()) - target_c) < 1e-12
            best_other = max(best_other, float(entropy(p)))
    print(f"max-entropy H = {H_star:.6f}   best of {n_tested} feasible neighbours = {best_other:.6f}")
    check("4b nothing feasible beats it", best_other < H_star,
          f"closest competitor was {H_star - best_other:.2e} short")
except Exception as _e:
    check("step 4", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **4c.** Interpret $\beta$. Set $T = 1/\beta$ and describe the limits $T \to 0$ and
# $T \to \infty$. Where have you seen this parameter in a sampling API, and where in knowledge
# distillation?

# %% [markdown]
# ---
# ## Step 5 — MLE = cross-entropy = forward KL
#
# Three quantities that differ only by a constant. Show it numerically, and identify the constant.

# %%
counts = torch.tensor([12.0, 30.0, 5.0, 53.0])     # observed category counts
logits = torch.randn(4, requires_grad=True)


def negative_log_likelihood(logits: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """-(1/N) Σ log p_θ(x_i), computed from counts."""
    # TODO
    raise NotImplementedError


def forward_kl(logits: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """D_KL(p_empirical || p_θ)."""
    # TODO
    raise NotImplementedError


try:
    p_emp = counts / counts.sum()
    nll = negative_log_likelihood(logits, counts)
    kl = forward_kl(logits, counts)
    H = entropy(p_emp)
    check("5 NLL = KL + H(empirical)", close(float(nll), float(kl) + float(H), atol=1e-10),
          f"NLL {float(nll):.5f} = KL {float(kl):.5f} + H {float(H):.5f}")

    # Fit by gradient descent and confirm the MLE is the empirical distribution.
    # Plain GD, not Adam: the objective is convex in the logits and a fixed step converges
    # geometrically. Try it with Adam(lr=0.1) afterwards and watch it stall around 1e-3 —
    # a non-decaying adaptive step size cannot reach an exact optimum.
    opt = torch.optim.SGD([logits], lr=1.0)
    for _ in range(5000):
        loss = negative_log_likelihood(logits, counts)
        opt.zero_grad(); loss.backward(); opt.step()
    p_fit = torch.softmax(logits, 0).detach()
    check("5 MLE recovers the empirical distribution", close(p_fit, p_emp, atol=1e-6),
          f"max abs error {float((p_fit - p_emp).abs().max()):.2e}")
    check("5 KL is driven to zero", float(forward_kl(logits, counts)) < 1e-6)

    # And the gradient identity that makes cross-entropy pleasant.
    logits2 = torch.randn(4, requires_grad=True)
    negative_log_likelihood(logits2, counts).backward()
    check("5 ∂L/∂logits = p - y", close(logits2.grad, torch.softmax(logits2, 0) - p_emp, atol=1e-10))
except Exception as _e:
    check("step 5", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# ---
# ## Step 6 — The attention backward pass, by hand
#
# This is the derivation from §3.2 of the article, and it is a genuine whiteboard question. With
# $S = QK^{\top}/\sqrt{d_k}$, $P = \mathrm{softmax}(S)$, $O = PV$:
#
# $$\bar V = P^{\top}\bar O,\quad \bar P = \bar O V^{\top},\quad
#   \bar S = P \odot (\bar P - \mathrm{rowsum}(\bar P \odot P)),\quad
#   \bar Q = \tfrac{1}{\sqrt{d_k}}\bar S K,\quad \bar K = \tfrac{1}{\sqrt{d_k}}\bar S^{\top}Q.$$
#
# Implement it with **no autograd**, and check against autograd in double precision.

# %%
def attention_forward(Q, K, V):
    """Return (O, P) for un-batched (T, d) inputs."""
    # TODO
    raise NotImplementedError


def attention_backward(Q, K, V, P, dO):
    """Return (dQ, dK, dV) using only matrix algebra — no autograd, no .backward()."""
    # TODO
    raise NotImplementedError


T, d_k, d_v = 7, 5, 4
Q = torch.randn(T, d_k)
K = torch.randn(T, d_k)
V = torch.randn(T, d_v)
dO = torch.randn(T, d_v)

try:
    O, P = attention_forward(Q, K, V)
    dQ, dK, dV = attention_backward(Q, K, V, P, dO)

    Qa, Ka, Va = (t.clone().requires_grad_(True) for t in (Q, K, V))
    Oa, _ = attention_forward(Qa, Ka, Va)
    Oa.backward(dO)

    check("6 dV", close(dV, Va.grad, atol=1e-10))
    check("6 dQ", close(dQ, Qa.grad, atol=1e-10))
    check("6 dK", close(dK, Ka.grad, atol=1e-10))

    # gradcheck is the real standard: finite differences in float64.
    ok = torch.autograd.gradcheck(lambda q, k, v: attention_forward(q, k, v)[0],
                                  (Q.clone().requires_grad_(True),
                                   K.clone().requires_grad_(True),
                                   V.clone().requires_grad_(True)), eps=1e-6, atol=1e-6)
    check("6 forward passes gradcheck", ok)
except Exception as _e:
    check("step 6", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **6b.** Two structural questions the derivation answers. Verify each numerically.
#
# 1. **Saturation kills gradients.** Scale `Q` by a large factor so the softmax is nearly one-hot,
#    and measure $\|\bar Q\|$. Relate this to the $1/\sqrt{d_k}$ argument.
# 2. **You never need to store $P$.** The correction term $\mathrm{rowsum}(\bar P \odot P)$ equals
#    $\mathrm{rowsum}(\bar O \odot O)$. Show it. This identity is what makes FlashAttention's
#    backward pass $O(T)$ in memory rather than $O(T^2)$ — module 07.

# %%
try:
    for scale in (1.0, 5.0, 20.0):
        Qs = Q * scale
        O_s, P_s = attention_forward(Qs, K, V)
        dQ_s, _, _ = attention_backward(Qs, K, V, P_s, dO)
        ent = float(-(P_s * (P_s + 1e-300).log()).sum(-1).mean())
        print(f"scale {scale:>5.1f}   entropy {ent:>7.4f} nats   ||dQ|| {float(dQ_s.norm()):>10.3e}")

    O_, P_ = attention_forward(Q, K, V)
    dP = dO @ V.T
    lhs = (dP * P_).sum(-1)
    rhs = (dO * O_).sum(-1)
    check("6b rowsum(dP∘P) == rowsum(dO∘O)", close(lhs, rhs, atol=1e-10))
except Exception as _e:
    check("step 6b", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# ---
# ## Step 7 — L2 regularisation is a Gaussian prior, with a constant
#
# For the linear-Gaussian model $y = Xw + \varepsilon$, $\varepsilon \sim \mathcal N(0, \sigma^2 I)$,
# with prior $w \sim \mathcal N(0, \tau^2 I)$, the posterior mean equals the ridge solution with
# $\lambda = \sigma^2/\tau^2$. Get the constant right — that is the part people fumble.

# %%
def ridge(X, y, lam):
    """argmin ||Xw - y||² + λ||w||²."""
    # TODO
    raise NotImplementedError


def posterior_mean(X, y, sigma2, tau2):
    """E[w | X, y] under the linear-Gaussian model above."""
    # TODO
    raise NotImplementedError


n, d = 50, 8
Xd = torch.randn(n, d)
w_true = torch.randn(d)
sigma2, tau2 = 0.25, 2.0
y = Xd @ w_true + math.sqrt(sigma2) * torch.randn(n)

try:
    lam = sigma2 / tau2
    check("7 ridge == posterior mean", close(ridge(Xd, y, lam), posterior_mean(Xd, y, sigma2, tau2), atol=1e-8))
    check("7 λ→0 recovers OLS", close(ridge(Xd, y, 1e-12), torch.linalg.lstsq(Xd, y).solution, atol=1e-5))
    check("7 stronger prior shrinks more",
          float(ridge(Xd, y, 10.0).norm()) < float(ridge(Xd, y, 0.1).norm()))
except Exception as _e:
    check("step 7", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **7b. (discussion)** What prior gives L1? What does it do to the posterior *mode* versus the
# posterior *mean*, and why does that distinction matter for whether L1 actually produces sparsity?

# %% [markdown]
# ---
# ## Revisiting the spine
#
# You have now seen attention at two levels: as tensor code (module 00) and as a differentiable
# bilinear-plus-softmax operator whose gradient you can derive (module 01). Module 02 asks for
# the third: a complete implementation in numpy, forward and backward, inside a transformer block
# that trains.
#
# **Stretch goals:**
#
# 1. Compute the spectrum of $W_QW_K^{\top}$ for a real pretrained model (e.g. GPT-2 small via
#    `transformers`) and plot the singular-value decay per head. How many heads are effectively
#    rank 1? Compare against the random-initialisation baseline.
# 2. Estimate the Lipschitz constant of self-attention as a function of $\|Q\|, \|K\|, \|V\|$, and
#    check it empirically. Is attention Lipschitz at all on unbounded inputs? (It is not — find out
#    why, and what that implies for training stability.)
# 3. Show empirically that the empirical Fisher information matrix approximates the Hessian of the
#    NLL at the optimum, on the categorical model of step 5.
