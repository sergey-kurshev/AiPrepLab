"""Reference solutions for exercise 01.  Run with `python solution.py`."""

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
from checks import check, report, close

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------- step 1
def truncated_svd(A: torch.Tensor, r: int) -> torch.Tensor:
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    r = min(r, S.numel())
    return (U[:, :r] * S[:r]) @ Vh[:r]


def predicted_error(A: torch.Tensor, r: int) -> float:
    S = torch.linalg.svdvals(A)
    return float(S[r:].pow(2).sum())


# ---------------------------------------------------------------- step 2
def qk_scores(X, W_Q, W_K):
    return (X @ W_Q) @ (X @ W_K).T


def gauge_transform(W_Q, W_K, R):
    # (W_Q R)(W_K R^{-T})ᵀ = W_Q R R^{-1} W_Kᵀ = W_Q W_Kᵀ.
    return W_Q @ R, W_K @ torch.linalg.inv(R).T


# 2c: the head is determined by the two *products* W_Q W_Kᵀ (the "QK circuit", which decides
# where attention goes) and W_V W_O (the "OV circuit", which decides what is written when it
# goes there) — d_model² numbers each, not the 4 × d_model × d_head of the factors.


# ---------------------------------------------------------------- step 3
def _contraction(lam: torch.Tensor, precond: torch.Tensor | None = None, n_steps: int = 300) -> float:
    """Geometric-mean contraction of ||x|| under GD with the optimal fixed step."""
    g = lam if precond is None else lam * precond
    eta = 2.0 / (g.max() + g.min())
    x = torch.ones_like(lam)
    rates = []
    for _ in range(n_steps):
        grad = lam * x                                  # ∇(½ xᵀ diag(λ) x)
        x_new = x - eta * (grad if precond is None else precond * grad)
        rates.append(float(x_new.norm() / x.norm()))
        x = x_new / x_new.norm()                        # renormalise to avoid underflow
    tail = rates[-n_steps // 4:]
    return math.exp(sum(math.log(r) for r in tail) / len(tail))


def gd_on_quadratic(kappa: float, n_steps: int = 300, dim: int = 2):
    # dim=2 gives the classic worst-case spectrum {1, κ}, where the bound
    # (κ-1)/(κ+1) is *attained*. With a dense spectrum the transient from the
    # mid-range modes dominates for a long time and the measured rate is faster
    # than the bound — the bound is worst-case over starting points, not typical.
    lam = torch.logspace(0, math.log10(kappa), dim)
    return _contraction(lam, n_steps=n_steps), (kappa - 1) / (kappa + 1)


def gd_preconditioned(kappa: float, n_steps: int = 300, dim: int = 2):
    # Preconditioning by 1/sqrt(λ) leaves the effective spectrum sqrt(λ), i.e. a
    # condition number of sqrt(κ): the preconditioner halves log κ. That is the
    # entire value proposition of a diagonal adaptive optimiser on a badly scaled
    # problem, and it is why Adam turns a hopeless κ=10⁴ into a workable κ=10².
    lam = torch.logspace(0, math.log10(kappa), dim)
    emp = _contraction(lam, precond=1.0 / lam.sqrt(), n_steps=n_steps)
    root = math.sqrt(kappa)
    return emp, (root - 1) / (root + 1)


# 3b: (i) Adam's m/√v is a diagonal preconditioner estimated online from gradient statistics.
#     (ii) LayerNorm fixes the scale of each activation vector, which keeps the per-layer Jacobian
#          — and hence the Gauss–Newton curvature — from developing wildly different scales.
#     (iii) A good init (He/Xavier) chooses variances so each layer's Jacobian is near-isometric,
#          keeping the product over depth well conditioned at step 0. All three target κ.


# ---------------------------------------------------------------- step 4
def entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * (p + 1e-300).log()).sum()


def beta_for_mean(z: torch.Tensor, c: float, lo: float = -50.0, hi: float = 50.0) -> float:
    # E_softmax(βz)[z] is strictly increasing in β (its derivative is Var_p[z] ≥ 0), so bisect.
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        m = float((torch.softmax(mid * z, 0) * z).sum())
        if m < c:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def max_entropy_distribution(z: torch.Tensor, c: float) -> torch.Tensor:
    return torch.softmax(beta_for_mean(z, c) * z, 0)


def feasible_directions(z: torch.Tensor) -> torch.Tensor:
    """Basis for {d : Σd = 0 and z·d = 0} — the tangent space of the feasible set.

    Moving along these keeps both constraints satisfied *exactly*, which makes the
    optimality test a clean one: no penalty weight to tune, no constraint drift.
    """
    M = torch.stack([torch.ones_like(z), z])            # (2, n)
    _, _, Vh = torch.linalg.svd(M, full_matrices=True)
    return Vh[2:]                                        # (n-2, n)


# 4c: T = 1/β. T → 0 concentrates all mass on argmax z (the greedy limit); T → ∞ gives the uniform
# distribution (β = 0, constraint inactive, unconstrained max entropy). It is the `temperature`
# parameter of every sampling API, and the τ in Hinton-style distillation, where a high T exposes
# the "dark knowledge" in the ratios between non-argmax logits.


# ---------------------------------------------------------------- step 5
def negative_log_likelihood(logits: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, 0)
    return -(counts * logp).sum() / counts.sum()


def forward_kl(logits: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    p_emp = counts / counts.sum()
    logp = torch.log_softmax(logits, 0)
    return (p_emp * (p_emp.log() - logp)).sum()


# ---------------------------------------------------------------- step 6
def attention_forward(Q, K, V):
    d_k = Q.shape[-1]
    S = Q @ K.T / math.sqrt(d_k)
    P = torch.softmax(S, dim=-1)
    return P @ V, P


def attention_backward(Q, K, V, P, dO):
    d_k = Q.shape[-1]
    dV = P.T @ dO
    dP = dO @ V.T
    # Softmax VJP, applied row-wise: dS = P ∘ (dP - rowsum(dP ∘ P)).
    dS = P * (dP - (dP * P).sum(-1, keepdim=True))
    dQ = dS @ K / math.sqrt(d_k)
    dK = dS.T @ Q / math.sqrt(d_k)
    return dQ, dK, dV


# ---------------------------------------------------------------- step 7
def ridge(X, y, lam):
    d = X.shape[1]
    return torch.linalg.solve(X.T @ X + lam * torch.eye(d, dtype=X.dtype), X.T @ y)


def posterior_mean(X, y, sigma2, tau2):
    # Posterior ∝ N(y | Xw, σ²I) N(w | 0, τ²I) ⇒ precision X ᵀX/σ² + I/τ²,
    # mean = (XᵀX/σ² + I/τ²)⁻¹ XᵀY/σ² = (XᵀX + (σ²/τ²)I)⁻¹ Xᵀy.
    d = X.shape[1]
    prec = X.T @ X / sigma2 + torch.eye(d, dtype=X.dtype) / tau2
    return torch.linalg.solve(prec, X.T @ y / sigma2)


# 7b: a Laplace prior p(w) ∝ exp(-|w|/b) gives L1 with λ = 1/b. It sparsifies the posterior *mode*
# (the MAP estimate has exact zeros, because |w| is non-differentiable at 0) but not the posterior
# *mean*, which is almost surely dense. So "L1 gives sparsity" is a statement about MAP estimation,
# not about Bayesian inference under a Laplace prior — a distinction worth making out loud.


# ---------------------------------------------------------------- self-test
def main() -> int:
    torch.manual_seed(0)

    A = torch.randn(40, 30)
    for r in (1, 5, 15, 30):
        Ar = truncated_svd(A, r)
        check(f"1 r={r} error", close(float((A - Ar).pow(2).sum()), predicted_error(A, r), atol=1e-8))

    d_model, d_head, T = 64, 8, 12
    X = torch.randn(T, d_model)
    W_Q = torch.randn(d_model, d_head) / math.sqrt(d_model)
    W_K = torch.randn(d_model, d_head) / math.sqrt(d_model)
    W = W_Q @ W_K.T
    check("2a rank", int(torch.linalg.matrix_rank(W)) <= d_head)
    check("2a bilinear form", close(qk_scores(X, W_Q, W_K), X @ W @ X.T))
    R = torch.randn(d_head, d_head) + d_head * torch.eye(d_head)
    Wq2, Wk2 = gauge_transform(W_Q, W_K, R)
    check("2b invariance", close(qk_scores(X, Wq2, Wk2), qk_scores(X, W_Q, W_K), atol=1e-8))
    check("2b factors changed", not close(Wq2, W_Q))

    for kappa in (2.0, 10.0, 100.0, 1e4):
        emp, theo = gd_on_quadratic(kappa)
        check(f"3 κ={kappa:.0f}", close(emp, theo, atol=1e-9), f"{emp:.8f} vs {theo:.8f}")
    for kappa in (100.0, 1e4):
        emp, theo = gd_preconditioned(kappa)
        check(f"3b κ={kappa:.0f} preconditioned rate is (√κ-1)/(√κ+1)",
              close(emp, theo, atol=1e-9), f"{emp:.8f} vs {theo:.8f}")
        check(f"3b κ={kappa:.0f} preconditioning helps", emp < gd_on_quadratic(kappa)[0])

    z = torch.tensor([-2.0, -0.5, 0.0, 1.0, 3.0])
    p_star = max_entropy_distribution(z, 1.0)
    check("4a constraint", close(float((p_star * z).sum()), 1.0, atol=1e-8))
    check("4a normalised", close(float(p_star.sum()), 1.0, atol=1e-12))
    # Anything else that is feasible must have strictly less entropy. Move along the
    # tangent space of the constraint set so feasibility is exact by construction.
    N = feasible_directions(z)
    H_star = float(entropy(p_star))
    worst_violation, n_tested = -math.inf, 0
    for _ in range(500):
        d = torch.randn(N.shape[0]) @ N
        d = d / d.norm()
        for t in (0.02, 0.05, 0.1):
            p = p_star + t * d
            if float(p.min()) <= 0:
                continue
            n_tested += 1
            assert abs(float(p.sum()) - 1.0) < 1e-12 and abs(float((p * z).sum()) - 1.0) < 1e-12
            worst_violation = max(worst_violation, float(entropy(p)) - H_star)
    check("4b max-entropy is maximal on the feasible set", worst_violation < 0,
          f"{n_tested} exactly-feasible perturbations, best was {worst_violation:.2e} worse")

    counts = torch.tensor([12.0, 30.0, 5.0, 53.0])
    logits = torch.randn(4, requires_grad=True)
    p_emp = counts / counts.sum()
    check("5 NLL = KL + H",
          close(float(negative_log_likelihood(logits, counts)),
                float(forward_kl(logits, counts)) + float(entropy(p_emp)), atol=1e-10))
    # Plain gradient descent, not Adam: the objective is convex in the logits with
    # gradient p - y, so a fixed step converges geometrically to machine precision.
    # Adam's non-decaying step size leaves it oscillating at ~1e-3 forever, which is a
    # useful thing to have seen once.
    opt = torch.optim.SGD([logits], lr=1.0)
    for _ in range(5000):
        loss = negative_log_likelihood(logits, counts)
        opt.zero_grad(); loss.backward(); opt.step()
    p_fit = torch.softmax(logits, 0).detach()
    check("5 MLE = empirical", close(p_fit, p_emp, atol=1e-4),
          f"max abs error {float((p_fit - p_emp).abs().max()):.2e}")
    l2 = torch.randn(4, requires_grad=True)
    negative_log_likelihood(l2, counts).backward()
    check("5 grad = p - y", close(l2.grad, torch.softmax(l2, 0) - p_emp, atol=1e-10))

    T, d_k, d_v = 7, 5, 4
    Q, K = torch.randn(T, d_k), torch.randn(T, d_k)
    V, dO = torch.randn(T, d_v), torch.randn(T, d_v)
    O, P = attention_forward(Q, K, V)
    dQ, dK, dV = attention_backward(Q, K, V, P, dO)
    Qa, Ka, Va = (t.clone().requires_grad_(True) for t in (Q, K, V))
    attention_forward(Qa, Ka, Va)[0].backward(dO)
    check("6 dV", close(dV, Va.grad, atol=1e-10))
    check("6 dQ", close(dQ, Qa.grad, atol=1e-10))
    check("6 dK", close(dK, Ka.grad, atol=1e-10))
    check("6 gradcheck", torch.autograd.gradcheck(
        lambda q, k, v: attention_forward(q, k, v)[0],
        (Q.clone().requires_grad_(True), K.clone().requires_grad_(True), V.clone().requires_grad_(True)),
        eps=1e-6, atol=1e-6))
    dP = dO @ V.T
    check("6b rowsum identity", close((dP * P).sum(-1), (dO * O).sum(-1), atol=1e-10))
    grads = []
    for scale in (1.0, 5.0, 20.0):
        _, Ps = attention_forward(Q * scale, K, V)
        grads.append(float(attention_backward(Q * scale, K, V, Ps, dO)[0].norm()))
    check("6b saturation shrinks the gradient", grads[2] < grads[0], f"{grads[0]:.3e} -> {grads[2]:.3e}")

    n, d = 50, 8
    Xd = torch.randn(n, d)
    sigma2, tau2 = 0.25, 2.0
    y = Xd @ torch.randn(d) + math.sqrt(sigma2) * torch.randn(n)
    check("7 ridge == posterior mean",
          close(ridge(Xd, y, sigma2 / tau2), posterior_mean(Xd, y, sigma2, tau2), atol=1e-8))
    check("7 λ→0 is OLS", close(ridge(Xd, y, 1e-12), torch.linalg.lstsq(Xd, y).solution, atol=1e-5))
    check("7 shrinkage", float(ridge(Xd, y, 10.0).norm()) < float(ridge(Xd, y, 0.1).norm()))

    return 0 if report() else 1


if __name__ == "__main__":
    raise SystemExit(main())
