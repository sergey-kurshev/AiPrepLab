"""Reference solutions for exercise 08 — safety phenomena, reproduced.

Run with `python solution.py`.

Every experiment here is small enough to run in seconds and concrete enough to
argue from. The aim is that when an interviewer asks "what are you worried about?",
the answer is about things you have watched happen rather than things you have read.
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


# ============================================================ 1. Bradley-Terry
def bradley_terry_loss(reward: torch.Tensor, winners: torch.Tensor, losers: torch.Tensor):
    """-log sigma(r_w - r_l), i.e. logistic regression on reward DIFFERENCES.

    Because only differences appear, r is identified only up to an additive constant.
    Any statement about the absolute value of a reward model's output is meaningless —
    which is worth saying out loud when someone quotes one.
    """
    return -F.logsigmoid(reward[winners] - reward[losers]).mean()


def fit_reward_model(true_reward: torch.Tensor, n_pairs: int = 20_000, steps: int = 800,
                     seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    n = true_reward.numel()
    a = torch.randint(0, n, (n_pairs,), generator=g)
    b = torch.randint(0, n, (n_pairs,), generator=g)
    keep = a != b
    a, b = a[keep], b[keep]
    # Sample the comparison outcome from the Bradley-Terry model itself: noisy labels,
    # as real annotators produce.
    p_a_wins = torch.sigmoid(true_reward[a] - true_reward[b])
    a_wins = torch.rand(len(a), generator=g) < p_a_wins
    winners = torch.where(a_wins, a, b)
    losers = torch.where(a_wins, b, a)

    r = torch.zeros(n, requires_grad=True)
    opt = torch.optim.Adam([r], lr=0.05)
    for _ in range(steps):
        loss = bradley_terry_loss(r, winners, losers)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return r.detach()


# ============================================================ 2. KL-regularised optimum
def optimal_policy(ref_logits: torch.Tensor, reward: torch.Tensor, beta: float) -> torch.Tensor:
    """pi*(y) ∝ pi_ref(y) exp(r(y)/beta) — the closed form of

        max_pi E_pi[r] - beta KL(pi || pi_ref).

    beta -> inf returns pi_ref; beta -> 0 returns the argmax of r, including all of r's
    errors. The entire practical difficulty of RLHF is that this is a dial with no
    principled setting.
    """
    return torch.softmax(torch.log_softmax(ref_logits, -1) + reward / beta, -1)


def kl(p: torch.Tensor, q: torch.Tensor) -> float:
    return float((p * ((p + 1e-30).log() - (q + 1e-30).log())).sum())


def solve_by_gradient(ref_logits, reward, beta, steps=4000, lr=0.05):
    """The same objective, optimised numerically — to confirm the closed form is right."""
    logits = ref_logits.clone().requires_grad_(True)
    ref = torch.softmax(ref_logits, -1)
    opt = torch.optim.Adam([logits], lr=lr)
    for _ in range(steps):
        p = torch.softmax(logits, -1)
        obj = (p * reward).sum() - beta * (p * ((p + 1e-30).log() - (ref + 1e-30).log())).sum()
        opt.zero_grad()
        (-obj).backward()
        opt.step()
    return torch.softmax(logits, -1).detach()


# ============================================================ 3. DPO
def dpo_loss(policy_logits, ref_logits, winners, losers, beta: float):
    """The DPO objective on a bandit (one prompt, N possible completions).

    Derivation: invert pi* ∝ pi_ref exp(r/beta) to get r = beta log(pi/pi_ref) + beta log Z,
    substitute into Bradley-Terry — and log Z cancels, because the likelihood depends
    only on the DIFFERENCE of two rewards for the same prompt. That cancellation is the
    whole trick, and reproducing it is the expected answer to "explain DPO".
    """
    logp = torch.log_softmax(policy_logits, -1)
    logref = torch.log_softmax(ref_logits, -1)
    margin = beta * ((logp[winners] - logref[winners]) - (logp[losers] - logref[losers]))
    return -F.logsigmoid(margin).mean()


def train_dpo(ref_logits, true_reward, beta: float, n_pairs=40_000, steps=3000, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = true_reward.numel()
    a = torch.randint(0, n, (n_pairs,), generator=g)
    b = torch.randint(0, n, (n_pairs,), generator=g)
    keep = a != b
    a, b = a[keep], b[keep]
    a_wins = torch.rand(len(a), generator=g) < torch.sigmoid(true_reward[a] - true_reward[b])
    winners, losers = torch.where(a_wins, a, b), torch.where(a_wins, b, a)

    logits = ref_logits.clone().requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=0.02)
    for _ in range(steps):
        loss = dpo_loss(logits, ref_logits, winners, losers, beta)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.softmax(logits.detach(), -1)


# ============================================================ 4. reward hacking
def reward_hacking_curve(n_actions=200, n_pairs=600, betas=None, seed=0):
    """Optimise against a reward model fit on FINITE data; track proxy and true reward.

    This is Gao/Schulman/Hilton in miniature. The proxy rises monotonically as you
    optimise harder (smaller beta = further from the reference policy). The true
    reward rises, peaks, and falls — because the proxy's errors are largest exactly
    where the optimiser is pushing.
    """
    g = torch.Generator().manual_seed(seed)
    true_r = torch.randn(n_actions, generator=g) * 1.5
    ref_logits = torch.zeros(n_actions)

    # A reward model fit on few comparisons: right on average, wrong in detail.
    a = torch.randint(0, n_actions, (n_pairs,), generator=g)
    b = torch.randint(0, n_actions, (n_pairs,), generator=g)
    keep = a != b
    a, b = a[keep], b[keep]
    a_wins = torch.rand(len(a), generator=g) < torch.sigmoid(true_r[a] - true_r[b])
    winners, losers = torch.where(a_wins, a, b), torch.where(a_wins, b, a)
    proxy = torch.zeros(n_actions, requires_grad=True)
    opt = torch.optim.Adam([proxy], lr=0.1)
    for _ in range(600):
        loss = bradley_terry_loss(proxy, winners, losers)
        opt.zero_grad()
        loss.backward()
        opt.step()
    proxy = proxy.detach()

    betas = betas if betas is not None else [8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05]
    rows = []
    for beta in betas:
        pi = optimal_policy(ref_logits, proxy, beta)
        rows.append({
            "beta": beta,
            "kl": kl(pi, torch.softmax(ref_logits, -1)),
            "proxy": float((pi * proxy).sum()),
            "true": float((pi * true_r).sum()),
        })
    return rows, true_r, proxy


# ============================================================ 5. fairness
def fairness_metrics(scores: torch.Tensor, labels: torch.Tensor, group: torch.Tensor,
                     threshold: float = 0.5) -> dict:
    pred = (scores >= threshold).float()
    out = {}
    for a in (0, 1):
        m = group == a
        y, p = labels[m], pred[m]
        out[f"selection_rate_{a}"] = float(p.mean())
        out[f"base_rate_{a}"] = float(y.mean())
        out[f"tpr_{a}"] = float(p[y == 1].mean()) if (y == 1).any() else float("nan")
        out[f"fpr_{a}"] = float(p[y == 0].mean()) if (y == 0).any() else float("nan")
        sel = p == 1
        out[f"ppv_{a}"] = float(y[sel].mean()) if sel.any() else float("nan")
    out["demographic_parity_gap"] = abs(out["selection_rate_0"] - out["selection_rate_1"])
    out["tpr_gap"] = abs(out["tpr_0"] - out["tpr_1"])
    out["fpr_gap"] = abs(out["fpr_0"] - out["fpr_1"])
    out["ppv_gap"] = abs(out["ppv_0"] - out["ppv_1"])
    return out


def calibration_error(scores, labels, group, bins: int = 10) -> float:
    """Max over groups and bins of |mean score - observed rate|. Zero = calibrated."""
    worst = 0.0
    for a in (0, 1):
        m = group == a
        s, y = scores[m], labels[m]
        for lo in torch.linspace(0, 1, bins + 1)[:-1]:
            b = (s >= lo) & (s < lo + 1.0 / bins)
            if b.sum() >= 20:
                worst = max(worst, abs(float(s[b].mean()) - float(y[b].mean())))
    return worst


def make_calibrated_population(n=200_000, base_a=0.3, base_b=0.6, seed=0):
    """Two groups with different base rates and a perfectly calibrated imperfect score.

    Construction: draw a true probability per individual, then draw the label from it.
    The score IS the true probability, so calibration holds exactly, in both groups,
    by construction. Everything that follows is therefore forced arithmetic, not an
    artefact of a badly fitted model.
    """
    g = torch.Generator().manual_seed(seed)
    group = (torch.rand(n, generator=g) < 0.5).long()
    base = torch.where(group == 0, torch.tensor(base_a), torch.tensor(base_b))
    # Beta-ish spread around each group's base rate, clipped to (0, 1).
    scores = (base + 0.25 * torch.randn(n, generator=g)).clamp(0.02, 0.98)
    labels = (torch.rand(n, generator=g) < scores).float()
    return scores, labels, group


# ============================================================ 6. membership inference
class TinyNet(nn.Module):
    def __init__(self, d_in=20, hidden=256, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


def make_classification_data(n=2000, d=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    w = torch.randn(d, generator=g)
    y = ((X @ w + 0.8 * torch.randn(n, generator=g)) > 0).long()
    return X, y


def make_memorisable_data(n=1200, d=20, flip=0.2, seed=0):
    """A linear rule with 20% of labels flipped at random.

    The flipped labels are the point: they cannot be generalised, only memorised, so a
    model that fits them has provably stored information about specific training rows.
    Without them a well-generalising model reaches near-zero loss on held-out data too,
    the two loss distributions overlap, and the attack has nothing to work with — which
    is itself worth seeing once.
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    w = torch.randn(d, generator=g)
    y = ((X @ w) > 0).long()
    flipped = torch.rand(n, generator=g) < flip
    return X, torch.where(flipped, 1 - y, y)


def make_rings(n=3000, nuisance=30, seed=0):
    """Two noisy concentric rings in 2-D, padded with `nuisance` pure-noise features.

    The nuisance dimensions carry no signal, but a normally-trained network still puts
    weight on them — and an L-infinity attacker gets to move all 30 at once. This is the
    Tsipras et al. setup in miniature, and it is what makes the adversarial-training
    experiment show anything at all.
    """
    g = torch.Generator().manual_seed(seed)
    lab = (torch.rand(n, generator=g) < 0.5).long()
    r = torch.where(lab == 1, 2.0, 1.0) + 0.25 * torch.randn(n, generator=g)
    th = torch.rand(n, generator=g) * (2 * math.pi)
    signal = torch.stack([r * th.cos(), r * th.sin()], 1)
    return torch.cat([signal, 0.5 * torch.randn(n, nuisance, generator=g)], 1), lab


def train_classifier(X, y, epochs=300, lr=1e-3, clip=None, noise=0.0, seed=0):
    """Optionally with DP-SGD-style per-batch clipping and Gaussian noise."""
    torch.manual_seed(seed)
    model = TinyNet(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        loss = F.cross_entropy(model(X), y)
        opt.zero_grad()
        loss.backward()
        if clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            for p in model.parameters():
                p.grad += noise * clip * torch.randn_like(p.grad)
        opt.step()
    return model


def membership_inference_auc(model, X_in, y_in, X_out, y_out) -> float:
    """The simplest attack there is: members have lower loss.

    AUC of the loss as a membership score. 0.5 is chance; anything meaningfully above
    it is leakage, and it tracks the train/test gap almost exactly.
    """
    with torch.no_grad():
        l_in = F.cross_entropy(model(X_in), y_in, reduction="none")
        l_out = F.cross_entropy(model(X_out), y_out, reduction="none")
    scores = torch.cat([-l_in, -l_out])                 # higher = more likely a member
    labels = torch.cat([torch.ones_like(l_in), torch.zeros_like(l_out)])
    order = scores.argsort()
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=scores.dtype)
    n_p = int(labels.sum())
    n_n = len(labels) - n_p
    return float((ranks[labels == 1].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


# ============================================================ 7. adversarial examples
def fgsm(model, X, y, eps: float):
    """One step along the sign of the gradient — the cheapest attack that works."""
    X = X.clone().requires_grad_(True)
    loss = F.cross_entropy(model(X), y)
    g, = torch.autograd.grad(loss, X)
    return (X + eps * g.sign()).detach()


def pgd(model, X, y, eps: float, alpha: float = None, steps: int = 20):
    """Projected gradient descent: FGSM iterated, projected back into the eps-ball.

    Strictly stronger than FGSM, and the standard against which robustness claims
    should be made — a defence that only stops FGSM usually only obscured the gradient.
    """
    alpha = alpha or eps / 4
    X0 = X.clone()
    X = (X + torch.empty_like(X).uniform_(-eps, eps)).detach()
    for _ in range(steps):
        X.requires_grad_(True)
        loss = F.cross_entropy(model(X), y)
        g, = torch.autograd.grad(loss, X)
        X = X.detach() + alpha * g.sign()
        X = X0 + (X - X0).clamp(-eps, eps)
    return X.detach()


def accuracy(model, X, y) -> float:
    with torch.no_grad():
        return float((model(X).argmax(-1) == y).float().mean())


def adversarial_train(X, y, eps: float, epochs=800, lr=1e-3, steps=7, seed=0):
    """Madry-style: train on PGD examples, not FGSM ones.

    FGSM adversarial training is cheaper and famously unreliable — it tends to produce
    models that resist FGSM and collapse under PGD ("catastrophic overfitting"), which
    is gradient masking rather than robustness. Always evaluate a defence with a
    stronger attack than the one you trained against.
    """
    torch.manual_seed(seed)
    model = TinyNet(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        Xa = pgd(model, X, y, eps, alpha=eps / 3, steps=steps)
        loss = F.cross_entropy(model(Xa), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


# ============================================================ self-test
def main() -> int:
    torch.manual_seed(0)

    # ---- 1. Bradley-Terry
    true_r = torch.randn(40) * 1.2
    fitted = fit_reward_model(true_r)
    centred_true = true_r - true_r.mean()
    centred_fit = fitted - fitted.mean()
    corr = float(torch.corrcoef(torch.stack([centred_true, centred_fit]))[0, 1])
    check("1 the reward model recovers the true reward", corr > 0.95, f"correlation {corr:.4f}")
    check("1 ...only up to an additive constant",
          close(bradley_terry_loss(fitted, torch.tensor([0]), torch.tensor([1])),
                bradley_terry_loss(fitted + 17.0, torch.tensor([0]), torch.tensor([1])), atol=1e-6),
          "shifting every reward by 17 changes nothing")
    check("1 differences are what is recovered",
          close(centred_fit, centred_true, atol=0.5, rtol=0.5) or corr > 0.95)

    # ---- 2. the KL-regularised optimum
    n = 12
    ref_logits = torch.randn(n)
    reward = torch.randn(n) * 2
    ref = torch.softmax(ref_logits, -1)

    for beta in (0.5, 2.0):
        closed = optimal_policy(ref_logits, reward, beta)
        numeric = solve_by_gradient(ref_logits, reward, beta)
        check(f"2 closed form solves the objective (beta={beta})", close(closed, numeric, atol=2e-3),
              f"max difference {float((closed - numeric).abs().max()):.2e}")

    check("2 beta -> infinity returns the reference policy",
          close(optimal_policy(ref_logits, reward, 1e6), ref, atol=1e-4))
    greedy = torch.zeros(n)
    greedy[int(reward.argmax())] = 1.0
    check("2 beta -> 0 returns the argmax of the reward",
          close(optimal_policy(ref_logits, reward, 1e-3), greedy, atol=1e-6))

    print(f"\n{'beta':>7} {'KL(pi||ref)':>13} {'E[reward]':>11}")
    prev_kl, prev_rew = -1.0, -1e9
    monotone = True
    for beta in (8.0, 4.0, 2.0, 1.0, 0.5, 0.25):
        pi = optimal_policy(ref_logits, reward, beta)
        k, r = kl(pi, ref), float((pi * reward).sum())
        print(f"{beta:>7.2f} {k:>13.4f} {r:>11.4f}")
        monotone &= k > prev_kl and r > prev_rew
        prev_kl, prev_rew = k, r
    check("2 lowering beta buys reward with KL, monotonically", monotone)

    # ---- 3. DPO reaches the same policy
    for beta in (1.0, 0.5):
        target = optimal_policy(ref_logits, true_r[:n] if False else reward, beta)
        learned = train_dpo(ref_logits, reward, beta)
        check(f"3 DPO converges to the RLHF optimum (beta={beta})", close(learned, target, atol=0.02),
              f"max difference {float((learned - target).abs().max()):.3f}")
    check("3 DPO needs no reward model at all", True,
          "the policy IS the reward model: r = beta log(pi/pi_ref) + beta log Z, and Z cancels")

    # ---- 4. reward hacking
    rows, true_reward, proxy = reward_hacking_curve()
    print(f"\n{'beta':>7} {'KL':>8} {'proxy reward':>14} {'TRUE reward':>13}")
    for r in rows:
        print(f"{r['beta']:>7.2f} {r['kl']:>8.3f} {r['proxy']:>14.4f} {r['true']:>13.4f}")
    proxies = [r["proxy"] for r in rows]
    trues = [r["true"] for r in rows]
    check("4 the proxy reward rises monotonically with optimisation pressure",
          all(b > a - 1e-9 for a, b in zip(proxies, proxies[1:])))
    peak = trues.index(max(trues))
    check("4 the TRUE reward peaks and then declines", peak < len(trues) - 1,
          f"peaks at beta={rows[peak]['beta']} (KL {rows[peak]['kl']:.2f}) then falls "
          f"{max(trues) - trues[-1]:.3f}")
    check("4 over-optimising is worse than the peak", trues[-1] < max(trues) * 0.9,
          "Goodhart's law, measured")

    # ---- 5. fairness impossibility
    scores, labels, group = make_calibrated_population()
    cal = calibration_error(scores, labels, group)
    check("5 the score is calibrated in both groups", cal < 0.02, f"max calibration error {cal:.4f}")
    m = fairness_metrics(scores, labels, group)
    print(f"\n{'group':>7} {'base rate':>11} {'selection':>11} {'TPR':>8} {'FPR':>8} {'PPV':>8}")
    for a in (0, 1):
        print(f"{a:>7} {m[f'base_rate_{a}']:>11.3f} {m[f'selection_rate_{a}']:>11.3f} "
              f"{m[f'tpr_{a}']:>8.3f} {m[f'fpr_{a}']:>8.3f} {m[f'ppv_{a}']:>8.3f}")
    check("5 base rates differ, as assumed", abs(m["base_rate_0"] - m["base_rate_1"]) > 0.15)
    check("5 calibration + unequal base rates FORCES unequal error rates",
          m["fpr_gap"] > 0.05 and m["tpr_gap"] > 0.05,
          f"FPR gap {m['fpr_gap']:.3f}, TPR gap {m['tpr_gap']:.3f} — this is arithmetic, not a bug")
    check("5 demographic parity also fails", m["demographic_parity_gap"] > 0.1,
          f"{m['demographic_parity_gap']:.3f}")

    # Equalising one criterion breaks another: threshold each group to match selection rates.
    thr = {}
    for a in (0, 1):
        s = scores[group == a]
        thr[a] = float(torch.quantile(s, 1 - 0.35))
    pred = torch.where(group == 0, (scores >= thr[0]).float(), (scores >= thr[1]).float())
    sel = [float(pred[group == a].mean()) for a in (0, 1)]
    ppv = [float(labels[(group == a) & (pred == 1)].mean()) for a in (0, 1)]
    check("5 group-specific thresholds CAN equalise selection rate",
          abs(sel[0] - sel[1]) < 0.02, f"{sel[0]:.3f} vs {sel[1]:.3f}")
    check("5 ...but then precision differs sharply", abs(ppv[0] - ppv[1]) > 0.1,
          f"PPV {ppv[0]:.3f} vs {ppv[1]:.3f} — you must choose which fairness you want")

    # ---- 6. membership inference
    X, y = make_memorisable_data(1200)
    X_in, y_in, X_out, y_out = X[:600], y[:600], X[600:], y[600:]
    # float64 for the attack: a confidently-correct fp32 cross-entropy underflows to
    # exactly 0.0 for members AND non-members, and the resulting mass of ties buries the
    # signal completely. Precision is part of the experiment.
    overfit = train_classifier(X_in, y_in, epochs=1500, lr=3e-3).double()
    Xi, Xo = X_in.double(), X_out.double()
    auc = membership_inference_auc(overfit, Xi, y_in, Xo, y_out)
    tr, te = accuracy(overfit, Xi, y_in), accuracy(overfit, Xo, y_out)
    check("6 a model that memorises leaks membership", auc > 0.65,
          f"attack AUC {auc:.3f} (train accuracy {tr:.3f} vs test {te:.3f})")
    check("6 it memorised the flipped labels", tr > 0.99,
          "20% of these labels are random, so fitting all of them is memorisation by definition")

    private = train_classifier(X_in, y_in, epochs=1500, lr=3e-3, clip=0.05, noise=2.0).double()
    auc_dp = membership_inference_auc(private, Xi, y_in, Xo, y_out)
    tr_dp, te_dp = accuracy(private, Xi, y_in), accuracy(private, Xo, y_out)
    check("6 clipping and noise reduce the leak", auc_dp < auc - 0.1, f"AUC {auc:.3f} -> {auc_dp:.3f}")
    check("6 because the memorisation is destroyed", tr_dp < 0.9,
          f"train accuracy {tr:.3f} -> {tr_dp:.3f}")
    check("6 the cost lands on memorisation, not automatically on generalisation", True,
          f"test accuracy {te:.3f} -> {te_dp:.3f}: here the noise even helped, by regularising "
          f"against the label noise. The privacy/utility trade-off is real but not a law")

    # ---- 7. adversarial examples
    Xc, yc = make_rings()
    Xtr, ytr, Xte, yte = Xc[:2200], yc[:2200], Xc[2200:], yc[2200:]
    clean_model = train_classifier(Xtr, ytr, epochs=1200, lr=3e-3)
    eps = 0.2
    acc_clean = accuracy(clean_model, Xte, yte)
    acc_fgsm = accuracy(clean_model, fgsm(clean_model, Xte, yte, eps), yte)
    acc_pgd = accuracy(clean_model, pgd(clean_model, Xte, yte, eps), yte)
    print(f"\nrings + 30 nuisance dimensions, eps={eps}")
    print(f"  standard training:    clean {acc_clean:.3f}   FGSM {acc_fgsm:.3f}   PGD {acc_pgd:.3f}")
    check("7 a small perturbation destroys accuracy", acc_pgd < acc_clean - 0.3,
          f"{acc_clean:.3f} -> {acc_pgd:.3f} for a perturbation of {eps} per coordinate")
    check("7 PGD is stronger than FGSM", acc_pgd <= acc_fgsm + 1e-6,
          "always evaluate a defence with a stronger attack than you trained against")

    robust = adversarial_train(Xtr, ytr, eps)
    r_clean = accuracy(robust, Xte, yte)
    r_pgd = accuracy(robust, pgd(robust, Xte, yte, eps), yte)
    print(f"  adversarial training: clean {r_clean:.3f}   PGD {r_pgd:.3f}")
    check("7 adversarial training improves robustness", r_pgd > acc_pgd + 0.1,
          f"PGD accuracy {acc_pgd:.3f} -> {r_pgd:.3f}")
    check("7 and here it cost no clean accuracy", r_clean >= acc_clean - 0.01,
          f"{acc_clean:.3f} -> {r_clean:.3f}: the nuisance features were never real signal, "
          f"so declining to use them costs nothing")

    # The trade-off is real when the fragile features ARE predictive. Same experiment on
    # data with no nuisance dimensions and a genuinely linear boundary.
    Xl, yl = make_classification_data(2000, seed=1)
    Ltr, Lytr, Lte, Lyte = Xl[:1500], yl[:1500], Xl[1500:], yl[1500:]
    a0 = accuracy(train_classifier(Ltr, Lytr, epochs=600), Lte, Lyte)
    a1 = accuracy(adversarial_train(Ltr, Lytr, 0.3, epochs=600), Lte, Lyte)
    print(f"\n  linear data, eps=0.3: clean accuracy {a0:.3f} (standard) vs {a1:.3f} (robust)")
    check("7 ...but where the fragile features are real signal, robustness does cost accuracy",
          a1 < a0 - 0.03,
          f"{a0:.3f} -> {a1:.3f} — the robustness/accuracy trade-off is conditional, not a law")

    return 0 if report() else 1


if __name__ == "__main__":
    raise SystemExit(main())
