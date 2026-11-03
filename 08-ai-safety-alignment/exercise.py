# %% [markdown]
# # Exercise 08 — Safety Phenomena You Can Watch Happen
#
# The safety round has no numerical answers, which is exactly why it is worth grounding your
# opinions in experiments. Seven of them, each running in seconds:
#
# 1. Fit a reward model from preferences, and find out in what sense it is identifiable.
# 2. Derive and verify the KL-regularised optimum $\pi^\ast \propto \pi_{\text{ref}}e^{r/\beta}$.
# 3. Implement DPO and show it reaches the *same* policy with no reward model and no RL.
# 4. **Reproduce reward hacking** — watch true reward peak and fall while proxy reward climbs.
# 5. Demonstrate the **fairness impossibility theorem** numerically.
# 6. Run a membership-inference attack, then defend against it.
# 7. Break a classifier with PGD, then adversarially train it — and find out when that costs
#    accuracy and when it does not.
#
# The goal is that "what are you worried about?" becomes a question about things you have seen,
# not things you have read.

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
import torch.nn as nn
import torch.nn.functional as F
from checks import check, report, close

torch.manual_seed(0)

# %% [markdown]
# ---
# ## Part 1 — Preferences into a reward
#
# Bradley–Terry: $P(y_1 \succ y_2) = \sigma(r_1 - r_2)$. Fitting it is logistic regression on
# reward *differences*.
#
# Implement the loss, then fit it to noisy comparisons drawn from a known true reward. Two things
# to check: that you recover the true reward, and **in what sense** you recover it. The second is
# the part interviewers probe.

# %%
def bradley_terry_loss(reward, winners, losers):
    """-log sigma(r_winner - r_loser), averaged."""
    # TODO
    raise NotImplementedError


def fit_reward_model(true_reward, n_pairs=20_000, steps=800, seed=0):
    """Sample comparisons from the Bradley-Terry model itself, then fit."""
    g = torch.Generator().manual_seed(seed)
    n = true_reward.numel()
    a, b = (torch.randint(0, n, (n_pairs,), generator=g) for _ in range(2))
    keep = a != b
    a, b = a[keep], b[keep]
    a_wins = torch.rand(len(a), generator=g) < torch.sigmoid(true_reward[a] - true_reward[b])
    winners, losers = torch.where(a_wins, a, b), torch.where(a_wins, b, a)
    # TODO: fit a free parameter vector `r` with Adam on bradley_terry_loss, return it detached.
    raise NotImplementedError


try:
    true_r = torch.randn(40) * 1.2
    fitted = fit_reward_model(true_r)
    corr = float(torch.corrcoef(torch.stack([true_r - true_r.mean(), fitted - fitted.mean()]))[0, 1])
    check("1 the reward model recovers the true reward", corr > 0.95, f"correlation {corr:.4f}")
    check("1 ...only up to an additive constant",
          close(bradley_terry_loss(fitted, torch.tensor([0]), torch.tensor([1])),
                bradley_terry_loss(fitted + 17.0, torch.tensor([0]), torch.tensor([1])), atol=1e-6),
          "adding 17 to every reward changes nothing, so absolute reward values are meaningless")
except Exception as _e:
    check("part 1", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# ---
# ## Part 2 — The KL-regularised optimum
#
# $$\max_\pi\ \mathbb{E}_\pi[r] - \beta\,D_{\mathrm{KL}}(\pi\,\|\,\pi_{\text{ref}})
#   \quad\Longrightarrow\quad
#   \pi^\ast(y) = \tfrac{1}{Z}\,\pi_{\text{ref}}(y)\,e^{r(y)/\beta}.$$
#
# Derive it (Lagrange multipliers, exactly as in exercise 01 step 4 — this is the same
# maximum-entropy calculation wearing different clothes), implement the closed form, and confirm
# it against a numerical optimisation of the objective.
#
# Then trace the frontier: as $\beta$ falls, how much KL do you spend, and how much reward do you
# buy?

# %%
def optimal_policy(ref_logits, reward, beta: float):
    # TODO — one line.
    raise NotImplementedError


def kl(p, q) -> float:
    return float((p * ((p + 1e-30).log() - (q + 1e-30).log())).sum())


def solve_by_gradient(ref_logits, reward, beta, steps=4000, lr=0.05):
    """Optimise the objective directly, to confirm the closed form. Given to you."""
    logits = ref_logits.clone().requires_grad_(True)
    ref = torch.softmax(ref_logits, -1)
    opt = torch.optim.Adam([logits], lr=lr)
    for _ in range(steps):
        p = torch.softmax(logits, -1)
        obj = (p * reward).sum() - beta * (p * ((p + 1e-30).log() - (ref + 1e-30).log())).sum()
        opt.zero_grad(); (-obj).backward(); opt.step()
    return torch.softmax(logits, -1).detach()


n = 12
ref_logits = torch.randn(n)
reward = torch.randn(n) * 2
ref = torch.softmax(ref_logits, -1)

try:
    for beta in (0.5, 2.0):
        check(f"2 the closed form solves the objective (beta={beta})",
              close(optimal_policy(ref_logits, reward, beta),
                    solve_by_gradient(ref_logits, reward, beta), atol=2e-3))
    check("2 beta -> infinity gives back the reference policy",
          close(optimal_policy(ref_logits, reward, 1e6), ref, atol=1e-4))
    greedy = torch.zeros(n); greedy[int(reward.argmax())] = 1.0
    check("2 beta -> 0 gives the argmax of the reward — including its errors",
          close(optimal_policy(ref_logits, reward, 1e-3), greedy, atol=1e-6))

    print(f"{'beta':>7} {'KL(pi||ref)':>13} {'E[reward]':>11}")
    for beta in (8.0, 4.0, 2.0, 1.0, 0.5, 0.25):
        pi = optimal_policy(ref_logits, reward, beta)
        print(f"{beta:>7.2f} {kl(pi, ref):>13.4f} {float((pi * reward).sum()):>11.4f}")
except Exception as _e:
    check("part 2", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **2b. (discussion)** The KL term is often described as "preventing the model from drifting too
# far". Give the sharper version: what *specifically* goes wrong at $\beta \to 0$, and why is it a
# property of the reward model rather than of the optimiser?

# %% [markdown]
# ---
# ## Part 3 — DPO: the same optimum, without the RL
#
# Invert the closed form:
# $r(x,y) = \beta\log\frac{\pi^\ast(y)}{\pi_{\text{ref}}(y)} + \beta\log Z(x)$.
#
# Substitute into Bradley–Terry. **$\log Z$ cancels**, because the likelihood only ever sees the
# difference between two rewards for the same prompt. What is left is a supervised loss:
#
# $$\mathcal{L} = -\log\sigma\!\left(\beta\log\tfrac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)}
#   - \beta\log\tfrac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\right).$$
#
# Do the cancellation on paper before implementing — it is the expected answer to "explain DPO",
# and it takes three lines.

# %%
def dpo_loss(policy_logits, ref_logits, winners, losers, beta: float):
    # TODO
    raise NotImplementedError


def train_dpo(ref_logits, true_reward, beta, n_pairs=40_000, steps=3000, seed=0):
    g = torch.Generator().manual_seed(seed)
    nn_ = true_reward.numel()
    a, b = (torch.randint(0, nn_, (n_pairs,), generator=g) for _ in range(2))
    keep = a != b
    a, b = a[keep], b[keep]
    a_wins = torch.rand(len(a), generator=g) < torch.sigmoid(true_reward[a] - true_reward[b])
    winners, losers = torch.where(a_wins, a, b), torch.where(a_wins, b, a)
    # TODO: optimise `logits` (initialised at ref_logits) on dpo_loss; return the softmax.
    raise NotImplementedError


try:
    for beta in (1.0, 0.5):
        target = optimal_policy(ref_logits, reward, beta)
        learned = train_dpo(ref_logits, reward, beta)
        check(f"3 DPO reaches the RLHF optimum (beta={beta})", close(learned, target, atol=0.02),
              f"max difference {float((learned - target).abs().max()):.3f}")
except Exception as _e:
    check("part 3", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **3b. (discussion)** You just showed DPO and RLHF reach the same policy *on a bandit with
# preferences sampled from the true reward*. Name three ways that setup is unlike real RLHF, and
# say which of them you would expect to matter most.

# %% [markdown]
# ---
# ## Part 4 — Reward hacking, reproduced
#
# The crucial experiment in this notebook. Fit a proxy reward model on a **small** number of
# comparisons — right on average, wrong in detail — then optimise against it at increasing
# strength and track both the proxy reward and the true reward.
#
# Predict the shapes before you run it. One of them is monotone; the other is not.

# %%
def reward_hacking_curve(n_actions=200, n_pairs=600, betas=None, seed=0):
    """Return rows of {beta, kl, proxy, true} for a policy optimised against a fitted proxy."""
    # TODO:
    #   1. draw a true reward over n_actions;
    #   2. sample only n_pairs comparisons from it and fit a proxy reward (part 1's machinery);
    #   3. for each beta, form the optimal policy against the PROXY;
    #   4. evaluate that policy under both the proxy and the TRUE reward.
    raise NotImplementedError


try:
    rows = reward_hacking_curve()
    print(f"{'beta':>7} {'KL':>8} {'proxy reward':>14} {'TRUE reward':>13}")
    for r in rows:
        bar = "#" * int(30 * r["true"] / max(x["true"] for x in rows))
        print(f"{r['beta']:>7.2f} {r['kl']:>8.3f} {r['proxy']:>14.4f} {r['true']:>13.4f}  {bar}")

    proxies = [r["proxy"] for r in rows]
    trues = [r["true"] for r in rows]
    check("4 the proxy rises monotonically with optimisation pressure",
          all(b > a - 1e-9 for a, b in zip(proxies, proxies[1:])))
    peak = trues.index(max(trues))
    check("4 the TRUE reward peaks and then declines", peak < len(trues) - 1,
          f"peaks at beta={rows[peak]['beta']} (KL {rows[peak]['kl']:.2f})")
    check("4 over-optimising is worse than stopping at the peak", trues[-1] < max(trues) * 0.9,
          "Goodhart's law, measured rather than quoted")
except Exception as _e:
    check("part 4", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **4b. (discussion)** You have a validation set of human preferences and an RLHF run in progress.
# Design the stopping rule. What do you monitor, and what confounder must you control for? (The
# well-known one in real systems is response length.)

# %% [markdown]
# ---
# ## Part 5 — The fairness impossibility theorem
#
# Three reasonable definitions: demographic parity, equalised odds, calibration. The theorem
# (Kleinberg et al.; Chouldechova, 2016): if base rates differ between groups and the classifier is
# imperfect, **calibration and equalised odds cannot both hold**.
#
# Demonstrate it. The construction matters: build a population where the score *is* the true
# probability, so calibration holds exactly **by construction** and anything that follows is forced
# arithmetic rather than a badly-fitted model.

# %%
def make_calibrated_population(n=200_000, base_a=0.3, base_b=0.6, seed=0):
    """Two groups with different base rates and a perfectly calibrated imperfect score."""
    # TODO: draw a per-individual probability centred on the group's base rate; let the
    #       score BE that probability; draw the label from it.
    raise NotImplementedError


def fairness_metrics(scores, labels, group, threshold=0.5) -> dict:
    """Per group: base rate, selection rate, TPR, FPR, PPV — plus the gaps."""
    # TODO
    raise NotImplementedError


def calibration_error(scores, labels, group, bins=10) -> float:
    """Max over groups and score bins of |mean score - observed rate|."""
    # TODO
    raise NotImplementedError


try:
    scores, labels, group = make_calibrated_population()
    cal = calibration_error(scores, labels, group)
    check("5 the score is calibrated in both groups", cal < 0.02, f"max calibration error {cal:.4f}")

    m = fairness_metrics(scores, labels, group)
    print(f"{'group':>7} {'base rate':>11} {'selection':>11} {'TPR':>8} {'FPR':>8} {'PPV':>8}")
    for a in (0, 1):
        print(f"{a:>7} {m[f'base_rate_{a}']:>11.3f} {m[f'selection_rate_{a}']:>11.3f} "
              f"{m[f'tpr_{a}']:>8.3f} {m[f'fpr_{a}']:>8.3f} {m[f'ppv_{a}']:>8.3f}")
    check("5 base rates differ, as assumed", abs(m["base_rate_0"] - m["base_rate_1"]) > 0.15)
    check("5 calibration + unequal base rates FORCE unequal error rates",
          m["fpr_gap"] > 0.05 and m["tpr_gap"] > 0.05,
          f"FPR gap {m['fpr_gap']:.3f}, TPR gap {m['tpr_gap']:.3f} — arithmetic, not a bug")
    check("5 demographic parity fails too", m["demographic_parity_gap"] > 0.1)
except Exception as _e:
    check("part 5", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **5b.** Now try to fix it. Pick per-group thresholds that equalise the selection rate (demographic
# parity), then measure precision in each group. You should find you have traded one disparity for
# another.

# %%
try:
    thr = {a: float(torch.quantile(scores[group == a], 1 - 0.35)) for a in (0, 1)}
    pred = torch.where(group == 0, (scores >= thr[0]).float(), (scores >= thr[1]).float())
    sel = [float(pred[group == a].mean()) for a in (0, 1)]
    ppv = [float(labels[(group == a) & (pred == 1)].mean()) for a in (0, 1)]
    print(f"selection rates equalised: {sel[0]:.3f} vs {sel[1]:.3f}")
    print(f"precision now differs:     {ppv[0]:.3f} vs {ppv[1]:.3f}")
    check("5b selection rates can be equalised", abs(sel[0] - sel[1]) < 0.02)
    check("5b but precision then diverges", abs(ppv[0] - ppv[1]) > 0.1)
except Exception as _e:
    check("part 5b", False, "needs part 5")
report()

# %% [markdown]
# **5c. (discussion)** Given the theorem, what is wrong with the instruction "make the model fair"?
# Who should choose which criterion, and on what grounds? This is the highest-value thing in the
# module to be able to say clearly.

# %% [markdown]
# ---
# ## Part 6 — Membership inference and a defence
#
# The simplest privacy attack: members of the training set have lower loss. Measure the AUC of
# loss-as-membership-score.
#
# Two implementation details that are part of the lesson. First, the data needs **label noise**:
# flipped labels cannot be generalised, only memorised, so a model that fits them has provably
# stored information about specific rows. Second, compute the losses in **float64** — a
# confidently-correct fp32 cross-entropy underflows to exactly 0.0 for members *and* non-members,
# and the resulting mass of ties buries the signal entirely.

# %%
class TinyNet(nn.Module):
    def __init__(self, d_in=20, hidden=256, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


def make_memorisable_data(n=1200, d=20, flip=0.2, seed=0):
    """A linear rule with `flip` of the labels randomly flipped."""
    # TODO
    raise NotImplementedError


def train_classifier(X, y, epochs=300, lr=1e-3, clip=None, noise=0.0, seed=0):
    """Full-batch training; if `clip` is given, clip the gradient norm and add
    Gaussian noise of scale `noise * clip` — DP-SGD's two ingredients."""
    # TODO
    raise NotImplementedError


def membership_inference_auc(model, X_in, y_in, X_out, y_out) -> float:
    """AUC of (negative loss) as a membership score. 0.5 is chance."""
    # TODO — you wrote an O(n log n) AUC in module 02; reuse it.
    raise NotImplementedError


def accuracy(model, X, y) -> float:
    with torch.no_grad():
        return float((model(X).argmax(-1) == y).float().mean())


try:
    X, y = make_memorisable_data(1200)
    X_in, y_in, X_out, y_out = X[:600], y[:600], X[600:], y[600:]
    overfit = train_classifier(X_in, y_in, epochs=1500, lr=3e-3).double()
    Xi, Xo = X_in.double(), X_out.double()
    auc = membership_inference_auc(overfit, Xi, y_in, Xo, y_out)
    tr, te = accuracy(overfit, Xi, y_in), accuracy(overfit, Xo, y_out)
    check("6 a memorising model leaks membership", auc > 0.65,
          f"attack AUC {auc:.3f}; train accuracy {tr:.3f} vs test {te:.3f}")
    check("6 it memorised the flipped labels", tr > 0.99)

    private = train_classifier(X_in, y_in, epochs=1500, lr=3e-3, clip=0.05, noise=2.0).double()
    auc_dp = membership_inference_auc(private, Xi, y_in, Xo, y_out)
    tr_dp, te_dp = accuracy(private, Xi, y_in), accuracy(private, Xo, y_out)
    check("6 clipping and noise reduce the leak", auc_dp < auc - 0.1, f"{auc:.3f} -> {auc_dp:.3f}")
    check("6 because memorisation is destroyed", tr_dp < 0.9, f"train accuracy {tr:.3f} -> {tr_dp:.3f}")
    print(f"\ntest accuracy {te:.3f} -> {te_dp:.3f}. Note what happened, and explain it.")
except Exception as _e:
    check("part 6", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **6b. (discussion)** The test accuracy did something you may not have predicted. Why? And what
# does that tell you about how to talk about the privacy/utility trade-off — is it a law?

# %% [markdown]
# ---
# ## Part 7 — Adversarial examples
#
# FGSM is one step along the sign of the gradient; PGD iterates it inside an $\varepsilon$-ball.
# **Always report PGD**: a defence that stops FGSM alone has usually just obscured the gradient.
#
# The data matters here. Use two concentric rings in 2-D padded with 30 pure-noise dimensions: the
# noise carries no signal, but a normally-trained network puts weight on it anyway, and an
# $L_\infty$ attacker gets to move all 30 coordinates at once.

# %%
def make_rings(n=3000, nuisance=30, seed=0):
    """Two noisy concentric rings in 2-D, padded with `nuisance` pure-noise features."""
    # TODO
    raise NotImplementedError


def fgsm(model, X, y, eps: float):
    # TODO
    raise NotImplementedError


def pgd(model, X, y, eps: float, alpha=None, steps=20):
    # TODO — random start, iterate, project back into the eps-ball each step.
    raise NotImplementedError


def adversarial_train(X, y, eps: float, epochs=800, lr=1e-3, steps=7, seed=0):
    """Madry-style: train on PGD examples, not FGSM ones."""
    # TODO
    raise NotImplementedError


try:
    Xc, yc = make_rings()
    Xtr, ytr, Xte, yte = Xc[:2200], yc[:2200], Xc[2200:], yc[2200:]
    base = train_classifier(Xtr, ytr, epochs=1200, lr=3e-3)
    eps = 0.2
    a_clean = accuracy(base, Xte, yte)
    a_fgsm = accuracy(base, fgsm(base, Xte, yte, eps), yte)
    a_pgd = accuracy(base, pgd(base, Xte, yte, eps), yte)
    print(f"standard training:    clean {a_clean:.3f}   FGSM {a_fgsm:.3f}   PGD {a_pgd:.3f}")
    check("7 a small perturbation destroys accuracy", a_pgd < a_clean - 0.3)
    check("7 PGD is stronger than FGSM", a_pgd <= a_fgsm + 1e-6)

    robust = adversarial_train(Xtr, ytr, eps)
    r_clean, r_pgd = accuracy(robust, Xte, yte), accuracy(robust, pgd(robust, Xte, yte, eps), yte)
    print(f"adversarial training: clean {r_clean:.3f}   PGD {r_pgd:.3f}")
    check("7 adversarial training improves robustness", r_pgd > a_pgd + 0.1)
    check("7 and here it cost no clean accuracy", r_clean >= a_clean - 0.01,
          "the nuisance features were never real signal, so declining to use them is free")
except Exception as _e:
    check("part 7", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **7b.** Now repeat it on data where the fragile features *are* real signal — the plain
# 20-dimensional linear problem, at $\varepsilon = 0.3$. The trade-off should reappear.
#
# Then state the conclusion precisely: the robustness/accuracy trade-off is **conditional on
# whether the non-robust features are predictive**, not a law of nature.

# %% [markdown]
# ---
# ## Part 8 — The discussion, written down
#
# Answer these in prose. They are the actual interview.
#
# **8a.** "What are you most worried about in current systems?" Give one concrete, measurable
# failure mode, a mitigation, and where the mitigation runs out. Ground it in something from this
# notebook.
#
# > *your answer*
#
# **8b.** "How would you evaluate whether a model is safe to deploy?" Distinguish capability
# evaluations, safety evaluations and red teaming, and name the limitation of each.
#
# > *your answer*
#
# **8c.** "Is RLHF enough?" Answer without either dismissing the question or reciting doom.
#
# > *your answer*
#
# **8d.** Prompt injection: why is it structurally harder than jailbreaking, and what would you do
# about it when deploying an agent with tools? Be specific about what your mitigations do and do
# not guarantee.
#
# > *your answer*
#
# **8e.** Pick one claim in this module that you think is over-stated, and argue against it.
#
# > *your answer*

# %% [markdown]
# ---
# **Stretch goals:**
#
# 1. Add a **length bias** to the reward-hacking experiment: make the proxy reward include a small
#   positive coefficient on "length" that the true reward does not have. Watch the optimised policy
#   become verbose, and implement a length-controlled evaluation that detects it.
# 2. Implement the DPO variant **IPO** (a bounded objective that does not run away when preferences
#   are deterministic) and compare its behaviour with DPO's on noiseless preference data.
# 3. Implement proper per-example gradient clipping (not per-batch) for DP-SGD using
#   `torch.func.vmap` and `grad`, and compare the privacy/utility curve with the cruder version here.
# 4. Build a tiny red-teaming loop: an attacker model searching a discrete token space against a
#   fixed classifier "target", and measure attack success against the search budget.
