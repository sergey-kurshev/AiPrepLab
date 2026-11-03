"""Training a network on a score instead of a label — the arithmetic behind sections 3 and 4.

Run with `python policy-gradient.py`.

Module 08's `exercise.py` builds the reward model, the closed-form policy and DPO.
It never runs reinforcement learning, because the closed form makes it unnecessary
on a three-answer problem. That is a reasonable shortcut, and it leaves a hole: the
reader never sees how a reward turns into a gradient. This file fills the hole.

Every table in sections 4 and 5 of `README.md` is printed below, from the same
toy problem the article uses elsewhere: three possible answers, a uniform reference
policy, and a reward of 1 for the first answer and 0 for the other two. The closed
form of section 5 gives the right answer, so every method here can be graded.
"""

from __future__ import annotations

import sys
import pathlib

_here = pathlib.Path(__file__).resolve().parent
for _c in (_here, *_here.parents):
    if (_c / "tools").is_dir():
        sys.path.insert(0, str(_c / "tools"))
        break

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from checks import check, report, close

REWARD = torch.tensor([1.0, 0.0, 0.0])       # r(y) for the three answers
REF_LOGITS = torch.zeros(3)                  # a uniform reference policy


# ===================================================== 1. the sample breaks the chain
def reward_of_a_sample(logits: torch.Tensor) -> torch.Tensor:
    """Sample an answer and look up its reward, the way a real training loop would."""
    y = Categorical(logits=logits).sample()
    return REWARD[y]


def exact_expected_reward(logits: torch.Tensor) -> torch.Tensor:
    """The same quantity in closed form. Only possible because there are three answers."""
    return (torch.softmax(logits, -1) * REWARD).sum()


# ===================================================== 2. the policy gradient
def true_gradient(logits: torch.Tensor) -> torch.Tensor:
    """d/dlogits of E[r], by enumerating all three answers. The grading key."""
    x = logits.detach().clone().requires_grad_(True)
    exact_expected_reward(x).backward()
    return x.grad.clone()


def reinforce_gradient(logits: torch.Tensor, n: int, baseline: float | str = 0.0,
                       generator: torch.Generator | None = None) -> torch.Tensor:
    """One Monte-Carlo estimate of the same gradient, from n sampled answers.

    baseline: a number, or "batch" for the mean reward of this batch of samples.
    """
    x = logits.detach().clone().requires_grad_(True)
    probs = torch.softmax(x, -1)
    y = torch.multinomial(probs.detach(), n, replacement=True, generator=generator)
    r = REWARD[y]
    b = r.mean() if baseline == "batch" else torch.tensor(float(baseline))
    logp = torch.log(probs[y])
    # Ascent on reward, so the loss is its negative: weighted cross-entropy on our
    # own samples, with the advantage as the weight.
    (-((r - b) * logp).mean()).backward()
    return -x.grad.clone()                   # undo the sign flip: this is the gradient of E[r]


PROBE_LOGITS = torch.tensor([0.4, -0.2, 0.1])   # a policy part-way through training
EXPECTED_R = float((torch.softmax(PROBE_LOGITS, -1) * REWARD).sum())


def estimator_error(n: int, baseline: float | str, trials: int, seed: int = 0) -> tuple:
    """RMS distance from the true gradient, and the bias, over independent estimates."""
    logits = PROBE_LOGITS
    truth = true_gradient(logits)
    g = torch.Generator().manual_seed(seed)
    est = torch.stack([reinforce_gradient(logits, n, baseline, g) for _ in range(trials)])
    rms = float(((est - truth) ** 2).sum(-1).mean().sqrt())
    bias = float((est.mean(0) - truth).norm())
    return rms, bias, truth


# ===================================================== 3. reuse, ratios and clipping
def ppo_surrogate(logits: torch.Tensor, old_logp: torch.Tensor, y: torch.Tensor,
                  adv: torch.Tensor, clip: float | None) -> torch.Tensor:
    """PPO's clipped objective for one batch. Returns a loss to minimise."""
    logp = torch.log_softmax(logits, -1)[y]
    ratio = torch.exp(logp - old_logp)
    if clip is None:
        return -(ratio * adv).mean()
    kept = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv)
    return -kept.mean()


def reuse_one_batch(epochs: int, clip: float | None, n: int = 256, lr: float = 0.2,
                    seed: int = 0) -> dict:
    """Take one batch of samples and keep updating on it. Does the policy run away?"""
    torch.manual_seed(seed)
    logits = REF_LOGITS.clone().requires_grad_(True)
    probs = torch.softmax(logits.detach(), -1)
    y = torch.multinomial(probs, n, replacement=True)
    old_logp = torch.log(probs[y])
    adv = REWARD[y] - REWARD[y].mean()
    opt = torch.optim.SGD([logits], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        ppo_surrogate(logits, old_logp, y, adv, clip).backward()
        opt.step()
    final = torch.softmax(logits.detach(), -1)
    ref = torch.softmax(REF_LOGITS, -1)
    return dict(policy=final,
                kl=float((final * (final / ref).log()).sum()),
                reward=float((final * REWARD).sum()),
                max_ratio=float(torch.exp(torch.log(final[y]) - old_logp).max()))


def clip_cases(clip: float = 0.2) -> list:
    """What the clip does in the four combinations of ratio and advantage sign."""
    out = []
    for ratio in (0.5, 1.5):
        for adv in (+1.0, -1.0):
            r = torch.tensor(float(ratio), requires_grad=True)
            kept = torch.minimum(r * adv, torch.clamp(r, 1 - clip, 1 + clip) * adv)
            kept.backward()
            out.append((ratio, adv, float(kept.detach()), float(r.grad)))
    return out


# ===================================================== 4. the whole loop
def optimal_policy(ref_logits: torch.Tensor, reward: torch.Tensor, beta: float):
    """Section 5's closed form, repeated here so this file stands alone."""
    return torch.softmax(torch.log_softmax(ref_logits, -1) + reward / beta, -1)


def ppo(beta: float = 1.0, iters: int = 600, n: int = 512, epochs: int = 4,
        clip: float | None = 0.2, lr: float = 0.1, seed: int = 0) -> torch.Tensor:
    """RLHF by the standard recipe: sample, score, subtract a baseline, clip, repeat.

    With epochs=1 and clip=None this is plain REINFORCE, which is why the article's
    table has both columns. The learning rate decays towards the end: without that,
    the estimator's noise keeps the policy jittering around the optimum instead of
    settling on it, and the answer stays about 0.08 off.
    """
    torch.manual_seed(seed)
    logits = REF_LOGITS.clone().requires_grad_(True)
    ref_logp = torch.log_softmax(REF_LOGITS, -1)
    opt = torch.optim.Adam([logits], lr=lr)
    schedule = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.02, iters)
    for _ in range(iters):
        with torch.no_grad():
            probs = torch.softmax(logits, -1)
            y = torch.multinomial(probs, n, replacement=True)
            old_logp = torch.log(probs[y])
            # The KL brake, applied inside the reward rather than as a separate term.
            shaped = REWARD[y] - beta * (old_logp - ref_logp[y])
            adv = shaped - shaped.mean()
        for _ in range(epochs):
            opt.zero_grad()
            ppo_surrogate(logits, old_logp, y, adv, clip).backward()
            opt.step()
        schedule.step()
    return torch.softmax(logits.detach(), -1)


# ===================================================== 5. credit over a sequence
VOCAB, LENGTH, TARGET = 4, 2, (0, 1)


def sequence_reward(tokens: torch.Tensor) -> torch.Tensor:
    """1 if the whole answer is exactly the target pair, 0 otherwise. Nothing in between."""
    want = torch.tensor(TARGET)
    return (tokens == want).all(-1).float()


def train_sequence(iters: int = 300, n: int = 512, lr: float = 0.2, seed: int = 0) -> dict:
    """REINFORCE where the reward arrives only after the last token."""
    torch.manual_seed(seed)
    logits = torch.zeros(LENGTH, VOCAB, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)
    start = None
    for it in range(iters):
        probs = torch.softmax(logits, -1)
        with torch.no_grad():
            tokens = torch.stack([torch.multinomial(probs[t], n, replacement=True)
                                  for t in range(LENGTH)], dim=-1)
        r = sequence_reward(tokens)
        adv = r - r.mean()                              # one number per answer...
        logp = torch.log(probs.gather(1, tokens.T)).T   # (n, LENGTH)
        # ...shared by every token in it. That is the credit-assignment problem.
        loss = -(adv.unsqueeze(-1) * logp).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it == 0:
            start = float(r.mean())
    final = torch.softmax(logits.detach(), -1)
    return dict(start=start, probs=final,
                p_target=float(final[0, TARGET[0]] * final[1, TARGET[1]]))


# ===================================================== 6. what the loop costs
def memory_bytes(params: int = 7_000_000_000) -> dict:
    """Weights and optimiser state for each recipe, in bytes per parameter times params.

    A trained model in bf16 with Adam holds: 2 bytes of weights, 2 of gradients,
    and 12 of fp32 master copy plus Adam's two moments. A frozen model holds 2.
    Activations are extra and depend on batch and sequence length.
    """
    trained, frozen = 2 + 2 + 12, 2
    recipes = {"PPO": dict(policy=trained, value=trained, reference=frozen, reward=frozen),
               "GRPO": dict(policy=trained, reference=frozen, reward=frozen),
               "DPO": dict(policy=trained, reference=frozen)}
    return {name: dict(per_param=sum(parts.values()),
                       gib=sum(parts.values()) * params / 2**30,
                       parts=parts)
            for name, parts in recipes.items()}


# ===================================================== main
def main() -> int:
    torch.manual_seed(0)

    # ---- 1. the sample is where the gradient stops
    print("1. a sampled reward carries no gradient\n")
    logits = torch.zeros(3, requires_grad=True)
    sampled = reward_of_a_sample(logits)
    exact = exact_expected_reward(logits)
    print(f"  r(sampled answer)  = {float(sampled):.1f}   requires_grad = {sampled.requires_grad}")
    print(f"  E[r] by enumeration= {exact.item():.4f}   requires_grad = {exact.requires_grad}")
    check("1 the reward of a sample is outside the autograd graph", not sampled.requires_grad,
          "sampling is a lookup into a table of rewards: no path back to the logits")
    try:
        sampled.backward()
        raised = None
    except RuntimeError as e:
        raised = str(e).split(",")[0]
    check("1 so calling backward on it fails", raised is not None, f"RuntimeError: {raised}")
    check("1 while the exact expectation differentiates fine", exact.requires_grad,
          "but enumerating every answer is only possible on a toy")

    # ---- 2. REINFORCE recovers the gradient it cannot backpropagate
    print("\n2. the estimator against the true gradient (1,000 trials each)\n")
    truth = true_gradient(PROBE_LOGITS)
    print(f"  true gradient: [{', '.join(f'{v:+.4f}' for v in truth)}]\n")
    print("  | n samples | RMS error | bias |")
    rows = {}
    for n in (4, 16, 64, 256, 1024):
        rms, bias, _ = estimator_error(n, 0.0, trials=1000)
        rows[n] = rms
        print(f"  | {n:>9} | {rms:9.4f} | {bias:.4f} |")
    check("2 the estimate is unbiased", estimator_error(1024, 0.0, 4000)[1] < 0.004,
          "the mean of many estimates lands on the true gradient")
    halving = rows[16] / rows[64]
    check("2 and its error falls as 1/sqrt(n)", 1.8 < halving < 2.2,
          f"16 -> 64 samples divides the error by {halving:.2f}, and sqrt(4) = 2")

    # ---- 3. the estimator is weighted cross-entropy on our own samples
    print("\n3. the same gradient, written as cross-entropy\n")
    g = torch.Generator().manual_seed(7)
    x = PROBE_LOGITS.clone().requires_grad_(True)
    probs = torch.softmax(x, -1)
    y = torch.multinomial(probs.detach(), 64, replacement=True, generator=g)
    r = REWARD[y]
    by_hand = torch.autograd.grad((r * torch.log(probs[y])).mean(), x, retain_graph=True)[0]
    by_ce = torch.autograd.grad(-(r * F.cross_entropy(x.expand(64, 3), y,
                                                      reduction="none")).mean(), x)[0]
    print(f"  REINFORCE, written out : [{', '.join(f'{v:+.4f}' for v in by_hand)}]")
    print(f"  reward-weighted cross-entropy: [{', '.join(f'{v:+.4f}' for v in by_ce)}]")
    check("3 they are the same computation", close(by_hand, by_ce, atol=1e-6),
          f"max difference {float((by_hand - by_ce).abs().max()):.1e}")

    # ---- 4. baselines
    print("\n4. what subtracting a baseline does (n = 16, 4,000 trials)\n")
    print("  | baseline | RMS error | bias |")
    base_rows = {}
    for label, b in (("none (b = 0)", 0.0), ("batch mean", "batch"),
                     (f"E[r] = {EXPECTED_R:.3f}", EXPECTED_R)):
        rms, bias, _ = estimator_error(16, b, trials=4000, seed=1)
        base_rows[label] = rms
        print(f"  | {label:<12} | {rms:9.4f} | {bias:.4f} |")
    gain = base_rows["none (b = 0)"] / base_rows["batch mean"]
    check("4 a baseline cuts the error", gain > 1.2,
          f"the batch mean divides the RMS error by {gain:.2f}")
    check("4 a constant baseline keeps the estimate unbiased",
          estimator_error(16, EXPECTED_R, 4000, seed=1)[1] < 0.006,
          "it subtracts a quantity that does not depend on which answer was sampled")

    print("\n  the batch mean does depend on the samples, so it carries a small bias\n")
    print("  | n samples | bias |")
    biases = {}
    for size in (4, 16, 64, 256):
        biases[size] = estimator_error(size, "batch", trials=4000, seed=2)[1]
        print(f"  | {size:>9} | {biases[size]:.4f} |")
    check("4 ...and that bias shrinks as the batch grows", biases[256] < biases[4] / 4,
          f"{biases[4]:.4f} at 4 samples against {biases[256]:.4f} at 256")

    # ---- 5. reusing a batch, with and without the clip
    print("\n5. forty gradient steps on one batch of 256 samples\n")
    print("  | clip | P(answer 1) | KL from reference | largest ratio |")
    runs = {}
    for label, c in (("none", None), ("0.2", 0.2)):
        res = reuse_one_batch(40, c)
        runs[label] = res
        print(f"  | {label:<4} | {float(res['policy'][0]):11.3f} | {res['kl']:17.3f} "
              f"| {res['max_ratio']:13.2f} |")
    check("5 without a clip, reuse runs the policy away from the batch it came from",
          runs["none"]["kl"] > 3 * runs["0.2"]["kl"],
          f"KL {runs['none']['kl']:.2f} without the clip against "
          f"{runs['0.2']['kl']:.2f} with it")

    print("\n  what the clip does, case by case (epsilon = 0.2)\n")
    print("  | ratio | advantage | objective | gradient |")
    for ratio, adv, kept, grad in clip_cases():
        print(f"  | {ratio:5.1f} | {adv:+9.0f} | {kept:9.2f} | {grad:8.1f} |")
    zeroed = [(ratio, adv) for ratio, adv, _, grad in clip_cases() if grad == 0.0]
    check("5 the clip switches off exactly the two cases that push further the same way",
          zeroed == [(0.5, -1.0), (1.5, 1.0)],
          "ratio already below 1 with a bad answer, or above 1 with a good one")

    # ---- 6. the whole loop lands on the closed form
    print("\n6. the loop against the closed-form answer\n")
    print("  | beta | plain REINFORCE      | PPO, 4 epochs a batch | closed form          |")
    for beta in (4.0, 1.0, 0.25):
        plain = ppo(beta, epochs=1, clip=None)
        clipped = ppo(beta)
        closed = optimal_policy(REF_LOGITS, REWARD, beta)
        cells = "| ".join(", ".join(f"{v:.3f}" for v in p) + " "
                          for p in (plain, clipped, closed))
        print(f"  | {beta:<4} | {cells}|")
        for name, got in (("REINFORCE", plain), ("PPO", clipped)):
            err = float((got - closed).abs().max())
            check(f"6 {name} reaches the closed-form policy (beta={beta})", err < 0.005,
                  f"max difference {err:.4f}")

    # ---- 7. a reward that arrives only at the end
    print("\n7. one reward for a two-token answer\n")
    seq = train_sequence()
    print(f"  P(correct answer) before training: {seq['start']:.4f}  (chance is 1/16 = 0.0625)")
    print(f"  P(correct answer) after training : {seq['p_target']:.4f}")
    print(f"  position 1: [{', '.join(f'{v:.3f}' for v in seq['probs'][0])}]")
    print(f"  position 2: [{', '.join(f'{v:.3f}' for v in seq['probs'][1])}]")
    check("7 a terminal reward still trains every token in the answer", seq["p_target"] > 0.9,
          f"{seq['start']:.3f} -> {seq['p_target']:.3f}, with both positions sharing one "
          f"advantage per answer")

    # ---- 8. what each recipe holds in memory
    print("\n8. weights and optimiser state for a 7B policy\n")
    mem = memory_bytes()
    print("  | recipe | models held | bytes per parameter | GiB |")
    for name, m in mem.items():
        parts = ", ".join(m["parts"])
        print(f"  | {name:<6} | {parts:<40} | {m['per_param']:19} | {m['gib']:5.0f} |")
    check("8 dropping the value model saves a quarter of the memory",
          close(mem["PPO"]["per_param"] - mem["GRPO"]["per_param"], 16, atol=0),
          f"{mem['PPO']['per_param']} bytes per parameter against {mem['GRPO']['per_param']}")

    return 0 if report() else 1


if __name__ == "__main__":
    raise SystemExit(main())
