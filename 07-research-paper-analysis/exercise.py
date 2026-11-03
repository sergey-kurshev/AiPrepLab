# %% [markdown]
# # Exercise 07 — Pass 3 on FlashAttention
#
# **Read the paper first.** Dao, Fu, Ermon, Rudra & Ré (2022), *FlashAttention: Fast and
# Memory-Efficient Exact Attention with IO-Awareness*, arXiv:2205.14135. Do passes 1 and 2 before
# opening this notebook; it is pass 3, and it will not teach you the paper if you have not read it.
#
# Two halves:
#
# - **Part A (code)** — reproduce the algorithm. Online softmax, tiled attention, the IO analysis,
#   and the recomputed backward pass. Everything is checkable exactly, because the paper's central
#   claim is that the answer is unchanged.
# - **Part B (writing)** — the structured critique. Answer it in prose, in this notebook. This is
#   the half the interview actually consists of.
#
# A note on expectations: your Python will not be fast. Block loops in Python lose to one fused
# cuBLAS call however good the algorithm is. Speed is not what you are reproducing — *exactness*
# and *bounded working set* are.

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
import torch.nn.functional as F
from checks import check, report, close

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

# %% [markdown]
# ---
# # Part A — Reproduce it
#
# ## A1. The online-softmax recurrence
#
# Everything rests on this. A softmax denominator looks like it needs a full pass over the row
# before anything can be normalised — which would force you to materialise the row. It does not:
#
# $$m^{\text{new}} = \max(m, \tilde m), \qquad
#   \ell^{\text{new}} = e^{m - m^{\text{new}}}\,\ell + e^{\tilde m - m^{\text{new}}}\,\tilde\ell.$$
#
# The factor $e^{m - m^{\text{new}}}$ is what lets you revise your estimate of the maximum *after*
# already accumulating a partial sum. Implement the streaming statistics and confirm they are
# exact for every block size — including a block size of 1, and a block size larger than the row.

# %%
def online_softmax_stats(z: torch.Tensor, block: int = 8):
    """Streaming (running max, running sum-of-exp) over a 1-D tensor, processed in blocks."""
    # TODO — start from m = -inf, l = 0.
    raise NotImplementedError


def online_softmax(z: torch.Tensor, block: int = 8) -> torch.Tensor:
    # TODO — one more pass using the final (m, l).
    raise NotImplementedError


try:
    z = torch.randn(64)
    for block in (1, 3, 8, 64, 128):
        check(f"A1 block {block} is exact", close(online_softmax(z, block), torch.softmax(z, 0), atol=1e-14))

    m, l = online_softmax_stats(z, 7)
    check("A1 the running max is the true max", close(m, z.max(), atol=1e-14))
    check("A1 the running sum is the shifted sum",
          close(l, (z - z.max()).exp().sum(), atol=1e-12))

    big = torch.tensor([1000.0, 1001.0, 999.0, 1002.0] * 8)
    check("A1 exact on inputs that overflow a naive softmax",
          close(online_softmax(big, 5), torch.softmax(big, 0), atol=1e-14) and bool(torch.isinf(big.exp()).any()),
          "exp(1000) is inf, and the streaming version is still exact")
except Exception as _e:
    check("A1", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# ## A2. Tiled attention
#
# Now the algorithm. Loop over blocks of queries; inside, loop over blocks of keys. Maintain
# $(m_i, \ell_i, O_i)$ per query row and apply the recurrence, **rescaling the accumulated output**
# whenever the running max changes:
#
# $$O_i \leftarrow e^{m_i - m^{\text{new}}} O_i + \tilde P V_j, \qquad
#   \ell_i \leftarrow e^{m_i - m^{\text{new}}}\ell_i + \mathrm{rowsum}(\tilde P)$$
#
# and divide by $\ell_i$ once, at the end of the row.
#
# Return $L = m + \log \ell$ as well — the per-row log-sum-exp. That single $O(T)$ vector is what
# replaces the $O(T^2)$ probability matrix in the backward pass.
#
# **Two traps.** First, with a causal mask a whole tile can be fully masked, giving
# $m_{\text{blk}} = -\infty$; guard the rescaling so you never evaluate $-\infty - (-\infty)$.
# Second, the exactness check is `atol=1e-13` — this is not an approximation, and if your version
# is only close to three decimals you have a bug, not a rounding difference.

# %%
def flash_attention_forward(Q, K, V, block_q: int = 16, block_k: int = 16, causal: bool = False):
    """Return (O, L, peak_tile_elements). Never materialise a T x T tensor."""
    # TODO
    raise NotImplementedError


def reference_attention(Q, K, V, causal: bool = False):
    """The textbook version, for comparison. Given to you."""
    s = Q @ K.T / math.sqrt(Q.shape[-1])
    if causal:
        s = s.masked_fill(torch.ones(*s.shape, dtype=torch.bool).triu(1), float("-inf"))
    p = torch.softmax(s, dim=-1)
    return p @ V, p


T, d = 64, 16
Q, K, V = (torch.randn(T, d) for _ in range(3))

try:
    ref, P = reference_attention(Q, K, V)
    for bq, bk in [(8, 8), (16, 32), (64, 64), (5, 7)]:
        O, L, peak = flash_attention_forward(Q, K, V, bq, bk)
        check(f"A2 tiles {bq}x{bk} are EXACT", close(O, ref, atol=1e-13),
              f"max difference {float((O - ref).abs().max()):.2e}")

    ref_c, _ = reference_attention(Q, K, V, causal=True)
    for bq, bk in [(8, 8), (16, 16), (7, 9)]:
        check(f"A2 causal tiles {bq}x{bk} are exact",
              close(flash_attention_forward(Q, K, V, bq, bk, causal=True)[0], ref_c, atol=1e-13))

    check("A2 agrees with torch's fused kernel",
          close(flash_attention_forward(Q, K, V)[0], F.scaled_dot_product_attention(Q, K, V), atol=1e-13))

    _, L, _ = flash_attention_forward(Q, K, V)
    check("A2 L is the row log-sum-exp",
          close(L, torch.logsumexp(Q @ K.T / math.sqrt(d), dim=-1), atol=1e-12))
    check("A2 and L is O(T)", L.numel() == T)
except Exception as _e:
    check("A2", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# ## A3. The memory claim, measured
#
# The paper's headline is linear rather than quadratic memory in sequence length. Verify it
# directly: `peak_tile` is the largest number of score entries alive at once. Sweep $T$ and
# confirm it does not move.

# %%
try:
    print(f"{'T':>7} {'reference P entries':>21} {'flash peak tile':>17} {'ratio':>9}")
    peaks = {}
    for T_ in (32, 64, 128, 256, 512):
        Qs, Ks, Vs = (torch.randn(T_, 16) for _ in range(3))
        _, _, peak = flash_attention_forward(Qs, Ks, Vs, 16, 16)
        peaks[T_] = peak
        print(f"{T_:>7} {T_ * T_:>21} {peak:>17} {T_ * T_ / peak:>8.0f}x")
    check("A3 the flash working set is constant in T", len(set(peaks.values())) == 1,
          f"constant at {list(peaks.values())[0]} score entries")
except Exception as _e:
    check("A3", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# ## A4. The IO analysis
#
# This is the part that makes it a paper rather than a trick. Count words moved between HBM and
# SRAM.
#
# - **Standard:** $\Theta(Td + T^2)$. The $T^2$ term is the score matrix crossing the HBM boundary
#   at every unfused kernel edge — written by the matmul, read and written by the softmax, read
#   again by the second matmul.
# - **FlashAttention:** with SRAM of $M$ words, a block of $B = M/(4d)$ rows fits, so there are
#   $T/B$ query blocks and each re-reads all of $K$ and $V$: $\Theta(T^2d^2/M)$.
#
# Then answer the question that a careless reader gets wrong: **does the advantage grow with $T$?**
# Take the ratio and work out its large-$T$ limit before you run the cell.

# %%
def standard_attention_hbm_words(T: int, d: int) -> float:
    # TODO
    raise NotImplementedError


def flash_attention_hbm_words(T: int, d: int, sram_words: float) -> float:
    # TODO
    raise NotImplementedError


def io_reduction_limit(d: int, sram_words: float) -> float:
    """The large-T limit of the standard/flash traffic ratio. Derive it on paper first."""
    # TODO
    raise NotImplementedError


try:
    sram = 100_000 / 2         # 100 KB of on-chip SRAM, in fp16 words
    print(f"{'T':>8} {'standard words':>16} {'flash words':>14} {'ratio':>8}")
    ratios = []
    for T_ in (1024, 4096, 16384, 65536):
        st = standard_attention_hbm_words(T_, 64)
        fl = flash_attention_hbm_words(T_, 64, sram)
        ratios.append(st / fl)
        print(f"{T_:>8} {st:>16.3e} {fl:>14.3e} {st / fl:>7.1f}x")
    print(f"\npredicted limit M/(2d^2) = {io_reduction_limit(64, sram):.1f}x")
    check("A4 flash moves less data", ratios[1] > 1.0)
    check("A4 the ratio converges rather than growing",
          close(ratios[-1], io_reduction_limit(64, sram), rtol=0.1),
          "the win is a constant factor M/(2d^2), NOT an asymptotic separation in T")
    check("A4 more SRAM buys more", flash_attention_hbm_words(4096, 64, 4 * sram) <
          flash_attention_hbm_words(4096, 64, sram))
    check("A4 a larger head dimension hurts",
          flash_attention_hbm_words(4096, 128, sram) > flash_attention_hbm_words(4096, 64, sram))
except Exception as _e:
    check("A4", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# ## A5. The backward pass, without storing P
#
# Backward needs $P$, which was never stored. Recompute it from $Q$, $K$ and $L$:
# $P = \exp(QK^\top/\sqrt d - L)$ — one line, because $L$ is exactly the normaliser.
#
# The remaining problem is the softmax correction term $\mathrm{rowsum}(\bar P \odot P)$, which
# looks like it needs the full matrix. It does not, because of the identity you proved in
# **module 01, step 6b**:
#
# $$\mathrm{rowsum}(\bar P \odot P) = \mathrm{rowsum}(\bar O \odot O).$$
#
# Both sides of that are $O(T)$. That lemma is what makes the backward pass fit in linear memory,
# and being able to state it is the difference between having read the paper and understanding it.

# %%
def flash_attention_backward(Q, K, V, O, L, dO):
    """Return (dQ, dK, dV), recomputing P rather than storing it."""
    # TODO
    raise NotImplementedError


try:
    Qg, Kg, Vg = (t.clone().requires_grad_(True) for t in (Q, K, V))
    dO = torch.randn(T, d)
    reference_attention(Qg, Kg, Vg)[0].backward(dO)

    O, L, _ = flash_attention_forward(Q, K, V)
    dQ, dK, dV = flash_attention_backward(Q, K, V, O, L, dO)
    check("A5 dQ matches autograd", close(dQ, Qg.grad, atol=1e-11))
    check("A5 dK matches autograd", close(dK, Kg.grad, atol=1e-11))
    check("A5 dV matches autograd", close(dV, Vg.grad, atol=1e-11))

    _, P_ref = reference_attention(Q, K, V)
    check("A5 P is recoverable from (Q, K, L) alone",
          close(torch.exp(Q @ K.T / math.sqrt(d) - L.unsqueeze(1)), P_ref, atol=1e-13))
    check("A5 the module-01 rowsum identity holds",
          close(((dO @ V.T) * P_ref).sum(-1), (dO * O).sum(-1), atol=1e-12))
except Exception as _e:
    check("A5", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# ---
# # Part B — The critique
#
# Answer these in prose, in the cells below. Keep each to a few sentences. **This half is the
# interview**; part A exists so that these answers are grounded in something you built.
#
# Aim for the calibrated register: *"the central idea is X, well supported by Y; I am less
# convinced by Z because..."* — specific, sourced, proportionate. Not "it's incremental"; not "the
# results are impressive".

# %% [markdown]
# **B1. The contribution, in two sentences, in your own words.** Do not paraphrase the abstract.
#
# > *your answer here*
#
# **B2. The key idea.** What is the single insight that, once you have it, makes everything else
# follow? (You implemented it in A1. State it in one sentence, without notation.)
#
# > *your answer*
#
# **B3. Why does the title say "exact"?** What was the surrounding literature doing, and why is
# "exact" a stronger and more checkable claim than "fast"? Name two of the approximate-attention
# papers it is arguing against.
#
# > *your answer*
#
# **B4. Which result is doing the real work?** Go through the paper's figures and tables and pick
# the one that, if it vanished, would most damage the argument. Which one would you delete?
#
# > *your answer*
#
# **B5. The lower bound.** The paper proves that no exact attention algorithm can do
# asymptotically better for all $M$. Why does that make this a substantially stronger paper than a
# speedup alone would be?
#
# > *your answer*
#
# **B6. What would you be sceptical about?** You measured one such thing in A4. Find two more:
# look at what the baseline actually was, at which shapes were benchmarked, and at what "memory"
# refers to in the headline claim.
#
# > *your answer*
#
# **B7. What does it NOT claim** that a casual reader would think it claims?
#
# > *your answer*
#
# **B8. The obvious follow-up.** Name one, then go and check whether it exists. (FlashAttention-2
# is the easy answer — so read its abstract and say what it implies was *wrong* with the first
# paper. That is the more interesting version of the question.)
#
# > *your answer*
#
# **B9. Rabe & Staats (2021)** made a closely related memory argument a year earlier. Read the
# abstract. What did FlashAttention add? This is exactly the kind of question an interviewer asks
# to see whether you know the literature or just the famous paper.
#
# > *your answer*
#
# **B10. Reproduction plan.** You have one week and eight GPUs. How would you verify the headline
# wall-clock speedup, what would you expect to go wrong first, and what would falsify the claim?
#
# > *your answer*

# %% [markdown]
# ---
# ## Revisiting the spine
#
# Attention: tensor code (00) → gradient derived (01) → implemented from scratch (02) → debugged
# (03) → sharded (04) → cached and quantised (05) → retrieved (06) → **and now the state of the
# art, reconstructed from the paper** (07).
#
# The satisfying part: the lemma that makes FlashAttention's backward pass work in linear memory is
# the one you proved in exercise 01 as a throwaway sub-step. Papers are usually made of things you
# already know, assembled by someone who noticed they fit together.
#
# **Stretch goals:**
#
# 1. Implement the *tiled* backward pass — block over K and V, accumulating `dQ` across blocks —
#    so that the backward never materialises a $T \times T$ tensor either. This is the part most
#    reimplementations skip, and where the bookkeeping actually bites.
# 2. Write the forward pass as a `torch.autograd.Function` with your `flash_attention_backward` as
#    its backward, and validate with `torch.autograd.gradcheck`.
# 3. Read FlashAttention-2 and implement its main change: swap the loop order so the outer loop is
#    over queries and the inner over keys (it already is, here — so work out what *else* changed,
#    and what that says about where the original was losing time).
# 4. Do all of this again with a different paper. A good next choice is GQA (arXiv:2305.13245),
#    whose claim you can also check numerically with what you built in module 05.
