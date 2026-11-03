"""Reference solutions for exercise 07 — FlashAttention, reproduced.

Run with `python solution.py`.

Nothing here is fast. It is a Python transcription of the algorithm, and Python
loops over blocks are slower than one fused cuBLAS call however good the algorithm
is. What it *is* is exact, memory-bounded and checkable — which is what pass 3 of
reading a paper is for.
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
import torch.nn.functional as F
from checks import check, report, close


# ============================================================ 1. online softmax
def online_softmax_stats(z: torch.Tensor, block: int = 8):
    """Streaming (max, sum-of-exp) over a 1-D row, in blocks.

    The recurrence that makes the whole paper work. Processing block by block:

        m_new = max(m, m_block)
        l_new = exp(m - m_new) * l + exp(m_block - m_new) * l_block

    The rescaling factor exp(m - m_new) is what lets you change your mind about the
    maximum after you have already accumulated a partial sum.
    """
    m = torch.tensor(float("-inf"), dtype=z.dtype)
    l = torch.tensor(0.0, dtype=z.dtype)
    for i in range(0, z.numel(), block):
        chunk = z[i:i + block]
        m_blk = chunk.max()
        m_new = torch.maximum(m, m_blk)
        l = torch.exp(m - m_new) * l + torch.exp(chunk - m_new).sum()
        m = m_new
    return m, l


def online_softmax(z: torch.Tensor, block: int = 8) -> torch.Tensor:
    m, l = online_softmax_stats(z, block)
    return torch.exp(z - m) / l


# ============================================================ 2. tiled attention
def flash_attention_forward(Q, K, V, block_q: int = 16, block_k: int = 16, causal: bool = False):
    """Attention without ever materialising the T x T score matrix.

    Returns (O, L) where L = m + log(l) is the log-sum-exp per row — the statistic the
    backward pass needs, and the only extra thing that has to be stored. It is O(T),
    not O(T^2).

    Working set at any moment: one (block_q, block_k) tile of scores, which is a
    constant independent of sequence length. That is the whole memory claim.
    """
    T_q, d = Q.shape
    T_k = K.shape[0]
    scale = 1.0 / math.sqrt(d)

    O = torch.zeros(T_q, V.shape[1], dtype=Q.dtype)
    m = torch.full((T_q,), float("-inf"), dtype=Q.dtype)
    l = torch.zeros(T_q, dtype=Q.dtype)
    peak_tile = 0

    for i in range(0, T_q, block_q):
        qi = Q[i:i + block_q]
        m_i = torch.full((qi.shape[0],), float("-inf"), dtype=Q.dtype)
        l_i = torch.zeros(qi.shape[0], dtype=Q.dtype)
        o_i = torch.zeros(qi.shape[0], V.shape[1], dtype=Q.dtype)

        for j in range(0, T_k, block_k):
            kj, vj = K[j:j + block_k], V[j:j + block_k]
            s = (qi @ kj.T) * scale                       # the ONLY quadratic object
            peak_tile = max(peak_tile, s.numel())
            if causal:
                rows = torch.arange(i, i + qi.shape[0]).unsqueeze(1)
                cols = torch.arange(j, j + kj.shape[0]).unsqueeze(0)
                s = s.masked_fill(cols > rows, float("-inf"))

            m_blk = s.amax(dim=-1)
            m_new = torch.maximum(m_i, m_blk)
            # A fully-masked tile gives m_blk = -inf; guard so -inf - -inf never happens.
            corr = torch.where(torch.isinf(m_i) & (m_i < 0), torch.zeros_like(m_i),
                               torch.exp(m_i - m_new))
            p = torch.exp(s - m_new.unsqueeze(1))
            p = torch.nan_to_num(p, nan=0.0)
            l_i = corr * l_i + p.sum(-1)
            o_i = corr.unsqueeze(1) * o_i + p @ vj
            m_i = m_new

        safe_l = torch.where(l_i > 0, l_i, torch.ones_like(l_i))
        O[i:i + block_q] = o_i / safe_l.unsqueeze(1)
        m[i:i + block_q] = m_i
        l[i:i + block_q] = l_i

    L = m + torch.log(torch.where(l > 0, l, torch.ones_like(l)))
    return O, L, peak_tile


def reference_attention(Q, K, V, causal: bool = False):
    """The textbook version, which materialises the whole T x T matrix."""
    s = Q @ K.T / math.sqrt(Q.shape[-1])
    if causal:
        T_q, T_k = s.shape
        s = s.masked_fill(torch.ones(T_q, T_k, dtype=torch.bool).triu(1), float("-inf"))
    p = torch.softmax(s, dim=-1)
    return p @ V, p


# ============================================================ 3. IO complexity
def standard_attention_hbm_words(T: int, d: int) -> float:
    """HBM traffic for the textbook algorithm: Theta(T d + T^2).

    The T^2 term is the score matrix, written once by the matmul kernel, read and
    written by the softmax kernel, and read again by the second matmul — every
    unfused kernel boundary is a round trip through HBM.
    """
    return 4 * T * d + 4 * T * T


def flash_attention_hbm_words(T: int, d: int, sram_words: float) -> float:
    """Theta(T^2 d^2 / M): the whole of K and V is re-read once per block of Q.

    With a block size of B = M/(4d) rows (Q, K, V and O tiles must all fit in SRAM),
    there are T/B query blocks, each re-reading 2Td words of K and V:
        4Td + 2Td * (T / B) = 4Td + 8 T^2 d^2 / M.

    Note what this means for the *ratio* against the standard algorithm: for large T
    it tends to M / (2 d^2), which is INDEPENDENT of T. The paper's "9x less memory
    traffic" is that constant, not an asymptotic separation in sequence length — a
    distinction worth catching on a careful read, and exactly the kind of thing an
    interviewer means by "what does the paper not claim?".
    """
    block = max(sram_words / (4 * d), 1.0)
    n_blocks_k = math.ceil(T / block)
    return 4 * T * d + 2 * T * d * n_blocks_k


def io_reduction_limit(d: int, sram_words: float) -> float:
    """The large-T limit of standard/flash traffic."""
    return sram_words / (2 * d * d)


# ============================================================ 4. backward
def flash_attention_backward(Q, K, V, O, L, dO):
    """Recompute P from (Q, K, L) instead of storing it.

    The softmax correction term needs rowsum(dP * P). Storing P would cost O(T^2);
    but rowsum(dP * P) == rowsum(dO * O), the identity from module 01 step 6b, and
    both of those are O(T). That lemma is what makes the backward pass fit in linear
    memory, and it is why exercise 01 asked you to prove it.
    """
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    s = Q @ K.T * scale
    P = torch.exp(s - L.unsqueeze(1))               # recomputed, never stored
    dV = P.T @ dO
    dP = dO @ V.T
    D = (dO * O).sum(-1, keepdim=True)              # == rowsum(dP * P)
    dS = P * (dP - D)
    return dS @ K * scale, dS.T @ Q * scale, dV


# ============================================================ self-test
def main() -> int:
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)

    # ---- 1. the recurrence
    z = torch.randn(64)
    check("1 online softmax matches the standard one", close(online_softmax(z), torch.softmax(z, 0), atol=1e-14))
    for block in (1, 3, 8, 64, 128):
        check(f"1 block size {block} gives the same answer",
              close(online_softmax(z, block), torch.softmax(z, 0), atol=1e-14))

    big = torch.tensor([1000.0, 1001.0, 999.0, 1002.0] * 8)
    check("1 it survives inputs that overflow a naive softmax",
          close(online_softmax(big, 5), torch.softmax(big, 0), atol=1e-14) and
          bool(torch.isinf(big.exp()).any()),
          "exp(1000) is inf, yet the streaming version is exact")

    m, l = online_softmax_stats(z, 7)
    check("1 the running stats are the real max and sum",
          close(m, z.max(), atol=1e-14) and close(l, (z - z.max()).exp().sum(), atol=1e-12))

    # ---- 2. tiled attention: exact, not approximate
    T, d = 64, 16
    Q, K, V = (torch.randn(T, d) for _ in range(3))
    ref, P = reference_attention(Q, K, V)
    for bq, bk in [(8, 8), (16, 32), (64, 64), (5, 7), (T, T)]:
        O, L, peak = flash_attention_forward(Q, K, V, bq, bk)
        check(f"2 tiles {bq}x{bk} reproduce attention exactly", close(O, ref, atol=1e-13),
              f"max difference {float((O - ref).abs().max()):.2e}")

    ref_c, _ = reference_attention(Q, K, V, causal=True)
    for bq, bk in [(8, 8), (16, 16), (7, 9)]:
        O, L, _ = flash_attention_forward(Q, K, V, bq, bk, causal=True)
        check(f"2 causal tiles {bq}x{bk} are exact", close(O, ref_c, atol=1e-13))

    check("2 it agrees with torch's fused kernel",
          close(flash_attention_forward(Q, K, V, 16, 16)[0],
                F.scaled_dot_product_attention(Q, K, V), atol=1e-13))

    # L is the log-sum-exp: the O(T) statistic that replaces the O(T^2) matrix.
    _, L, _ = flash_attention_forward(Q, K, V, 16, 16)
    lse = torch.logsumexp(Q @ K.T / math.sqrt(d), dim=-1)
    check("2 L is the row log-sum-exp", close(L, lse, atol=1e-12))
    check("2 and it is O(T), not O(T^2)", L.numel() == T)

    # ---- 3. the memory claim, measured
    print(f"\n{'T':>6} {'reference P entries':>21} {'flash peak tile':>17} {'ratio':>8}")
    peaks = {}
    for T_ in (32, 64, 128, 256):
        Qs, Ks, Vs = (torch.randn(T_, 16) for _ in range(3))
        _, _, peak = flash_attention_forward(Qs, Ks, Vs, 16, 16)
        peaks[T_] = peak
        print(f"{T_:>6} {T_ * T_:>21} {peak:>17} {T_ * T_ / peak:>7.0f}x")
    check("3 the flash working set does not grow with T",
          len(set(peaks.values())) == 1, f"constant at {peaks[32]} score entries")
    check("3 while the reference grows quadratically", 256 ** 2 / 32 ** 2 == 64)

    # ---- 3b. IO complexity
    print(f"\n{'T':>7} {'standard words':>16} {'flash words':>14} {'reduction':>11}")
    sram = 100_000 / 2          # 100 KB of on-chip SRAM, in fp16 words
    for T_ in (1024, 4096, 16384):
        st = standard_attention_hbm_words(T_, 64)
        fl = flash_attention_hbm_words(T_, 64, sram)
        print(f"{T_:>7} {st:>16.3e} {fl:>14.3e} {st / fl:>10.1f}x")
    check("3b flash moves less data at long context",
          flash_attention_hbm_words(4096, 64, sram) < standard_attention_hbm_words(4096, 64))
    # The reduction tends to M/(2d^2) and does NOT keep growing with T. Say so out loud:
    # it is the most commonly misread part of the paper.
    ratios = [standard_attention_hbm_words(T_, 64) / flash_attention_hbm_words(T_, 64, sram)
              for T_ in (4096, 16384, 65536)]
    check("3b the reduction converges to M/(2d^2), not to infinity",
          close(ratios[-1], io_reduction_limit(64, sram), rtol=0.15),
          f"ratios {[round(r, 1) for r in ratios]} -> limit {io_reduction_limit(64, sram):.1f}")
    check("3b it is ~9x at the paper's d=64, M=100KB", 6 < ratios[-1] < 14, f"{ratios[-1]:.1f}x")
    check("3b a bigger SRAM helps",
          flash_attention_hbm_words(4096, 64, 4 * sram) < flash_attention_hbm_words(4096, 64, sram))

    # ---- 4. backward without storing P
    Qg, Kg, Vg = (t.clone().requires_grad_(True) for t in (Q, K, V))
    out, _ = reference_attention(Qg, Kg, Vg)
    dO = torch.randn(T, d)
    out.backward(dO)

    O, L, _ = flash_attention_forward(Q, K, V, 16, 16)
    dQ, dK, dV = flash_attention_backward(Q, K, V, O, L, dO)
    check("4 dQ matches autograd", close(dQ, Qg.grad, atol=1e-11))
    check("4 dK matches autograd", close(dK, Kg.grad, atol=1e-11))
    check("4 dV matches autograd", close(dV, Vg.grad, atol=1e-11))

    _, P_ref = reference_attention(Q, K, V)
    P_recomputed = torch.exp(Q @ K.T / math.sqrt(d) - L.unsqueeze(1))
    check("4 P is recoverable from (Q, K, L) alone", close(P_recomputed, P_ref, atol=1e-13),
          "so it never has to be stored")
    dP = dO @ V.T
    check("4 the module-01 rowsum identity holds",
          close((dP * P_ref).sum(-1), (dO * O).sum(-1), atol=1e-12),
          "rowsum(dP*P) == rowsum(dO*O): the lemma the backward pass rests on")

    return 0 if report() else 1


if __name__ == "__main__":
    raise SystemExit(main())
