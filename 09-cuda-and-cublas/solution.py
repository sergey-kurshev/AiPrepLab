"""Reference solutions for exercise 09 — the GPU-free half.

Run with `python solution.py`.

The CUDA sources in `kernels/` are the other half; this file contains the parts you
can check without a GPU, which is more of the module than you might expect. Kernel
bugs are almost always *index arithmetic* bugs, and index arithmetic can be emulated
exactly in numpy — so the tiling logic of a GEMM or an attention kernel is verifiable
on any machine. What needs the hardware is the performance, not the correctness.
"""

from __future__ import annotations

import math
import subprocess
import sys
import pathlib

_here = pathlib.Path(__file__).resolve().parent
for _c in (_here, *_here.parents):
    if (_c / "tools").is_dir():
        sys.path.insert(0, str(_c / "tools"))
        break

import numpy as np
from checks import check, report, close

KERNELS = _here / "kernels"


# ============================================================ 1. roofline
def ridge_point(peak_flops: float, peak_bandwidth: float) -> float:
    """FLOPs per byte at which a device stops being memory-bound."""
    return peak_flops / peak_bandwidth


def attainable_flops(intensity: float, peak_flops: float, peak_bandwidth: float) -> float:
    """The roofline: min(peak compute, bandwidth * arithmetic intensity)."""
    return min(peak_flops, peak_bandwidth * intensity)


def gemm_intensity(m: int, n: int, k: int, bytes_per_elem: int = 4) -> float:
    """2mnk FLOPs against (mk + kn + mn) elements moved, assuming perfect reuse."""
    flops = 2.0 * m * n * k
    moved = bytes_per_elem * (m * k + k * n + m * n)
    return flops / moved


def naive_gemm_intensity(n: int, bytes_per_elem: int = 4) -> float:
    """What the level-0 kernel actually achieves: every thread re-reads A and B from
    global memory, so 2n^3 elements are moved for 2n^3 FLOPs."""
    return 2.0 * n ** 3 / (bytes_per_elem * 2.0 * n ** 3)


def tiled_gemm_intensity(n: int, tile: int, bytes_per_elem: int = 4) -> float:
    """Tiling by `tile` divides the global traffic by `tile` — the whole point."""
    return naive_gemm_intensity(n, bytes_per_elem) * tile


# ============================================================ 2. coalescing
def transactions_per_warp(addresses: np.ndarray, bytes_per_elem: int = 4,
                          line_bytes: int = 128) -> int:
    """How many memory transactions one warp's accesses require.

    The hardware coalesces a warp's 32 accesses into as few `line_bytes` lines as it
    can. A warp reading consecutive floats touches one line; a warp reading with
    stride 32 touches 32. Same instruction count, same bytes *requested*, thirty-two
    times the bytes actually *moved*.
    """
    lines = {(int(a) * bytes_per_elem) // line_bytes for a in addresses}
    return len(lines)


def coalescing_efficiency(addresses: np.ndarray, bytes_per_elem: int = 4,
                          line_bytes: int = 128) -> float:
    """Useful bytes divided by bytes actually fetched."""
    useful = len(addresses) * bytes_per_elem
    fetched = transactions_per_warp(addresses, bytes_per_elem, line_bytes) * line_bytes
    return useful / fetched


# ============================================================ 3. occupancy
def occupancy(threads_per_block: int, regs_per_thread: int, shared_per_block: int,
              max_threads_per_sm: int = 2048, max_regs_per_sm: int = 65536,
              max_shared_per_sm: int = 102400, max_blocks_per_sm: int = 32,
              warp_size: int = 32) -> dict:
    """Resident warps per SM, and which resource is the binding constraint.

    The answer to "why is my occupancy low" is always one of three numbers, and this
    tells you which. Note that low occupancy is not automatically a problem: a
    register-tiled GEMM deliberately trades occupancy for instruction-level
    parallelism and wins.
    """
    warps_per_block = math.ceil(threads_per_block / warp_size)
    by_threads = max_threads_per_sm // threads_per_block
    by_regs = max_regs_per_sm // (regs_per_thread * threads_per_block) if regs_per_thread else 1 << 30
    by_shared = max_shared_per_sm // shared_per_block if shared_per_block else 1 << 30
    blocks = min(by_threads, by_regs, by_shared, max_blocks_per_sm)
    limiter = min(
        [("threads", by_threads), ("registers", by_regs),
         ("shared memory", by_shared), ("blocks/SM", max_blocks_per_sm)],
        key=lambda kv: kv[1])[0]
    return {
        "blocks_per_sm": blocks,
        "warps_per_sm": blocks * warps_per_block,
        "occupancy": blocks * warps_per_block / (max_threads_per_sm / warp_size),
        "limiter": limiter,
    }


# ============================================================ 4. GEMM tiling, emulated
def emulate_tiled_gemm(A: np.ndarray, B: np.ndarray, TS: int = 32) -> np.ndarray:
    """Execute the level-2 kernel's index arithmetic exactly, in numpy.

    Every line here mirrors a line of `02_sgemm.cu::sgemm_tiled`, including the
    zero-padding of out-of-range loads. If this reproduces A @ B then the indexing is
    right, and the only things left to get wrong on the GPU are the two
    __syncthreads() calls — which a single-threaded emulation cannot model, and which
    are therefore the two lines to re-read by eye.
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = np.zeros((M, N), dtype=A.dtype)

    for by in range((M + TS - 1) // TS):
        for bx in range((N + TS - 1) // TS):
            acc = np.zeros((TS, TS), dtype=np.float64)
            for t in range((K + TS - 1) // TS):
                As = np.zeros((TS, TS), dtype=np.float64)
                Bs = np.zeros((TS, TS), dtype=np.float64)
                for ty in range(TS):
                    for tx in range(TS):
                        row, a_col = by * TS + ty, t * TS + tx
                        b_row, col = t * TS + ty, bx * TS + tx
                        if row < M and a_col < K:
                            As[ty, tx] = A[row, a_col]
                        if b_row < K and col < N:
                            Bs[ty, tx] = B[b_row, col]
                acc += As @ Bs                      # the inner k loop, vectorised
            for ty in range(TS):
                for tx in range(TS):
                    row, col = by * TS + ty, bx * TS + tx
                    if row < M and col < N:
                        C[row, col] = acc[ty, tx]
    return C


def emulate_register_tiled_gemm(A, B, BM=64, BN=64, BK=8, TM=4, TN=4):
    """The level-3 kernel's index arithmetic, including its cooperative loads.

    This is the emulation that earns its keep: the loading indices (innerRowA,
    strideA, ...) are chosen so that consecutive threads read consecutive addresses,
    and getting them wrong produces a kernel that runs, coalesces badly, and computes
    the wrong answer only for some shapes.
    """
    M, K = A.shape
    _, N = B.shape
    assert M % BM == 0 and N % BN == 0 and K % BK == 0, "this kernel requires exact tiling"
    num_threads = (BM // TM) * (BN // TN)
    stride_a = num_threads // BK
    stride_b = num_threads // BN
    C = np.zeros((M, N), dtype=np.float64)

    for cRow in range(M // BM):
        for cCol in range(N // BN):
            acc = np.zeros((num_threads, TM * TN))
            for bk in range(0, K, BK):
                As = np.zeros((BM, BK))
                Bs = np.zeros((BK, BN))
                for tid in range(num_threads):
                    inner_row_a, inner_col_a = tid // BK, tid % BK
                    for off in range(0, BM, stride_a):
                        As[inner_row_a + off, inner_col_a] = \
                            A[cRow * BM + inner_row_a + off, bk + inner_col_a]
                    inner_row_b, inner_col_b = tid // BN, tid % BN
                    for off in range(0, BK, stride_b):
                        Bs[inner_row_b + off, inner_col_b] = \
                            B[bk + inner_row_b + off, cCol * BN + inner_col_b]
                for tid in range(num_threads):
                    thread_row, thread_col = tid // (BN // TN), tid % (BN // TN)
                    regM = As[thread_row * TM:(thread_row + 1) * TM, :]      # (TM, BK)
                    regN = Bs[:, thread_col * TN:(thread_col + 1) * TN]      # (BK, TN)
                    acc[tid] += (regM @ regN).reshape(-1)
            for tid in range(num_threads):
                thread_row, thread_col = tid // (BN // TN), tid % (BN // TN)
                for i in range(TM):
                    for j in range(TN):
                        C[cRow * BM + thread_row * TM + i,
                          cCol * BN + thread_col * TN + j] = acc[tid, i * TN + j]
    return C


def gemm_write_map(M, N, BM=64, BN=64, TM=4, TN=4) -> np.ndarray:
    """Which (block, thread) writes each element of C? Every cell must be written once."""
    num_threads = (BM // TM) * (BN // TN)
    counts = np.zeros((M, N), dtype=int)
    for cRow in range(M // BM):
        for cCol in range(N // BN):
            for tid in range(num_threads):
                tr, tc = tid // (BN // TN), tid % (BN // TN)
                for i in range(TM):
                    for j in range(TN):
                        counts[cRow * BM + tr * TM + i, cCol * BN + tc * TN + j] += 1
    return counts


# ============================================================ 5. fused attention, emulated
def emulate_flash_kernel(Q, K, V, tile: int = 32, causal: bool = False):
    """`04_attention.cu` in numpy, including the running (m, l) and the rescaling.

    Returns (O, peak_score_elements). The second value is the point: it is `tile`,
    regardless of sequence length.
    """
    T, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    O = np.zeros((T, D))
    peak = 0

    for qi in range(T):
        q = Q[qi]
        acc = np.zeros(D)
        m, l = -np.inf, 0.0
        n_keys = qi + 1 if causal else T
        for kt in range(0, n_keys, tile):
            tile_len = min(tile, n_keys - kt)
            sK = K[kt:kt + tile_len]
            sV = V[kt:kt + tile_len]
            s = (sK @ q) * scale                    # the only quadratic object
            peak = max(peak, s.size)
            m_new = max(m, float(s.max()))
            corr = math.exp(m - m_new) if np.isfinite(m) else 0.0
            p = np.exp(s - m_new)
            l = corr * l + float(p.sum())
            acc = corr * acc + p @ sV
            m = m_new
        O[qi] = acc / l
    return O, peak


def reference_attention(Q, K, V, causal: bool = False):
    T, D = Q.shape
    s = Q @ K.T / math.sqrt(D)
    if causal:
        s = np.where(np.tril(np.ones((T, T), dtype=bool)), s, -np.inf)
    s = s - s.max(-1, keepdims=True)
    p = np.exp(s)
    return (p / p.sum(-1, keepdims=True)) @ V


# ============================================================ 6. driving the binaries
def binary_path(name: str) -> pathlib.Path | None:
    for candidate in (KERNELS / f"{name}.exe", KERNELS / name):
        if candidate.exists():
            return candidate
    return None


def run_kernel_binary(name: str, *args) -> tuple[bool, str]:
    path = binary_path(name)
    if path is None:
        return False, f"{name} not built (see kernels/Makefile or kernels/build.ps1)"
    try:
        r = subprocess.run([str(path), *args], capture_output=True, text=True, timeout=600)
        return r.returncode == 0, r.stdout + r.stderr
    except Exception as exc:                                  # noqa: BLE001
        return False, f"failed to run {name}: {exc}"


# ============================================================ self-test
def main() -> int:
    rng = np.random.default_rng(0)

    # ---- 1. roofline
    h100_flops, h100_bw = 990e12, 3.35e12
    ridge = ridge_point(h100_flops, h100_bw)
    check("1 the H100 ridge point is ~300 FLOP/byte", 250 < ridge < 350, f"{ridge:.0f}")
    check("1 vector add is hopelessly memory-bound",
          attainable_flops(1 / 12, h100_flops, h100_bw) < h100_flops / 100,
          "1 FLOP per 12 bytes: 0.03% of peak compute is the ceiling")
    check("1 a large GEMM is compute-bound",
          gemm_intensity(4096, 4096, 4096) > ridge,
          f"intensity {gemm_intensity(4096, 4096, 4096):.0f} FLOP/byte with perfect reuse")
    check("1 ...but only with reuse: the naive kernel is not",
          naive_gemm_intensity(4096) < 1.0, f"{naive_gemm_intensity(4096):.2f} FLOP/byte")
    check("1 tiling multiplies intensity by the tile width",
          close(tiled_gemm_intensity(4096, 32), 32 * naive_gemm_intensity(4096), rtol=1e-9),
          f"{naive_gemm_intensity(4096):.2f} -> {tiled_gemm_intensity(4096, 32):.1f} FLOP/byte")

    # ---- 2. coalescing
    consecutive = np.arange(32)
    strided = np.arange(32) * 32
    check("2 a consecutive warp read is one transaction",
          transactions_per_warp(consecutive) == 1)
    check("2 a stride-32 read is 32 transactions",
          transactions_per_warp(strided) == 32,
          "identical instruction count, 32x the bytes moved")
    check("2 efficiency collapses accordingly",
          close(coalescing_efficiency(consecutive), 1.0, rtol=1e-9) and
          coalescing_efficiency(strided) < 0.05,
          f"{coalescing_efficiency(consecutive):.2f} vs {coalescing_efficiency(strided):.3f}")
    # The naive GEMM's A reads, exactly: 32 lanes each reading a different row.
    K_dim = 1024
    naive_a = np.arange(32) * K_dim
    check("2 the naive GEMM's A reads are worst-case",
          transactions_per_warp(naive_a) == 32,
          "threadIdx.x selecting the row is the whole bug in level 0")
    check("2 the coalesced GEMM's B reads are perfect",
          transactions_per_warp(np.arange(32)) == 1)

    # ---- 3. occupancy
    o_small = occupancy(256, 32, 8192)
    check("3 a modest kernel reaches full occupancy", o_small["occupancy"] >= 1.0,
          f"{o_small['warps_per_sm']} warps/SM, limited by {o_small['limiter']}")
    o_reg = occupancy(256, 128, 8192)
    check("3 register pressure cuts occupancy", o_reg["occupancy"] < o_small["occupancy"],
          f"{o_reg['occupancy']:.2f}, limited by {o_reg['limiter']}")
    check("3 ...and the limiter is correctly identified", o_reg["limiter"] == "registers")
    o_shared = occupancy(256, 32, 48 * 1024)
    check("3 shared memory can be the limiter", o_shared["limiter"] == "shared memory",
          f"occupancy {o_shared['occupancy']:.2f}")
    check("3 the single-warp attention kernel wastes the SM",
          occupancy(32, 40, 16 * 1024)["occupancy"] < 0.6,
          "one warp per block: see 04_attention.cu's own list of what it costs")

    # ---- 4. GEMM tiling
    A = rng.standard_normal((96, 80))
    B = rng.standard_normal((80, 112))
    check("4 the shared-memory tiling reproduces A @ B",
          close(emulate_tiled_gemm(A, B, 32), A @ B, atol=1e-10))
    check("4 ...including with ragged edges (96x80x112 is not tile-aligned)",
          close(emulate_tiled_gemm(A, B, 16), A @ B, atol=1e-10))

    A2 = rng.standard_normal((128, 64))
    B2 = rng.standard_normal((64, 128))
    check("4 the register tiling reproduces A @ B",
          close(emulate_register_tiled_gemm(A2, B2), A2 @ B2, atol=1e-10))
    counts = gemm_write_map(128, 128)
    check("4 every output element is written exactly once", bool((counts == 1).all()),
          "no gaps, no double writes — the check a race condition would fail")

    # ---- 5. attention
    T, D = 96, 32
    Qa = rng.standard_normal((T, D))
    Ka = rng.standard_normal((T, D))
    Va = rng.standard_normal((T, D))
    for causal in (False, True):
        O, peak = emulate_flash_kernel(Qa, Ka, Va, tile=32, causal=causal)
        ref = reference_attention(Qa, Ka, Va, causal=causal)
        check(f"5 the fused kernel is exact (causal={causal})", close(O, ref, atol=1e-10),
              f"max difference {float(np.abs(O - ref).max()):.2e}")
    peaks = {t: emulate_flash_kernel(Qa[:t], Ka[:t], Va[:t], tile=32)[1] for t in (32, 64, 96)}
    check("5 the score working set does not grow with T", len(set(peaks.values())) == 1,
          f"constant at {peaks[32]} elements while T^2 grows from {32**2} to {96**2}")
    for tile in (8, 16, 32):
        O, _ = emulate_flash_kernel(Qa, Ka, Va, tile=tile, causal=True)
        check(f"5 tile size {tile} gives the same answer",
              close(O, reference_attention(Qa, Ka, Va, causal=True), atol=1e-10))

    # ---- 6. the binaries, if they were built
    print()
    built = [n for n in ("01_vector_add", "02_sgemm", "03_softmax", "04_attention")
             if binary_path(n) is not None]
    if not built:
        print("No compiled kernels found. The GPU-free checks above are the portable half;")
        print("build the CUDA half with `make` (Linux) or `.\\build.ps1` (Windows) in kernels/.")
    else:
        for name in built:
            ok, out = run_kernel_binary(name)
            print(f"--- {name} ---")
            print(out.strip()[:2000])
            check(f"6 {name} passes its own checks", ok)

    return 0 if report() else 1


if __name__ == "__main__":
    raise SystemExit(main())
