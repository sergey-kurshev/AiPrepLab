# %% [markdown]
# # Exercise 09 — CUDA and cuBLAS
#
# Two halves, and **both are worth doing**:
#
# - **The portable half (this notebook).** Roofline arithmetic, coalescing analysis, an occupancy
#   calculator, and numpy emulations of the exact index arithmetic in the CUDA kernels. Kernel bugs
#   are almost always indexing bugs, and indexing can be verified on any machine — so this half
#   catches most of what goes wrong.
# - **The CUDA half (`kernels/`).** Four self-checking programs. Build with `make` (Linux) or
#   `.\build.ps1` (Windows — it needs the MSVC build tools, because `nvcc` uses a host compiler for
#   everything outside a kernel). This notebook runs them if they are built.
#
# If you have no GPU, do the portable half and read the `.cu` files. Almost everything that
# transfers to an interview is in the reasoning, not the wall-clock number.

# %%
import sys
import pathlib
import subprocess

_here = pathlib.Path.cwd()
for _c in (_here, *_here.parents):
    if (_c / "tools").is_dir():
        sys.path.insert(0, str(_c / "tools"))
        break

import math
import numpy as np
from checks import check, report, close

KERNELS = (_here if (_here / "kernels").is_dir() else _here / "09-cuda-and-cublas") / "kernels"
H100_FLOPS, H100_BW = 990e12, 3.35e12
print("kernel sources:", KERNELS)
print("built binaries:", sorted(p.name for p in KERNELS.glob("0*") if p.suffix in ("", ".exe")))

# %% [markdown]
# ---
# ## Part 1 — The roofline
#
# Every performance claim in modules 04, 05 and 07 is a statement about this plot. Implement it,
# then place the operations you have written this course on it.

# %%
def ridge_point(peak_flops: float, peak_bandwidth: float) -> float:
    """FLOPs per byte at which a device stops being memory-bound."""
    # TODO
    raise NotImplementedError


def attainable_flops(intensity: float, peak_flops: float, peak_bandwidth: float) -> float:
    # TODO
    raise NotImplementedError


def naive_gemm_intensity(n: int, bytes_per_elem: int = 4) -> float:
    """Level 0: every thread re-reads A and B from global memory for every k."""
    # TODO
    raise NotImplementedError


def tiled_gemm_intensity(n: int, tile: int, bytes_per_elem: int = 4) -> float:
    """Tiling by `tile` divides global traffic by `tile`."""
    # TODO
    raise NotImplementedError


try:
    ridge = ridge_point(H100_FLOPS, H100_BW)
    print(f"H100 ridge point: {ridge:.0f} FLOP/byte\n")
    print(f"{'operation':<34} {'intensity':>11} {'regime':>14} {'% of peak':>11}")
    ops = [
        ("vector add", 1 / 12),
        ("softmax (fp32)", 1 / 8),
        ("decode step, batch 1 (module 05)", 1.0),
        ("decode step, batch 256", 256.0),
        ("naive SGEMM n=4096", naive_gemm_intensity(4096)),
        ("tiled SGEMM, 32x32 tiles", tiled_gemm_intensity(4096, 32)),
        ("tiled + 4x4 register tiles", tiled_gemm_intensity(4096, 32) * 4),
    ]
    for name, ai in ops:
        att = attainable_flops(ai, H100_FLOPS, H100_BW)
        print(f"{name:<34} {ai:>11.2f} {'memory' if ai < ridge else 'compute':>14} "
              f"{100 * att / H100_FLOPS:>10.1f}%")

    check("1a the ridge point is ~300 FLOP/byte", 250 < ridge < 350)
    check("1a batch-1 decode is memory-bound", 1.0 < ridge / 100)
    check("1b tiling multiplies intensity by the tile width",
          close(tiled_gemm_intensity(4096, 32), 32 * naive_gemm_intensity(4096), rtol=1e-9))
except Exception as _e:
    check("part 1", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **1b. (discussion)** Using only the table above, explain in one sentence each:
# why quantisation speeds up decoding but not training; why batching helps throughput so much;
# and why FlashAttention is faster despite doing *more* arithmetic than the standard algorithm.

# %% [markdown]
# ---
# ## Part 2 — Coalescing
#
# The number-one performance bug in hand-written kernels, and the cheapest to fix. The hardware
# coalesces a warp's 32 accesses into as few 128-byte lines as possible.
#
# Implement the transaction counter, then use it to explain the level-0 → level-1 speedup in
# `02_sgemm.cu` — which changes no arithmetic at all.

# %%
def transactions_per_warp(addresses, bytes_per_elem=4, line_bytes=128) -> int:
    """How many distinct memory lines a warp's 32 accesses touch."""
    # TODO
    raise NotImplementedError


def coalescing_efficiency(addresses, bytes_per_elem=4, line_bytes=128) -> float:
    """Useful bytes / bytes actually fetched."""
    # TODO
    raise NotImplementedError


try:
    K_dim = 1024
    patterns = {
        "consecutive (perfect)": np.arange(32),
        "stride 2": np.arange(32) * 2,
        "stride 32": np.arange(32) * 32,
        "naive GEMM: A[row*K + k]": np.arange(32) * K_dim,
        "coalesced GEMM: B[k*N + col]": np.arange(32),
        "reversed": np.arange(31, -1, -1),
    }
    print(f"{'access pattern':<32} {'transactions':>13} {'efficiency':>12}")
    for name, addr in patterns.items():
        print(f"{name:<32} {transactions_per_warp(addr):>13} {coalescing_efficiency(addr):>11.1%}")

    check("2 consecutive reads take one transaction", transactions_per_warp(np.arange(32)) == 1)
    check("2 stride-32 reads take 32", transactions_per_warp(np.arange(32) * 32) == 32)
    check("2 reversed order is still one transaction", transactions_per_warp(np.arange(31, -1, -1)) == 1,
          "coalescing cares about which LINES are touched, not about ordering")
    check("2 the naive GEMM's A reads are worst-case",
          transactions_per_warp(np.arange(32) * K_dim) == 32)
except Exception as _e:
    check("part 2", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# ---
# ## Part 3 — Occupancy, and why it is not the goal
#
# Resident warps per SM, limited by whichever of three resources runs out first. Build the
# calculator, then use it on the kernels in `kernels/`.
#
# Then answer the question that separates people who have tuned kernels from people who have read
# about them: the register-tiled GEMM has *lower* occupancy than the shared-memory one and is much
# faster. Why?

# %%
def occupancy(threads_per_block, regs_per_thread, shared_per_block,
              max_threads_per_sm=2048, max_regs_per_sm=65536,
              max_shared_per_sm=102400, max_blocks_per_sm=32, warp_size=32) -> dict:
    """Return blocks_per_sm, warps_per_sm, occupancy, and the binding constraint."""
    # TODO
    raise NotImplementedError


try:
    configs = [
        ("modest kernel", 256, 32, 8192),
        ("register-heavy (level 3 GEMM)", 256, 128, 8192),
        ("shared-heavy", 256, 32, 48 * 1024),
        ("04_attention.cu (one warp/block)", 32, 40, 16 * 1024),
    ]
    print(f"{'configuration':<36} {'warps/SM':>9} {'occupancy':>11} {'limited by':>16}")
    for name, t, r, s in configs:
        o = occupancy(t, r, s)
        print(f"{name:<36} {o['warps_per_sm']:>9} {o['occupancy']:>10.0%} {o['limiter']:>16}")

    check("3 a modest kernel reaches full occupancy", occupancy(256, 32, 8192)["occupancy"] >= 1.0)
    check("3 register pressure is correctly identified",
          occupancy(256, 128, 8192)["limiter"] == "registers")
    check("3 shared memory can be the limiter",
          occupancy(256, 32, 48 * 1024)["limiter"] == "shared memory")
    check("3 one warp per block wastes the SM", occupancy(32, 40, 16 * 1024)["occupancy"] < 0.6)
except Exception as _e:
    check("part 3", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# ---
# ## Part 4 — Emulating the GEMM kernels
#
# Read `kernels/02_sgemm.cu` first. Then transcribe its index arithmetic — *exactly*, including the
# zero-padded out-of-range loads — into numpy and check it reproduces `A @ B`.
#
# This is not busywork. It is the fastest way to debug a kernel: if the emulation is right and the
# kernel is wrong, the bug is in synchronisation or memory scope, which is a much smaller search
# space than "somewhere in the indexing".

# %%
def emulate_tiled_gemm(A, B, TS=32):
    """Level 2: shared-memory tiling. Mirror sgemm_tiled line for line."""
    # TODO
    raise NotImplementedError


try:
    rng = np.random.default_rng(0)
    A = rng.standard_normal((96, 80))
    B = rng.standard_normal((80, 112))
    check("4a tiled GEMM reproduces A @ B", close(emulate_tiled_gemm(A, B, 32), A @ B, atol=1e-10))
    check("4a ...with ragged edges too", close(emulate_tiled_gemm(A, B, 16), A @ B, atol=1e-10),
          "96x80x112 is not tile-aligned, which is exactly where bounds bugs live")
except Exception as _e:
    check("4a", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **4b. The register-tiled kernel.** Harder, and worth the effort: its cooperative load indices
# (`innerRowA`, `strideA`, …) are chosen so consecutive threads read consecutive addresses, and
# getting them wrong yields a kernel that runs, coalesces badly, and is wrong only for some shapes.
#
# Also write the *write map*: which thread writes each element of C. Every cell must be written
# exactly once — zero means a gap, two means a race.

# %%
def emulate_register_tiled_gemm(A, B, BM=64, BN=64, BK=8, TM=4, TN=4):
    # TODO
    raise NotImplementedError


def gemm_write_map(M, N, BM=64, BN=64, TM=4, TN=4):
    """Count how many (block, thread) pairs write each element of C."""
    # TODO
    raise NotImplementedError


try:
    A2 = rng.standard_normal((128, 64))
    B2 = rng.standard_normal((64, 128))
    check("4b register-tiled GEMM reproduces A @ B",
          close(emulate_register_tiled_gemm(A2, B2), A2 @ B2, atol=1e-10))
    counts = gemm_write_map(128, 128)
    check("4b every output element is written exactly once", bool((counts == 1).all()),
          f"min {counts.min()}, max {counts.max()}")
except Exception as _e:
    check("4b", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# ---
# ## Part 5 — Emulating the fused attention kernel
#
# Read `kernels/04_attention.cu`, then transcribe it. It is module 07's algorithm with the memory
# hierarchy made explicit: `sK`/`sV` are the shared-memory tiles, `acc`/`m`/`l` are registers.
#
# Track the peak number of score elements alive. It should equal the tile size, at any sequence
# length — the claim you verified in Python in module 07, now true of the kernel you are about to
# compile.

# %%
def emulate_flash_kernel(Q, K, V, tile=32, causal=False):
    """Return (O, peak_score_elements)."""
    # TODO
    raise NotImplementedError


def reference_attention(Q, K, V, causal=False):
    T, D = Q.shape
    s = Q @ K.T / math.sqrt(D)
    if causal:
        s = np.where(np.tril(np.ones((T, T), dtype=bool)), s, -np.inf)
    s = s - s.max(-1, keepdims=True)
    p = np.exp(s)
    return (p / p.sum(-1, keepdims=True)) @ V


try:
    T, D = 96, 32
    Qa, Ka, Va = (rng.standard_normal((T, D)) for _ in range(3))
    for causal in (False, True):
        O, peak = emulate_flash_kernel(Qa, Ka, Va, 32, causal)
        check(f"5 exact (causal={causal})", close(O, reference_attention(Qa, Ka, Va, causal), atol=1e-10))
    for tile in (8, 16, 32):
        check(f"5 tile {tile} gives the same answer",
              close(emulate_flash_kernel(Qa, Ka, Va, tile, True)[0],
                    reference_attention(Qa, Ka, Va, True), atol=1e-10))
    peaks = {t: emulate_flash_kernel(Qa[:t], Ka[:t], Va[:t], 32)[1] for t in (32, 64, 96)}
    print("peak score elements by T:", peaks)
    check("5 the working set is constant in T", len(set(peaks.values())) == 1)
except Exception as _e:
    check("part 5", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# ---
# ## Part 6 — The CUDA half
#
# Build and run. Each binary checks itself against a CPU reference and prints a timing table.
#
# ```
# cd kernels
# make                # Linux / WSL
# .\build.ps1 -Run    # Windows (needs the MSVC build tools)
# ```
#
# Set `ARCH` to your compute capability (`make ARCH=sm_86`) to avoid a JIT pause on first launch.

# %%
def run_binary(name, *args):
    for candidate in (KERNELS / f"{name}.exe", KERNELS / name):
        if candidate.exists():
            r = subprocess.run([str(candidate), *args], capture_output=True, text=True, timeout=900)
            return r.returncode == 0, r.stdout + r.stderr
    return None, f"{name} not built"


for name in ("01_vector_add", "02_sgemm", "03_softmax", "04_attention"):
    ok, out = run_binary(name)
    print(f"=== {name} ===")
    if ok is None:
        print(f"  {out}")
    else:
        print(out.strip()[:3000])
        check(f"6 {name} passes its own checks", ok)
    print()
report()

# %% [markdown]
# ### What to record
#
# For `02_sgemm`, fill in this table from your own run and keep it — it is a concrete, quotable
# result about hardware you have actually used:
#
# | level | GFLOP/s | % of cuBLAS | what changed |
# |---|---|---|---|
# | 0 naive | | | |
# | 1 coalesced | | | one index swapped |
# | 2 shared tiles | | | K/32 fewer global reads |
# | 3 register tiles | | | 4x reuse of each shared value |
# | cuBLAS | | 100% | |
#
# Then answer: levels 0 and 1 do *identical arithmetic*. What is the gap, and what is the
# one-sentence explanation? (You computed it in part 2.)

# %% [markdown]
# ---
# ## Exercises in the kernels themselves
#
# In rough order of value:
#
# 1. **`01_vector_add.cu`:** sweep the block size over 32…1024 and find the knee. Explain it with
#    your occupancy calculator from part 3.
# 2. **`02_sgemm.cu`:** add a level 4 with `float4` vectorised loads. Expect 10–20%. Then try
#    double buffering: prefetch tile $t+1$ while computing tile $t$.
# 3. **`03_softmax.cu`:** the online kernel still reads the input twice. Make it genuinely
#    single-pass for rows that fit in shared memory, and measure the difference.
# 4. **`04_attention.cu`:** process `BLOCK_M` query rows per block instead of one, so the K/V tiles
#    are reused across queries. This is the single biggest win available in that file, and it is
#    what a real implementation does.
# 5. Profile with `ncu --set full ./02_sgemm` and find where each level's time actually goes.
#    Compare the profiler's verdict with your prediction — the gap between them is the learning.
# 6. Rewrite `04_attention.cu` in **Triton** and compare. It will take a tenth of the code and
#    probably beat your CUDA; that is the honest state of the field, and knowing both is the point.

# %% [markdown]
# ---
# ## The spine, complete
#
# | module | attention as |
# |---|---|
# | 00 | tensor operations, broadcasting, autograd |
# | 01 | a low-rank bilinear form with a derivable gradient |
# | 02 | numpy code you wrote, forward and backward |
# | 03 | four planted bugs and the tests that catch them |
# | 04 | a computation sharded across devices |
# | 05 | a KV cache and a quantisation target |
# | 06 | soft retrieval, next to the hard kind |
# | 07 | a paper, reproduced from its recurrence |
# | 08 | the thing being aligned |
# | 09 | **a kernel, racing cuBLAS** |
#
# One function, ten levels. That is the thing worth being able to talk about for an hour.
