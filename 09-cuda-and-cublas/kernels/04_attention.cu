// 04 — fused attention: the last level of the spine.
//
// One kernel computes  O = softmax(Q K^T / sqrt(d)) V  without ever writing the
// T x T score matrix to global memory. Q, K and V tiles live in shared memory; the
// running (m, l) statistics and the output accumulator live in registers; the online
// softmax recurrence from module 07 stitches the tiles together.
//
// This is FlashAttention's algorithm, written for legibility rather than speed. It
// will lose to PyTorch's fused kernel by a wide margin, and the reasons are worth
// knowing precisely -- they are listed at the bottom of this file. The deliverable is
// the understanding, not the benchmark.
//
// Build:  nvcc -O3 -o 04_attention 04_attention.cu
// Run:    ./04_attention

#include "common.cuh"
#include <vector>

#define D 64            // head dimension
#define TILE 32         // keys processed per iteration; also the block size (one warp)
#define FULL_MASK 0xffffffffu

// One block per query row, one warp (32 threads) per block. Keeping the block to a
// single warp means every reduction is a warp shuffle and no __syncthreads() is
// needed for them -- a deliberate simplification that costs occupancy and buys
// clarity. Improving this is the first stretch goal.
//
// Each thread owns D / TILE = 2 components of the output accumulator.
__global__ void flash_attention_kernel(const float* __restrict__ Q,
                                       const float* __restrict__ K,
                                       const float* __restrict__ V,
                                       float* __restrict__ O,
                                       int T, int causal) {
    const int qi = blockIdx.x;                 // which query row
    const int tid = threadIdx.x;               // 0..TILE-1
    const int per_thread = D / TILE;

    __shared__ float sQ[D];
    __shared__ float sK[TILE][D];
    __shared__ float sV[TILE][D];

    // Cooperative, coalesced load of this block's query vector.
    for (int i = tid; i < D; i += TILE) sQ[i] = Q[size_t(qi) * D + i];
    __syncwarp();

    float acc[D / TILE];
    for (int i = 0; i < per_thread; ++i) acc[i] = 0.f;
    float m = -INFINITY, l = 0.f;

    const float scale = rsqrtf(float(D));
    const int n_keys = causal ? (qi + 1) : T;

    for (int kt = 0; kt < n_keys; kt += TILE) {
        const int tile_len = min(TILE, n_keys - kt);

        // Load the K and V tiles. Index linearly over (row, col) so consecutive
        // threads touch consecutive addresses: TILE*D elements, TILE threads.
        for (int idx = tid; idx < TILE * D; idx += TILE) {
            int r = idx / D, c = idx % D;
            bool live = r < tile_len;
            sK[r][c] = live ? K[size_t(kt + r) * D + c] : 0.f;
            sV[r][c] = live ? V[size_t(kt + r) * D + c] : 0.f;
        }
        __syncwarp();

        // Each thread scores one key of the tile against the query.
        float s = -INFINITY;
        if (tid < tile_len) {
            float dot = 0.f;
            for (int c = 0; c < D; ++c) dot += sQ[c] * sK[tid][c];
            s = dot * scale;
        }

        // Tile maximum, by warp shuffle. __shfl_xor_sync gives every lane the result,
        // which is what we want here (no broadcast step needed).
        float tile_max = s;
        for (int off = 16; off > 0; off >>= 1)
            tile_max = fmaxf(tile_max, __shfl_xor_sync(FULL_MASK, tile_max, off));

        const float m_new = fmaxf(m, tile_max);
        // Guard the very first tile, where m is -inf: exp(-inf - finite) = 0 is fine,
        // but if the whole tile were masked out we would hit -inf minus -inf.
        const float corr = isfinite(m) ? __expf(m - m_new) : 0.f;
        const float p = (tid < tile_len) ? __expf(s - m_new) : 0.f;

        float tile_sum = p;
        for (int off = 16; off > 0; off >>= 1)
            tile_sum += __shfl_xor_sync(FULL_MASK, tile_sum, off);

        l = corr * l + tile_sum;

        // Rescale the accumulator, then add this tile's contribution. Each thread
        // owns `per_thread` output components and sums over all keys in the tile, so
        // it needs every lane's p -- fetch them with shuffles rather than shared
        // memory.
        for (int i = 0; i < per_thread; ++i) acc[i] *= corr;
        for (int j = 0; j < TILE; ++j) {
            const float pj = __shfl_sync(FULL_MASK, p, j);
            if (pj == 0.f) continue;
            for (int i = 0; i < per_thread; ++i) acc[i] += pj * sV[j][tid * per_thread + i];
        }
        m = m_new;
        __syncwarp();
    }

    const float inv_l = (l > 0.f) ? 1.f / l : 0.f;
    for (int i = 0; i < per_thread; ++i)
        O[size_t(qi) * D + tid * per_thread + i] = acc[i] * inv_l;
}

// ---------------------------------------------------------------- reference
void attention_cpu(const float* Q, const float* K, const float* V, float* O,
                   int T, int causal) {
    std::vector<double> s(T);
    const double scale = 1.0 / std::sqrt(double(D));
    for (int i = 0; i < T; ++i) {
        int n = causal ? i + 1 : T;
        double m = -INFINITY;
        for (int j = 0; j < n; ++j) {
            double dot = 0.0;
            for (int c = 0; c < D; ++c) dot += double(Q[size_t(i) * D + c]) * double(K[size_t(j) * D + c]);
            s[j] = dot * scale;
            m = std::max(m, s[j]);
        }
        double l = 0.0;
        for (int j = 0; j < n; ++j) { s[j] = std::exp(s[j] - m); l += s[j]; }
        for (int c = 0; c < D; ++c) {
            double acc = 0.0;
            for (int j = 0; j < n; ++j) acc += s[j] * double(V[size_t(j) * D + c]);
            O[size_t(i) * D + c] = float(acc / l);
        }
    }
}

int main(int argc, char** argv) {
    print_device_info();

    const int T = (argc > 1) ? std::atoi(argv[1]) : 1024;
    const size_t count = size_t(T) * D, bytes = count * sizeof(float);

    std::vector<float> hQ(count), hK(count), hV(count), hO(count), hRef(count);
    // The casts to int are not decoration. i is size_t, so without them the
    // subtraction is unsigned: every value below the offset wraps to about 1.8e19,
    // the softmax saturates to one-hot, and the check below passes on garbage.
    for (size_t i = 0; i < count; ++i) {
        hQ[i] = float(int(i * 37 % 211) - 105) / 105.f;
        hK[i] = float(int(i * 53 % 197) - 98) / 98.f;
        hV[i] = float(int(i * 29 % 173) - 86) / 86.f;
    }

    float *dQ, *dK, *dV, *dO;
    for (float** p : {&dQ, &dK, &dV, &dO}) CUDA_CHECK(cudaMalloc(p, bytes));
    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), bytes, cudaMemcpyHostToDevice));

    // FLOPs: 2*T*T*D for QK^T plus 2*T*T*D for PV. Causal halves it.
    std::printf("T = %d, D = %d\n\n", T, D);
    std::printf("%-22s %11s %12s %10s\n", "variant", "time (ms)", "GFLOP/s", "check");

    bool ok = true;
    for (int causal : {0, 1}) {
        CUDA_CHECK(cudaMemset(dO, 0, bytes));
        float ms = time_kernel_ms([&] {
            flash_attention_kernel<<<T, TILE>>>(dQ, dK, dV, dO, T, causal);
        });
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaMemcpy(hO.data(), dO, bytes, cudaMemcpyDeviceToHost));
        attention_cpu(hQ.data(), hK.data(), hV.data(), hRef.data(), T, causal);

        double flops = 4.0 * T * T * D / (causal ? 2.0 : 1.0);
        std::printf("%-22s %11.3f %12.1f\n", causal ? "causal" : "full", ms,
                    flops / (ms * 1e-3) / 1e9);
        ok &= allclose(hO.data(), hRef.data(), count, 1e-3f, 1e-4f, causal ? "causal" : "full");
    }

    std::printf("\nThe peak memory footprint of the score matrix here is TILE floats per\n");
    std::printf("block -- %d bytes -- and it does NOT grow with T. That is the claim from\n", TILE * 4);
    std::printf("module 07, now true of an actual kernel.\n");

    std::printf("\nWhy this loses to PyTorch's fused kernel, in order of how much it costs:\n");
    std::printf("  1. One warp per block: occupancy is ~1/8 of what it could be, and the\n");
    std::printf("     __shfl-based V accumulation serialises over 32 lanes.\n");
    std::printf("  2. No tensor cores. The QK^T and PV products are scalar FMAs; wmma would\n");
    std::printf("     do a 16x16x16 block per instruction.\n");
    std::printf("  3. No double buffering: the K/V tile load and the compute do not overlap.\n");
    std::printf("  4. Scalar loads. float4 would move 16 bytes per instruction.\n");
    std::printf("  5. One query row per block wastes the reuse of the K/V tiles across the\n");
    std::printf("     BLOCK_M query rows that a real implementation processes together.\n");

    for (float* p : {dQ, dK, dV, dO}) CUDA_CHECK(cudaFree(p));
    std::printf("\n%s\n", ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    return ok ? 0 : 1;
}
