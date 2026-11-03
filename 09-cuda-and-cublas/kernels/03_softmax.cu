// 03 — reductions, warp shuffles, and the online softmax.
//
// Softmax over the rows of an (R x N) matrix. Three implementations:
//
//   naive_3pass   max, then sum, then normalise: three passes over the row
//   fused_2pass   logsumexp trick, still two reads of the row
//   online_1pass  the FlashAttention recurrence: one read, carrying (m, l)
//
// Softmax is purely memory-bound (about 1 FLOP per byte), so passes over the data
// are the cost. Going from three passes to one is close to a 3x speedup, which is
// exactly the observation module 07's paper is built on.
//
// Build:  nvcc -O3 -o 03_softmax 03_softmax.cu
// Run:    ./03_softmax

#include "common.cuh"
#include <vector>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffffu

// ---------------------------------------------------------------- warp primitives
// __shfl_down_sync exchanges a register directly between lanes of a warp: no shared
// memory, no __syncthreads(), five instructions to reduce 32 values to one. This is
// the single most useful primitive in reduction kernels.
__device__ __forceinline__ float warp_reduce_max(float v) {
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(FULL_MASK, v, off));
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        v += __shfl_down_sync(FULL_MASK, v, off);
    return v;
}

// Block reduction: reduce within each warp, write one value per warp to shared
// memory, then have the first warp reduce those. The result is broadcast back through
// shared memory so every thread has it.
template <bool IS_MAX>
__device__ float block_reduce(float v, float identity) {
    __shared__ float partial[32];
    __shared__ float result;
    const int lane = threadIdx.x % WARP_SIZE;
    const int wid = threadIdx.x / WARP_SIZE;
    const int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    v = IS_MAX ? warp_reduce_max(v) : warp_reduce_sum(v);
    if (lane == 0) partial[wid] = v;
    __syncthreads();

    if (wid == 0) {
        float x = (lane < n_warps) ? partial[lane] : identity;
        x = IS_MAX ? warp_reduce_max(x) : warp_reduce_sum(x);
        if (lane == 0) result = x;
    }
    __syncthreads();
    return result;
}

// ---------------------------------------------------------------- kernels
// One block per row throughout, so the reductions stay inside a block.

__global__ void softmax_naive_3pass(const float* in, float* out, int N) {
    const float* row = in + size_t(blockIdx.x) * N;
    float* orow = out + size_t(blockIdx.x) * N;

    float m = -INFINITY;
    for (int i = threadIdx.x; i < N; i += blockDim.x) m = fmaxf(m, row[i]);
    m = block_reduce<true>(m, -INFINITY);

    float l = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) l += __expf(row[i] - m);
    l = block_reduce<false>(l, 0.f);

    for (int i = threadIdx.x; i < N; i += blockDim.x) orow[i] = __expf(row[i] - m) / l;
}

// Same two reductions, but the exponentials in the second pass are reused via the
// log-sum-exp form, saving one exp per element in the final pass.
__global__ void softmax_fused_2pass(const float* in, float* out, int N) {
    const float* row = in + size_t(blockIdx.x) * N;
    float* orow = out + size_t(blockIdx.x) * N;

    float m = -INFINITY, l = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) m = fmaxf(m, row[i]);
    m = block_reduce<true>(m, -INFINITY);
    for (int i = threadIdx.x; i < N; i += blockDim.x) l += __expf(row[i] - m);
    l = block_reduce<false>(l, 0.f);

    const float logZ = m + __logf(l);
    for (int i = threadIdx.x; i < N; i += blockDim.x) orow[i] = __expf(row[i] - logZ);
}

// The online version. Each thread maintains its OWN (m, l) over the elements it
// visits, using the recurrence from module 07; then the per-thread pairs are combined
// with the same recurrence in the reduction. One pass over the input to build the
// statistics, one to write the output.
//
// Combining two (m, l) pairs:
//     m = max(m1, m2),  l = exp(m1 - m) * l1 + exp(m2 - m) * l2
__global__ void softmax_online_1pass(const float* in, float* out, int N) {
    const float* row = in + size_t(blockIdx.x) * N;
    float* orow = out + size_t(blockIdx.x) * N;

    float m = -INFINITY, l = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float x = row[i];
        float m_new = fmaxf(m, x);
        l = l * __expf(m - m_new) + __expf(x - m_new);
        m = m_new;
    }

    // Reduce the (m, l) pairs across the block, pair-wise, with the same recurrence.
    __shared__ float s_m[32], s_l[32];
    __shared__ float bm, bl;
    const int lane = threadIdx.x % WARP_SIZE, wid = threadIdx.x / WARP_SIZE;
    const int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        float m_o = __shfl_down_sync(FULL_MASK, m, off);
        float l_o = __shfl_down_sync(FULL_MASK, l, off);
        float m_new = fmaxf(m, m_o);
        l = l * __expf(m - m_new) + l_o * __expf(m_o - m_new);
        m = m_new;
    }
    if (lane == 0) { s_m[wid] = m; s_l[wid] = l; }
    __syncthreads();

    if (threadIdx.x == 0) {
        float mm = -INFINITY, ll = 0.f;
        for (int w = 0; w < n_warps; ++w) {
            float m_new = fmaxf(mm, s_m[w]);
            ll = ll * __expf(mm - m_new) + s_l[w] * __expf(s_m[w] - m_new);
            mm = m_new;
        }
        bm = mm;
        bl = ll;
    }
    __syncthreads();

    const float logZ = bm + __logf(bl);
    for (int i = threadIdx.x; i < N; i += blockDim.x) orow[i] = __expf(row[i] - logZ);
}

// ---------------------------------------------------------------- reference
void softmax_cpu(const float* in, float* out, int R, int N) {
    for (int r = 0; r < R; ++r) {
        const float* row = in + size_t(r) * N;
        float m = -INFINITY;
        for (int i = 0; i < N; ++i) m = std::max(m, row[i]);
        double l = 0.0;
        for (int i = 0; i < N; ++i) l += std::exp(double(row[i]) - m);
        for (int i = 0; i < N; ++i) out[size_t(r) * N + i] = float(std::exp(double(row[i]) - m) / l);
    }
}

int main() {
    print_device_info();

    const int R = 4096, N = 4096;
    const size_t count = size_t(R) * N, bytes = count * sizeof(float);
    std::vector<float> h_in(count), h_out(count), h_ref(count);
    for (size_t i = 0; i < count; ++i) h_in[i] = float((i * 37 % 1999)) / 200.f - 5.f;
    // Include a row of large values, to catch an implementation that skips the shift.
    for (int i = 0; i < N; ++i) h_in[i] = 1000.f + float(i % 17);
    softmax_cpu(h_in.data(), h_ref.data(), R, N);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    std::printf("%d x %d, %.0f MB per pass\n\n", R, N, bytes / 1e6);
    std::printf("%-22s %11s %12s %10s\n", "kernel", "time (ms)", "GB/s", "passes");

    struct K { const char* name; void (*fn)(const float*, float*, int); double passes; };
    K kernels[] = {
        {"naive 3-pass",  softmax_naive_3pass, 4.0},   // 3 reads + 1 write
        {"fused 2-pass",  softmax_fused_2pass, 3.0},   // 2 reads + 1 write
        {"online 1-pass", softmax_online_1pass, 2.0},  // 1 read + 1 write... plus one re-read
    };

    bool ok = true;
    for (const auto& k : kernels) {
        CUDA_CHECK(cudaMemset(d_out, 0, bytes));
        float ms = time_kernel_ms([&] { k.fn<<<R, threads>>>(d_in, d_out, N); });
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        std::printf("%-22s %11.3f %12.1f %10.0f\n", k.name, ms,
                    k.passes * bytes / (ms * 1e-3) / 1e9, k.passes);
        // __expf is the fast-math intrinsic: ~2 ulp, not 0.5. A 1e-5 relative tolerance
        // is the right expectation, and demanding more would mean giving up the speed.
        ok &= allclose(h_out.data(), h_ref.data(), count, 1e-4f, 1e-6f, k.name);
    }

    std::printf("\nQuestions:\n");
    std::printf("  1. The online kernel still reads the input twice. Why, and what would it\n");
    std::printf("     take to make it genuinely single-pass? (Hint: you would have to keep the\n");
    std::printf("     row in registers or shared memory - which is exactly what FlashAttention\n");
    std::printf("     does with its tiles.)\n");
    std::printf("  2. Replace __expf with expf and re-measure. What did the accuracy buy?\n");
    std::printf("  3. The first row contains values around 1000. Remove the max subtraction\n");
    std::printf("     and see what happens.\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    std::printf("\n%s\n", ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    return ok ? 0 : 1;
}
