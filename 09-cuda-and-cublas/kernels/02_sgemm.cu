// 02 — the matmul ladder, measured against cuBLAS.
//
// Four kernels, each one step up the memory hierarchy, each computing exactly the
// same C = A*B for row-major square matrices. What changes is only where the data
// lives when it is used, and how many times it is re-read.
//
//   level 0  naive                one thread per output, uncoalesced
//   level 1  coalesced            same algorithm, threadIdx.x mapped to the column
//   level 2  shared-memory tiled  each A element read K/BK times instead of K
//   level 3  register tiled       each shared value used TM (or TN) times
//
// Report percentages of cuBLAS, not milliseconds. "My kernel takes 4 ms" says
// nothing; "my kernel reaches 38% of cuBLAS" is an engineering statement.
//
// Build:  nvcc -O3 -o 02_sgemm 02_sgemm.cu -lcublas
// Run:    ./02_sgemm [n]

#include "common.cuh"
#include <cublas_v2.h>
#include <vector>

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t _s = (call);                                             \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                      \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", int(_s),         \
                         __FILE__, __LINE__);                                   \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// ================================================================ level 0: naive
// threadIdx.x selects the ROW. Consecutive threads in a warp therefore read
// consecutive rows of A (stride K) and the SAME element of B -- the A reads are
// maximally uncoalesced. Identical arithmetic to level 1; several times slower.
__global__ void sgemm_naive(int M, int N, int K, const float* A, const float* B, float* C) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

// ================================================================ level 1: coalesced
// One character of difference in effect: threadIdx.x now selects the COLUMN, so the
// 32 lanes of a warp read 32 consecutive elements of B (one transaction) and
// broadcast the same element of A (also one transaction).
__global__ void sgemm_coalesced(int M, int N, int K, const float* A, const float* B, float* C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

// ================================================================ level 2: shared tiles
// Stage a TS x TS tile of A and of B in shared memory, then compute from there.
// Each element of A now crosses the HBM boundary K/TS times instead of K times:
// arithmetic intensity multiplied by TS.
//
// The two __syncthreads() are both mandatory and both easy to get wrong. The first
// makes the loads visible to every thread; the second stops a fast thread from
// overwriting the tile while a slow one is still reading it. Delete either and the
// result is wrong intermittently, which is the worst kind of wrong.
template <int TS>
__global__ void sgemm_tiled(int M, int N, int K, const float* A, const float* B, float* C) {
    __shared__ float As[TS][TS];
    __shared__ float Bs[TS][TS + 1];   // +1 padding: avoids 32-way bank conflicts

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TS + ty;
    int col = blockIdx.x * TS + tx;

    float acc = 0.f;
    for (int t = 0; t < (K + TS - 1) / TS; ++t) {
        int a_col = t * TS + tx;
        int b_row = t * TS + ty;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.f;
        __syncthreads();

        for (int k = 0; k < TS; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// ================================================================ level 3: register tiles
// The same idea, one level higher: each thread computes a TM x TN patch of C held in
// registers, so every value fetched from shared memory is used TM (or TN) times
// instead of once.
//
// Note the deliberate trade: this kernel uses many registers per thread and therefore
// runs at LOWER occupancy than level 2 -- and it is much faster, because the
// instruction-level parallelism inside each thread hides latency better than extra
// resident warps would. "Maximise occupancy" is a heuristic, not a rule.
//
// Requires M, N divisible by BM/BN and K divisible by BK. Production kernels handle
// the ragged edges; here we assert, because an unchecked assumption in a kernel is a
// silently wrong answer.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_register_tiled(int M, int N, int K,
                                     const float* A, const float* B, float* C) {
    const int cRow = blockIdx.y, cCol = blockIdx.x;
    const int numThreads = (BM / TM) * (BN / TN);

    const int threadRow = threadIdx.x / (BN / TN);
    const int threadCol = threadIdx.x % (BN / TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // Loading indices: each thread fetches several elements per tile, chosen so that
    // consecutive threads load consecutive addresses (coalesced) in both A and B.
    const int innerRowA = threadIdx.x / BK, innerColA = threadIdx.x % BK;
    const int strideA = numThreads / BK;
    const int innerRowB = threadIdx.x / BN, innerColB = threadIdx.x % BN;
    const int strideB = numThreads / BN;

    float acc[TM * TN] = {0.f};
    float regM[TM], regN[TN];

    for (int bk = 0; bk < K; bk += BK) {
        for (int off = 0; off < BM; off += strideA)
            As[(innerRowA + off) * BK + innerColA] = A[(innerRowA + off) * K + innerColA];
        for (int off = 0; off < BK; off += strideB)
            Bs[(innerRowB + off) * BN + innerColB] = B[(innerRowB + off) * N + innerColB];
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int dot = 0; dot < BK; ++dot) {
            for (int i = 0; i < TM; ++i) regM[i] = As[(threadRow * TM + i) * BK + dot];
            for (int j = 0; j < TN; ++j) regN[j] = Bs[dot * BN + threadCol * TN + j];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j) acc[i * TN + j] += regM[i] * regN[j];
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            C[(threadRow * TM + i) * N + threadCol * TN + j] = acc[i * TN + j];
}

// ================================================================ CPU reference
void sgemm_cpu(int M, int N, int K, const float* A, const float* B, float* C) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;     // double accumulation so the reference is the trustworthy one
            for (int k = 0; k < K; ++k) acc += double(A[i * K + k]) * double(B[k * N + j]);
            C[i * N + j] = float(acc);
        }
}

// ================================================================ driver
int main(int argc, char** argv) {
    print_device_info();

    const int n = (argc > 1) ? std::atoi(argv[1]) : 1024;
    if (n % 64 != 0) { std::printf("n must be a multiple of 64\n"); return 1; }
    const int M = n, N = n, K = n;
    const size_t bytes = size_t(n) * n * sizeof(float);
    const double gflop = 2.0 * n * n * n / 1e9;

    std::vector<float> h_A(n * n), h_B(n * n), h_C(n * n), h_ref(n * n);
    for (int i = 0; i < n * n; ++i) {
        h_A[i] = float((i * 37 % 211) - 105) / 105.f;
        h_B[i] = float((i * 53 % 197) - 98) / 98.f;
    }
    // The CPU reference is O(n^3) on one core; only compute it for modest sizes.
    const bool verify = (n <= 1024);
    if (verify) sgemm_cpu(M, N, K, h_A.data(), h_B.data(), h_ref.data());

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.f, beta = 0.f;

    std::printf("n = %d   %.2f GFLOP per multiply\n\n", n, gflop);
    std::printf("%-26s %12s %12s %12s\n", "kernel", "time (ms)", "GFLOP/s", "% of cuBLAS");

    bool ok = true;
    auto run = [&](const char* name, auto&& launch) -> double {
        CUDA_CHECK(cudaMemset(d_C, 0, bytes));
        float ms = time_kernel_ms(launch);
        CUDA_CHECK_KERNEL();
        if (verify) {
            CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));
            // fp32 accumulation over K terms: relative error ~ sqrt(K) * eps is normal.
            ok &= allclose(h_C.data(), h_ref.data(), size_t(n) * n, 1e-3f, 1e-3f, name);
        }
        return gflop / (ms * 1e-3);
    };

    // cuBLAS first, so everything else can be quoted as a fraction of it.
    //
    // THE COLUMN-MAJOR TRAP: cuBLAS is column-major, our data is row-major. A
    // row-major MxN matrix is bit-identical to a column-major NxM one, so we compute
    // C^T = B^T A^T by passing B and A in the opposite order with swapped dimensions.
    // Get this wrong and you get a transposed answer, or a plausible-looking matrix of
    // entirely wrong numbers.
    double cublas_gflops = run("cuBLAS SgemmEx", [&] {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha,
                                 d_B, N,
                                 d_A, K,
                                 &beta, d_C, N));
    });
    std::printf("%-26s %12.3f %12.1f %11.1f%%\n", "cuBLAS", gflop / cublas_gflops * 1e3,
                cublas_gflops, 100.0);

    struct Row { const char* name; double gflops; };
    std::vector<Row> rows;

    {
        dim3 block(32, 32), grid((M + 31) / 32, (N + 31) / 32);
        rows.push_back({"0 naive (uncoalesced)",
            run("0 naive", [&] { sgemm_naive<<<grid, block>>>(M, N, K, d_A, d_B, d_C); })});
    }
    {
        dim3 block(32, 32), grid((N + 31) / 32, (M + 31) / 32);
        rows.push_back({"1 coalesced",
            run("1 coalesced", [&] { sgemm_coalesced<<<grid, block>>>(M, N, K, d_A, d_B, d_C); })});
    }
    {
        constexpr int TS = 32;
        dim3 block(TS, TS), grid((N + TS - 1) / TS, (M + TS - 1) / TS);
        rows.push_back({"2 shared-memory tiled",
            run("2 tiled", [&] { sgemm_tiled<TS><<<grid, block>>>(M, N, K, d_A, d_B, d_C); })});
    }
    {
        constexpr int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4;
        dim3 block((BM / TM) * (BN / TN)), grid(N / BN, M / BM);
        rows.push_back({"3 register tiled 4x4",
            run("3 register tiled", [&] {
                sgemm_register_tiled<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, d_A, d_B, d_C);
            })});
    }

    std::printf("\n");
    for (const auto& r : rows)
        std::printf("%-26s %12.3f %12.1f %11.1f%%\n", r.name, gflop / r.gflops * 1e3,
                    r.gflops, 100.0 * r.gflops / cublas_gflops);

    std::printf("\nQuestions:\n");
    std::printf("  1. Levels 0 and 1 do IDENTICAL arithmetic. Explain the gap in one sentence.\n");
    std::printf("  2. Level 3 has lower occupancy than level 2. Why is it faster?\n");
    std::printf("  3. What is left between level 3 and cuBLAS? (vectorised loads, double\n");
    std::printf("     buffering, tensor cores - estimate how much each is worth.)\n");
    std::printf("  4. Re-run with n=2048 and n=4096. Do the ratios hold?\n");

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    if (verify) std::printf("\n%s\n", ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    else std::printf("\n(n > 1024: skipped CPU verification)\n");
    return (verify && !ok) ? 1 : 0;
}
