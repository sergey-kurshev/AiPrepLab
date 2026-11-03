// 01 — the execution model, and your first bandwidth measurement.
//
// Vector addition does 1 FLOP per 12 bytes moved (two reads, one write). Its
// arithmetic intensity is 0.083, which is far below any GPU's ridge point, so it is
// purely a memory-bandwidth benchmark. A correct implementation should reach 70-90%
// of the device's peak bandwidth; if yours does not, the reason is in how the
// threads are indexed, not in the arithmetic.
//
// Build:  nvcc -O3 -o 01_vector_add 01_vector_add.cu
// Run:    ./01_vector_add

#include "common.cuh"
#include <vector>

// ---------------------------------------------------------------- kernels

// The canonical form. `blockIdx.x * blockDim.x + threadIdx.x` gives each thread a
// unique global index; the guard is needed because n is rarely a multiple of the
// block size, and writing past the end is a silent corruption rather than a crash.
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// A grid-stride loop: the grid is sized to the device rather than to the data, and
// each thread handles several elements. This decouples launch configuration from
// problem size, works for any n, and is the pattern to default to.
__global__ void vector_add_grid_stride(const float* a, const float* b, float* c, int n) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        c[i] = a[i] + b[i];
}

// DELIBERATELY BAD, and instructive. Exactly the same elements are read and written,
// exactly once each -- only the assignment of elements to threads changes. Within each
// group of 1024 elements the (warp, lane) mapping is transposed, so consecutive lanes
// of a warp are 32 floats (128 bytes) apart instead of adjacent. One coalesced
// 128-byte transaction per warp becomes 32 separate ones.
//
// Same work, same total bytes, roughly an order of magnitude less bandwidth. This is
// the number-one performance bug in hand-written kernels.
__global__ void vector_add_uncoalesced(const float* a, const float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp = tid / 32, lane = tid % 32;
    int group = warp / 32, wid = warp % 32;
    int i = group * 1024 + lane * 32 + wid;
    if (i < n) c[i] = a[i] + b[i];
}

// ---------------------------------------------------------------- driver
int main() {
    print_device_info();

    const int n = 1 << 24;                    // 16.7M elements = 64 MB per array
    const size_t bytes = size_t(n) * sizeof(float);

    std::vector<float> h_a(n), h_b(n), h_c(n), h_ref(n);
    for (int i = 0; i < n; ++i) {
        h_a[i] = float(i % 1000) * 0.001f;
        h_b[i] = float((i * 7) % 1000) * 0.002f;
        h_ref[i] = h_a[i] + h_b[i];
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    std::printf("n = %d elements, %.0f MB moved per call\n\n", n, 3.0 * bytes / 1e6);
    std::printf("%-22s %10s %14s\n", "kernel", "time (ms)", "GB/s");

    struct Entry { const char* name; void (*fn)(const float*, const float*, float*, int); int blocks; };
    Entry entries[] = {
        {"vector_add",          vector_add,             blocks},
        {"grid-stride",         vector_add_grid_stride, 1024},
        {"uncoalesced",         vector_add_uncoalesced, blocks},
    };

    bool all_ok = true;
    for (const auto& e : entries) {
        CUDA_CHECK(cudaMemset(d_c, 0, bytes));
        float ms = time_kernel_ms([&] { e.fn<<<e.blocks, threads>>>(d_a, d_b, d_c, n); });
        CUDA_CHECK_KERNEL();
        double gbps = 3.0 * bytes / (ms * 1e-3) / 1e9;
        std::printf("%-22s %10.3f %14.1f\n", e.name, ms, gbps);

        CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
        // All three compute the same result -- which is the point of the comparison.
        all_ok &= allclose(h_c.data(), h_ref.data(), n, 1e-5f, 1e-6f, e.name);
    }

    std::printf("\nQuestions to answer before moving on:\n");
    std::printf("  1. What fraction of peak bandwidth did the coalesced version reach?\n");
    std::printf("  2. Why is the uncoalesced version slower when it does identical work?\n");
    std::printf("  3. Sweep `threads` over 32, 64, 128, 256, 512, 1024. Where is the knee, and why?\n");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    std::printf("\n%s\n", all_ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    return all_ok ? 0 : 1;
}
