// Shared helpers for every kernel in this module: error checking, timing, and a
// pass/fail comparison against a CPU reference.
//
// The error-checking macro is not optional decoration. CUDA API calls and kernel
// launches fail asynchronously and silently; without a check you get wrong numbers
// rather than an error, which is the same failure mode as every other bug in this
// course.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d -> %s\n",              \
                         cudaGetErrorName(_err), __FILE__, __LINE__,            \
                         cudaGetErrorString(_err));                             \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// A kernel launch returns void. Errors surface either at launch (bad config) or
// asynchronously during execution, so check both.
#define CUDA_CHECK_KERNEL()                                                     \
    do {                                                                        \
        CUDA_CHECK(cudaGetLastError());                                         \
        CUDA_CHECK(cudaDeviceSynchronize());                                    \
    } while (0)

// Wall-clock timing done properly: warm up, then average over `iters` with CUDA
// events, which measure on the device timeline rather than the host's.
template <typename F>
float time_kernel_ms(F&& fn, int warmup = 3, int iters = 20) {
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;
}

// Relative comparison. Note the tolerance: fp32 accumulation order differs between
// your kernel and the reference, so bitwise equality is NOT the expectation. A
// relative error around 1e-5 for a few-thousand-element dot product is correct
// behaviour, not a bug -- see module 00 on floating-point associativity.
inline bool allclose(const float* a, const float* b, size_t n,
                     float rtol = 1e-4f, float atol = 1e-4f, const char* label = "") {
    double worst = 0.0;
    size_t worst_i = 0;
    for (size_t i = 0; i < n; ++i) {
        double diff = std::fabs(double(a[i]) - double(b[i]));
        double tol = atol + rtol * std::fabs(double(b[i]));
        if (diff / tol > worst) { worst = diff / tol; worst_i = i; }
    }
    bool ok = worst <= 1.0;
    std::printf("  [%s] %s  (worst element %zu: %.6g vs %.6g)\n",
                ok ? "PASS" : "FAIL", label, worst_i, a[worst_i], b[worst_i]);
    return ok;
}

inline void print_device_info() {
    int dev = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    // CUDA 13 removed memoryClockRate and memoryBusWidth from cudaDeviceProp. The
    // attribute query below is the supported route and works on CUDA 11 onwards.
    int clock_khz = 0, bus_bits = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev));
    // Peak memory bandwidth = bus width (bits) * memory clock (kHz) * 2 (DDR) / 8 bits.
    double bw = 2.0 * clock_khz * (bus_bits / 8.0) / 1.0e6;
    std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    std::printf("  SMs %d | shared/block %zu KB | peak HBM bandwidth ~%.0f GB/s\n\n",
                prop.multiProcessorCount, prop.sharedMemPerBlock / 1024, bw);
}
