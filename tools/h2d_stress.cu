// Background host-to-device copy load for E3g step 3: does PCIe upload traffic slow CPU experts reading the same RAM?
// Copies a pinned host buffer to the GPU in a loop and prints the achieved GB/s every few seconds.
//
// Build: nvcc -O2 -o h2d_stress tools/h2d_stress.cu
// Usage: h2d_stress [MiB per copy, default 64] [seconds, default 0 = forever]
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char ** argv) {
    const size_t mib  = argc > 1 ? strtoul(argv[1], nullptr, 10) : 64;
    const double secs = argc > 2 ? atof(argv[2]) : 0;
    const size_t n    = mib << 20;
    void * host = nullptr;
    void * dev  = nullptr;
    if (cudaMallocHost(&host, n) != cudaSuccess || cudaMalloc(&dev, n) != cudaSuccess) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }
    using clk = std::chrono::steady_clock;
    const auto start = clk::now();
    auto last = start;
    size_t bytes = 0;
    for (;;) {
        cudaMemcpy(dev, host, n, cudaMemcpyHostToDevice);
        bytes += n;
        const auto now = clk::now();
        const double dt = std::chrono::duration<double>(now - last).count();
        if (dt >= 5) {
            printf("%.2f GB/s\n", bytes / dt / 1e9);
            fflush(stdout);
            bytes = 0;
            last  = now;
        }
        if (secs > 0 && std::chrono::duration<double>(now - start).count() >= secs) {
            break;
        }
    }
    return 0;
}
