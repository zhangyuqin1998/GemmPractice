#include <cuda_runtime.h>

#include "utils.h"

__global__ void Kernel(const float *A, const float *B, float *C, uint64_t m, uint64_t n, uint64_t k) {
    uint64_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < m && x < n) {
        float sum = 0.0f;
        for (uint64_t t = 0; t < k; ++t) {
            sum += A[y * k + t] * B[x * k + t];
        }
        C[y * n + x] = sum;
    }
}

class GemmCudaCore_1 : public GemmBase {
 public:
    using GemmBase::GemmBase; // 继承基类的构造函数

    void LaunchKernel(
        const float *d_A,
        const float *d_B,
        float *d_C, 
        uint64_t m, uint64_t n, uint64_t k) override
{
        dim3 blockdim(16, 16);
        dim3 griddim((n + blockdim.x - 1) / blockdim.x,
                        (m + blockdim.y - 1) / blockdim.y);
        Kernel<<<griddim, blockdim>>>(d_A, d_B, d_C, m, n, k);
    }
};



int main() {
    GemmCudaCore_1 gemm("GemmCudaCore_1");
    // mxnxk
    gemm.RunProfile(128, 128, 128);
    gemm.RunProfile(128, 256, 256);
    gemm.RunProfile(256, 512, 512);
    gemm.RunProfile(512, 256, 512);
    gemm.RunProfile(512, 512, 512);
    gemm.RunProfile(512, 512, 1024);
    gemm.RunProfile(512, 512, 2048);
    gemm.RunProfile(1024, 1024, 2048);
    gemm.RunProfile(1024, 256, 2048);
    gemm.RunProfile(256, 1024, 2048);
    gemm.RunProfile(2048, 2048, 2048);
    gemm.RunProfile(4096, 4096, 4096);

    return 0;
}
