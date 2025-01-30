#include <cuda_runtime.h>

#include "utils.h"

__global__ void Kernel(float *A, float *B, float *C, int m, int n, int k) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < m && x < n) {
        float sum = 0.0f;
        for (int t = 0; t < k; ++t) {
            sum += A[y * k + t] * B[x * k + t];
        }
        C[y * n + x] = sum;
    }
}

class GemmCudaCore_1 : public GemmBase {
 public:
    using GemmBase::GemmBase; // 继承基类的构造函数

    void LaunchKernel(
        float *d_A,
        float *d_B,
        float *d_C, 
        int m, int n, int k) override
{
        dim3 blockdim(16, 16);
        dim3 griddim((n + blockdim.x - 1) / blockdim.x,
                        (m + blockdim.y - 1) / blockdim.y);
        Kernel<<<griddim, blockdim>>>(d_A, d_B, d_C, m, n, k);
    }
};



int main() {
    GemmCudaCore_1 gemm("GemmCudaCore_1");
    gemm.RunProfile(128, 128, 128);
    gemm.RunProfile(128, 256, 256);
    gemm.RunProfile(256, 512, 512);
    gemm.RunProfile(512, 256, 512);
    gemm.RunProfile(512, 512, 512);
    gemm.RunProfile(512, 512, 1024);

    return 0;
}
