#include <cuda_runtime.h>

#include "utils.h"

constexpr uint64_t TILE_SIZE = 16;

__global__ void Kernel(const half *A, const half *B, half *C, uint64_t m, uint64_t n, uint64_t k) {
    __shared__ half s_A[TILE_SIZE][TILE_SIZE];
    __shared__ half s_B[TILE_SIZE][TILE_SIZE];

    uint64_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;

    half sum = 0.0f;
    for (uint64_t k_tile = 0; k_tile < k; k_tile += TILE_SIZE) {
        s_A[threadIdx.y][threadIdx.x] = A[y * k + k_tile + threadIdx.x];
        s_B[threadIdx.x][threadIdx.y] = B[x * k + k_tile + threadIdx.y];
        __syncthreads();

        if (y < m && x < n) {
            for (uint64_t t = 0; t < TILE_SIZE; ++t) {
                sum += (s_A[threadIdx.y][t] * s_B[threadIdx.x][t]);
            }
        }
        __syncthreads();
    }

    if (y < m && x < n) {
        C[y * n + x] = (sum);
    }
}

class GemmCudaCore_2 : public GemmBase {
 public:
    using GemmBase::GemmBase; // 继承基类的构造函数

    void LaunchKernel(
        const half *d_A,
        const half *d_B,
        half *d_C, 
        uint64_t m, uint64_t n, uint64_t k) override
{
        dim3 blockdim(TILE_SIZE, TILE_SIZE);
        dim3 griddim((n + blockdim.x - 1) / blockdim.x,
                        (m + blockdim.y - 1) / blockdim.y);
        Kernel<<<griddim, blockdim>>>(d_A, d_B, d_C, m, n, k);
    }
};



int main() {
    // 使用共享内存
    GemmCudaCore_2 gemm("GemmCudaCore_2");
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
