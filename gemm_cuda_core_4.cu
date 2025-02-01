#include <cuda_runtime.h>

#include "utils.h"

constexpr uint64_t TILE_SIZE = 16;


template <class T>
__device__ __inline__ void LDVectorized(half *dst, const half *src) {
    *(reinterpret_cast<T*>(dst)) = *(reinterpret_cast<const T*>(src));
}

template <int BBits, int MBase, int SShift = BBits>
struct Swizzle
{
    __device__ __inline__ static uint64_t apply(uint64_t x, uint64_t y, uint64_t rows, uint64_t cols) {
        uint64_t offset = x + y * cols;
        uint64_t one = 1;
        uint64_t bit_msk = (one << BBits) - one;
        uint64_t yyy_msk = bit_msk << (MBase + SShift);
        uint64_t msk_sft = SShift;
        return offset ^ ((offset & yyy_msk) >> msk_sft);;
    }
};

__global__ void Kernel(const half *A, const half *B, half *C, uint64_t m, uint64_t n, uint64_t k) {
    __shared__ half s_A[TILE_SIZE][TILE_SIZE];
    __shared__ half s_B[TILE_SIZE][TILE_SIZE];

    uint64_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;

    half sum = 0.0f;
    for (uint64_t k_tile = 0; k_tile < k; k_tile += TILE_SIZE) {
        if (threadIdx.x < TILE_SIZE / 8) {
            uint64_t new_offset = Swizzle<2, 3, 3>::apply(threadIdx.x * 8, threadIdx.y, TILE_SIZE, TILE_SIZE);
            LDVectorized<float4>(&(s_A[new_offset / TILE_SIZE][new_offset % TILE_SIZE]), &(A[y * k + k_tile + threadIdx.x * 8]));
        }
        if (threadIdx.x < TILE_SIZE / 8) {
            uint64_t new_offset = Swizzle<2, 3, 3>::apply(threadIdx.x * 8, threadIdx.y, TILE_SIZE, TILE_SIZE);
            LDVectorized<float4>(&(s_B[new_offset / TILE_SIZE][new_offset % TILE_SIZE]), &(B[(blockIdx.x * blockDim.x + threadIdx.y) * k + k_tile + threadIdx.x * 8]));
        }
        __syncthreads();

        if (y < m && x < n) {
#pragma unroll
            for (uint64_t t = 0; t < TILE_SIZE; t += 8) {
                half f_A[8];
                half f_B[8];
                uint64_t new_offset_sa = Swizzle<2, 3, 3>::apply(t, threadIdx.y, TILE_SIZE, TILE_SIZE);
                uint64_t new_offset_sb = Swizzle<2, 3, 3>::apply(t, threadIdx.x, TILE_SIZE, TILE_SIZE);
                LDVectorized<float4>(f_A, &(s_A[new_offset_sa / TILE_SIZE][new_offset_sa % TILE_SIZE])); 
                LDVectorized<float4>(f_B, &(s_B[new_offset_sb / TILE_SIZE][new_offset_sb % TILE_SIZE])); 
#pragma unroll
                for (uint64_t i = 0; i < 8; ++i) {
                    sum += (f_A[i] * f_B[i]);
                }
            }
        }
        __syncthreads();
    }

    if (y < m && x < n) {
        C[y * n + x] = (sum);
    }
}

class GemmCudaCore_4 : public GemmBase {
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
    // 解决bank conflict
    GemmCudaCore_4 gemm("GemmCudaCore_4");
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
