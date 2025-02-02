#include <cuda_runtime.h>
#include <mma.h>

#include "utils.h"

constexpr uint64_t TILE_SIZE = 16;

template <class T>
__device__ __inline__ void STVectorized(float *dst, const float *src) {
    *(reinterpret_cast<T*>(dst)) = *(reinterpret_cast<const T*>(src));
}

__global__ void Kernel(const half *A, const half *B, float *C, uint64_t m, uint64_t n, uint64_t k) {
    const size_t laneid = threadIdx.x % 32;
    
    __shared__ half s_A[TILE_SIZE][TILE_SIZE];
    __shared__ half s_B[TILE_SIZE][TILE_SIZE];

    // 要做2次16x8x16才能得到16x16的结果, 但2次用到的A矩阵的数据是一样的
    // 所以FragmentA是一维矩阵即可
    uint32_t FragmentA[4];
    uint32_t FragmentB[2][2];
    float    FragmentC[2][4];
    memset(FragmentC, 0, sizeof(FragmentC));


    uint64_t tiled_m = blockIdx.y * TILE_SIZE;
    uint64_t tiled_n = blockIdx.x * TILE_SIZE;

    // ldmatrix指令每个线程加载8个元素, 每个matrix加载8x8个元素, 一条ldmatrix最多加载4个8x8的matrix
    uint64_t mn_in_tile = laneid / (TILE_SIZE / 8);
    uint64_t k_in_tile = (laneid % (TILE_SIZE / 8)) * 8;
    uint64_t m_idx = tiled_m + mn_in_tile;
    uint64_t n_idx = tiled_n + mn_in_tile;

    for (uint64_t k_tile = 0; k_tile < k; k_tile += TILE_SIZE) {
        uint64_t k_idx = k_tile + k_in_tile;

        auto a_src_addr = &A[(m_idx) * k + k_idx];
        uint32_t sa_dst_addr = __cvta_generic_to_shared(&s_A[mn_in_tile][k_in_tile]);
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(sa_dst_addr), "l"(a_src_addr), "n"(16));

        auto b_src_addr = &B[(n_idx) * k + k_idx];
        uint32_t sb_dst_addr = __cvta_generic_to_shared(&s_B[mn_in_tile][k_in_tile]);
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(sb_dst_addr), "l"(b_src_addr), "n"(16));

        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group %0;\n" ::"n"(0));

        __syncthreads();

        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(FragmentA[0]), "=r"(FragmentA[1]), "=r"(FragmentA[2]), "=r"(FragmentA[3])
                 : "l"(__cvta_generic_to_shared(&s_A[laneid % 16][laneid / 16 * 8])));

        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(FragmentB[0][0]), "=r"(FragmentB[1][0]), "=r"(FragmentB[0][1]), "=r"(FragmentB[1][1])
                 : "l"(__cvta_generic_to_shared(&s_B[laneid % 16][laneid / 16 * 8])));
        
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,  %1,  %2,  %3},{%4,  %5,  %6,  %7}, {%8,  %9}, {%10, %11, %12, %13};\n"
                    : "=f"(FragmentC[i][0]), "=f"(FragmentC[i][1]), "=f"(FragmentC[i][2]), "=f"(FragmentC[i][3])
                    : "r"(FragmentA[0]), "r"(FragmentA[1]), "r"(FragmentA[2]), "r"(FragmentA[3]),
                            "r"(FragmentB[i][0]), "r"(FragmentB[i][1]),
                            "f"(FragmentC[i][0]), "f"(FragmentC[i][1]), "f"(FragmentC[i][2]), "f"(FragmentC[i][3]));
        }
        __syncthreads();
    }

    // 根据mma.sync中C的排布, 获取8x8 C matirx中每个线程持有的元素
    uint64_t m_in_tile = laneid / 4;
    uint64_t n_in_tile = (laneid % 4) * 2;

    STVectorized<float2>(&C[(tiled_m + m_in_tile) * n + tiled_n + n_in_tile], &FragmentC[0][0]);
    STVectorized<float2>(&C[(tiled_m + m_in_tile + 8) * n + tiled_n + n_in_tile], &FragmentC[0][2]);
    STVectorized<float2>(&C[(tiled_m + m_in_tile) * n + tiled_n + n_in_tile + 8], &FragmentC[1][0]);
    STVectorized<float2>(&C[(tiled_m + m_in_tile + 8) * n + tiled_n + n_in_tile + 8], &FragmentC[1][2]);    
}



class GemmTensorCore_2 : public GemmBase {
 public:
    using GemmBase::GemmBase; // 继承基类的构造函数

    void LaunchKernel(
        const half *d_A,
        const half *d_B,
        float *d_C, 
        uint64_t m, uint64_t n, uint64_t k) override
{
        dim3 blockdim(32);
        dim3 griddim((n + TILE_SIZE - 1) / TILE_SIZE,
                        (m + TILE_SIZE - 1) / TILE_SIZE);
        Kernel<<<griddim, blockdim>>>(d_A, d_B, d_C, m, n, k);
    }
};

int main() {
    GemmTensorCore_2 gemm("GemmTensorCore_2");
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
