#include <cuda_runtime.h>
#include <mma.h>

#include "utils.h"

constexpr uint64_t TILE_SIZE = 16;

template <class T>
__device__ __inline__ void STVectorized(float *dst, const float *src) {
  *(reinterpret_cast<T *>(dst)) = *(reinterpret_cast<const T *>(src));
}

template <int BBits, int MBase, int SShift = BBits>
struct Swizzle {
  __device__ __inline__ static uint64_t apply(uint64_t x, uint64_t y,
                                              uint64_t rows, uint64_t cols) {
    uint64_t offset = x + y * cols;
    uint64_t one = 1;
    uint64_t bit_msk = (one << BBits) - one;
    uint64_t yyy_msk = bit_msk << (MBase + SShift);
    uint64_t msk_sft = SShift;
    return offset ^ ((offset & yyy_msk) >> msk_sft);
    ;
  }
};

__device__ __inline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __inline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

template <class SrcT>
__device__ __inline__ void cp_async_128B(uint32_t dst_addr, SrcT src_addr) {
  constexpr uint32_t nbyte = 16;
  asm volatile(
      "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
        ::"r"(dst_addr), "l"(src_addr), "n"(nbyte));
}

template <class SrcT>
__device__ __inline__ void ldmatrix_sync_x4_m8n8(uint32_t &r0, uint32_t &r1,
                                                 uint32_t &r2, uint32_t &r3,
                                                 SrcT addr) {
  asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "l"(addr));
}

template <class FragmentA, class FragmentB, class FragmentC>
__device__ __inline__ void mma16x16x16(FragmentA frag_a, FragmentB frag_b,
                                       FragmentC frag_c) {
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,  %1,  %2, %3}, "
          "{%4,  %5,  %6,  %7}, {%8,  %9}, {%10, %11, %12, %13};\n"
            : "=f"(frag_c[i][0]), "=f"(frag_c[i][1]), "=f"(frag_c[i][2]), "=f"(frag_c[i][3])
            : "r"(frag_a[0]), "r"(frag_a[1]), "r"(frag_a[2]), "r"(frag_a[3]),
              "r"(frag_b[i][0]), "r"(frag_b[i][1]),
              "f"(frag_c[i][0]), "f"(frag_c[i][1]), "f"(frag_c[i][2]), "f"(frag_c[i][3]));
  }
}

__global__ void Kernel(const half *A, const half *B, float *C, uint64_t m,
                       uint64_t n, uint64_t k) {
  const size_t laneid = threadIdx.x % 32;

  __shared__ half s_A[2][TILE_SIZE][TILE_SIZE];
  __shared__ half s_B[2][TILE_SIZE][TILE_SIZE];

  // 要做2次16x8x16才能得到16x16的结果, 但2次用到的A矩阵的数据是一样的
  // 所以FragmentA是一维矩阵即可
  uint32_t frag_a[4];
  uint32_t frag_b[2][2];
  float frag_c[2][4];
  memset(frag_c, 0, sizeof(frag_c));

  uint64_t tiled_m = blockIdx.y * TILE_SIZE;
  uint64_t tiled_n = blockIdx.x * TILE_SIZE;

  // ldmatrix指令每个线程加载8个元素, 每个matrix加载8x8个元素,
  // 一条ldmatrix最多加载4个8x8的matrix
  uint64_t mn_in_tile = laneid / (TILE_SIZE / 8);
  uint64_t k_in_tile = (laneid % (TILE_SIZE / 8)) * 8;
  uint64_t m_idx = tiled_m + mn_in_tile;
  uint64_t n_idx = tiled_n + mn_in_tile;

  uint64_t st_seizzled_offset =
      Swizzle<2, 3, 3>::apply(k_in_tile, mn_in_tile, TILE_SIZE, TILE_SIZE);
  uint64_t ld_swizzled_offset = Swizzle<2, 3, 3>::apply(
      laneid / 16 * 8, laneid % 16, TILE_SIZE, TILE_SIZE);
  uint64_t st_seizzled_y = st_seizzled_offset / TILE_SIZE;
  uint64_t ld_seizzled_y = ld_swizzled_offset / TILE_SIZE;
  uint64_t st_seizzled_x = st_seizzled_offset % TILE_SIZE;
  uint64_t ld_seizzled_x = ld_swizzled_offset % TILE_SIZE;

  auto a_src_addr = &A[(m_idx)*k + k_in_tile];
  uint32_t sa_dst_addr = __cvta_generic_to_shared(&s_A[0][st_seizzled_y][st_seizzled_x]);
  cp_async_128B(sa_dst_addr, a_src_addr);

  auto b_src_addr = &B[(n_idx)*k + k_in_tile];
  uint32_t sb_dst_addr = __cvta_generic_to_shared(&s_B[0][st_seizzled_y][st_seizzled_x]);
  cp_async_128B(sb_dst_addr, b_src_addr);

  cp_async_commit_group();

  for (uint64_t k_tile = 0; k_tile < k; k_tile += TILE_SIZE) {
    uint64_t k_idx = k_tile + k_in_tile + TILE_SIZE;
    if (k_idx < k) {
      auto a_src_addr = &A[(m_idx)*k + k_idx];
      uint32_t sa_dst_addr = __cvta_generic_to_shared(
          &s_A[((k_tile >> 4) + 1) & 1][st_seizzled_y][st_seizzled_x]);
      cp_async_128B(sa_dst_addr, a_src_addr);

      auto b_src_addr = &B[(n_idx)*k + k_idx];
      uint32_t sb_dst_addr = __cvta_generic_to_shared(
          &s_B[((k_tile >> 4) + 1) & 1][st_seizzled_y][st_seizzled_x]);
      cp_async_128B(sb_dst_addr, b_src_addr);
    }

    cp_async_commit_group();
    cp_async_wait_group<1>();

    __syncthreads();

    uint64_t ldmatrix_sa_addr = __cvta_generic_to_shared(
        &s_A[(k_tile >> 4) & 1][ld_seizzled_y][ld_seizzled_x]);
    ldmatrix_sync_x4_m8n8(frag_a[0], frag_a[1], frag_a[2], frag_a[3],
                          ldmatrix_sa_addr);

    uint64_t ldmatrix_sb_addr = __cvta_generic_to_shared(
        &s_B[(k_tile >> 4) & 1][ld_seizzled_y][ld_seizzled_x]);
    ldmatrix_sync_x4_m8n8(frag_b[0][0], frag_b[1][0], frag_b[0][1],
                          frag_b[1][1], ldmatrix_sb_addr);

    mma16x16x16(frag_a, frag_b, frag_c);

    __syncthreads();
  }

  // 根据mma.sync中C的排布, 获取8x8 C matirx中每个线程持有的元素
  uint64_t m_in_tile = laneid / 4;
  uint64_t n_in_tile = (laneid % 4) * 2;

  STVectorized<float2>(&C[(tiled_m + m_in_tile) * n + tiled_n + n_in_tile],&frag_c[0][0]);
  STVectorized<float2>(&C[(tiled_m + m_in_tile + 8) * n + tiled_n + n_in_tile],&frag_c[0][2]);
  STVectorized<float2>(&C[(tiled_m + m_in_tile) * n + tiled_n + n_in_tile + 8],&frag_c[1][0]);
  STVectorized<float2>(&C[(tiled_m + m_in_tile + 8) * n + tiled_n + n_in_tile + 8],&frag_c[1][2]);
}

class GemmTensorCore_3 : public GemmBase {
 public:
  using GemmBase::GemmBase;  // 继承基类的构造函数

  void LaunchKernel(const half *d_A, const half *d_B, float *d_C, uint64_t m,
                    uint64_t n, uint64_t k) override {
    dim3 blockdim(32);
    dim3 griddim((n + TILE_SIZE - 1) / TILE_SIZE,
                 (m + TILE_SIZE - 1) / TILE_SIZE);
    Kernel<<<griddim, blockdim>>>(d_A, d_B, d_C, m, n, k);
  }
};

int main() {
  GemmTensorCore_3 gemm("GemmTensorCore_3");
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
