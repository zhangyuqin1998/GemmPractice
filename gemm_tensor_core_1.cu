#include <cuda_runtime.h>
#include <mma.h>

#include "utils.h"

constexpr uint64_t TILE_SIZE = 16;

__global__ void Kernel(const half *A, const half *B, float *C, uint64_t m,
                       uint64_t n, uint64_t k) {
  // Declare WMMA fragments
  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, half,
                 wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, half,
                 wmma::col_major>
      b_frag;
  wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, float>
      acc_frag;

  // Initialize the accumulator to zero
  wmma::fill_fragment(acc_frag, 0.0f);

  // Calculate thread coordinates
  uint64_t y_base = blockIdx.y * TILE_SIZE;
  uint64_t x_base = blockIdx.x * TILE_SIZE;

  // Loop over tiles
  for (uint64_t k_tile = 0; k_tile < k; k_tile += TILE_SIZE) {
    // Load fragments from shared memory
    wmma::load_matrix_sync(a_frag, &A[y_base * k + k_tile], k);
    wmma::load_matrix_sync(b_frag, &B[x_base * k + k_tile], k);

    // Perform matrix multiplication using WMMA
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  if (y_base < m && x_base < n) {
    wmma::store_matrix_sync(&C[y_base * n + x_base], acc_frag, n,
                            wmma::mem_row_major);
  }
}

class GemmTensorCore_1 : public GemmBase {
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
  GemmTensorCore_1 gemm("GemmTensorCore_1");
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
