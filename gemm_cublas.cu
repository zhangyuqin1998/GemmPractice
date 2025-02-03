#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "utils.h"

class GemmCublas : public GemmBase {
 public:
  using GemmBase::GemmBase;  // 继承基类的构造函数

  void LaunchKernel(const half *d_A, const half *d_B, float *d_C, uint64_t m,
                    uint64_t n, uint64_t k) override {
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        d_B,
        CUDA_R_16F,
        k,
        d_A,
        CUDA_R_16F,
        k,
        &beta,
        d_C,
        CUDA_R_32F,
        n,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT
    );

  }
};

int main() {
  GemmCublas gemm("GemmCublas");
  // mxnxk
  // gemm.RunProfile(32, 32, 16);
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
