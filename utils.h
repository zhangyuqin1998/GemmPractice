#pragma once

#include <cuda_fp16.h>

#include <cstdlib>
#include <ctime>
#include <iostream>

#define CHECKCUDA(call)                                                 \
  do {                                                                  \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
      std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ \
                << ": " << cudaGetErrorString(err) << std::endl;        \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  } while (0)

// Initialize matrix with random values
void initMatrix(half* matrix, uint64_t size) {
  for (uint64_t i = 0; i < size; ++i) {
    float random01 = static_cast<float>(rand()) / RAND_MAX;
    float randomMinus1To1 = random01 * 2 - 1;
    matrix[i] = __float2half(randomMinus1To1);
  }
}

__global__ void gemmBaselineGPU(const half* A, const half* B, float* C,
                                uint64_t m, uint64_t n, uint64_t k) {
  uint64_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;

  if (y < m && x < n) {
    float sum = 0.0f;
    for (uint64_t t = 0; t < k; ++t) {
      sum += __half2float(A[y * k + t] * B[x * k + t]);
    }
    C[y * n + x] = (sum);
  }
}

// Compare two matrices
bool compareMatrices(const float* mat1, const float* mat2, uint64_t size,
                     float tolerance = 5e-2) {
  for (uint64_t i = 0; i < size; ++i) {
    float val1 = (mat1[i]);
    float val2 = (mat2[i]);
    if (fabs(val1 - val2) > tolerance) {
      std::cout << "Matrices differ at " << i << ", mat1[i]=" << val1
                << ",mat2[i]=" << val2;
      std::cout << ", diff=" << fabs(val1 - val2) << std::endl;
      return false;
    }
  }
  return true;
}

class GemmBase {
 public:
  GemmBase(std::string name) : name(name) {}

  virtual ~GemmBase() {}

  virtual void LaunchKernel(const half* d_A, const half* d_B, float* d_C,
                            uint64_t m, uint64_t n, uint64_t k) = 0;

  void RunProfile(uint64_t m, uint64_t n, uint64_t k) {
    half* d_A;
    half* d_B;
    float* d_C;
    float* d_C_ref;
    half* h_A = new half[m * k];
    half* h_B = new half[n * k];
    float* h_C = new float[m * n];
    float* h_C_ref = new float[m * n];

    CHECKCUDA(cudaMalloc(&d_A, m * k * sizeof(half)));
    CHECKCUDA(cudaMalloc(&d_B, n * k * sizeof(half)));
    CHECKCUDA(cudaMalloc(&d_C, m * n * sizeof(float)));
    CHECKCUDA(cudaMalloc(&d_C_ref, m * n * sizeof(float)));

    srand(42);
    initMatrix(h_A, m * k);
    initMatrix(h_B, n * k);

    CHECKCUDA(
        cudaMemcpy(d_A, h_A, m * k * sizeof(half), cudaMemcpyHostToDevice));
    CHECKCUDA(
        cudaMemcpy(d_B, h_B, n * k * sizeof(half), cudaMemcpyHostToDevice));

    for (uint64_t i = 0; i < 10; ++i) {
      // warm up
      LaunchKernel(d_A, d_B, d_C, m, n, k);
    }
    // Create CUDA event for timing
    cudaEvent_t start, stop;
    CHECKCUDA(cudaEventCreate(&start));
    CHECKCUDA(cudaEventCreate(&stop));

    // Start recording
    CHECKCUDA(cudaEventRecord(start));
    for (uint64_t i = 0; i < 10; ++i) {
      LaunchKernel(d_A, d_B, d_C, m, n, k);
    }
    CHECKCUDA(cudaGetLastError());

    // Stop recording
    CHECKCUDA(cudaEventRecord(stop));
    CHECKCUDA(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    CHECKCUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Compute reference result on GPU
    dim3 baseline_blockdim(16, 16);
    dim3 baseline_griddim((n + baseline_blockdim.x - 1) / baseline_blockdim.x,
                          (m + baseline_blockdim.y - 1) / baseline_blockdim.y);
    gemmBaselineGPU<<<baseline_griddim, baseline_blockdim>>>(d_A, d_B, d_C_ref,
                                                             m, n, k);

    // Copy result back to host
    CHECKCUDA(
        cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECKCUDA(cudaMemcpy(h_C_ref, d_C_ref, m * n * sizeof(float),
                         cudaMemcpyDeviceToHost));

    // gemmBaselineCPU(h_A, h_B, h_C_ref, m, n, k);
    float gflops = (float)(m * n * k * 2) / (milliseconds / 10.f) / 1e6f;
    // Compare results
    if (compareMatrices(h_C, h_C_ref, m * n)) {
      std::cout << "Kernel " << name << ", shape=" << m << "x" << n << "x" << k;
      std::cout << " execution time: " << milliseconds << " ms, "
                << "gflops: " << gflops << std::endl;
    }

    // Clean up
    CHECKCUDA(cudaEventDestroy(start));
    CHECKCUDA(cudaEventDestroy(stop));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    CHECKCUDA(cudaFree(d_A));
    CHECKCUDA(cudaFree(d_B));
    CHECKCUDA(cudaFree(d_C));
    CHECKCUDA(cudaFree(d_C_ref));
  }

 protected:
  std::string name;
};