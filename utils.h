#pragma once

#include <iostream>
#include <cstdlib>
#include <ctime>

#define CHECKCUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


// Initialize matrix with random values
void initMatrix(float* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// CPU implementation of matrix multiplication for verification
void gemmCPU(const float* A, const float* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int t = 0; t < k; ++t) {
                sum += A[i * k + t] * B[j * k + t];
            }
            C[i * n + j] = sum;
        }
    }
}

// Compare two matrices
bool compareMatrices(const float* mat1, const float* mat2, int size, float tolerance = 1e-3) {
    for (int i = 0; i < size; ++i) {
        if (fabs(mat1[i] - mat2[i]) > tolerance) {
            std::cout << "Matrices differ at " << i << ", mat1[i]=" << mat1[i] << ",mat2[i]=" << mat2[i];
            std::cout << ", diff=" << fabs(mat1[i] - mat2[i]) << std::endl;
            return false;
        }
    }
    return true;
}


class GemmBase {
 public:
    GemmBase(std::string name) : name(name) {}

    virtual ~GemmBase() {}

    virtual void LaunchKernel(
        float *d_A,
        float *d_B,
        float *d_C,
        int m, int n, int k) = 0;

    void RunProfile(int m, int n, int k) {
        float* d_A;
        float* d_B;
        float* d_C;
        float* h_A = new float[m * k];
        float* h_B = new float[n * k];
        float* h_C = new float[m * n];
        float* h_C_ref = new float[m * n];

        CHECKCUDA(cudaMalloc(&d_A, m * k * sizeof(float)));
        CHECKCUDA(cudaMalloc(&d_B, n * k * sizeof(float)));
        CHECKCUDA(cudaMalloc(&d_C, m * n * sizeof(float)));

        srand(time(nullptr));
        initMatrix(h_A, m * k);
        initMatrix(h_B, n * k);

        CHECKCUDA(cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice));
        CHECKCUDA(cudaMemcpy(d_B, h_B, n * k * sizeof(float), cudaMemcpyHostToDevice));

        for (int i = 0; i < 10; ++i) {
            // warm up
            LaunchKernel(d_A, d_B, d_C, m, n, k);
        }
        // Create CUDA event for timing
        cudaEvent_t start, stop;
        CHECKCUDA(cudaEventCreate(&start));
        CHECKCUDA(cudaEventCreate(&stop));

         // Start recording
        CHECKCUDA(cudaEventRecord(start));
        for (int i = 0; i < 10; ++i) {
            LaunchKernel(d_A, d_B, d_C, m, n, k);
        }
        CHECKCUDA(cudaGetLastError());

        // Stop recording
        CHECKCUDA(cudaEventRecord(stop));
        CHECKCUDA(cudaEventSynchronize(stop));

        // Calculate elapsed time
        float milliseconds = 0;
        CHECKCUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        std::cout << "Kernel " << name << ", shape=" << m << "x" << n << "x" << k << " execution time: " << milliseconds << " ms" << std::endl;

        // Copy result back to host
        CHECKCUDA(cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));

        // Compute reference result on CPU
        gemmCPU(h_A, h_B, h_C_ref, m, n, k);

        // Compare results
        if (compareMatrices(h_C, h_C_ref, m * n)) {
            // std::cout << "Results of " << name << " are correct!" << std::endl;
        } else {
            std::cout << "Results of " << name << " are incorrect!" << std::endl;
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
    }

 protected:
  std::string name;
};