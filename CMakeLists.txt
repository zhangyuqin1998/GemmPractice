cmake_minimum_required(VERSION 3.18)

project(GemmPractice CUDA)

function(add_gemm_executable executable_name source_file)
    add_executable(${executable_name} ${source_file})
    set_target_properties(${executable_name} PROPERTIES 
                            CUDA_STANDARD 11
                            CUDA_STANDARD_REQUIRED YES
                            CUDA_ARCHITECTURES 90)
endfunction()

add_gemm_executable(gemm_cuda_core_1 gemm_cuda_core_1.cu)
add_gemm_executable(gemm_cuda_core_2 gemm_cuda_core_2.cu)