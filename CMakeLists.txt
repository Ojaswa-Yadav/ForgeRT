cmake_minimum_required(VERSION 3.20)
project(ForgeRT LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Detect GPU architecture; fall back to Ampere/Ada/Hopper if cmake can't query
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
endif()

find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# Common include paths
# ---------------------------------------------------------------------------
set(FORGERT_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/runtime/include
    ${CMAKE_SOURCE_DIR}/kernels
    ${CMAKE_SOURCE_DIR}/third_party
    ${CUDAToolkit_INCLUDE_DIRS}
)

# ---------------------------------------------------------------------------
# Kernel library  (CUDA)
# ---------------------------------------------------------------------------
add_library(forgert_kernels STATIC
    kernels/rmsnorm.cu
    kernels/rope.cu
    kernels/attention_decode.cu
    kernels/elementwise.cu
    kernels/sampling.cu
)
target_include_directories(forgert_kernels PUBLIC ${FORGERT_INCLUDE_DIRS})
set_target_properties(forgert_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(forgert_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        -lineinfo
        -Xptxas=-v
    >
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
target_link_libraries(forgert_kernels PUBLIC CUDA::cudart CUDA::cublas)

# ---------------------------------------------------------------------------
# Runtime library  (C++)
# ---------------------------------------------------------------------------
add_library(forgert_runtime STATIC
    runtime/model.cpp
    runtime/executor.cpp
    runtime/kv_cache.cpp
)
target_include_directories(forgert_runtime PUBLIC ${FORGERT_INCLUDE_DIRS})
target_link_libraries(forgert_runtime PUBLIC forgert_kernels CUDA::cudart CUDA::cublas)

# ---------------------------------------------------------------------------
# Main CLI executable
# ---------------------------------------------------------------------------
add_executable(forgert runtime/main.cpp)
target_include_directories(forgert PRIVATE ${FORGERT_INCLUDE_DIRS})
target_link_libraries(forgert PRIVATE forgert_runtime)

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
add_executable(bench_attention benchmarks/bench_attention.cpp)
target_include_directories(bench_attention PRIVATE ${FORGERT_INCLUDE_DIRS})
target_link_libraries(bench_attention PRIVATE forgert_runtime)

add_executable(bench_gemm benchmarks/bench_gemm.cpp)
target_include_directories(bench_gemm PRIVATE ${FORGERT_INCLUDE_DIRS})
target_link_libraries(bench_gemm PRIVATE forgert_runtime)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
add_executable(test_kernels tests/test_kernels.cpp)
target_include_directories(test_kernels PRIVATE ${FORGERT_INCLUDE_DIRS})
target_link_libraries(test_kernels PRIVATE forgert_runtime)
