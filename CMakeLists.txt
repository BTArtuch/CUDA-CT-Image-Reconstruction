cmake_minimum_required(VERSION 3.10)
project(CT_Reconstruction LANGUAGES CXX CUDA)

# Request C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enforce GTX 750 Ti Architecture
set(CMAKE_CUDA_ARCHITECTURES "50")

# Enable aggressive compiler optimizations for a fair benchmark
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")

# C++ Parallel Beam Target
add_executable(fbp_parallel_cpp src/cpp/fbp_parallel.cpp)

# CUDA Parallel Beam Target
add_executable(fbp_parallel_cuda src/cuda/fbp_parallel.cu)
# Ensure nvcc compiles with high optimization and fast-math intrinsics
target_compile_options(fbp_parallel_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)