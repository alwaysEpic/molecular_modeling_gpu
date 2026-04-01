cmake_minimum_required(VERSION 3.18)
project(mc_sim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# Default to sm_75 (Turing), override with -DCMAKE_CUDA_ARCHITECTURES=86 etc.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

add_executable(mc_sim
    src/main.cu
    src/timing.c
)

target_include_directories(mc_sim PRIVATE src)
