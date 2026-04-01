cmake_minimum_required(VERSION 3.18)
project(mc_sim LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)

# CPU-only reference build (no CUDA required)
add_executable(mc_sim_cpu
    src/main_cpu.cpp
    src/simulation_cpu.cpp
    src/timing.c
)
target_include_directories(mc_sim_cpu PRIVATE src)

# GPU build (requires CUDA toolkit)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 11)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()

    add_executable(mc_sim
        src/main.cu
        src/simulation_cpu.cpp
        src/simulation_gpu.cu
        src/timing.c
    )
    target_include_directories(mc_sim PRIVATE src)

    # Fast math: uses hardware intrinsics for sqrt, sin, cos, etc.
    # ~2-7x faster for math-heavy kernels with no qualitative impact
    # on Brownian motion statistics (Januszewski & Kostur 2009, Section 6)
    target_compile_options(mc_sim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
else()
    message(STATUS "CUDA not found -- only building mc_sim_cpu")
endif()
