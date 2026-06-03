cmake_minimum_required(VERSION 3.18)
project(rocm-serve LANGUAGES CXX)

# ──────────────────────────────────────────────────────────────
# rocm-serve: Custom HIP kernels for ROCm/MI300X
#
# This build system compiles the high-performance custom kernels
# used by rocm-serve for PagedAttention, quantized GEMM, and
# all-reduce operations optimized for CDNA 3 architecture.
# ──────────────────────────────────────────────────────────────

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ROCm configuration
if(NOT DEFINED HIP_PATH)
    set(HIP_PATH "/opt/rocm")
endif()

list(APPEND CMAKE_PREFIX_PATH "${HIP_PATH}")
find_package(HIP REQUIRED)

# Compiler flags for CDNA 3 (MI300X)
set(HIP_ARCHITECTURES "gfx942")  # MI300X = gfx942
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --offload-arch=${HIP_ARCHITECTURES}")
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3 -ffast-math")
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --hipstdpar")
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -D__HIP_PLATFORM_AMD__")

# Include directories
set(KERNELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ── Kernel Sources ────────────────────────────────────────────

set(KERNEL_SOURCES
    ${KERNELS_DIR}/paged_attention.cu
    ${KERNELS_DIR}/quantized_gemm.cu
    ${KERNELS_DIR}/all_reduce.cu
    ${KERNELS_DIR}/topk_sampling.cu
    ${KERNELS_DIR}/rms_norm.cu
    ${KERNELS_DIR}/rotary_embedding.cu
)

# ── Build Targets ─────────────────────────────────────────────

add_library(rocm_kernels STATIC ${KERNEL_SOURCES})
target_include_directories(rocm_kernels PUBLIC ${INCLUDE_DIR})
target_link_libraries(rocm_kernels PUBLIC hip::device)

# Install
install(TARGETS rocm_kernels
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(DIRECTORY ${INCLUDE_DIR}/ DESTINATION include/rocm-serve)

# ── Tests ─────────────────────────────────────────────────────

option(BUILD_TESTS "Build kernel tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_executable(test_paged_attention tests/test_paged_attention.cpp)
    target_link_libraries(test_paged_attention rocm_kernels)
    add_test(NAME paged_attention_test COMMAND test_paged_attention)
endif()
