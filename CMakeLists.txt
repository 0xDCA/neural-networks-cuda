cmake_minimum_required(VERSION 3.4)

find_package(CUDA QUIET REQUIRED)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

list(
  APPEND CMAKE_CXX_FLAGS
  "-std=c++11"
)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_60,code=sm_60
)

set(CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(neural-networks)
add_subdirectory(runner)
add_subdirectory(test)
