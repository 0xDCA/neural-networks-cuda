cmake_minimum_required(VERSION 3.4)

project(neural-network-test)

set(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/build/neural-network-test)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    --cudart static
)

include_directories(
    ${CMAKE_SOURCE_DIR}/neural-networks
    lib
)

link_directories(
    ${CMAKE_SOURCE_DIR}/build/lib
)

cuda_add_executable(
    neural-network-test
    ApproxMatrixMatcher.h
    ApproxMatrixMatcher.cpp
    main.cpp
)

target_link_libraries(
    neural-network-test
    neural-networks
    ${CUDA_LIBRARY}
)
