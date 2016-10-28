cmake_minimum_required(VERSION 3.4)

project(runner)

set(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/build/runner)

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
    runner
    data-util.h
    data-util.cpp
    main.cpp
)

target_link_libraries(
    runner
    neural-networks
    ${CUDA_LIBRARY}
)

add_dependencies(runner neural-networks)
