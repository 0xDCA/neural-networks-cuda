cmake_minimum_required(VERSION 3.4)

project(neural-networks)

set(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/build/neural-networks)

cuda_add_library(
    neural-networks
    cuda-helpers.cu
    CudaFeedforwardNeuralNetwork.h
    CudaFeedforwardNeuralNetwork.cpp
    int-util.h
    int-util.cpp
    ManagedMatrix.h
    TrainResult.h
    TrainSettings.h
)
