cmake_minimum_required(VERSION 3.19)

project(betaVLLM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(third_party/tokenizers-cpp)

add_executable(betavllm
    src/main.cpp
    src/tokenizer.cpp
)

target_include_directories(betavllm PRIVATE
    include
)

target_link_libraries(betavllm PRIVATE
    CUDA::cudart
    CUDA::cublas
    tokenizers_cpp
)
