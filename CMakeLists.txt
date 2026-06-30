cmake_minimum_required(VERSION 3.19)

project(betaVLLM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Skip FetchContent update checks for downloaded dependencies" FORCE)

if(NOT DEFINED CARGO_HOME)
    if("$ENV{CARGO_HOME}" STREQUAL "")
        set(CARGO_HOME "$ENV{HOME}/.cargo")
    else()
        set(CARGO_HOME "$ENV{CARGO_HOME}")
    endif()
endif()

set(_BETAVLLM_RUSTUP_CARGO "${CARGO_HOME}/bin/cargo")
if(EXISTS "${_BETAVLLM_RUSTUP_CARGO}")
    set(CARGO_HOME "${CARGO_HOME}" CACHE PATH "Cargo home used by tokenizers-cpp" FORCE)
    set(CARGO_EXECUTABLE "${_BETAVLLM_RUSTUP_CARGO}" CACHE FILEPATH "Cargo executable used by tokenizers-cpp" FORCE)
    message(STATUS "Using Cargo: ${CARGO_EXECUTABLE}")
endif()

add_subdirectory(third_party/tokenizers-cpp)

add_library(betavllm_core STATIC
    src/tokenizer.cpp
    src/input_ids.cpp
    src/activation_buffer.cpp
    src/layers/embedding.cu
    src/layers/rmsnorm.cu
)

target_include_directories(betavllm_core PUBLIC
    include
)

target_link_libraries(betavllm_core PUBLIC
    CUDA::cudart
    CUDA::cublas
    tokenizers_cpp
)

add_executable(betavllm
    src/main.cpp
)

target_link_libraries(betavllm PRIVATE
    betavllm_core
)

option(BUILD_BETAVLLM_TESTS "Build betaVLLM test executables" ON)
if(BUILD_BETAVLLM_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
