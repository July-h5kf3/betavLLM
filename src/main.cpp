#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <queue>
#include <iostream>

#define JSON_USE_IMPLICIT_CONVERSIONS 0

#include "model_config.hpp"

#include "model_loader.hpp"



int checkGPUStatus()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if(device_count == 0)
    {
        std::cerr<< "No CUDA device found\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Global memory: " << prop.totalGlobalMem / B_TO_MB << " MB\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;

    size_t free_mem;
    size_t total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Free memory: " << free_mem / B_TO_GB << "GB, total memory: " << total_mem / B_TO_GB << "GB\n";
    return 0;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <model.safetensors>\n";
        retrun 1;
    }

    std::string model_path = argv[1];
    if(checkGPUStatus() != 0)
    {
        return 1;
    }
    cublasHandle_t cublas_handle;
    cublasStatus_t status = cublasCreate(&cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS init failed, status: " << status << "\n";
        return 1;
    }
    Weights weights;
    if(loadWeights(model_path, weights) != 0)
    {
        std::cerr<<"Load Weights failed \n";
        return 1;
    }
    return 0;
}
