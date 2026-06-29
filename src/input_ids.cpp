#include "input_ids.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace betavllm
{
namespace
{
    void checkCuda(cudaError_t error, const char* message)
    {
        if (error != cudaSuccess)
        {
            throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(error));
        }
    }
}

    InputIds::InputIds(std::vector<int32_t> ids)
    : cpu_ids(std::move(ids)), num_tokens(cpu_ids.size())
    {
        if (num_tokens == 0)
        {
            return;
        }

        const size_t bytes = num_tokens * sizeof(int32_t);
        checkCuda(cudaMalloc(&gpu_ids, bytes), "cudaMalloc input ids failed");
        cudaError_t copy_error = cudaMemcpy(
            gpu_ids,
            cpu_ids.data(),
            bytes,
            cudaMemcpyHostToDevice
        );
        if (copy_error != cudaSuccess)
        {
            cudaFree(gpu_ids);
            gpu_ids = nullptr;
            checkCuda(copy_error, "cudaMemcpy input ids failed");
        }
    }

    InputIds::~InputIds()
    {
        if (gpu_ids)
        {
            cudaFree(gpu_ids);
        }
    }

    InputIds::InputIds(InputIds&& other) noexcept
    : cpu_ids(std::move(other.cpu_ids)),
      gpu_ids(other.gpu_ids),
      num_tokens(other.num_tokens)
    {
        other.gpu_ids = nullptr;
        other.num_tokens = 0;
    }

    InputIds& InputIds::operator=(InputIds&& other) noexcept
    {
        if (this != &other)
        {
            if (gpu_ids)
            {
                cudaFree(gpu_ids);
            }

            cpu_ids = std::move(other.cpu_ids);
            gpu_ids = other.gpu_ids;
            num_tokens = other.num_tokens;

            other.gpu_ids = nullptr;
            other.num_tokens = 0;
        }

        return *this;
    }
}
