#include "activation_buffer.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

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

    ActivationBuffer::ActivationBuffer(size_t num_tokens_, int hidden_size_)
    : num_tokens(num_tokens_), hidden_size(hidden_size_)
    {
        if (num_tokens == 0 || hidden_size <= 0)
        {
            return;
        }

        const size_t bytes = num_tokens * static_cast<size_t>(hidden_size) * sizeof(__nv_bfloat16);
        checkCuda(cudaMalloc(&hidden_states, bytes), "cudaMalloc hidden_states failed");
    }

    ActivationBuffer::~ActivationBuffer()
    {
        if (hidden_states)
        {
            cudaFree(hidden_states);
        }
    }

    ActivationBuffer::ActivationBuffer(ActivationBuffer&& other) noexcept
    : hidden_states(other.hidden_states),
      num_tokens(other.num_tokens),
      hidden_size(other.hidden_size)
    {
        other.hidden_states = nullptr;
        other.num_tokens = 0;
        other.hidden_size = 0;
    }

    ActivationBuffer& ActivationBuffer::operator=(ActivationBuffer&& other) noexcept
    {
        if (this != &other)
        {
            if (hidden_states)
            {
                cudaFree(hidden_states);
            }

            hidden_states = other.hidden_states;
            num_tokens = other.num_tokens;
            hidden_size = other.hidden_size;

            other.hidden_states = nullptr;
            other.num_tokens = 0;
            other.hidden_size = 0;
        }

        return *this;
    }
}
