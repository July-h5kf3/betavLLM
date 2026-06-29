#pragma once

#include <cstddef>
#include <cuda_bf16.h>


namespace betavllm
{
    struct ActivationBuffer
    {
        __nv_bfloat16* hidden_states = nullptr;
        size_t num_tokens = 0;
        int hidden_size = 0;

        ActivationBuffer(size_t num_tokens, int hidden_size);
        ~ActivationBuffer();

        ActivationBuffer(const ActivationBuffer&) = delete;
        ActivationBuffer& operator=(const ActivationBuffer&) = delete;

        ActivationBuffer(ActivationBuffer&& other) noexcept;
        ActivationBuffer& operator=(ActivationBuffer&& other) noexcept;
    };
}