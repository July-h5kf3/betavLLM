#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "input_ids.hpp"
#include "activation_buffer.hpp"

namespace betavllm
{
    void embeddingGather(
        const InputIds& input_ids,
        const __nv_bfloat16* embed_tokens,
        ActivationBuffer& activations
    );
}
