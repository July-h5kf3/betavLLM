#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "input_ids.hpp"
#include "activation_buffer.hpp"

namespace betavllm
{
    void rmsNorm(
        const ActivationBuffer& input_hidden,
        const __nv_bfloat16* weight,
        ActivationBuffer& output
    );
}
