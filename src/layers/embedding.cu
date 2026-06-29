#include "layers/embedding.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace
{
__global__ void embeddingGatherKernel(
    const int32_t* input_ids,
    const __nv_bfloat16* embed_tokens,
    __nv_bfloat16* hidden_states,
    int hidden_size
)
{
    const int token_idx = blockIdx.x;
    const int hidden_idx = threadIdx.x;
    const int token_id = input_ids[token_idx];

    for (int offset = hidden_idx; offset < hidden_size; offset += blockDim.x)
    {
        hidden_states[token_idx * hidden_size + offset] =
            embed_tokens[token_id * hidden_size + offset];
    }
}
}

namespace betavllm
{
void embeddingGather(
    const InputIds& input_ids,
    const __nv_bfloat16* embed_tokens,
    ActivationBuffer& activations)
{
    if (input_ids.num_tokens == 0 || activations.hidden_size <= 0)
    {
        return;
    }

    embeddingGatherKernel<<<input_ids.num_tokens, 1024>>>(
        input_ids.gpu_ids,
        embed_tokens,
        activations.hidden_states,
        activations.hidden_size
    );
}
}
