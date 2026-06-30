#include "layers/rmsnorm.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace
{
__global__ void rmsNormKernel(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int hidden_size
)
{
    __shared__ float rms_vector[1024];
    rms_vector[threadIdx.x] = 0;
    for(int offset = threadIdx.x;offset < hidden_size;offset += blockDim.x)
    {
        float x = __bfloat162float(input[blockIdx.x * hidden_size + offset]);
        rms_vector[threadIdx.x] += x * x;
    }
    __syncthreads();
    for(int stride = blockDim.x / 2;stride > 0; stride >>= 1)
    {
        if(threadIdx.x < stride)
        {
            rms_vector[threadIdx.x] += rms_vector[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0)
    {
        rms_vector[0] = rsqrtf(rms_vector[0] / hidden_size + 1e-5f);
    }
    __syncthreads();
    for(int offset = threadIdx.x;offset < hidden_size;offset += blockDim.x)
    {
        float x = __bfloat162float(input[blockIdx.x * hidden_size + offset]);
        float w = __bfloat162float(weight[offset]);
        output[blockIdx.x * hidden_size + offset] = __float2bfloat16(x * rms_vector[0] * w);
    }
}
}

namespace betavllm
{
    void rmsNorm(
        const ActivationBuffer& input_hidden,
        const __nv_bfloat16* weight,
        ActivationBuffer& output
    )
    {
        if(input_hidden.num_tokens == 0 || output.hidden_size == 0)return;
        rmsNormKernel<<<input_hidden.num_tokens,1024>>>(
            input_hidden.hidden_states,
            weight,
            output.hidden_states,
            input_hidden.hidden_size
        );
    }
}
