#include "layers/rmsnorm.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace
{
__device__ __forceinline__ float warpReduceSum(float value)
{
    constexpr unsigned mask = 0xFFFFFFFF;
    for(int offset = 16; offset > 0; offset >>= 1)
    {
        value += __shfl_down_sync(mask, value, offset);
    }
    return value;
}
__global__ void rmsNormKernel(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int hidden_size
)
{
    __shared__ float warp_sums[32];
    float sum = 0.0;
    for(int offset = threadIdx.x;offset < hidden_size;offset += blockDim.x)
    {
        float x = __bfloat162float(input[blockIdx.x * hidden_size + offset]);
        sum += x * x;
    }
    sum = warpReduceSum(sum);
    
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
   if(lane == 0)
   {
    warp_sums[warp_id] = sum;
   }
   __syncthreads();
   float block_sum = 0.0;
   if(warp_id == 0)
   {
    block_sum = warp_sums[lane];
    block_sum = warpReduceSum(block_sum);
    if(lane == 0)
    {
        warp_sums[0] = rsqrtf(block_sum / hidden_size + 1e-6f);
    }
   }
   __syncthreads();
   const float inv_rms = warp_sums[0];
   for(int  offset = threadIdx.x;offset < hidden_size;offset += blockDim.x)
   {
    float x = __bfloat162float(input[blockIdx.x * hidden_size + offset]);
    float w = __bfloat162float(weight[offset]);
    output[blockIdx.x * hidden_size + offset] = __float2bfloat16(x * inv_rms * w);
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
