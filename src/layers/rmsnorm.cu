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
    __shared__ __nv_bfloat162 input_cache[1024];
    float sum = 0.0;

    const __nv_bfloat162* input2 = reinterpret_cast<const __nv_bfloat162*>(input + blockIdx.x * hidden_size);
    const __nv_bfloat162* weight2 = reinterpret_cast<const __nv_bfloat162*>(weight);
    __nv_bfloat162* output2 = reinterpret_cast<__nv_bfloat162*>(output + blockIdx.x * hidden_size);

    for(int offset = threadIdx.x;offset < hidden_size / 2;offset += blockDim.x)
    {
        const __nv_bfloat162 x2 = input2[offset];

        const float2 x = __bfloat1622float2(x2);
        input_cache[offset] = x2;
        sum += x.x * x.x + x.y * x.y;
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
   for(int  offset = threadIdx.x;offset < hidden_size / 2;offset += blockDim.x)
   {
    const __nv_bfloat162 x2 = input_cache[offset];
    const __nv_bfloat162 w2 = weight2[offset];

    const float2 x = __bfloat1622float2(x2);
    const float2 w = __bfloat1622float2(w2);
    output2[offset] = __floats2bfloat162_rn(
            x.x * inv_rms * w.x,
            x.y * inv_rms * w.y);
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
