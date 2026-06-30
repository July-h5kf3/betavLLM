#include "model_loader.hpp"
#include "test_utils.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <string>

namespace
{
bool hasCudaDevice()
{
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0)
    {
        std::cerr << betavllm::test::passFail(false)
                  << ": no CUDA device available: " << cudaGetErrorString(error) << "\n";
        return false;
    }
    return true;
}
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <model_dir>\n";
        return 2;
    }
    if (!hasCudaDevice())
    {
        return 1;
    }

    Weights weights{};
    const std::string model_dir = argv[1];
    if (loadWeights(model_dir, weights) != 0)
    {
        std::cerr << betavllm::test::passFail(false) << ": loadWeights returned non-zero\n";
        return 1;
    }

    bool ok = weights.embed_tokens != nullptr && weights.norm != nullptr;
    for (int layer = 0; layer < N_LAYERS; ++layer)
    {
        ok = ok &&
             weights.input_layernorm[layer] != nullptr &&
             weights.post_attn_layernorms[layer] != nullptr &&
             weights.mlp_gate_proj[layer] != nullptr &&
             weights.mlp_up_proj[layer] != nullptr &&
             weights.mlp_down_proj[layer] != nullptr &&
             weights.w_q[layer] != nullptr &&
             weights.w_k[layer] != nullptr &&
             weights.w_v[layer] != nullptr &&
             weights.w_o[layer] != nullptr;
    }

    if (!ok)
    {
        std::cerr << betavllm::test::passFail(false)
                  << ": at least one required weight pointer is null\n";
        return 1;
    }

    std::cout << betavllm::test::passFail(true) << ": model weights loaded\n";
    std::cout << "embed_tokens=" << static_cast<const void*>(weights.embed_tokens)
              << ", norm=" << static_cast<const void*>(weights.norm)
              << ", layers=" << N_LAYERS << "\n";
    return 0;
}
