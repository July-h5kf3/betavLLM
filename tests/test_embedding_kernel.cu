#include "activation_buffer.hpp"
#include "input_ids.hpp"
#include "layers/embedding.hpp"
#include "test_utils.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
void checkCuda(cudaError_t error, const char* message)
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(error));
    }
}

float bf16ToFloat(__nv_bfloat16 value)
{
    return __bfloat162float(value);
}
}

int main()
{
    try
    {
        constexpr int vocab_size = 1024;
        constexpr int hidden_size = 512;
        constexpr int num_tokens = 512;
        constexpr int iterations = 200;

        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> value_dist(-1.0f, 1.0f);
        std::uniform_int_distribution<int32_t> id_dist(0, vocab_size - 1);

        std::vector<__nv_bfloat16> embed_tokens(vocab_size * hidden_size);
        for (auto& value : embed_tokens)
        {
            value = __float2bfloat16(value_dist(rng));
        }

        std::vector<int32_t> token_ids(num_tokens);
        for (auto& id : token_ids)
        {
            id = id_dist(rng);
        }

        std::vector<__nv_bfloat16> expected(num_tokens * hidden_size);

        auto cpu_start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < iterations; ++iter)
        {
            for (int token = 0; token < num_tokens; ++token)
            {
                const int32_t token_id = token_ids[token];
                for (int hidden = 0; hidden < hidden_size; ++hidden)
                {
                    expected[token * hidden_size + hidden] =
                        embed_tokens[token_id * hidden_size + hidden];
                }
            }
        }
        auto cpu_end = std::chrono::high_resolution_clock::now();
        const double cpu_ms =
            std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

        __nv_bfloat16* gpu_embed_tokens = nullptr;
        checkCuda(cudaMalloc(&gpu_embed_tokens, embed_tokens.size() * sizeof(__nv_bfloat16)),
                  "cudaMalloc embed_tokens failed");
        checkCuda(cudaMemcpy(gpu_embed_tokens,
                             embed_tokens.data(),
                             embed_tokens.size() * sizeof(__nv_bfloat16),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy embed_tokens failed");

        betavllm::InputIds input_ids(token_ids);
        betavllm::ActivationBuffer output(num_tokens, hidden_size);

        cudaEvent_t start;
        cudaEvent_t stop;
        checkCuda(cudaEventCreate(&start), "cudaEventCreate start failed");
        checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop failed");

        betavllm::embeddingGather(input_ids, gpu_embed_tokens, output);
        checkCuda(cudaGetLastError(), "embeddingGather warmup launch failed");
        checkCuda(cudaDeviceSynchronize(), "embeddingGather warmup sync failed");

        checkCuda(cudaEventRecord(start), "cudaEventRecord start failed");
        for (int iter = 0; iter < iterations; ++iter)
        {
            betavllm::embeddingGather(input_ids, gpu_embed_tokens, output);
        }
        checkCuda(cudaEventRecord(stop), "cudaEventRecord stop failed");
        checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop failed");

        float gpu_ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&gpu_ms, start, stop), "cudaEventElapsedTime failed");

        std::vector<__nv_bfloat16> actual(expected.size());
        checkCuda(cudaMemcpy(actual.data(),
                             output.hidden_states,
                             actual.size() * sizeof(__nv_bfloat16),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy output failed");

        float max_abs_error = 0.0f;
        for (size_t i = 0; i < actual.size(); ++i)
        {
            const float error = std::fabs(bf16ToFloat(actual[i]) - bf16ToFloat(expected[i]));
            max_abs_error = std::max(max_abs_error, error);
        }

        checkCuda(cudaEventDestroy(start), "cudaEventDestroy start failed");
        checkCuda(cudaEventDestroy(stop), "cudaEventDestroy stop failed");
        checkCuda(cudaFree(gpu_embed_tokens), "cudaFree embed_tokens failed");

        const double speedup = gpu_ms > 0.0f ? cpu_ms / static_cast<double>(gpu_ms) : 0.0;
        const bool correct = max_abs_error == 0.0f;
        std::cout << "embedding_kernel correctness: "
                  << betavllm::test::passFail(correct) << "\n";
        std::cout << "max_abs_error: " << max_abs_error << "\n";
        std::cout << "cpu_ms: " << cpu_ms << "\n";
        std::cout << "gpu_ms: " << gpu_ms << "\n";
        std::cout << "speedup: " << betavllm::test::speedup(speedup) << "\n";

        return correct ? 0 : 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << betavllm::test::passFail(false) << ": " << error.what() << "\n";
        return 1;
    }
}
