#include "activation_buffer.hpp"
#include "layers/rmsnorm.hpp"
#include "test_utils.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
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

void cpuRmsNorm(
    const std::vector<__nv_bfloat16>& input,
    const std::vector<__nv_bfloat16>& weight,
    std::vector<float>& output,
    int num_tokens,
    int hidden_size)
{
    constexpr float eps = 1e-5f;
    for (int token = 0; token < num_tokens; ++token)
    {
        float sum = 0.0f;
        for (int hidden = 0; hidden < hidden_size; ++hidden)
        {
            const float x = __bfloat162float(input[token * hidden_size + hidden]);
            sum += x * x;
        }

        const float inv_rms = 1.0f / std::sqrt(sum / hidden_size + eps);
        for (int hidden = 0; hidden < hidden_size; ++hidden)
        {
            const float x = __bfloat162float(input[token * hidden_size + hidden]);
            const float w = __bfloat162float(weight[hidden]);
            output[token * hidden_size + hidden] = x * inv_rms * w;
        }
    }
}
}

int main()
{
    try
    {
        constexpr int num_tokens = 256;
        constexpr int hidden_size = 2048;
        constexpr int iterations = 100;
        constexpr float tolerance = 2.5e-2f;

        std::mt19937 rng(5678);
        std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
        std::uniform_real_distribution<float> weight_dist(0.25f, 1.25f);

        std::vector<__nv_bfloat16> input(num_tokens * hidden_size);
        std::vector<__nv_bfloat16> weight(hidden_size);
        for (auto& value : input)
        {
            value = __float2bfloat16(input_dist(rng));
        }
        for (auto& value : weight)
        {
            value = __float2bfloat16(weight_dist(rng));
        }

        std::vector<float> expected(num_tokens * hidden_size);
        auto cpu_start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < iterations; ++iter)
        {
            cpuRmsNorm(input, weight, expected, num_tokens, hidden_size);
        }
        auto cpu_end = std::chrono::high_resolution_clock::now();
        const double cpu_ms =
            std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

        betavllm::ActivationBuffer gpu_input(num_tokens, hidden_size);
        betavllm::ActivationBuffer gpu_output(num_tokens, hidden_size);
        checkCuda(cudaMemcpy(gpu_input.hidden_states,
                             input.data(),
                             input.size() * sizeof(__nv_bfloat16),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy input failed");

        __nv_bfloat16* gpu_weight = nullptr;
        checkCuda(cudaMalloc(&gpu_weight, weight.size() * sizeof(__nv_bfloat16)),
                  "cudaMalloc weight failed");
        checkCuda(cudaMemcpy(gpu_weight,
                             weight.data(),
                             weight.size() * sizeof(__nv_bfloat16),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy weight failed");

        cudaEvent_t start;
        cudaEvent_t stop;
        checkCuda(cudaEventCreate(&start), "cudaEventCreate start failed");
        checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop failed");

        betavllm::rmsNorm(gpu_input, gpu_weight, gpu_output);
        checkCuda(cudaGetLastError(), "rmsNorm warmup launch failed");
        checkCuda(cudaDeviceSynchronize(), "rmsNorm warmup sync failed");

        checkCuda(cudaEventRecord(start), "cudaEventRecord start failed");
        for (int iter = 0; iter < iterations; ++iter)
        {
            betavllm::rmsNorm(gpu_input, gpu_weight, gpu_output);
        }
        checkCuda(cudaEventRecord(stop), "cudaEventRecord stop failed");
        checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop failed");

        float gpu_ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&gpu_ms, start, stop), "cudaEventElapsedTime failed");

        std::vector<__nv_bfloat16> actual_bf16(input.size());
        checkCuda(cudaMemcpy(actual_bf16.data(),
                             gpu_output.hidden_states,
                             actual_bf16.size() * sizeof(__nv_bfloat16),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy output failed");

        float max_abs_error = 0.0f;
        for (size_t i = 0; i < actual_bf16.size(); ++i)
        {
            const float error = std::fabs(__bfloat162float(actual_bf16[i]) - expected[i]);
            max_abs_error = std::max(max_abs_error, error);
        }

        checkCuda(cudaEventDestroy(start), "cudaEventDestroy start failed");
        checkCuda(cudaEventDestroy(stop), "cudaEventDestroy stop failed");
        checkCuda(cudaFree(gpu_weight), "cudaFree weight failed");

        const bool correct = max_abs_error <= tolerance;
        const double speedup = gpu_ms > 0.0f ? cpu_ms / static_cast<double>(gpu_ms) : 0.0;
        std::cout << "rmsnorm_kernel correctness: "
                  << betavllm::test::passFail(correct) << "\n";
        std::cout << "max_abs_error: " << max_abs_error << "\n";
        std::cout << "tolerance: " << tolerance << "\n";
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
