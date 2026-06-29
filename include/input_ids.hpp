#pragma once

#include <cstdint>
#include <vector>
#include <cstddef>

namespace betavllm 
{
    struct InputIds
    {
        std::vector<int32_t> cpu_ids;
        int32_t* gpu_ids = nullptr;
        size_t num_tokens = 0;

        explicit InputIds(std::vector<int32_t> ids);
        ~InputIds();

        InputIds(const InputIds&) = delete;
        InputIds& operator=(const InputIds&) = delete;
        InputIds(InputIds&& other) noexcept;
        InputIds& operator=(InputIds&& other) noexcept;
    };
}