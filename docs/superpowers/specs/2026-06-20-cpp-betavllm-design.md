# C++ betaVLLM 迁移设计

日期：2026-06-20

## 目标

把 betaVLLM 逐步改造成 C++ 版本，目标有两个：

1. 学习 nano-vLLM/vLLM 背后的核心系统设计。
2. 在学习过程中保留可量化的提速路径，而不是只写一个玩具版本。

第一版应该先把 CPU 侧 runtime 迁到 C++，同时把模型执行后端做成可替换接口。这样可以先把 scheduler、KV block manager、batch 组织和请求生命周期做清楚、测清楚，再进入更重的 CUDA 和模型执行部分。

## 第一版不做什么

第一版暂时不实现：

- Tensor parallel。
- CUDA Graph 捕获和回放。
- 自定义 safetensors 权重加载器。
- 自定义 CUDA attention kernel。
- 多模型架构支持。
- 完整替换 PyTorch/FlashAttention。

这些内容可以作为后续阶段，等 runtime 行为正确之后再逐步推进。

## 当前 Python 参考实现的组件

当前 Python 版本已经移动到 `alpha/` 目录，后续作为参考实现使用，不纳入 betaVLLM 的 git 跟踪。它的主要组件如下：

- `alpha/nanovllm/llm.py`：公开的 `LLM` 包装。
- `alpha/nanovllm/engine/llm_engine.py`：请求入口、tokenizer、主生成循环、worker 启动。
- `alpha/nanovllm/engine/sequence.py`：单条请求的运行状态。
- `alpha/nanovllm/engine/scheduler.py`：waiting/running 队列、prefill/decode 调度、chunked prefill、抢占。
- `alpha/nanovllm/engine/block_manager.py`：paged KV block 分配、prefix cache 哈希表、引用计数。
- `alpha/nanovllm/engine/model_runner.py`：模型初始化、KV cache 分配、输入准备、CUDA Graph 路径、forward 调用、采样。
- `alpha/nanovllm/models/qwen3.py` 和 `alpha/nanovllm/layers/`：Qwen3 模型结构以及 PyTorch/Triton/FlashAttention 层。

C++ 重写应该先从 runtime/control 层开始，而不是直接重写整个模型执行栈。

## 推荐架构

新增一个 `cpp/` 目录：

```text
cpp/
  CMakeLists.txt
  include/betavllm/
    block_manager.h
    config.h
    llm_engine.h
    model_runner.h
    sampling_params.h
    scheduler.h
    sequence.h
  src/
    block_manager.cpp
    llm_engine.cpp
    scheduler.cpp
    sequence.cpp
  tests/
    test_block_manager.cpp
    test_scheduler.cpp
    test_engine_fake_runner.cpp
```

C++ runtime 的概念结构保持和 Python 版一致：

```text
LLMEngine
  -> Scheduler
      -> BlockManager
      -> Sequence
  -> ModelRunner interface
```

第一阶段的 `ModelRunner` 只定义接口：

```cpp
class ModelRunner {
public:
    virtual ~ModelRunner() = default;

    virtual std::vector<int> run(
        const std::vector<Sequence*>& seqs,
        bool is_prefill
    ) = 0;
};
```

这样早期可以用 fake runner 测 runtime，后续再接 Python/PyTorch、libtorch 或原生 CUDA 后端。

## 核心类设计

### Config

第一阶段只保留会影响调度的字段：

- `max_num_batched_tokens`
- `max_num_seqs`
- `max_model_len`
- `kvcache_block_size`
- `num_kvcache_blocks`
- `eos`

模型路径、dtype、Hugging Face config、GPU memory utilization、tensor parallel size 等字段等接真实后端时再加入。

### SamplingParams

先支持：

- `temperature`
- `max_tokens`
- `ignore_eos`

fake runner 不需要真的采样，但这些字段会影响请求结束条件，所以应该从第一版就存在。

### Sequence

`Sequence` 保存一条请求的运行状态：

- `seq_id`
- `status`
- `token_ids`
- `last_token`
- `num_prompt_tokens`
- `num_cached_tokens`
- `num_scheduled_tokens`
- `block_table`
- 从 `SamplingParams` 复制来的采样字段

它需要提供和 Python property 对齐的辅助函数：

- `num_tokens()`
- `num_completion_tokens()`
- `num_blocks()`
- `last_block_num_tokens()`
- `block(i)`
- `append_token(token_id)`
- `is_finished()`

### BlockManager

第一版 C++ `BlockManager` 要对齐 Python 行为：

- 维护 `blocks`。
- 维护 `free_block_ids`。
- 维护 `used_block_ids`。
- 维护 `hash_to_block_id`。
- 分配和释放物理 KV block。
- 通过 rolling block hash 复用完整 prefix-cache block。
- 对共享 cached block 做引用计数。

哈希需要确定性并可测试。第一阶段可以先使用 C++ 确定性哈希；如果要和 Python 的 prefix cache 行为完全逐项对齐，则引入 C++ xxhash 依赖。

### Scheduler

C++ `Scheduler` 要保留 Python 版状态机：

```text
waiting queue: 新请求，以及被抢占后需要重新 prefill 的请求
running queue: 已完成 prefill、正在逐 token decode 的请求
```

`schedule()` 返回：

```cpp
struct ScheduleOutput {
    std::vector<Sequence*> seqs;
    bool is_prefill;
};
```

必须支持的行为：

- 有 waiting 请求可调度时，prefill 优先于 decode。
- 遵守 `max_num_seqs`。
- 遵守 `max_num_batched_tokens`。
- 只允许 prefill batch 中第一条 sequence 被 chunked prefill。
- prefill 前分配 KV block。
- prompt 全部 prefill 完成后，把请求从 waiting 移到 running。
- decode 时需要保证每条请求可以 append 一个 token。
- KV block 不足时，从 running 队尾抢占其它请求。
- 请求结束后释放 KV block。
- 除非 `ignore_eos = true`，否则遇到 EOS 结束。
- `num_completion_tokens == max_tokens` 时结束。

### LLMEngine

第一版 C++ `LLMEngine` 不依赖 tokenizer 和真实模型，直接接收 token id prompt：

```cpp
void add_request(
    std::vector<int> prompt_token_ids,
    SamplingParams sampling_params
);

std::vector<RequestOutput> step();

std::vector<RequestOutput> generate(
    std::vector<std::vector<int>> prompts,
    SamplingParams sampling_params
);
```

文本 tokenization 和 decode 可以先留在 C++ runtime 之外。这样第一阶段能集中验证调度和 KV cache 行为。

## 数据流

### Prefill

1. 用户传入 token id prompt。
2. Engine 创建 `Sequence`。
3. Scheduler 把它放入 `waiting`。
4. Scheduler 从 waiting 中挑选本轮 prefill 请求。
5. BlockManager 分配或复用 KV block。
6. Scheduler 设置 `num_scheduled_tokens`。
7. ModelRunner 收到 scheduled sequences 和 `is_prefill = true`。
8. Scheduler 根据返回 token 做 postprocess。
9. 如果 prefill 尚未完成，请求继续留在 waiting，等待下一轮 chunk。
10. 如果 prefill 已完成，请求进入 running，并 append 本轮采样 token。

### Decode

1. Scheduler 从 `running` 中挑选 decode 请求。
2. BlockManager 保证每条请求可以 append 一个 token。
3. KV block 不足时，Scheduler 抢占其它 running 请求。
4. ModelRunner 收到 scheduled sequences 和 `is_prefill = false`。
5. Scheduler append 采样 token。
6. 完成的请求释放 KV block 并输出结果。
7. 未完成的请求继续留在 running。

## 第一阶段后端策略

先使用 fake backend：

```cpp
class FakeModelRunner : public ModelRunner {
public:
    std::vector<int> run(
        const std::vector<Sequence*>& seqs,
        bool is_prefill
    ) override;
};
```

fake runner 可以返回确定性 token，例如：

- 在达到 `max_tokens` 前一直返回固定非 EOS token。
- 在指定测试场景中返回 EOS。
- 根据 `seq_id` 返回不同 token，方便调试多请求行为。

fake backend 测通后，再进入真实 backend：

```text
CppRuntimeModelRunner
  option A: 调 Python 版 ModelRunner
  option B: libtorch 后端
  option C: 原生 CUDA 后端
```

建议先选 A 或 B，再选 C。这样 C++ scheduler 可以先驱动真实推理，不必立刻重写所有模型 kernel。

## 测试策略

### 单元测试

需要测试：

- Sequence append 和 block 计算。
- Block allocate/deallocate。
- Prefix cache 复用。
- 引用计数。
- Scheduler prefill batching。
- Chunked prefill。
- Decode batching。
- Preemption。
- EOS 结束。
- `max_tokens` 结束。

### Python 行为对齐测试

第一阶段把 `alpha/` 中的 Python 实现作为行为 oracle。

同样的 synthetic prompts 和 config 下，比较：

- scheduled sequence ids
- `is_prefill`
- `num_scheduled_tokens`
- block tables
- cached token counts
- fake sampling 下的最终 completion token ids

这样可以防止迁移到 C++ 后 scheduler 行为发生隐性漂移。

### Benchmark 测试

优化前先加 baseline：

- scheduler-only synthetic workload
- block allocation/deallocation workload
- fake runner engine loop
- 后续真实模型 backend 的端到端 generation

报告时优先给加速比，同时保留必要的 profile 解释。

## 里程碑

### Milestone 1: C++ Scheduler Core

交付：

- C++ `Sequence`
- C++ `BlockManager`
- C++ `Scheduler`
- 核心调度行为单元测试

成功标准：

- C++ scheduler 在 synthetic cases 上和 Python scheduler 行为一致。

### Milestone 2: Fake Generation Engine

交付：

- C++ `LLMEngine`
- `ModelRunner` interface
- `FakeModelRunner`
- 端到端 fake `generate()` demo

成功标准：

- 多个 prompts 能在 C++ engine loop 中被调度、decode、完成并返回。

### Milestone 3: Real Backend Bridge

交付：

- 一个能让 C++ runtime 调用真实模型执行的 backend bridge。
- backend 保持在 `ModelRunner` 接口后面。

成功标准：

- C++ scheduler 可以驱动某一个模型路径完成真实 token generation。

### Milestone 4: Runtime Optimization

针对性优化：

- 减少 Python object overhead。
- 减少重复 tensor 构造开销。
- 优化 block table 和 slot mapping 构造。
- 把 scheduler/runtime overhead 与 GPU/model time 分开测量。

成功标准：

- 同 workload 下相对原 Python runtime 有可测加速；或者 profile 清楚说明剩余时间主要被 GPU/model execution 主导。

## 推荐实现顺序

1. 新增 C++ 项目骨架。
2. 实现 `SamplingParams`、`SequenceStatus`、`Sequence`。
3. 实现 `Block` 和 `BlockManager`。
4. 实现 `Scheduler`。
5. 添加 scheduler 和 block-manager 测试。
6. 实现 `ModelRunner` 接口和 `FakeModelRunner`。
7. 实现 fake `LLMEngine`。
8. 添加 Python parity tests 或 trace comparison scripts。
9. 添加 scheduler/runtime benchmarks。
10. fake engine 正确后，再设计真实 backend bridge。

## 实施前需要确定的选择

这些选择可以在写 implementation plan 前确定：

- 测试框架：GoogleTest、Catch2，或先用 assert-based tests。
- 构建方式：独立 `cpp/CMakeLists.txt`，还是和 Python extension build 集成。
- 后端桥接：先 Python bridge，还是先 libtorch。
- 哈希策略：和 Python xxhash 完全对齐，还是第一阶段用 C++ 确定性哈希。

推荐默认值：

- 使用 CMake。
- 如果依赖安装方便，用 GoogleTest 或 Catch2；否则先用 assert-based executable tests。
- 如果引入依赖不麻烦，prefix cache 哈希使用 xxhash 对齐 Python。
- 等 scheduler 和 fake engine 稳定后，再引入 libtorch。
