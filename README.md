# 从0开始用C++搭建一个推理引擎

## 环境要求

- CMake 3.19+
- 支持 CUDA 的 NVIDIA GPU
- CUDA Toolkit
- C++17 编译器
- Rust/Cargo，用于构建 `third_party/tokenizers-cpp`

## 构建

推荐直接使用项目脚本：

```bash
./runall.sh kernels
```

这条命令会自动执行 CMake configure、build，并运行不依赖模型目录的 CUDA kernel 测试。

也可以手动构建：

```bash
cmake -S . -B build -DBUILD_BETAVLLM_TESTS=ON
cmake --build build -j
```

## 基本测试指令

运行所有不依赖模型文件的 kernel 测试：

```bash
./runall.sh kernels
```

单独测试 Embedding kernel：

```bash
./runall.sh embedding
```

单独测试 RMSNorm kernel：

```bash
./runall.sh rmsnorm
```

如果本地有 Hugging Face 模型目录，可以运行权重加载和 tokenizer 测试：

```bash
./runall.sh model_loader /path/to/model
./runall.sh tokenizer /path/to/model
```

也可以一次运行全部测试：

```bash
./runall.sh all /path/to/model
```

使用 CTest 运行已注册测试：

```bash
ctest --test-dir build --output-on-failure
```

## 运行主程序

```bash
./build/betavllm /path/to/model
```

其中 `/path/to/model` 应该是包含模型权重、配置和 tokenizer 文件的模型目录。
