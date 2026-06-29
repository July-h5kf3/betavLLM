#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <fstream>
#include "model_config.hpp"
#include "json.hpp"

using json = nlohmann::json;
#pragma once

struct Weights
{
    __nv_bfloat16 *embed_tokens;
    __nv_bfloat16 *input_layernorm[N_LAYERS];
    __nv_bfloat16 *mlp_gate_proj[N_LAYERS];
    __nv_bfloat16 *mlp_up_proj[N_LAYERS];
    __nv_bfloat16 *mlp_down_proj[N_LAYERS];
    __nv_bfloat16 *post_attn_layernorms[N_LAYERS];
    __nv_bfloat16 *w_k[N_LAYERS];
    __nv_bfloat16 *w_o[N_LAYERS];
    __nv_bfloat16 *w_q[N_LAYERS];
    __nv_bfloat16 *w_v[N_LAYERS];
    __nv_bfloat16 *norm;
};

int loadWeights(const std::string& model_dir , Weights &weights)
{
    /*
        这里先暂时硬编码为Llama3.2的权重地址，后续可以改为args参数
    */
    std::string model_path = model_dir + "/model.safetensors";
    std::ifstream safetensors_file(model_path,std::ios_base::binary);
    if(!safetensors_file.is_open())
    {
        std::cout << "Can't open model.safetensors file: "<<model_path<<"\n";
        safetensors_file.close();
        return 1;
    }
    if(DEBUG)
    {
        std::cout<<"[DEBUG] Loading weights from: "<< model_path<<std::endl;
    }

    uint64_t header_size;
    safetensors_file.read(reinterpret_cast<char *>(&header_size), 8);

    if(DEBUG)
    {
        std::cout<<"[DEBUG] Safetensors header size: "<<header_size<<std::endl;
    }

    std::string header;
    header.resize(header_size);
    safetensors_file.read(header.data(), header_size);

    std::unordered_map<std::string, uint64_t> offsets;
    json header_json = json::parse(header);
    uint64_t max_offsets = 0;

    int tensor_count = 0;
    for(auto &[key,value] : header_json.items())
    {
        if(key == "__metadata__")
        {
            continue;
        }
        tensor_count++;

        uint64_t offsets_begin = value["data_offsets"][0].get<uint64_t>();
        uint64_t offset_end = value["data_offsets"][1].get<uint64_t>();

        offsets[key] = offsets_begin;
        max_offsets = std::max(max_offsets,offset_end);
    }

    if(DEBUG)
    {
        std::cout<<"[DEBUG] Tensor Count: "<<tensor_count<<std::endl;
        std::cout<<"[DEBUG] Total weight MB: "<<max_offsets / B_TO_MB <<std::endl;
    }

    void *model_weights;
    cudaMalloc(&model_weights, max_offsets);

    std::vector<char> model_weights_cpu;
    model_weights_cpu.resize(max_offsets);
    safetensors_file.read(model_weights_cpu.data(),max_offsets);

    cudaMemcpy(model_weights, model_weights_cpu.data(), max_offsets, cudaMemcpyHostToDevice);
    safetensors_file.close();

    weights.embed_tokens = (__nv_bfloat16 *)((char *)model_weights + offsets["model.embed_tokens.weight"]);
    weights.norm = (__nv_bfloat16 *)((char *)model_weights + offsets["model.norm.weight"]);
    for(int i = 0;i < N_LAYERS;i++)
    {
        std::string layer_prefix = "model.layers." + std::to_string(i) + ".";

        weights.input_layernorm[i] = (__nv_bfloat16 *)((char *)model_weights + offsets[layer_prefix + "input_layernorm.weight"]);
        weights.post_attn_layernorms[i] = (__nv_bfloat16 *)((char *)model_weights + offsets[layer_prefix + "post_attention_layernorm.weight"]);

        weights.mlp_gate_proj[i] = (__nv_bfloat16 *)((char *)model_weights + offsets[layer_prefix + "mlp.gate_proj.weight"]);
        weights.mlp_up_proj[i] = (__nv_bfloat16 *)((char *)model_weights + offsets[layer_prefix + "mlp.up_proj.weight"]);
        weights.mlp_down_proj[i] = (__nv_bfloat16 *)((char *)model_weights + offsets[layer_prefix + "mlp.down_proj.weight"]);

        weights.w_q[i] = (__nv_bfloat16 *)((char *)model_weights + offsets[layer_prefix + "self_attn.q_proj.weight"]);
        weights.w_k[i] = (__nv_bfloat16 *)((char *)model_weights + offsets[layer_prefix + "self_attn.k_proj.weight"]);
        weights.w_v[i] = (__nv_bfloat16 *)((char *)model_weights + offsets[layer_prefix + "self_attn.v_proj.weight"]);
        weights.w_o[i] = (__nv_bfloat16 *)((char *)model_weights + offsets[layer_prefix + "self_attn.o_proj.weight"]);
    }
    return 0;
}
