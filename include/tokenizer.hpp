#pragma once

#include <memory>
#include <cstdint>
#include <string>
#include <vector>

namespace tokenizers
{
    class Tokenizer;
}
namespace betavllm 
{
class HFTokenizer 
{
    public:
        explicit HFTokenizer(const std::string& model_dir);
        ~HFTokenizer();
        std::vector<int32_t> encode(const std::string& text) const;
        std::string decode(const std::vector<int32_t>& ids) const;
        
        int32_t bos_id() const;
        int32_t eos_id() const;
        int32_t eot_id() const;
        size_t vocab_size() const;
    private:
        std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
};
}
