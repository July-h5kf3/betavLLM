#include "tokenizer.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "tokenizers_cpp.h"

namespace betavllm 
{
    static std::string readFile(const std::string& path)
    {
        std::ifstream file(path, std::ios::binary);
        std::ostream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }
    HFTokenizer::HFTokenizer(const std::string& model_dir)
    {
        std::string tokenizer_path = model_dir + "/tokenizeer.json";
        std::string blob = readFile(tokenizer_path);
        tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(blob);
        if (!tokenizer_)
        {
            throw std::runtime_error("Failed to load!");
        }
    }

    std::vector<int32_t> HFTokenizer::encode(const std::string& text) const
    {
        return tokenizer_->Encode(text);
    }

    std::string HFTokenizer::decode(const std::vector<int32_t>& ids) const
    {
        return tokenizer_->Decode(ids);
    }

    int32_t HFTokenizer::bos_id() const
    {
        return tokenizer_ ->TokenToId("<|begin_of_text|>");
    }
    int32_t HFTokenizer::eos_id() const 
    {
    return tokenizer_->TokenToId("<|end_of_text|>");
    }

    int32_t HFTokenizer::eot_id() const 
    {
    return tokenizer_->TokenToId("<|eot_id|>");
    }

    size_t HFTokenizer::vocab_size() const 
    {
    return tokenizer_->GetVocabSize();
    }
}