#include "tokenizer.hpp"
#include "test_utils.hpp"

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <model_dir> [prompt]\n";
        return 2;
    }

    const std::string model_dir = argv[1];
    const std::string prompt = argc >= 3 ? argv[2] : "Hello world!";

    betavllm::HFTokenizer tokenizer(model_dir);
    std::vector<int32_t> ids = tokenizer.encode(prompt);
    std::string decoded = tokenizer.decode(ids);

    std::cout << "prompt: " << prompt << "\n";
    std::cout << "ids:";
    for (int32_t id : ids)
    {
        std::cout << " " << id;
    }
    std::cout << "\n";
    std::cout << "decoded: " << decoded << "\n";
    std::cout << "vocab_size: " << tokenizer.vocab_size() << "\n";
    std::cout << "bos/eos/eot: "
              << tokenizer.bos_id() << "/"
              << tokenizer.eos_id() << "/"
              << tokenizer.eot_id() << "\n";

    if (ids.empty() || decoded.empty() || tokenizer.vocab_size() == 0)
    {
        std::cerr << betavllm::test::passFail(false)
                  << ": tokenizer produced empty output\n";
        return 1;
    }

    std::cout << betavllm::test::passFail(true)
              << ": tokenizer loaded and round-tripped a prompt\n";
    return 0;
}
