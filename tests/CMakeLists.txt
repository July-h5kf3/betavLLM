set(BETAVLLM_TEST_MODEL_DIR "" CACHE PATH "Optional model directory for model/tokenizer tests")

add_executable(test_model_loader
    test_model_loader.cpp
)
target_link_libraries(test_model_loader PRIVATE
    betavllm_core
)

add_executable(test_tokenizer
    test_tokenizer.cpp
)
target_link_libraries(test_tokenizer PRIVATE
    betavllm_core
)

add_executable(test_embedding_kernel
    test_embedding_kernel.cu
)
target_link_libraries(test_embedding_kernel PRIVATE
    betavllm_core
)

add_executable(test_rmsnorm_kernel
    test_rmsnorm_kernel.cu
)
target_link_libraries(test_rmsnorm_kernel PRIVATE
    betavllm_core
)

add_test(NAME embedding_kernel COMMAND test_embedding_kernel)
add_test(NAME rmsnorm_kernel COMMAND test_rmsnorm_kernel)

if(BETAVLLM_TEST_MODEL_DIR)
    add_test(NAME model_loader COMMAND test_model_loader ${BETAVLLM_TEST_MODEL_DIR})
    add_test(NAME tokenizer COMMAND test_tokenizer ${BETAVLLM_TEST_MODEL_DIR})
endif()
