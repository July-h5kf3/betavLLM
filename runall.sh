#!/usr/bin/env bash
set -u

GREEN=$'\033[32m'
RED=$'\033[31m'
RESET=$'\033[0m'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
TARGET="${1:-all}"
MODEL_DIR="${2:-${BETAVLLM_MODEL_DIR:-}}"

green() {
    printf "%s%s%s\n" "${GREEN}" "$1" "${RESET}"
}

red() {
    printf "%s%s%s\n" "${RED}" "$1" "${RESET}"
}

usage() {
    cat <<EOF
Usage:
  ./runall.sh [all|model_loader|tokenizer|embedding|rmsnorm|kernels] [model_dir]

Examples:
  ./runall.sh all /path/to/model
  ./runall.sh embedding
  ./runall.sh rmsnorm
  BETAVLLM_MODEL_DIR=/path/to/model ./runall.sh tokenizer

Environment:
  BUILD_DIR              CMake build directory, default: ./build
  BETAVLLM_MODEL_DIR     Model directory used by model_loader/tokenizer tests
EOF
}

if [[ "${TARGET}" == "-h" || "${TARGET}" == "--help" ]]; then
    usage
    exit 0
fi

if ! command -v cmake >/dev/null 2>&1; then
    red "FAIL: cmake not found"
    exit 1
fi

cmake_args=(-S "${ROOT_DIR}" -B "${BUILD_DIR}" -DBUILD_BETAVLLM_TESTS=ON)
if [[ -n "${MODEL_DIR}" ]]; then
    cmake_args+=("-DBETAVLLM_TEST_MODEL_DIR=${MODEL_DIR}")
fi

if ! cmake "${cmake_args[@]}"; then
    red "FAIL: cmake configure failed"
    exit 1
fi

if ! cmake --build "${BUILD_DIR}" -j; then
    red "FAIL: cmake build failed"
    exit 1
fi

run_test() {
    local name="$1"
    shift

    printf "\n==> %s\n" "${name}"
    if "$@"; then
        green "PASS: ${name}"
        return 0
    fi

    red "FAIL: ${name}"
    return 1
}

run_model_loader() {
    if [[ -z "${MODEL_DIR}" ]]; then
        red "FAIL: model_loader requires model_dir"
        return 1
    fi
    run_test "model_loader" "${BUILD_DIR}/tests/test_model_loader" "${MODEL_DIR}"
}

run_tokenizer() {
    if [[ -z "${MODEL_DIR}" ]]; then
        red "FAIL: tokenizer requires model_dir"
        return 1
    fi
    run_test "tokenizer" "${BUILD_DIR}/tests/test_tokenizer" "${MODEL_DIR}"
}

run_embedding() {
    run_test "embedding" "${BUILD_DIR}/tests/test_embedding_kernel"
}

run_rmsnorm() {
    run_test "rmsnorm" "${BUILD_DIR}/tests/test_rmsnorm_kernel"
}

status=0
case "${TARGET}" in
    all)
        run_model_loader || status=1
        run_tokenizer || status=1
        run_embedding || status=1
        run_rmsnorm || status=1
        ;;
    kernels)
        run_embedding || status=1
        run_rmsnorm || status=1
        ;;
    model_loader|model|weights)
        run_model_loader || status=1
        ;;
    tokenizer|tok)
        run_tokenizer || status=1
        ;;
    embedding|embed)
        run_embedding || status=1
        ;;
    rmsnorm|rms)
        run_rmsnorm || status=1
        ;;
    *)
        red "FAIL: unknown target '${TARGET}'"
        usage
        exit 1
        ;;
esac

if [[ "${status}" -eq 0 ]]; then
    green "ALL REQUESTED TESTS PASSED"
else
    red "SOME REQUESTED TESTS FAILED"
fi

exit "${status}"
