cmake_minimum_required(VERSION 3.14)
project(moe-trace CXX)

# Build against an existing llama.cpp checkout and build tree:
#   cmake -B build -DLLAMA_DIR=$HOME/src/llama.cpp && cmake --build build
set(LLAMA_DIR "$ENV{HOME}/src/llama.cpp" CACHE PATH "llama.cpp source tree")
set(LLAMA_BUILD "${LLAMA_DIR}/build" CACHE PATH "llama.cpp build tree")

add_executable(moe-trace moe-trace.cpp)
target_compile_features(moe-trace PRIVATE cxx_std_17)
target_include_directories(moe-trace PRIVATE
    ${LLAMA_DIR}/include ${LLAMA_DIR}/common ${LLAMA_DIR}/ggml/include ${LLAMA_DIR}/vendor)
target_link_directories(moe-trace PRIVATE ${LLAMA_BUILD}/bin ${LLAMA_BUILD}/common)
target_link_libraries(moe-trace PRIVATE llama-common llama-common-base llama ggml ggml-base)
set_target_properties(moe-trace PROPERTIES BUILD_RPATH "${LLAMA_BUILD}/bin")
