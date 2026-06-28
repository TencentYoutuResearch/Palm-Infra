cmake_minimum_required(VERSION 3.14)
project(mlllm VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------

option(MLLLM_BUILD_TESTS "Build tests" ON)
option(MLLLM_BUILD_EXAMPLE "Build example CLI" ON)

# ---------------------------------------------------------------------------
# library sources
# ---------------------------------------------------------------------------

set(SOURCES
    # graph
    graph/buffer_pool.cpp
    graph/mmap_file.cpp
    graph/io.cpp
    graph/execute.cpp
    # kernels
    kernels/matmul.cpp
    kernels/norm.cpp
    kernels/rope.cpp
    kernels/attention.cpp
    kernels/gdn.cpp
    kernels/gdn_decode.cpp
    kernels/gdn_prefill.cpp
    kernels/threading.cpp
    # engine
    engine/engine.cpp
    engine/tokenizer.cpp
)

set(HEADERS
    kernels/tensor.h
    kernels/matmul.h
    kernels/norm.h
    kernels/rope.h
    kernels/attention.h
    kernels/threading.h
    graph/buffer_pool.h
    graph/mmap_file.h
    graph/graph.h
    graph/io.cpp   # graph_load/graph_save declared in graph.h
    graph/execute.h
)

add_library(mlllm_lib STATIC ${SOURCES})
target_include_directories(mlllm_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann
)
target_link_libraries(mlllm_lib PUBLIC)

# ---------------------------------------------------------------------------
# platform detection for SIMD
# ---------------------------------------------------------------------------

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    message(STATUS "Target: ARM64 (NEON enabled)")
    target_compile_definitions(mlllm_lib PRIVATE __ARM_NEON=1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    message(STATUS "Target: ARM32")
    target_compile_definitions(mlllm_lib PRIVATE __ARM_NEON=1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    message(STATUS "Target: x86_64 (scalar fallback)")
else()
    message(STATUS "Target: ${CMAKE_SYSTEM_PROCESSOR} (scalar fallback)")
endif()

# Apple-specific: link Accelerate for optional BLAS (future use)
if(APPLE)
    find_library(ACCELERATE_LIBRARY Accelerate)
    if(ACCELERATE_LIBRARY)
        target_link_libraries(mlllm_lib PUBLIC ${ACCELERATE_LIBRARY})
    endif()
endif()

# ---------------------------------------------------------------------------
# example CLI
# ---------------------------------------------------------------------------

if(MLLLM_BUILD_EXAMPLE)
    add_executable(mlllm_chat
        examples/main.cpp
        examples/cli_common.cpp
    )
    target_link_libraries(mlllm_chat PRIVATE mlllm_lib)

    add_executable(mlllm_bench
        examples/bench.cpp
        examples/cli_common.cpp
    )
    target_link_libraries(mlllm_bench PRIVATE mlllm_lib)
endif()

# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

if(MLLLM_BUILD_TESTS)
    enable_testing()

    function(add_mlllm_test name src)
        add_executable(${name} ${src})
        target_link_libraries(${name} PRIVATE mlllm_lib)
        add_test(NAME ${name} COMMAND ${name})
    endfunction()

    add_mlllm_test(test_tensor        tests/test_tensor.cpp)
    add_mlllm_test(test_buffer_pool   tests/test_buffer_pool.cpp)
    add_mlllm_test(test_mmap_file     tests/test_mmap_file.cpp)
    add_mlllm_test(test_graph         tests/test_graph.cpp)
    add_mlllm_test(test_io            tests/test_io.cpp)
    add_mlllm_test(test_execute       tests/test_execute.cpp)
    add_mlllm_test(test_matmul        tests/test_matmul.cpp)
    add_mlllm_test(bench_matmul       tests/bench_matmul.cpp)
    add_mlllm_test(bench_sdpa         tests/bench_sdpa.cpp)
    add_mlllm_test(test_norm          tests/test_norm.cpp)
    add_mlllm_test(test_rope          tests/test_rope.cpp)
    add_mlllm_test(test_attention     tests/test_attention.cpp)
    add_mlllm_test(test_gdn           tests/test_gdn.cpp)
    add_mlllm_test(test_engine        tests/test_engine.cpp)
    add_mlllm_test(test_tokenizer     tests/test_tokenizer.cpp)
    add_mlllm_test(test_e2e           tests/test_e2e.cpp)
endif()

# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------

install(TARGETS mlllm_lib ARCHIVE DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/mlllm)
