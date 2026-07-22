cmake_minimum_required(VERSION 3.14)
project(mollm VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckCXXSourceCompiles)
include(CheckCXXSourceRuns)

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------

option(MOLLM_BUILD_TESTS "Build tests" ON)
option(MOLLM_BUILD_EXAMPLE "Build example CLI" ON)
option(MOLLM_ARM_NATIVE "Build ARM64 kernels with -mcpu=native for local benchmarking" OFF)
option(MOLLM_METAL "Build the Metal GPU backend (Apple only)" OFF)
set(MOLLM_ARM_I8MM "AUTO" CACHE STRING
    "ARM64 i8mm kernel target mode: AUTO, ON, or OFF")
set_property(CACHE MOLLM_ARM_I8MM PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${MOLLM_ARM_I8MM}" MOLLM_ARM_I8MM_MODE)
if(NOT MOLLM_ARM_I8MM_MODE MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "MOLLM_ARM_I8MM must be AUTO, ON, or OFF (got '${MOLLM_ARM_I8MM}')")
endif()

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
    kernels/moe.cpp
    kernels/moe_ssd.cpp
    kernels/rwkv.cpp
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
    kernels/moe.h
    kernels/moe_ssd.h
    kernels/rwkv.h
    kernels/threading.h
    graph/buffer_pool.h
    graph/mmap_file.h
    graph/graph.h
    graph/io.cpp   # graph_load/graph_save declared in graph.h
    graph/execute.h
)

add_library(mollm_lib STATIC ${SOURCES})
target_include_directories(mollm_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann
)
target_link_libraries(mollm_lib PUBLIC)

# ---------------------------------------------------------------------------
# platform detection for SIMD
# ---------------------------------------------------------------------------

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    message(STATUS "Target: ARM64 (NEON enabled)")
    target_compile_definitions(mollm_lib PRIVATE __ARM_NEON=1)
    if(MOLLM_ARM_NATIVE)
        message(STATUS "ARM64 native target enabled (-mcpu=native)")
        target_compile_options(mollm_lib PRIVATE -mcpu=native)
    elseif(NOT MOLLM_ARM_I8MM_MODE STREQUAL "OFF")
        set(MOLLM_ARM_I8MM_FLAGS "-march=armv8.6-a+i8mm+fp16+fp16fml")
        set(_mollm_saved_required_flags "${CMAKE_REQUIRED_FLAGS}")
        set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${MOLLM_ARM_I8MM_FLAGS}")
        set(_mollm_i8mm_probe_source "
            #include <arm_neon.h>
            #ifndef __ARM_FEATURE_MATMUL_INT8
            #error __ARM_FEATURE_MATMUL_INT8 is not defined
            #endif
            int main() {
                int32x4_t acc = vdupq_n_s32(0);
                int8x16_t a = vdupq_n_s8(1);
                int8x16_t b = vdupq_n_s8(1);
                acc = vmmlaq_s32(acc, a, b);
                return vaddvq_s32(acc) == 0;
            }
        ")
        check_cxx_source_compiles("${_mollm_i8mm_probe_source}" MOLLM_ARM_I8MM_COMPILES)
        if(MOLLM_ARM_I8MM_COMPILES AND NOT CMAKE_CROSSCOMPILING)
            check_cxx_source_runs("${_mollm_i8mm_probe_source}" MOLLM_ARM_I8MM_RUNS)
        elseif(MOLLM_ARM_I8MM_COMPILES AND CMAKE_CROSSCOMPILING)
            set(MOLLM_ARM_I8MM_RUNS TRUE)
            message(STATUS "ARM64 i8mm runtime probe skipped while cross-compiling")
        else()
            set(MOLLM_ARM_I8MM_RUNS FALSE)
        endif()
        set(CMAKE_REQUIRED_FLAGS "${_mollm_saved_required_flags}")

        if(MOLLM_ARM_I8MM_COMPILES AND MOLLM_ARM_I8MM_RUNS)
            message(STATUS "ARM64 i8mm target enabled (${MOLLM_ARM_I8MM_FLAGS})")
            target_compile_options(mollm_lib PRIVATE ${MOLLM_ARM_I8MM_FLAGS})
        elseif(MOLLM_ARM_I8MM_MODE STREQUAL "ON")
            message(FATAL_ERROR
                "MOLLM_ARM_I8MM=ON requested, but compiler/target does not support "
                "${MOLLM_ARM_I8MM_FLAGS} with __ARM_FEATURE_MATMUL_INT8")
        else()
            message(STATUS "ARM64 i8mm target not available; using DOTPROD/NEON kernels")
        endif()
    else()
        message(STATUS "ARM64 i8mm target disabled by MOLLM_ARM_I8MM=OFF")
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    message(STATUS "Target: ARM32")
    target_compile_definitions(mollm_lib PRIVATE __ARM_NEON=1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    message(STATUS "Target: x86_64 (scalar fallback)")
else()
    message(STATUS "Target: ${CMAKE_SYSTEM_PROCESSOR} (scalar fallback)")
endif()

# Apple-specific: link Accelerate for optional BLAS (future use)
if(APPLE)
    find_library(ACCELERATE_LIBRARY Accelerate)
    if(ACCELERATE_LIBRARY)
        target_link_libraries(mollm_lib PUBLIC ${ACCELERATE_LIBRARY})
    endif()
endif()

# ---------------------------------------------------------------------------
# Metal GPU backend (optional, Apple only)
# ---------------------------------------------------------------------------

if(MOLLM_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "MOLLM_METAL requires an Apple platform")
    endif()
    enable_language(OBJCXX)

    find_library(METAL_LIBRARY Metal REQUIRED)
    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)

    # Compile the shader library: .metal -> .air -> default.metallib
    set(MOLLM_METAL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/mollm.metal)
    set(MOLLM_METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/mollm.air)
    set(MOLLM_METALLIB  ${CMAKE_CURRENT_BINARY_DIR}/default.metallib)

    # Probe: does the toolchain support the Metal 4 tensor API (mpp::tensor_ops)?
    # If so, compile with -std=metal4.0 -DMOLLM_METAL_TENSOR so the fast
    # tensor-API GEMM kernel is built. The backend still checks GPU family at
    # runtime (M5/A19+) before dispatching to it, falling back to simdgroup.
    set(MOLLM_METAL_FLAGS "")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E echo "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nkernel void _p(){}"
        OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/_tensor_probe.metal)
    execute_process(
        COMMAND xcrun -sdk macosx metal -std=metal4.0 -c
                ${CMAKE_CURRENT_BINARY_DIR}/_tensor_probe.metal
                -o ${CMAKE_CURRENT_BINARY_DIR}/_tensor_probe.air
        RESULT_VARIABLE MOLLM_METAL_TENSOR_RC
        OUTPUT_QUIET ERROR_QUIET)
    if(MOLLM_METAL_TENSOR_RC EQUAL 0)
        message(STATUS "Metal 4 tensor API available — enabling tensor GEMM kernel")
        set(MOLLM_METAL_FLAGS -std=metal4.0 -DMOLLM_METAL_TENSOR)
        target_compile_definitions(mollm_lib PUBLIC MOLLM_METAL_TENSOR=1)
    else()
        message(STATUS "Metal 4 tensor API not available — simdgroup GEMM only")
    endif()

    add_custom_command(
        OUTPUT ${MOLLM_METAL_AIR}
        COMMAND xcrun -sdk macosx metal ${MOLLM_METAL_FLAGS} -c ${MOLLM_METAL_SRC}
                -I ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal
                -o ${MOLLM_METAL_AIR}
        DEPENDS ${MOLLM_METAL_SRC}
                ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/metal_common.h
        COMMENT "Compiling Metal shaders (mollm.metal -> mollm.air)"
        VERBATIM)
    add_custom_command(
        OUTPUT ${MOLLM_METALLIB}
        COMMAND xcrun -sdk macosx metallib ${MOLLM_METAL_AIR} -o ${MOLLM_METALLIB}
        DEPENDS ${MOLLM_METAL_AIR}
        COMMENT "Linking Metal library (default.metallib)"
        VERBATIM)
    add_custom_target(mollm_metallib ALL DEPENDS ${MOLLM_METALLIB})
    add_dependencies(mollm_lib mollm_metallib)

    target_sources(mollm_lib PRIVATE
        engine/metal_backend.mm
        graph/metal_pool.mm)
    # The Objective-C++ sources use ARC bridge-cast ownership semantics.
    set_source_files_properties(
        engine/metal_backend.mm graph/metal_pool.mm
        PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    target_compile_definitions(mollm_lib PUBLIC
        MOLLM_METAL=1
        MOLLM_METALLIB_PATH="${MOLLM_METALLIB}")
    target_link_libraries(mollm_lib PUBLIC ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})
    message(STATUS "Metal backend enabled (metallib: ${MOLLM_METALLIB})")
endif()

# ---------------------------------------------------------------------------
# example CLI
# ---------------------------------------------------------------------------

if(MOLLM_BUILD_EXAMPLE)
    add_executable(mollm_chat
        examples/main.cpp
        examples/cli_common.cpp
    )
    target_link_libraries(mollm_chat PRIVATE mollm_lib)

    add_executable(mollm_bench
        examples/bench.cpp
        examples/cli_common.cpp
    )
    target_link_libraries(mollm_bench PRIVATE mollm_lib)

    add_executable(mollm_ppl
        examples/ppl.cpp
    )
    target_link_libraries(mollm_ppl PRIVATE mollm_lib)

    if(UNIX)
        add_executable(mollm_server
            examples/server.cpp
            examples/cli_common.cpp
        )
        target_link_libraries(mollm_server PRIVATE mollm_lib)
    endif()
endif()

add_executable(mollm-quantize tools/quantize_weight.cpp)
target_link_libraries(mollm-quantize PRIVATE Threads::Threads)

# Backwards-compatible build target name used in older docs/scripts. The
# produced binary is build/mollm-quantize.
add_custom_target(mollm_quantize_weight DEPENDS mollm-quantize)

# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

if(MOLLM_BUILD_TESTS)
    enable_testing()

    function(add_mollm_test name src)
        add_executable(${name} ${src})
        target_link_libraries(${name} PRIVATE mollm_lib)
        add_test(NAME ${name} COMMAND ${name})
    endfunction()

    add_mollm_test(test_tensor        tests/test_tensor.cpp)
    add_mollm_test(test_buffer_pool   tests/test_buffer_pool.cpp)
    add_mollm_test(test_mmap_file     tests/test_mmap_file.cpp)
    add_mollm_test(test_graph         tests/test_graph.cpp)
    add_mollm_test(test_io            tests/test_io.cpp)
    add_mollm_test(test_execute       tests/test_execute.cpp)
    add_mollm_test(test_matmul        tests/test_matmul.cpp)
    add_mollm_test(test_moe           tests/test_moe.cpp)
    add_mollm_test(test_moe_ssd       tests/test_moe_ssd.cpp)
    add_mollm_test(bench_matmul       tests/bench_matmul.cpp)
    add_mollm_test(bench_sdpa         tests/bench_sdpa.cpp)
    add_mollm_test(bench_recurrent    tests/bench_recurrent.cpp)
    add_mollm_test(test_norm          tests/test_norm.cpp)
    add_mollm_test(test_rope          tests/test_rope.cpp)
    add_mollm_test(test_attention     tests/test_attention.cpp)
    add_mollm_test(test_gdn           tests/test_gdn.cpp)
    add_mollm_test(test_rwkv_e2e      tests/test_rwkv_e2e.cpp)
    add_mollm_test(test_engine        tests/test_engine.cpp)
    add_mollm_test(test_tokenizer     tests/test_tokenizer.cpp)
    add_mollm_test(test_e2e           tests/test_e2e.cpp)
    add_mollm_test(test_quantized_e2e tests/test_quantized_e2e.cpp)
    add_mollm_test(test_thread_pool_stress tests/test_thread_pool_stress.cpp)

    if(MOLLM_METAL)
        add_mollm_test(test_metal_ops tests/test_metal_ops.cpp)
    endif()
endif()

# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------

install(TARGETS mollm_lib ARCHIVE DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/mollm)
