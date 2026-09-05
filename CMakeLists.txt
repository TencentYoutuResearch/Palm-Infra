cmake_minimum_required(VERSION 3.14)
project(mollm VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckCXXSourceCompiles)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------

option(MOLLM_BUILD_TESTS "Build tests" ON)
option(MOLLM_BUILD_EXAMPLE "Build example CLI" ON)
option(MOLLM_ARM_NATIVE "Build ARM64 kernels with -mcpu=native for local benchmarking" OFF)
option(MOLLM_METAL "Build the Metal GPU backend (Apple only)" OFF)
option(MOLLM_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(MOLLM_CUDA "Build the CUDA correctness backend" OFF)
set(MOLLM_ARM_I8MM "AUTO" CACHE STRING
    "ARM64 i8mm kernel target mode: AUTO, ON, or OFF")
set_property(CACHE MOLLM_ARM_I8MM PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${MOLLM_ARM_I8MM}" MOLLM_ARM_I8MM_MODE)
if(NOT MOLLM_ARM_I8MM_MODE MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "MOLLM_ARM_I8MM must be AUTO, ON, or OFF (got '${MOLLM_ARM_I8MM}')")
endif()

if(MOLLM_SANITIZERS)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(FATAL_ERROR "MOLLM_SANITIZERS requires GCC or Clang")
    endif()
    set(MOLLM_SANITIZER_FLAGS
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
        -fno-sanitize-recover=all)
    add_compile_options(${MOLLM_SANITIZER_FLAGS})
    add_link_options(${MOLLM_SANITIZER_FLAGS})
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64|arm")
    set(MOLLM_CPU_PLATFORM_SOURCE kernels/cpu_platform_arm.cpp)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(MOLLM_CPU_PLATFORM_SOURCE
        kernels/cpu_platform_x86.cpp
        kernels/matmul_x86_avx2.cpp
        kernels/matmul_x86_avx512.cpp
        kernels/matmul_x86_vnni.cpp
        kernels/gdn_x86_avx512.cpp
        kernels/attention_x86_avx512.cpp)
else()
    set(MOLLM_CPU_PLATFORM_SOURCE kernels/cpu_platform_scalar.cpp)
endif()

# ---------------------------------------------------------------------------
# library sources
# ---------------------------------------------------------------------------

set(SOURCES
    # graph
    graph/buffer_pool.cpp
    graph/mmap_file.cpp
    graph/io.cpp
    graph/cpu_backend.cpp
    graph/execute.cpp
    # kernels
    kernels/matmul.cpp
    kernels/matmul_dense.cpp
    kernels/matmul_pack.cpp
    kernels/matmul_profile.cpp
    kernels/matmul_quant.cpp
    kernels/matmul_sparse.cpp
    kernels/matmul_w4.cpp
    kernels/matmul_w8.cpp
    kernels/matmul_mxfp4.cpp
    kernels/matmul_nvfp4.cpp
    ${MOLLM_CPU_PLATFORM_SOURCE}
    kernels/elementwise.cpp
    kernels/layout.cpp
    kernels/norm.cpp
    kernels/rope.cpp
    kernels/attention.cpp
    kernels/shortconv.cpp
    kernels/gdn.cpp
    kernels/gdn_decode.cpp
    kernels/gdn_prefill.cpp
    kernels/gated_residual.cpp
    kernels/ple.cpp
    kernels/deepseek_v4_attention.cpp
    kernels/hyper_connection.cpp
    kernels/moe.cpp
    kernels/moe_routing.cpp
    kernels/moe_ssd.cpp
    kernels/moe_ssd_io.cpp
    kernels/moe_ssd_predict.cpp
    kernels/trace.cpp
    kernels/rwkv.cpp
    kernels/threading.cpp
    # engine
    engine/byte_ranges.cpp
    engine/engine.cpp
    engine/input_prep.cpp
    engine/model_loader.cpp
    engine/package.cpp
    engine/sampler.cpp
    engine/tokenizer.cpp
    engine/tokenizer_rwkv.cpp
    engine/vision.cpp
    engine/weight_metadata.cpp
)

set(PUBLIC_ENGINE_HEADERS
    engine/accelerator_backend.h
    engine/backend.h
    engine/engine.h
    engine/sampler.h
    engine/tokenizer.h
)

set(PUBLIC_GRAPH_HEADERS
    graph/buffer_pool.h
    graph/mmap_file.h
    graph/graph.h
    graph/execute.h
)

set(PUBLIC_KERNEL_HEADERS
    kernels/cpu_platform.h
    kernels/moe_ssd.h
    kernels/prepared_weight.h
    kernels/tensor.h
    kernels/threading.h
)

add_library(mollm_lib STATIC ${SOURCES})
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set_source_files_properties(
            kernels/matmul_x86_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
        set_source_files_properties(
            kernels/matmul_x86_avx512.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx512f;-mfma;-mf16c")
        set_source_files_properties(
            kernels/matmul_x86_vnni.cpp
            PROPERTIES COMPILE_OPTIONS
                       "-mavx512f;-mavx512bw;-mavx512vl;-mavx512vnni;-mfma")
        set_source_files_properties(
            kernels/gdn_x86_avx512.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx512f;-mfma")
        set_source_files_properties(
            kernels/attention_x86_avx512.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx512f;-mfma")
    else()
        message(FATAL_ERROR
            "The x86 SIMD providers currently require GCC or Clang")
    endif()
endif()
add_library(mollm::mollm ALIAS mollm_lib)
set_target_properties(mollm_lib PROPERTIES EXPORT_NAME mollm)
target_include_directories(mollm_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/mollm>
)
target_link_libraries(mollm_lib PUBLIC Threads::Threads)
if(MOLLM_SANITIZERS)
    # Static-library consumers need the sanitizer runtimes at final link time.
    target_link_options(mollm_lib INTERFACE ${MOLLM_SANITIZER_FLAGS})
endif()

# ---------------------------------------------------------------------------
# platform detection for SIMD
# ---------------------------------------------------------------------------

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    message(STATUS "Target: ARM64 (NEON enabled)")
    target_compile_definitions(mollm_lib PUBLIC MOLLM_CPU_ARM_NEON=1)
    target_compile_definitions(mollm_lib PRIVATE __ARM_NEON=1)
    # Baseline NEON/DOTPROD code and optional i8mm code are compiled into
    # separate objects.  The baseline therefore remains runnable on ARMv8.2
    # CPUs even when the build also contains i8mm kernels.
    set(MOLLM_ARM_BASELINE_FLAGS "-march=armv8.2-a+dotprod+fp16+fp16fml")
    if(APPLE)
        set(MOLLM_ARM_BASELINE_FLAGS "")
    endif()
    if(MOLLM_ARM_NATIVE)
        message(STATUS "ARM64 native target enabled (-mcpu=native)")
        target_compile_options(mollm_lib PRIVATE -mcpu=native)
    elseif(MOLLM_ARM_BASELINE_FLAGS)
        target_compile_options(mollm_lib PRIVATE ${MOLLM_ARM_BASELINE_FLAGS})
    endif()

    if(NOT MOLLM_ARM_I8MM_MODE STREQUAL "OFF")
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
        set(CMAKE_REQUIRED_FLAGS "${_mollm_saved_required_flags}")

        if(MOLLM_ARM_I8MM_COMPILES)
            add_library(mollm_arm_i8mm OBJECT
                kernels/matmul_w4_i8mm.cpp
                kernels/matmul_w8.cpp)
            target_include_directories(mollm_arm_i8mm PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}
                ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann)
            target_compile_definitions(mollm_arm_i8mm PRIVATE
                MOLLM_CPU_ARM_NEON=1
                MOLLM_ARM_I8MM_KERNELS=1
                MOLLM_W8_I8MM_ONLY=1
                __ARM_NEON=1)
            target_compile_options(mollm_arm_i8mm PRIVATE
                ${MOLLM_ARM_I8MM_FLAGS})
            target_sources(mollm_lib PRIVATE
                $<TARGET_OBJECTS:mollm_arm_i8mm>)
            target_compile_definitions(mollm_lib PRIVATE
                MOLLM_ARM_I8MM_KERNELS=1)
            message(STATUS
                "ARM64 i8mm kernels embedded for runtime dispatch "
                "(${MOLLM_ARM_I8MM_FLAGS})")
        elseif(MOLLM_ARM_I8MM_MODE STREQUAL "ON")
            message(FATAL_ERROR
                "MOLLM_ARM_I8MM=ON requested, but the compiler does not support "
                "${MOLLM_ARM_I8MM_FLAGS}")
        else()
            message(STATUS
                "ARM64 i8mm kernels unavailable; using the NEON baseline")
        endif()
    else()
        message(STATUS
            "ARM64 i8mm kernels disabled by MOLLM_ARM_I8MM=OFF")
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    message(STATUS "Target: ARM32")
    target_compile_definitions(mollm_lib PUBLIC MOLLM_CPU_ARM_NEON=1)
    target_compile_definitions(mollm_lib PRIVATE __ARM_NEON=1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    message(STATUS "Target: x86_64 (runtime scalar/AVX2/AVX-512 provider)")
    target_compile_definitions(mollm_lib PUBLIC MOLLM_CPU_ARM_NEON=0)
else()
    message(STATUS "Target: ${CMAKE_SYSTEM_PROCESSOR} (scalar fallback)")
    target_compile_definitions(mollm_lib PUBLIC MOLLM_CPU_ARM_NEON=0)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_definitions(mollm_lib PRIVATE MOLLM_CPU_X86_SIMD=1)
endif()

# Apple-specific: link Accelerate for optional BLAS (future use)
if(APPLE)
    find_library(ACCELERATE_LIBRARY Accelerate)
    find_library(COREFOUNDATION_LIBRARY CoreFoundation REQUIRED)
    find_library(COREGRAPHICS_LIBRARY CoreGraphics REQUIRED)
    find_library(IMAGEIO_LIBRARY ImageIO REQUIRED)
    if(ACCELERATE_LIBRARY)
        target_link_libraries(mollm_lib PUBLIC ${ACCELERATE_LIBRARY})
        target_compile_definitions(mollm_lib PRIVATE ACCELERATE_NEW_LAPACK)
    endif()
    target_link_libraries(
        mollm_lib PUBLIC ${COREFOUNDATION_LIBRARY} ${COREGRAPHICS_LIBRARY}
                         ${IMAGEIO_LIBRARY})
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
        "$<BUILD_INTERFACE:MOLLM_METALLIB_PATH=\"${MOLLM_METALLIB}\">"
        "$<INSTALL_INTERFACE:MOLLM_METALLIB_PATH=\"${CMAKE_INSTALL_FULL_DATADIR}/mollm/default.metallib\">")
    target_link_libraries(mollm_lib PUBLIC ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})
    install(FILES ${MOLLM_METALLIB}
            DESTINATION ${CMAKE_INSTALL_DATADIR}/mollm)
    message(STATUS "Metal backend enabled (metallib: ${MOLLM_METALLIB})")
endif()

# ---------------------------------------------------------------------------
# CUDA GPU backend (optional)
# ---------------------------------------------------------------------------

if(MOLLM_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # Emit SASS for common datacenter/consumer generations and retain PTX
        # for forward-compatible driver JIT on newer GPUs.
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_sources(mollm_lib PRIVATE
        engine/cuda_attention.cu
        engine/cuda_backend.cu
        engine/cuda_matmul.cu
        engine/cuda_memory.cu
        engine/cuda_ops.cu
        engine/cuda_recurrent.cu)
    target_compile_definitions(mollm_lib PUBLIC MOLLM_CUDA=1)
    target_link_libraries(mollm_lib PUBLIC CUDA::cudart CUDA::cublas)
    set_target_properties(mollm_lib PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION OFF)
    message(STATUS "CUDA correctness backend enabled")
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

    add_executable(mollm_mcqa
        examples/mcqa.cpp
    )
    target_link_libraries(mollm_mcqa PRIVATE mollm_lib)

    if(UNIX)
        add_executable(mollm_server
            examples/server.cpp
            examples/cli_common.cpp
        )
        target_link_libraries(mollm_server PRIVATE mollm_lib)
    endif()
endif()

add_executable(mollm-quantize tools/quantize_weight.cpp)
target_include_directories(mollm-quantize PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mollm-quantize PRIVATE Threads::Threads)

# Backwards-compatible build target name used in older docs/scripts. The
# produced binary is build/mollm-quantize.
add_custom_target(mollm_quantize_weight DEPENDS mollm-quantize)

# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

if(MOLLM_BUILD_TESTS)
    enable_testing()

    configure_file(
        cmake/RunPackageConsumer.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/RunPackageConsumer.cmake
        @ONLY)

    function(add_mollm_test name src)
        add_executable(${name} ${src})
        target_link_libraries(${name} PRIVATE mollm_lib)
        add_test(NAME ${name} COMMAND ${name})
    endfunction()

    add_mollm_test(test_tensor        tests/test_tensor.cpp)
    add_mollm_test(test_cpu_platform  tests/test_cpu_platform.cpp)
    add_mollm_test(test_byte_ranges   tests/test_byte_ranges.cpp)
    add_mollm_test(test_buffer_pool   tests/test_buffer_pool.cpp)
    add_mollm_test(test_mmap_file     tests/test_mmap_file.cpp)
    add_mollm_test(test_weight_metadata tests/test_weight_metadata.cpp)
    add_mollm_test(test_sampler       tests/test_sampler.cpp)
    add_executable(vision_probe       tests/vision_probe.cpp)
    target_link_libraries(vision_probe PRIVATE mollm_lib)
    add_mollm_test(test_graph         tests/test_graph.cpp)
    add_mollm_test(test_io            tests/test_io.cpp)
    add_mollm_test(test_execute       tests/test_execute.cpp)
    add_mollm_test(test_elementwise   tests/test_elementwise.cpp)
    add_mollm_test(test_layout        tests/test_layout.cpp)
    add_mollm_test(test_matmul        tests/test_matmul.cpp)
    add_mollm_test(test_nvfp4         tests/test_nvfp4.cpp)
    add_mollm_test(test_moe_routing  tests/test_moe_routing.cpp)
    add_mollm_test(test_moe           tests/test_moe.cpp)
    add_mollm_test(test_moe_ssd       tests/test_moe_ssd.cpp)
    add_mollm_test(test_hyper_connection tests/test_hyper_connection.cpp)
    add_mollm_test(test_gated_residual tests/test_gated_residual.cpp)
    add_mollm_test(test_ple            tests/test_ple.cpp)
    add_mollm_test(test_deepseek_v4_attention tests/test_deepseek_v4_attention.cpp)
    add_mollm_test(test_trace         tests/test_trace.cpp)
    add_mollm_test(bench_matmul       tests/bench_matmul.cpp)
    add_mollm_test(bench_sdpa         tests/bench_sdpa.cpp)
    add_mollm_test(bench_recurrent    tests/bench_recurrent.cpp)
    add_mollm_test(test_norm          tests/test_norm.cpp)
    add_mollm_test(test_rope          tests/test_rope.cpp)
    add_mollm_test(test_attention     tests/test_attention.cpp)
    add_mollm_test(test_shortconv     tests/test_shortconv.cpp)
    add_mollm_test(test_gdn           tests/test_gdn.cpp)
    add_mollm_test(test_rwkv_e2e      tests/test_rwkv_e2e.cpp)
    add_mollm_test(test_engine        tests/test_engine.cpp)
    add_mollm_test(test_tokenizer     tests/test_tokenizer.cpp)
    add_mollm_test(test_e2e           tests/test_e2e.cpp)
    add_mollm_test(test_quantized_e2e tests/test_quantized_e2e.cpp)
    add_mollm_test(test_thread_pool_stress tests/test_thread_pool_stress.cpp)
    if(MOLLM_CUDA)
        add_mollm_test(test_cuda_backend tests/test_cuda_backend.cpp)
        set_tests_properties(test_cuda_backend PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        function(add_mollm_python_test name script)
            add_test(
                NAME ${name}
                COMMAND ${Python3_EXECUTABLE}
                        ${CMAKE_CURRENT_SOURCE_DIR}/tests/${script})
        endfunction()

        add_mollm_python_test(test_shape_propagation
                              test_shape_propagation.py)
        add_mollm_python_test(test_qwen3_graph
                              test_qwen3_graph.py)
        add_mollm_python_test(test_qwen3_moe_graph
                              test_qwen3_moe_graph.py)
        add_mollm_python_test(test_qwen35_graph
                              test_qwen35_graph.py)
        add_mollm_python_test(test_qwen4_exp_graph
                              test_qwen4_exp_graph.py)
        add_mollm_python_test(test_qwen35_quant_policy
                              test_qwen35_quant_policy.py)
        add_mollm_python_test(test_qwen35_moe_quant_policy
                              test_qwen35_moe_quant_policy.py)
        add_mollm_python_test(test_qwen35_moe_projection_fusion
                              test_qwen35_moe_projection_fusion.py)
        add_mollm_python_test(test_youtu_quant_policy
                              test_youtu_quant_policy.py)
        add_mollm_python_test(test_rwkv7_quant_policy
                              test_rwkv7_quant_policy.py)
        add_mollm_python_test(test_moe_expert_storage_metadata
                              test_moe_expert_storage_metadata.py)
        add_mollm_python_test(test_nvfp4_stream test_nvfp4_stream.py)
        add_mollm_python_test(test_w4g32_stream test_w4g32_stream.py)
        add_test(
            NAME test_canonical_w4_storage
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_canonical_w4_storage.py
                    $<TARGET_FILE:mollm-quantize>)
        add_test(
            NAME test_mtp_package
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mtp_package.py)
        add_test(
            NAME test_hy_v3_adapter
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hy_v3_adapter.py)
        if(TARGET mollm_bench)
            add_test(
                NAME bench_qwen35_mtp
                COMMAND ${Python3_EXECUTABLE}
                        ${CMAKE_CURRENT_SOURCE_DIR}/tests/bench_qwen35_mtp.py
                        $<TARGET_FILE:mollm_bench>)
            set_tests_properties(bench_qwen35_mtp
                                 PROPERTIES SKIP_RETURN_CODE 77
                                            RUN_SERIAL TRUE
                                            TIMEOUT 1800
                                            LABELS "performance;cpu;real-fixture")
        endif()
        add_executable(test_tiny_cpu_inference tests/test_tiny_cpu_inference.cpp)
        target_link_libraries(test_tiny_cpu_inference PRIVATE mollm_lib)
        add_dependencies(test_tiny_cpu_inference mollm-quantize)
        add_test(
            NAME test_tiny_qwen3_cpu_inference
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_tiny_qwen3_cpu_e2e.py
                    ${CMAKE_CURRENT_SOURCE_DIR}/models/converter.py
                    $<TARGET_FILE:mollm-quantize>
                    $<TARGET_FILE:test_tiny_cpu_inference>)
        if(MOLLM_CUDA)
            set(TINY_DENSE_E2E_LABELS "cpu;cuda;e2e")
        else()
            set(TINY_DENSE_E2E_LABELS "cpu;e2e")
        endif()
        set_tests_properties(test_tiny_qwen3_cpu_inference
                             PROPERTIES SKIP_RETURN_CODE 77
                                        LABELS "${TINY_DENSE_E2E_LABELS}")
    else()
        message(STATUS "Python3 not found; tiny CPU inference E2E test disabled")
    endif()
    add_test(
        NAME test_package_install
        COMMAND ${CMAKE_COMMAND}
                -P ${CMAKE_CURRENT_BINARY_DIR}/RunPackageConsumer.cmake)

    if(MOLLM_METAL)
        add_mollm_test(test_metal_ops tests/test_metal_ops.cpp)

        # Developer-only microbenchmark for the decode-time Metal SSD-MoE
        # selected-expert kernel. It is intentionally not registered with
        # CTest because it allocates full 122B expert working sets.
        add_executable(bench_metal_moe tests/bench_metal_moe.mm)
        target_link_libraries(bench_metal_moe PRIVATE mollm_lib)
        set_source_files_properties(
            tests/bench_metal_moe.mm
            PROPERTIES COMPILE_OPTIONS "-fobjc-arc")

        add_executable(
            bench_metal_w4_gemv tests/bench_metal_w4_gemv.mm)
        target_link_libraries(bench_metal_w4_gemv PRIVATE mollm_lib)
        set_source_files_properties(
            tests/bench_metal_w4_gemv.mm
            PROPERTIES COMPILE_OPTIONS "-fobjc-arc")

        add_executable(
            bench_metal_w4_gemm tests/bench_metal_w4_gemm.mm)
        target_link_libraries(bench_metal_w4_gemm PRIVATE mollm_lib)
        set_source_files_properties(
            tests/bench_metal_w4_gemm.mm
            PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    endif()
endif()

# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------

install(TARGETS mollm_lib
        EXPORT mollmTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mollm)
install(FILES ${PUBLIC_ENGINE_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mollm/engine)
install(FILES ${PUBLIC_GRAPH_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mollm/graph)
install(FILES ${PUBLIC_KERNEL_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mollm/kernels)

configure_package_config_file(
    cmake/mollmConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/mollmConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mollm)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/mollmConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(EXPORT mollmTargets
        FILE mollmTargets.cmake
        NAMESPACE mollm::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mollm)
install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/mollmConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/mollmConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mollm)
