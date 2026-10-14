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
option(MOLLM_CUDA_DECODE_SEGMENTS "Experimental CUDA decode segment replay (per-thread default stream)" OFF)
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
    set(MOLLM_CPU_PLATFORM_SOURCE backends/cpu/platform/arm.cpp)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(MOLLM_CPU_PLATFORM_SOURCE
        backends/cpu/platform/x86.cpp
        kernels/cpu/x86/matmul_x86_avx2.cpp
        kernels/cpu/x86/matmul_x86_avx512.cpp
        kernels/cpu/x86/matmul_x86_vnni.cpp
        kernels/cpu/x86/gdn_x86_avx512.cpp
        kernels/cpu/x86/attention_x86_avx512.cpp)
else()
    set(MOLLM_CPU_PLATFORM_SOURCE backends/cpu/platform/scalar.cpp)
endif()

# ---------------------------------------------------------------------------
# library sources
# ---------------------------------------------------------------------------

set(SOURCES
    # Metal-specific weight conversion is written in portable C++ so it can
    # also be validated by the CPU-only test target.
    backends/metal/weight_layout.cpp
    # runtime
    runtime/host_buffer_pool.cpp
    runtime/threading.cpp
    runtime/trace.cpp
    # storage
    storage/mapped_file.cpp
    storage/byte_ranges.cpp
    storage/ssd_expert_cache/provider.cpp
    storage/ssd_expert_cache/cache.cpp
    storage/ssd_expert_cache/io.cpp
    # graph
    graph/io.cpp
    backends/cpu/attention_ops.cpp
    backends/cpu/backend.cpp
    backends/cpu/host_compute.cpp
    backends/cpu/deepseek_v4_ops.cpp
    backends/cpu/layout_ops.cpp
    backends/cpu/matmul_ops.cpp
    backends/cpu/moe_ops.cpp
    backends/cpu/normalization_ops.cpp
    backends/cpu/recurrent_ops.cpp
    backends/cpu/ssd_prefetch.cpp
    graph/execute.cpp
    # kernels
    kernels/cpu/matmul/matmul.cpp
    kernels/cpu/matmul/matmul_dense.cpp
    kernels/cpu/matmul/matmul_pack.cpp
    kernels/cpu/matmul/matmul_profile.cpp
    kernels/cpu/matmul/matmul_quant.cpp
    kernels/cpu/matmul/matmul_sparse.cpp
    kernels/cpu/matmul/matmul_w4.cpp
    kernels/cpu/matmul/matmul_w8.cpp
    kernels/cpu/matmul/matmul_mxfp4.cpp
    kernels/cpu/matmul/matmul_nvfp4.cpp
    ${MOLLM_CPU_PLATFORM_SOURCE}
    kernels/cpu/elementwise.cpp
    kernels/cpu/layout.cpp
    kernels/cpu/norm.cpp
    kernels/cpu/rope.cpp
    kernels/cpu/attention.cpp
    kernels/cpu/recurrent/shortconv.cpp
    kernels/cpu/recurrent/gdn.cpp
    kernels/cpu/recurrent/gdn_decode.cpp
    kernels/cpu/recurrent/gdn_prefill.cpp
    kernels/cpu/gated_residual.cpp
    kernels/cpu/ple.cpp
    kernels/cpu/models/deepseek_v4_attention.cpp
    kernels/cpu/hyper_connection.cpp
    kernels/cpu/moe/moe.cpp
    kernels/cpu/moe/moe_routing.cpp
    kernels/cpu/models/rwkv.cpp
    # backend selection
    backends/factory.cpp
    # engine
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
    engine/engine.h
    engine/sampler.h
    engine/tokenizer.h
)

set(PUBLIC_CORE_HEADERS
    core/tensor.h
    core/fp16.h
    core/rwkv_params.h
    core/gdn_params.h
    core/attention_params.h
    core/shortconv_params.h
    core/activation.h
    core/bf16.h
    core/quant_layouts.h
    core/prepared_weight.h
    core/cache_layout.h
)

set(PUBLIC_RUNTIME_HEADERS
    runtime/expert_provider.h
    runtime/backend.h
    runtime/accelerator_backend.h
    runtime/threading.h
    runtime/trace.h
    runtime/host_buffer_pool.h
    runtime/host_compute.h
)

set(PUBLIC_CPU_BACKEND_HEADERS
    backends/cpu/backend.h
    backends/cpu/platform.h
)

set(PUBLIC_STORAGE_HEADERS
    storage/mapped_file.h
    storage/byte_ranges.h
)

set(PUBLIC_GRAPH_HEADERS
    graph/graph.h
    graph/execute.h
)

add_library(mollm_lib STATIC ${SOURCES})
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set_source_files_properties(
            kernels/cpu/x86/matmul_x86_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
        set_source_files_properties(
            kernels/cpu/x86/matmul_x86_avx512.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx512f;-mfma;-mf16c")
        set_source_files_properties(
            kernels/cpu/x86/matmul_x86_vnni.cpp
            PROPERTIES COMPILE_OPTIONS
                       "-mavx512f;-mavx512bw;-mavx512vl;-mavx512vnni;-mfma")
        set_source_files_properties(
            kernels/cpu/x86/gdn_x86_avx512.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx512f;-mfma")
        set_source_files_properties(
            kernels/cpu/x86/attention_x86_avx512.cpp
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
                kernels/cpu/arm/matmul_w4_i8mm.cpp
                kernels/cpu/matmul/matmul_w8.cpp)
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

    # Compile each shader domain independently, then link one metallib.
    set(MOLLM_METAL_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/matmul.metal
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/moe.metal
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/attention.metal
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/elementwise.metal
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/layout.metal
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/normalization.metal
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/recurrent.metal
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/rotary.metal
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/selection.metal)
    set(MOLLM_METALLIB ${CMAKE_CURRENT_BINARY_DIR}/default.metallib)

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
    set(MOLLM_METAL_AIR_FILES "")
    foreach(MOLLM_METAL_SOURCE IN LISTS MOLLM_METAL_SOURCES)
        get_filename_component(MOLLM_METAL_DOMAIN
                               ${MOLLM_METAL_SOURCE} NAME_WE)
        set(MOLLM_METAL_DOMAIN_AIR
            ${CMAKE_CURRENT_BINARY_DIR}/${MOLLM_METAL_DOMAIN}.air)
        add_custom_command(
            OUTPUT ${MOLLM_METAL_DOMAIN_AIR}
            COMMAND xcrun -sdk macosx metal ${MOLLM_METAL_FLAGS}
                    -c ${MOLLM_METAL_SOURCE}
                    -I ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal
                    -o ${MOLLM_METAL_DOMAIN_AIR}
            DEPENDS ${MOLLM_METAL_SOURCE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/metal_common.h
            COMMENT "Compiling Metal shader domain: ${MOLLM_METAL_DOMAIN}"
            VERBATIM)
        list(APPEND MOLLM_METAL_AIR_FILES ${MOLLM_METAL_DOMAIN_AIR})
    endforeach()
    add_custom_command(
        OUTPUT ${MOLLM_METALLIB}
        COMMAND xcrun -sdk macosx metallib ${MOLLM_METAL_AIR_FILES}
                -o ${MOLLM_METALLIB}
        DEPENDS ${MOLLM_METAL_AIR_FILES}
        COMMENT "Linking Metal library (default.metallib)"
        VERBATIM)
    add_custom_target(mollm_metallib ALL DEPENDS ${MOLLM_METALLIB})
    add_dependencies(mollm_lib mollm_metallib)

    target_sources(mollm_lib PRIVATE
        backends/metal/attention_ops.mm
        backends/metal/backend.mm
        backends/metal/buffer_pool.mm
        backends/metal/command_context.mm
        backends/metal/elementwise_ops.mm
        backends/metal/layout_ops.mm
        backends/metal/lm_head.mm
        backends/metal/matmul_ops.mm
        backends/metal/moe_dispatch.mm
        backends/metal/moe_host_ops.mm
        backends/metal/moe_ops.mm
        backends/metal/normalization_ops.mm
        backends/metal/pipeline_cache.mm
        backends/metal/recurrent_ops.mm
        backends/metal/resource_store.mm
        backends/metal/rotary_ops.mm
        backends/metal/ssd_expert_cache.mm
        backends/metal/ssd_shared_expert.mm)
    # The Objective-C++ sources use ARC bridge-cast ownership semantics.
    set_source_files_properties(
        backends/metal/attention_ops.mm
        backends/metal/backend.mm backends/metal/buffer_pool.mm
        backends/metal/command_context.mm
        backends/metal/elementwise_ops.mm
        backends/metal/layout_ops.mm
        backends/metal/lm_head.mm
        backends/metal/matmul_ops.mm
        backends/metal/moe_dispatch.mm
        backends/metal/moe_host_ops.mm
        backends/metal/moe_ops.mm
        backends/metal/normalization_ops.mm
        backends/metal/pipeline_cache.mm
        backends/metal/recurrent_ops.mm
        backends/metal/resource_store.mm
        backends/metal/rotary_ops.mm
        backends/metal/ssd_expert_cache.mm
        backends/metal/ssd_shared_expert.mm
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
        backends/cuda/backend.cu
        backends/cuda/device_runtime.cu
        backends/cuda/dispatch.cu
        backends/cuda/lm_head.cu
        backends/cuda/moe.cu
        kernels/cuda/attention.cu
        kernels/cuda/qsa.cu
        kernels/cuda/matmul.cu
        kernels/cuda/moe.cu
        kernels/cuda/elementwise.cu
        kernels/cuda/recurrent.cu)
    target_compile_definitions(mollm_lib PUBLIC MOLLM_CUDA=1)
    if(MOLLM_CUDA_DECODE_SEGMENTS)
        target_compile_options(mollm_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
        target_compile_definitions(mollm_lib PRIVATE MOLLM_CUDA_DECODE_SEGMENTS=1)
        # C++ consumers also call cudart (copies, events and memset). Keep
        # their implicit stream consistent with the kernels compiled by nvcc.
        target_compile_definitions(mollm_lib PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM=1)
    endif()
    target_link_libraries(mollm_lib PUBLIC CUDA::cudart CUDA::cublas CUDA::cublasLt)
    set_target_properties(mollm_lib PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION OFF)
    message(STATUS "CUDA correctness backend enabled")
    if(MOLLM_BUILD_TESTS)
        add_executable(bench_cuda_dense_lt_gemm tests/bench_cuda_dense_lt_gemm.cu)
        add_executable(bench_cuda_nvfp4_batch_plan tests/bench_cuda_nvfp4_batch_plan.cu)
        target_link_libraries(bench_cuda_nvfp4_batch_plan PRIVATE mollm_lib)
        set_target_properties(bench_cuda_nvfp4_batch_plan PROPERTIES CUDA_STANDARD 17)
        target_link_libraries(bench_cuda_dense_lt_gemm PRIVATE mollm_lib CUDA::cublasLt)
        set_target_properties(bench_cuda_dense_lt_gemm PROPERTIES CUDA_STANDARD 17)
        add_executable(bench_cuda_projections tests/bench_cuda_projections.cpp)
        target_link_libraries(bench_cuda_projections PRIVATE mollm_lib)
        add_executable(bench_cuda_q8_traffic tests/bench_cuda_q8_traffic.cu)
        target_link_libraries(bench_cuda_q8_traffic PRIVATE mollm_lib)
        set_target_properties(bench_cuda_q8_traffic PROPERTIES CUDA_STANDARD 17)
        add_executable(bench_cuda_q4_mma tests/bench_cuda_q4_mma.cu)
        add_executable(bench_cuda_sdpa_prefill tests/bench_cuda_sdpa_prefill.cu)
        add_executable(bench_cuda_sdpa_validated_prefix tests/bench_cuda_sdpa_validated_prefix.cu)
        add_executable(bench_cuda_group_rms_half tests/bench_cuda_group_rms_half.cu)
        add_executable(bench_cuda_gr_deferred_rms tests/bench_cuda_gr_deferred_rms.cu)
        target_link_libraries(bench_cuda_gr_deferred_rms PRIVATE mollm_lib)
        target_compile_options(bench_cuda_gr_deferred_rms PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
        set_target_properties(bench_cuda_gr_deferred_rms PROPERTIES CUDA_STANDARD 17)
        add_executable(test_cuda_nvfp4_async_weights tests/test_cuda_nvfp4_async_weights.cu)
        target_link_libraries(test_cuda_nvfp4_async_weights PRIVATE mollm_lib)
        target_compile_options(test_cuda_nvfp4_async_weights PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
        set_target_properties(test_cuda_nvfp4_async_weights PROPERTIES CUDA_STANDARD 17)
        add_executable(test_cuda_nvfp4_gate_code_repack tests/test_cuda_nvfp4_gate_code_repack.cu)
        add_executable(test_cuda_nvfp4_gate_scale_repack tests/test_cuda_nvfp4_gate_scale_repack.cu)
        target_link_libraries(test_cuda_nvfp4_gate_scale_repack PRIVATE mollm_lib)
        target_compile_options(test_cuda_nvfp4_gate_scale_repack PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
        set_target_properties(test_cuda_nvfp4_gate_scale_repack PROPERTIES CUDA_STANDARD 17)
        target_link_libraries(test_cuda_nvfp4_gate_code_repack PRIVATE mollm_lib)
        target_compile_options(test_cuda_nvfp4_gate_code_repack PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
        set_target_properties(test_cuda_nvfp4_gate_code_repack PROPERTIES CUDA_STANDARD 17)
        add_executable(test_cuda_nvfp4_gate_code_decode tests/test_cuda_nvfp4_gate_code_decode.cu)
        target_link_libraries(test_cuda_nvfp4_gate_code_decode PRIVATE mollm_lib)
        target_compile_options(test_cuda_nvfp4_gate_code_decode PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
        set_target_properties(test_cuda_nvfp4_gate_code_decode PROPERTIES CUDA_STANDARD 17)
        add_executable(bench_cuda_nvfp4_bf16_output tests/bench_cuda_nvfp4_bf16_output.cu)
        add_executable(bench_cuda_nvfp4_w4a16 tests/bench_cuda_nvfp4_w4a16.cu)
        target_link_libraries(bench_cuda_nvfp4_w4a16 PRIVATE mollm_lib)
        target_compile_options(bench_cuda_nvfp4_w4a16 PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
        set_target_properties(bench_cuda_nvfp4_w4a16 PROPERTIES CUDA_STANDARD 17)
        add_executable(bench_cuda_nvfp4_w4a16_routes tests/bench_cuda_nvfp4_w4a16_routes.cu)
        target_link_libraries(bench_cuda_nvfp4_w4a16_routes PRIVATE mollm_lib)
        target_compile_options(bench_cuda_nvfp4_w4a16_routes PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
        set_target_properties(bench_cuda_nvfp4_w4a16_routes PROPERTIES CUDA_STANDARD 17)
        target_link_libraries(bench_cuda_nvfp4_bf16_output PRIVATE mollm_lib)
        target_compile_options(bench_cuda_nvfp4_bf16_output PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
        set_target_properties(bench_cuda_nvfp4_bf16_output PROPERTIES CUDA_STANDARD 17)
        add_executable(bench_cuda_hc_fused_up_reduce tests/bench_cuda_hc_fused_up_reduce.cu)
        add_executable(bench_cuda_hc_compact_carry tests/bench_cuda_hc_compact_carry.cu)
        add_executable(bench_cuda_moe_hc_inject tests/bench_cuda_moe_hc_inject.cu)
        target_link_libraries(bench_cuda_moe_hc_inject PRIVATE mollm_lib)
        target_compile_options(bench_cuda_moe_hc_inject PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
        set_target_properties(bench_cuda_moe_hc_inject PROPERTIES CUDA_STANDARD 17)
        target_link_libraries(bench_cuda_hc_compact_carry PRIVATE mollm_lib)
        target_compile_options(bench_cuda_hc_compact_carry PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
        set_target_properties(bench_cuda_hc_compact_carry PROPERTIES CUDA_STANDARD 17)
        target_link_libraries(bench_cuda_hc_fused_up_reduce PRIVATE mollm_lib CUDA::cublas)
        target_compile_options(bench_cuda_hc_fused_up_reduce PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
        set_target_properties(bench_cuda_hc_fused_up_reduce PROPERTIES CUDA_STANDARD 17)
        add_executable(bench_cuda_nvfp4_bf16_gate tests/bench_cuda_nvfp4_bf16_gate.cu)
        add_executable(bench_cuda_moe_route_windows tests/bench_cuda_moe_route_windows.cu)
        target_link_libraries(bench_cuda_moe_route_windows PRIVATE mollm_lib)
        target_compile_options(bench_cuda_moe_route_windows PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
        set_target_properties(bench_cuda_moe_route_windows PROPERTIES CUDA_STANDARD 17)
        target_link_libraries(bench_cuda_nvfp4_bf16_gate PRIVATE mollm_lib)
        target_compile_options(bench_cuda_nvfp4_bf16_gate PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
        set_target_properties(bench_cuda_nvfp4_bf16_gate PROPERTIES CUDA_STANDARD 17)
        target_link_libraries(bench_cuda_group_rms_half PRIVATE mollm_lib)
        target_compile_options(bench_cuda_group_rms_half PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
        set_target_properties(bench_cuda_group_rms_half PROPERTIES CUDA_STANDARD 17)
        target_link_libraries(bench_cuda_sdpa_validated_prefix PRIVATE mollm_lib)
        target_compile_options(bench_cuda_sdpa_validated_prefix PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
        set_target_properties(bench_cuda_sdpa_validated_prefix PROPERTIES CUDA_STANDARD 17)
        add_executable(bench_cuda_gr_reduce tests/bench_cuda_gr_reduce.cu)
        add_executable(bench_cuda_gdn_recurrence tests/bench_cuda_gdn_recurrence.cu)
        add_executable(bench_cuda_gdn_chunkwise tests/bench_cuda_gdn_chunkwise.cu)
        target_link_libraries(bench_cuda_gdn_chunkwise PRIVATE mollm_lib)
        target_compile_options(bench_cuda_gdn_chunkwise PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
        set_target_properties(bench_cuda_gdn_chunkwise PROPERTIES CUDA_STANDARD 17)
        add_executable(bench_cuda_nvfp4_quantize tests/bench_cuda_nvfp4_quantize.cu)
        target_link_libraries(bench_cuda_nvfp4_quantize PRIVATE mollm_lib)
        target_compile_options(bench_cuda_nvfp4_quantize PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
        set_target_properties(bench_cuda_nvfp4_quantize PROPERTIES CUDA_STANDARD 17)
        target_link_libraries(bench_cuda_gdn_recurrence PRIVATE mollm_lib)
        target_compile_options(bench_cuda_gdn_recurrence PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
        set_target_properties(bench_cuda_gdn_recurrence PROPERTIES CUDA_STANDARD 17)
        target_link_libraries(bench_cuda_gr_reduce PRIVATE mollm_lib)
        target_compile_options(bench_cuda_gr_reduce PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
        set_target_properties(bench_cuda_gr_reduce PROPERTIES CUDA_STANDARD 17)
        target_link_libraries(bench_cuda_sdpa_prefill PRIVATE mollm_lib)
        target_compile_options(bench_cuda_sdpa_prefill PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
        set_target_properties(bench_cuda_sdpa_prefill PROPERTIES CUDA_STANDARD 17)
        target_link_libraries(bench_cuda_q4_mma PRIVATE mollm_lib)
        set_target_properties(bench_cuda_q4_mma PROPERTIES CUDA_STANDARD 17)
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
            add_executable(bench_cuda_nvfp4_activation tests/bench_cuda_nvfp4_activation.cu)
            target_link_libraries(bench_cuda_nvfp4_activation PRIVATE mollm_lib)
            target_compile_options(bench_cuda_nvfp4_activation PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
            set_target_properties(bench_cuda_nvfp4_activation PROPERTIES CUDA_STANDARD 17)
            add_executable(bench_cuda_nvfp4_mma tests/bench_cuda_nvfp4_mma.cu)
            target_link_libraries(bench_cuda_nvfp4_mma PRIVATE mollm_lib)
            target_compile_options(bench_cuda_nvfp4_mma PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
            set_target_properties(bench_cuda_nvfp4_mma PROPERTIES CUDA_STANDARD 17)
            add_executable(bench_cuda_nvfp4_bf16_tile tests/bench_cuda_nvfp4_bf16_tile.cu)
            target_link_libraries(bench_cuda_nvfp4_bf16_tile PRIVATE mollm_lib)
            target_compile_options(bench_cuda_nvfp4_bf16_tile PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
            set_target_properties(bench_cuda_nvfp4_bf16_tile PROPERTIES CUDA_STANDARD 17 CUDA_ARCHITECTURES 121a)
        endif()
    endif()
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
    add_mollm_test(test_metal_weight_layout tests/test_metal_weight_layout.cpp)
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
    add_mollm_test(test_expert_provider tests/test_expert_provider.cpp)
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
    add_mollm_test(test_rwkv_ops      tests/test_rwkv_ops.cpp)
    add_mollm_test(test_execution_lifecycle tests/test_execution_lifecycle.cpp)
    add_mollm_test(test_node_dump_row tests/test_node_dump_row.cpp)
    set_tests_properties(test_node_dump_row PROPERTIES
        ENVIRONMENT "MOLLM_DUMP_NODES=1;MOLLM_DUMP_FIRST_ROW=1;MOLLM_DUMP_FIRST_ROW_MAX_SEQ=5;MOLLM_DUMP_ROW_OFFSET=2"
        PASS_REGULAR_EXPRESSION "12.00000 14.00000 16.00000 sum=42")
    add_test(NAME test_node_dump_row_shortconv COMMAND test_node_dump_row --dim-major)
    set_tests_properties(test_node_dump_row_shortconv PROPERTIES
        ENVIRONMENT "MOLLM_DUMP_NODES=1;MOLLM_DUMP_FIRST_ROW=1;MOLLM_DUMP_FIRST_ROW_MAX_SEQ=5;MOLLM_DUMP_ROW_OFFSET=2"
        PASS_REGULAR_EXPRESSION "2.00000 102.00000 202.00000 sum=306")
    add_test(NAME test_node_dump_row_bounds COMMAND test_node_dump_row)
    set_tests_properties(test_node_dump_row_bounds PROPERTIES
        ENVIRONMENT "MOLLM_DUMP_NODES=1;MOLLM_DUMP_FIRST_ROW=1;MOLLM_DUMP_FIRST_ROW_MAX_SEQ=5;MOLLM_DUMP_ROW_OFFSET=5"
        PASS_REGULAR_EXPRESSION "dump does not modify output" FAIL_REGULAR_EXPRESSION "ROW graph=")
    add_mollm_test(test_hc_deferred_rms_plan tests/test_hc_deferred_rms_plan.cpp)
    add_mollm_test(test_cuda_prefill_batch_policy tests/test_cuda_prefill_batch_policy.cpp)
    add_mollm_test(test_cuda_host_memory_watchdog tests/test_cuda_host_memory_watchdog.cpp)
    add_mollm_test(test_engine        tests/test_engine.cpp)
    add_mollm_test(test_recurrent_checkpoint tests/test_recurrent_checkpoint.cpp)
    if(MOLLM_CUDA)
        add_test(NAME test_recurrent_checkpoint_device COMMAND test_recurrent_checkpoint)
        set_tests_properties(test_recurrent_checkpoint_device PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_MTP_DEVICE_CHECKPOINT=1" LABELS "cuda")
    endif()
    add_mollm_test(test_tokenizer     tests/test_tokenizer.cpp)
    add_mollm_test(test_e2e           tests/test_e2e.cpp)
    add_mollm_test(test_quantized_e2e tests/test_quantized_e2e.cpp)
    add_mollm_test(test_thread_pool_stress tests/test_thread_pool_stress.cpp)
    add_executable(test_bench_repeat_args tests/test_bench_repeat_args.cpp examples/cli_common.cpp)
    target_link_libraries(test_bench_repeat_args PRIVATE mollm_lib)
    add_test(NAME test_bench_repeat_args COMMAND test_bench_repeat_args)
    add_executable(test_ppl_decode_schedule tests/test_ppl_decode_schedule.cpp examples/cli_common.cpp)
    target_link_libraries(test_ppl_decode_schedule PRIVATE mollm_lib)
    add_test(NAME test_ppl_decode_schedule COMMAND test_ppl_decode_schedule)
    if(MOLLM_CUDA)
        add_executable(test_cuda_qsa_selected_attention tests/test_cuda_qsa_selected_attention.cu)
        add_executable(test_cuda_qsa_indexer tests/test_cuda_qsa_indexer.cu)
        add_executable(test_cuda_qsa_graph tests/test_cuda_qsa_graph.cpp)
        add_executable(test_qsa_engine tests/test_qsa_engine.cpp)
        target_link_libraries(test_qsa_engine PRIVATE mollm_lib)
        target_link_libraries(test_cuda_qsa_graph PRIVATE mollm_lib)
        add_test(NAME test_cuda_qsa_graph COMMAND test_cuda_qsa_graph)
        set_tests_properties(test_cuda_qsa_graph PROPERTIES
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE TIMEOUT 180)
        target_link_libraries(test_cuda_qsa_indexer PRIVATE mollm_lib)
        set_target_properties(test_cuda_qsa_indexer PROPERTIES CUDA_STANDARD 17)
        add_test(NAME test_cuda_qsa_indexer COMMAND test_cuda_qsa_indexer)
        set_tests_properties(test_cuda_qsa_indexer PROPERTIES
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE TIMEOUT 180)
        target_link_libraries(test_cuda_qsa_selected_attention PRIVATE mollm_lib)
        set_target_properties(test_cuda_qsa_selected_attention PROPERTIES CUDA_STANDARD 17)
        add_test(NAME test_cuda_qsa_selected_attention COMMAND test_cuda_qsa_selected_attention)
        set_tests_properties(test_cuda_qsa_selected_attention PROPERTIES
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE TIMEOUT 180)
        if(NOT TARGET bench_cuda_gr_reduce)
            add_executable(bench_cuda_gr_reduce tests/bench_cuda_gr_reduce.cu)
            target_link_libraries(bench_cuda_gr_reduce PRIVATE mollm_lib)
            target_compile_options(bench_cuda_gr_reduce PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
            set_target_properties(bench_cuda_gr_reduce PROPERTIES CUDA_STANDARD 17)
        endif()
        add_test(NAME test_cuda_sigmoid_gr_reduce COMMAND bench_cuda_gr_reduce --check-only)
        if(TARGET bench_cuda_group_rms_half)
            add_test(NAME test_cuda_nvfp4_async_weights COMMAND test_cuda_nvfp4_async_weights)
            set_tests_properties(test_cuda_nvfp4_async_weights PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_nvfp4_group_code_kmajor COMMAND test_cuda_nvfp4_async_weights)
            set_tests_properties(test_cuda_nvfp4_group_code_kmajor PROPERTIES
                ENVIRONMENT "MOLLM_TEST_NVFP4_GROUP_CODE_KMAJOR=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_nvfp4_gate_code_repack COMMAND test_cuda_nvfp4_gate_code_repack)
            set_tests_properties(test_cuda_nvfp4_gate_code_repack PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_nvfp4_gate_code_decode COMMAND test_cuda_nvfp4_gate_code_decode)
            add_test(NAME test_cuda_nvfp4_gate_scale_repack COMMAND test_cuda_nvfp4_gate_scale_repack)
            set_tests_properties(test_cuda_nvfp4_gate_scale_repack PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_nvfp4_gate_scale_decode COMMAND test_cuda_nvfp4_gate_code_decode)
            set_tests_properties(test_cuda_nvfp4_gate_scale_decode PROPERTIES
                ENVIRONMENT "MOLLM_TEST_NVFP4_RESIDENT_GATE_SCALE_VIEW=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_nvfp4_gate_scale_grouped COMMAND test_cuda_nvfp4_async_weights)
            set_tests_properties(test_cuda_nvfp4_gate_scale_grouped PROPERTIES
                ENVIRONMENT "MOLLM_TEST_NVFP4_GROUP_CODE_KMAJOR=1;MOLLM_TEST_NVFP4_RESIDENT_GATE_SCALE_VIEW=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_nvfp4_gate_code_decode PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_nvfp4_bf16_output COMMAND bench_cuda_nvfp4_bf16_output --check-only)
            add_test(NAME test_cuda_hc_fused_up_reduce COMMAND bench_cuda_hc_fused_up_reduce --check-only)
            add_test(NAME test_cuda_moe_hc_inject_probe COMMAND bench_cuda_moe_hc_inject --boundaries)
            set_tests_properties(test_cuda_moe_hc_inject_probe PROPERTIES
                SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS cuda)
            add_test(NAME test_cuda_moe_hc_inject_no_branch COMMAND bench_cuda_moe_hc_inject --boundaries)
            set_tests_properties(test_cuda_moe_hc_inject_no_branch PROPERTIES
                SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS cuda
                ENVIRONMENT "MOLLM_TEST_MOE_HC_PUBLISH_BRANCH=0")
            add_test(NAME test_cuda_hc_fused_up_reduce_pitch328 COMMAND bench_cuda_hc_fused_up_reduce --check-only)
            add_test(NAME test_cuda_hc_fused_up_reduce_pitch336 COMMAND bench_cuda_hc_fused_up_reduce --check-only)
            add_test(NAME test_cuda_hc_fused_up_reduce_columns32 COMMAND bench_cuda_hc_fused_up_reduce --check-only)
            add_test(NAME test_cuda_hc_fused_up_reduce_columns32_pitch336 COMMAND bench_cuda_hc_fused_up_reduce --check-only)
            add_test(NAME test_cuda_hc_fused_up_reduce_compact_columns32_pitch336 COMMAND bench_cuda_hc_fused_up_reduce --check-only)
            set_tests_properties(test_cuda_hc_fused_up_reduce_compact_columns32_pitch336 PROPERTIES
                ENVIRONMENT "MOLLM_TEST_CUDA_HC_UP_INPUT_PITCH=336;MOLLM_TEST_CUDA_HC_UP_TILE_COLUMNS=32;MOLLM_TEST_CUDA_HC_UP_COMPACT_CARRY=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_hc_fused_up_reduce_columns32_pitch328 COMMAND bench_cuda_hc_fused_up_reduce --check-only)
            set_tests_properties(test_cuda_hc_fused_up_reduce_columns32_pitch328 PROPERTIES
                ENVIRONMENT "MOLLM_TEST_CUDA_HC_UP_TILE_COLUMNS=32;MOLLM_TEST_CUDA_HC_UP_INPUT_PITCH=328"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_hc_fused_up_reduce_columns32 PROPERTIES
                ENVIRONMENT "MOLLM_TEST_CUDA_HC_UP_TILE_COLUMNS=32"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_hc_fused_up_reduce_columns32_pitch336 PROPERTIES
                ENVIRONMENT "MOLLM_TEST_CUDA_HC_UP_TILE_COLUMNS=32;MOLLM_TEST_CUDA_HC_UP_INPUT_PITCH=336"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_hc_fused_up_reduce_pitch328 PROPERTIES
                ENVIRONMENT "MOLLM_TEST_CUDA_HC_UP_INPUT_PITCH=328"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_hc_fused_up_reduce_pitch336 PROPERTIES
                ENVIRONMENT "MOLLM_TEST_CUDA_HC_UP_INPUT_PITCH=336"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_hc_fused_up_reduce PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_nvfp4_bf16_gate COMMAND bench_cuda_nvfp4_bf16_gate --check-only)
            add_test(NAME test_cuda_nvfp4_mma_activation_fused COMMAND bench_cuda_nvfp4_bf16_gate --check-only)
            add_test(NAME test_cuda_nvfp4_mma_activation_staged COMMAND bench_cuda_nvfp4_bf16_gate --check-only)
            set_tests_properties(test_cuda_nvfp4_mma_activation_staged PROPERTIES
                ENVIRONMENT "MOLLM_TEST_NVFP4_GATE_VECTOR_QUANTIZE=1;MOLLM_TEST_NVFP4_MMA_ACTIVATION_FUSED=1;MOLLM_TEST_NVFP4_GATE_STAGE_INPUTS=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_nvfp4_mma_activation_staged48 COMMAND bench_cuda_nvfp4_bf16_gate --check-only)
            add_test(NAME test_cuda_nvfp4_mma_activation_logits_pad COMMAND bench_cuda_nvfp4_bf16_gate --check-only)
            add_test(NAME test_cuda_nvfp4_mma_activation_preplanned COMMAND bench_cuda_nvfp4_bf16_gate --check-only)
            add_test(NAME test_cuda_nvfp4_mma_activation_weight_sf COMMAND bench_cuda_nvfp4_bf16_gate --check-only)
            add_test(NAME test_cuda_nvfp4_mma_activation_input_sf COMMAND bench_cuda_nvfp4_bf16_gate --check-only)
            add_test(NAME test_cuda_nvfp4_mma_activation_both_sf COMMAND bench_cuda_nvfp4_bf16_gate --check-only)
            add_test(NAME test_cuda_nvfp4_mma_activation_compact_logits COMMAND bench_cuda_nvfp4_bf16_gate --check-only)
            add_test(NAME test_cuda_nvfp4_mma_activation_token_quantization COMMAND bench_cuda_nvfp4_bf16_gate --check-only)
            add_test(NAME test_cuda_nvfp4_mma_activation_token_quantization_scatter COMMAND bench_cuda_nvfp4_bf16_gate --check-only)
            add_test(NAME test_cuda_nvfp4_mma_activation_token_pair_words COMMAND bench_cuda_nvfp4_bf16_gate --check-only)
            add_test(NAME test_cuda_nvfp4_mma_activation_token_pair_words_scatter COMMAND bench_cuda_nvfp4_bf16_gate --check-only)
            add_test(NAME test_cuda_nvfp4_mma_weight_locality_control COMMAND bench_cuda_nvfp4_bf16_gate --check-only)
            set_tests_properties(test_cuda_nvfp4_mma_weight_locality_control PROPERTIES
                ENVIRONMENT "MOLLM_TEST_NVFP4_GATE_VECTOR_QUANTIZE=1;MOLLM_TEST_NVFP4_MMA_ACTIVATION_FUSED=1;MOLLM_TEST_NVFP4_GATE_INPUT_SF_KMAJOR=1;MOLLM_TEST_NVFP4_GATE_WEIGHT_SF_KMAJOR=1;MOLLM_TEST_NVFP4_GATE_COMPACT_LOGITS=1;MOLLM_TEST_NVFP4_GATE_TOKEN_QUANTIZE=1;MOLLM_TEST_NVFP4_GATE_TOKEN_SCATTER=1;MOLLM_TEST_NVFP4_GATE_TOKEN_PAIR_WORDS=1;MOLLM_TEST_NVFP4_GATE_WEIGHT_LOCALITY=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            foreach(locality_scope codes scales)
                add_test(NAME test_cuda_nvfp4_mma_weight_locality_${locality_scope} COMMAND bench_cuda_nvfp4_bf16_gate --check-only)
                set_tests_properties(test_cuda_nvfp4_mma_weight_locality_${locality_scope} PROPERTIES
                    ENVIRONMENT "MOLLM_TEST_NVFP4_GATE_VECTOR_QUANTIZE=1;MOLLM_TEST_NVFP4_MMA_ACTIVATION_FUSED=1;MOLLM_TEST_NVFP4_GATE_INPUT_SF_KMAJOR=1;MOLLM_TEST_NVFP4_GATE_WEIGHT_SF_KMAJOR=1;MOLLM_TEST_NVFP4_GATE_COMPACT_LOGITS=1;MOLLM_TEST_NVFP4_GATE_TOKEN_QUANTIZE=1;MOLLM_TEST_NVFP4_GATE_TOKEN_SCATTER=1;MOLLM_TEST_NVFP4_GATE_TOKEN_PAIR_WORDS=1;MOLLM_TEST_NVFP4_GATE_WEIGHT_LOCALITY=${locality_scope}"
                    SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            endforeach()
            set_tests_properties(test_cuda_nvfp4_mma_activation_token_pair_words PROPERTIES
                ENVIRONMENT "MOLLM_TEST_NVFP4_GATE_VECTOR_QUANTIZE=1;MOLLM_TEST_NVFP4_MMA_ACTIVATION_FUSED=1;MOLLM_TEST_NVFP4_GATE_INPUT_SF_KMAJOR=1;MOLLM_TEST_NVFP4_GATE_WEIGHT_SF_KMAJOR=1;MOLLM_TEST_NVFP4_GATE_COMPACT_LOGITS=1;MOLLM_TEST_NVFP4_GATE_TOKEN_QUANTIZE=1;MOLLM_TEST_NVFP4_GATE_TOKEN_PAIR_WORDS=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_nvfp4_mma_activation_token_pair_words_scatter PROPERTIES
                ENVIRONMENT "MOLLM_TEST_NVFP4_GATE_VECTOR_QUANTIZE=1;MOLLM_TEST_NVFP4_MMA_ACTIVATION_FUSED=1;MOLLM_TEST_NVFP4_GATE_INPUT_SF_KMAJOR=1;MOLLM_TEST_NVFP4_GATE_WEIGHT_SF_KMAJOR=1;MOLLM_TEST_NVFP4_GATE_COMPACT_LOGITS=1;MOLLM_TEST_NVFP4_GATE_TOKEN_QUANTIZE=1;MOLLM_TEST_NVFP4_GATE_TOKEN_SCATTER=1;MOLLM_TEST_NVFP4_GATE_TOKEN_PAIR_WORDS=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_nvfp4_mma_activation_token_quantization_scatter PROPERTIES
                ENVIRONMENT "MOLLM_TEST_NVFP4_GATE_VECTOR_QUANTIZE=1;MOLLM_TEST_NVFP4_MMA_ACTIVATION_FUSED=1;MOLLM_TEST_NVFP4_GATE_INPUT_SF_KMAJOR=1;MOLLM_TEST_NVFP4_GATE_WEIGHT_SF_KMAJOR=1;MOLLM_TEST_NVFP4_GATE_COMPACT_LOGITS=1;MOLLM_TEST_NVFP4_GATE_TOKEN_QUANTIZE=1;MOLLM_TEST_NVFP4_GATE_TOKEN_SCATTER=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_nvfp4_gate_code_kmajor COMMAND bench_cuda_nvfp4_bf16_gate --check-only)
            add_test(NAME test_cuda_nvfp4_gate_token_code_kmajor COMMAND bench_cuda_nvfp4_bf16_gate --check-only)
            set_tests_properties(test_cuda_nvfp4_gate_code_kmajor PROPERTIES
                ENVIRONMENT "MOLLM_TEST_NVFP4_GATE_VECTOR_QUANTIZE=1;MOLLM_TEST_NVFP4_MMA_ACTIVATION_FUSED=1;MOLLM_TEST_NVFP4_GATE_INPUT_SF_KMAJOR=1;MOLLM_TEST_NVFP4_GATE_WEIGHT_SF_KMAJOR=1;MOLLM_TEST_NVFP4_GATE_COMPACT_LOGITS=1;MOLLM_TEST_NVFP4_GATE_TOKEN_QUANTIZE=1;MOLLM_TEST_NVFP4_GATE_TOKEN_SCATTER=1;MOLLM_TEST_NVFP4_GATE_TOKEN_PAIR_WORDS=1;MOLLM_TEST_NVFP4_GATE_CODE_KMAJOR=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            get_property(gate_code_test_env TEST test_cuda_nvfp4_gate_code_kmajor PROPERTY ENVIRONMENT)
            set_tests_properties(test_cuda_nvfp4_gate_token_code_kmajor PROPERTIES
                ENVIRONMENT "${gate_code_test_env};MOLLM_TEST_NVFP4_GATE_TOKEN_CODE_KMAJOR=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_nvfp4_gate_token_hoist COMMAND bench_cuda_nvfp4_bf16_gate --check-only)
            set_tests_properties(test_cuda_nvfp4_gate_token_hoist PROPERTIES
                ENVIRONMENT "${gate_code_test_env};MOLLM_TEST_NVFP4_GATE_TOKEN_CODE_KMAJOR=1;MOLLM_TEST_NVFP4_GATE_TOKEN_HOIST=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_nvfp4_mma_activation_token_quantization PROPERTIES
                ENVIRONMENT "MOLLM_TEST_NVFP4_GATE_VECTOR_QUANTIZE=1;MOLLM_TEST_NVFP4_MMA_ACTIVATION_FUSED=1;MOLLM_TEST_NVFP4_GATE_INPUT_SF_KMAJOR=1;MOLLM_TEST_NVFP4_GATE_WEIGHT_SF_KMAJOR=1;MOLLM_TEST_NVFP4_GATE_COMPACT_LOGITS=1;MOLLM_TEST_NVFP4_GATE_TOKEN_QUANTIZE=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_nvfp4_mma_activation_compact_logits PROPERTIES
                ENVIRONMENT "MOLLM_TEST_NVFP4_GATE_VECTOR_QUANTIZE=1;MOLLM_TEST_NVFP4_MMA_ACTIVATION_FUSED=1;MOLLM_TEST_NVFP4_GATE_INPUT_SF_KMAJOR=1;MOLLM_TEST_NVFP4_GATE_WEIGHT_SF_KMAJOR=1;MOLLM_TEST_NVFP4_GATE_COMPACT_LOGITS=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_nvfp4_mma_activation_staged_both_sf COMMAND bench_cuda_nvfp4_bf16_gate --check-only)
            set_tests_properties(test_cuda_nvfp4_mma_activation_staged_both_sf PROPERTIES
                ENVIRONMENT "MOLLM_TEST_NVFP4_GATE_VECTOR_QUANTIZE=1;MOLLM_TEST_NVFP4_MMA_ACTIVATION_FUSED=1;MOLLM_TEST_NVFP4_GATE_STAGE_INPUTS=1;MOLLM_TEST_NVFP4_GATE_INPUT_SF_KMAJOR=1;MOLLM_TEST_NVFP4_GATE_WEIGHT_SF_KMAJOR=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_nvfp4_mma_activation_input_sf PROPERTIES
                ENVIRONMENT "MOLLM_TEST_NVFP4_GATE_VECTOR_QUANTIZE=1;MOLLM_TEST_NVFP4_MMA_ACTIVATION_FUSED=1;MOLLM_TEST_NVFP4_GATE_INPUT_SF_KMAJOR=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_nvfp4_mma_activation_both_sf PROPERTIES
                ENVIRONMENT "MOLLM_TEST_NVFP4_GATE_VECTOR_QUANTIZE=1;MOLLM_TEST_NVFP4_MMA_ACTIVATION_FUSED=1;MOLLM_TEST_NVFP4_GATE_INPUT_SF_KMAJOR=1;MOLLM_TEST_NVFP4_GATE_WEIGHT_SF_KMAJOR=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_nvfp4_mma_activation_weight_sf PROPERTIES
                ENVIRONMENT "MOLLM_TEST_NVFP4_GATE_VECTOR_QUANTIZE=1;MOLLM_TEST_NVFP4_MMA_ACTIVATION_FUSED=1;MOLLM_TEST_NVFP4_GATE_WEIGHT_SF_KMAJOR=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_nvfp4_mma_activation_preplanned PROPERTIES
                ENVIRONMENT "MOLLM_TEST_NVFP4_GATE_VECTOR_QUANTIZE=1;MOLLM_TEST_NVFP4_MMA_ACTIVATION_FUSED=1;MOLLM_TEST_NVFP4_GATE_PREPLANNED=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_nvfp4_mma_activation_weights_skew COMMAND bench_cuda_nvfp4_bf16_gate --check-only)
            set_tests_properties(test_cuda_nvfp4_mma_activation_weights_skew PROPERTIES
                ENVIRONMENT "MOLLM_TEST_NVFP4_GATE_VECTOR_QUANTIZE=1;MOLLM_TEST_NVFP4_MMA_ACTIVATION_FUSED=1;MOLLM_TEST_NVFP4_GATE_SKEW_WEIGHTS=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_nvfp4_mma_activation_logits_pad PROPERTIES
                ENVIRONMENT "MOLLM_TEST_NVFP4_GATE_VECTOR_QUANTIZE=1;MOLLM_TEST_NVFP4_MMA_ACTIVATION_FUSED=1;MOLLM_TEST_NVFP4_GATE_PAD_LOGITS=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_nvfp4_mma_activation_staged48 PROPERTIES
                ENVIRONMENT "MOLLM_TEST_NVFP4_GATE_VECTOR_QUANTIZE=1;MOLLM_TEST_NVFP4_MMA_ACTIVATION_FUSED=1;MOLLM_TEST_NVFP4_GATE_STAGE_INPUTS=1;MOLLM_TEST_NVFP4_GATE_STAGE_INPUT_PITCH=48"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_moe_route_windows COMMAND bench_cuda_moe_route_windows --check-only)
            set_tests_properties(test_cuda_moe_route_windows PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_nvfp4_mma_activation_fused PROPERTIES
                ENVIRONMENT "MOLLM_TEST_NVFP4_GATE_VECTOR_QUANTIZE=1;MOLLM_TEST_NVFP4_MMA_ACTIVATION_FUSED=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_dense_lt_policy COMMAND bench_cuda_dense_lt_gemm --policy-only)
            add_test(NAME test_cuda_dense_full_lt COMMAND bench_cuda_dense_lt_gemm --logical-policy-only)
            set_tests_properties(test_cuda_dense_full_lt PROPERTIES
                ENVIRONMENT "MOLLM_TEST_CUDA_DENSE_WIDE_FULL_LT=1;MOLLM_TEST_CUDA_DENSE_WIDE_ROW_TILE=0"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_dense_full_lt_default COMMAND bench_cuda_dense_lt_gemm --logical-policy-only)
            add_test(NAME test_cuda_dense_wide_tile23 COMMAND bench_cuda_dense_lt_gemm --logical-policy-only)
            set_tests_properties(test_cuda_dense_wide_tile23 PROPERTIES
                ENVIRONMENT "MOLLM_TEST_CUDA_DENSE_WIDE_FULL_LT=1;MOLLM_TEST_CUDA_DENSE_WIDE_ROW_TILE=0;MOLLM_TEST_CUDA_DENSE_WIDE_TILE=23"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_dense_wide_tile23_invalid COMMAND bench_cuda_dense_lt_gemm --logical-policy-only)
            set_tests_properties(test_cuda_dense_wide_tile23_invalid PROPERTIES
                ENVIRONMENT "MOLLM_TEST_CUDA_DENSE_WIDE_FULL_LT=1;MOLLM_TEST_CUDA_DENSE_WIDE_ROW_TILE=0;MOLLM_TEST_CUDA_DENSE_WIDE_TILE=24"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_dense_wide_tile23_short COMMAND bench_cuda_dense_lt_gemm --policy-only)
            set_tests_properties(test_cuda_dense_wide_tile23_short PROPERTIES
                ENVIRONMENT "MOLLM_TEST_CUDA_DENSE_WIDE_FULL_LT=1;MOLLM_TEST_CUDA_DENSE_WIDE_ROW_TILE=0;MOLLM_TEST_CUDA_DENSE_WIDE_TILE=23"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_dense_wide_tile23_without_full COMMAND bench_cuda_dense_lt_gemm --logical-policy-only)
            set_tests_properties(test_cuda_dense_wide_tile23_without_full PROPERTIES
                ENVIRONMENT "MOLLM_TEST_CUDA_DENSE_WIDE_FULL_LT=0;MOLLM_TEST_CUDA_DENSE_WIDE_ROW_TILE=0;MOLLM_TEST_CUDA_DENSE_WIDE_TILE=23"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_dense_full_lt_default PROPERTIES
                ENVIRONMENT "MOLLM_TEST_CUDA_DENSE_WIDE_FULL_LT=0;MOLLM_TEST_CUDA_DENSE_WIDE_ROW_TILE=0"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_dense_window_partition COMMAND bench_cuda_dense_lt_gemm --window-merge-check)
            add_test(NAME test_cuda_dense_wide_row_tile COMMAND bench_cuda_dense_lt_gemm --window-merge-check)
            set_tests_properties(test_cuda_dense_wide_row_tile PROPERTIES
                ENVIRONMENT "MOLLM_TEST_CUDA_DENSE_WIDE_ROW_TILE=1024"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_hc_gate_row_tile COMMAND bench_cuda_dense_lt_gemm --hc-gate-tile-check)
            set_tests_properties(test_cuda_hc_gate_row_tile PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_dense_window_balance COMMAND bench_cuda_dense_lt_gemm --window-balance-check)
            set_tests_properties(test_cuda_dense_window_partition test_cuda_dense_window_balance PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_nvfp4_parallel_batch_plan COMMAND bench_cuda_nvfp4_batch_plan --check-only)
            set_tests_properties(test_cuda_nvfp4_parallel_batch_plan PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_dense_lt_policy PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_nvfp4_bf16_gate PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_nvfp4_bf16_gate_early_inputs COMMAND bench_cuda_nvfp4_bf16_gate --check-only)
            add_test(NAME test_cuda_nvfp4_bf16_gate_vector_quantize COMMAND bench_cuda_nvfp4_bf16_gate --check-only)
            set_tests_properties(test_cuda_nvfp4_bf16_gate_vector_quantize PROPERTIES
                ENVIRONMENT "MOLLM_TEST_NVFP4_GATE_VECTOR_QUANTIZE=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_nvfp4_bf16_gate_early_inputs PROPERTIES
                ENVIRONMENT "MOLLM_TEST_NVFP4_EARLY_INPUTS=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_nvfp4_bf16_output PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_group_rms_half COMMAND bench_cuda_group_rms_half)
            add_test(NAME test_cuda_group_rms_half_cached COMMAND bench_cuda_group_rms_half --cached-input)
            add_test(NAME test_cuda_gr_inject_rms_half COMMAND bench_cuda_group_rms_half --inject-rms)
            add_test(NAME test_cuda_gr_deferred_rms COMMAND bench_cuda_gr_deferred_rms --check-only)
            set_tests_properties(test_cuda_gr_deferred_rms PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_group_rms_half test_cuda_group_rms_half_cached test_cuda_gr_inject_rms_half PROPERTIES
                ENVIRONMENT "MOLLM_GROUP_RMS_HALF_CHECK_ONLY=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        endif()
        if(TARGET bench_cuda_sdpa_prefill)
            add_test(NAME test_cuda_sdpa_gemm_ordered_pv COMMAND bench_cuda_sdpa_prefill --gemm-ordered-pv-check)
            set_tests_properties(test_cuda_sdpa_gemm_ordered_pv PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        endif()
        if(TARGET bench_cuda_sdpa_validated_prefix)
            add_test(NAME test_cuda_sdpa_mid_pv_guards
                COMMAND bench_cuda_sdpa_validated_prefix --native-pv-gemm-mid-sanitize)
            set_tests_properties(test_cuda_sdpa_mid_pv_guards PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_sdpa_native_pv_gemm COMMAND bench_cuda_sdpa_validated_prefix --native-pv-gemm-sanitize)
            add_test(NAME test_cuda_sdpa_native_pv_capacity32k
                COMMAND ${CMAKE_COMMAND} -E env MOLLM_SDPA_VALIDATED_PREFIX_CHECK_ONLY=1
                    $<TARGET_FILE:bench_cuda_sdpa_validated_prefix> --native-pv-capacity32k)
            add_test(NAME test_cuda_sdpa_native_pv_capacity32k_sanitize
                COMMAND bench_cuda_sdpa_validated_prefix --native-pv-capacity32k-sanitize)
            set_tests_properties(test_cuda_sdpa_native_pv_capacity32k test_cuda_sdpa_native_pv_capacity32k_sanitize
                PROPERTIES SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE TIMEOUT 240)
            foreach(head_lane_fixture IN ITEMS reuse2 reuse2-long reuse2-sanitize)
                add_test(NAME test_cuda_sdpa_head_lane_${head_lane_fixture}
                    COMMAND ${CMAKE_COMMAND} -E env MOLLM_SDPA_VALIDATED_PREFIX_CHECK_ONLY=1
                        $<TARGET_FILE:bench_cuda_sdpa_validated_prefix> --head-lane-${head_lane_fixture})
                set_tests_properties(test_cuda_sdpa_head_lane_${head_lane_fixture} PROPERTIES
                    SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE TIMEOUT 240)
            endforeach()
            set_tests_properties(test_cuda_sdpa_native_pv_gemm PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_sdpa_full_batch COMMAND bench_cuda_sdpa_validated_prefix --full-batch)
            add_test(NAME test_cuda_sdpa_full_batch_group2 COMMAND bench_cuda_sdpa_validated_prefix --full-batch-group2)
            add_test(NAME test_cuda_sdpa_full_batch_group4 COMMAND bench_cuda_sdpa_validated_prefix --full-batch-group4)
            add_test(NAME test_cuda_sdpa_full_batch_heads4 COMMAND bench_cuda_sdpa_validated_prefix --full-batch-heads4)
            set_tests_properties(test_cuda_sdpa_full_batch test_cuda_sdpa_full_batch_group2 test_cuda_sdpa_full_batch_group4 test_cuda_sdpa_full_batch_heads4 PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_sdpa_prefix_query_pair COMMAND bench_cuda_sdpa_validated_prefix --pair-queries)
            add_test(NAME test_cuda_sdpa_prefix_query_group4 COMMAND bench_cuda_sdpa_validated_prefix --pair-queries4)
            add_test(NAME test_cuda_sdpa_prefix_query_group4_long_shared COMMAND bench_cuda_sdpa_validated_prefix --pair-queries4-long-shared-small)
            add_test(NAME test_cuda_sdpa_prefix_head_group4 COMMAND bench_cuda_sdpa_validated_prefix --group-heads4-small)
            set_tests_properties(test_cuda_sdpa_prefix_head_group4 PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_sdpa_prefix_query_group4_long_shared PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_sdpa_prefix_query_pair test_cuda_sdpa_prefix_query_group4 PROPERTIES
                ENVIRONMENT "MOLLM_SDPA_VALIDATED_PREFIX_CHECK_ONLY=1" SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_sdpa_pv_dimension_pair_guards COMMAND bench_cuda_sdpa_validated_prefix --pair-pv)
            set_tests_properties(test_cuda_sdpa_pv_dimension_pair_guards PROPERTIES
                ENVIRONMENT "MOLLM_SDPA_VALIDATED_PREFIX_CHECK_ONLY=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_sdpa_pv_unroll_guards COMMAND bench_cuda_sdpa_validated_prefix --pv-unroll)
            set_tests_properties(test_cuda_sdpa_pv_unroll_guards PROPERTIES
                ENVIRONMENT "MOLLM_SDPA_VALIDATED_PREFIX_CHECK_ONLY=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_sdpa_validated_prefix_long_shared_zero_pv_guards COMMAND bench_cuda_sdpa_validated_prefix --long-shared-zero-pv)
            set_tests_properties(test_cuda_sdpa_validated_prefix_long_shared_zero_pv_guards PROPERTIES
                ENVIRONMENT "MOLLM_SDPA_VALIDATED_PREFIX_CHECK_ONLY=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_sdpa_validated_zero_pv_guards COMMAND bench_cuda_sdpa_validated_prefix --skip-zero-pv)
            set_tests_properties(test_cuda_sdpa_validated_zero_pv_guards PROPERTIES
                ENVIRONMENT "MOLLM_SDPA_VALIDATED_PREFIX_CHECK_ONLY=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_sdpa_validated_prefix_guards COMMAND bench_cuda_sdpa_validated_prefix)
            set_tests_properties(test_cuda_sdpa_validated_prefix_guards PROPERTIES
                ENVIRONMENT "MOLLM_SDPA_VALIDATED_PREFIX_CHECK_ONLY=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        endif()
        if(TARGET bench_cuda_nvfp4_quantize)
            add_test(NAME test_cuda_nvfp4_quantize_guards COMMAND bench_cuda_nvfp4_quantize)
            add_test(NAME test_cuda_nvfp4_shared_max_cache_guards COMMAND bench_cuda_nvfp4_quantize)
            set_tests_properties(test_cuda_nvfp4_shared_max_cache_guards PROPERTIES
                ENVIRONMENT "MOLLM_NVFP4_SHARED_MAX_PROBE=1;MOLLM_NVFP4_QUANTIZE_CHECK_ONLY=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_nvfp4_quantize_guards PROPERTIES
                ENVIRONMENT "MOLLM_NVFP4_QUANTIZE_CHECK_ONLY=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        endif()
        if(TARGET bench_cuda_gdn_recurrence)
            add_test(NAME test_cuda_gdn_chunkwise_prototype COMMAND bench_cuda_gdn_chunkwise --check-only)
            add_test(NAME test_cuda_gdn_chunkwise_column_solve COMMAND bench_cuda_gdn_chunkwise --check-only --column-solve --small-only)
            add_test(NAME test_cuda_gdn_chunkwise_coalesced_values COMMAND bench_cuda_gdn_chunkwise --check-only --coalesced-values --small-only)
            add_test(NAME test_cuda_gdn_chunkwise_dual_projection COMMAND bench_cuda_gdn_chunkwise --check-only --dual-projection --small-only)
            add_test(NAME test_cuda_gdn_chunkwise_shared_grams COMMAND bench_cuda_gdn_chunkwise --check-only --shared-grams --small-only)
            add_test(NAME test_cuda_gdn_chunkwise_shared_grams_combo COMMAND bench_cuda_gdn_chunkwise --check-only --shared-grams --dual-projection --coalesced-values --small-only)
            add_test(NAME test_cuda_gdn_chunkwise_staged_coefficients COMMAND bench_cuda_gdn_chunkwise --check-only --staged-coefficients --small-only)
            add_test(NAME test_cuda_gdn_chunkwise_logical_shapes
                COMMAND bench_cuda_gdn_chunkwise --check-only --staged-coefficients --logical-shapes)
            set_tests_properties(test_cuda_gdn_chunkwise_logical_shapes PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_gdn_chunkwise_tensor_coeff_combo COMMAND bench_cuda_gdn_chunkwise --check-only --tensor-coeff-combo --small-only)
            set_tests_properties(test_cuda_gdn_chunkwise_tensor_coeff_combo PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_gdn_chunkwise_staged_coefficients PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_gdn_chunkwise_shared_grams test_cuda_gdn_chunkwise_shared_grams_combo PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_gdn_chunkwise_dual_projection PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_gdn_chunkwise_coalesced_values PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_gdn_chunkwise_column_solve PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            set_tests_properties(test_cuda_gdn_chunkwise_prototype PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_gdn_chunkwise_tf32_tile16 COMMAND bench_cuda_gdn_chunkwise --check-only --tensor-core)
            add_test(NAME test_cuda_gdn_chunkwise_tf32_tile32 COMMAND bench_cuda_gdn_chunkwise --check-only --value-tile32)
            add_test(NAME test_cuda_gdn_chunkwise_tf32_prepare COMMAND bench_cuda_gdn_chunkwise --check-only --tensor-prepare)
            add_test(NAME test_cuda_gdn_chunkwise_tf32_padded COMMAND bench_cuda_gdn_chunkwise --check-only --padded)
            add_test(NAME test_cuda_gdn_chunkwise_shared_update COMMAND bench_cuda_gdn_chunkwise --check-only --shared-update)
            add_test(NAME test_cuda_gdn_chunkwise_shared_eight COMMAND bench_cuda_gdn_chunkwise --check-only --shared-update --eight-warps)
            add_test(NAME test_cuda_gdn_chunkwise_half_products COMMAND bench_cuda_gdn_chunkwise --check-only --half-products)
            add_test(NAME test_cuda_gdn_chunkwise_half_eight COMMAND bench_cuda_gdn_chunkwise --check-only --half-products --eight-warps)
            add_test(NAME test_cuda_gdn_chunkwise_tensor_coeff COMMAND bench_cuda_gdn_chunkwise --check-only --tensor-coeff)
            set_tests_properties(test_cuda_gdn_chunkwise_tf32_tile16 test_cuda_gdn_chunkwise_tf32_tile32
                test_cuda_gdn_chunkwise_tf32_prepare test_cuda_gdn_chunkwise_tf32_padded
                test_cuda_gdn_chunkwise_shared_update test_cuda_gdn_chunkwise_shared_eight
                test_cuda_gdn_chunkwise_half_products test_cuda_gdn_chunkwise_half_eight
                test_cuda_gdn_chunkwise_tensor_coeff PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_gdn_quad_values_guards COMMAND bench_cuda_gdn_recurrence --quad-values)
            set_tests_properties(test_cuda_gdn_quad_values_guards PROPERTIES
                ENVIRONMENT "MOLLM_GDN_RECURRENCE_CHECK_ONLY=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_gdn_paired_values_guards COMMAND bench_cuda_gdn_recurrence --paired-values)
            set_tests_properties(test_cuda_gdn_paired_values_guards PROPERTIES
                ENVIRONMENT "MOLLM_GDN_RECURRENCE_CHECK_ONLY=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        endif()
        if(TARGET bench_cuda_nvfp4_activation)
            add_test(NAME test_cuda_nvfp4_activation_prepare COMMAND bench_cuda_nvfp4_activation)
            set_tests_properties(test_cuda_nvfp4_activation_prepare PROPERTIES
                ENVIRONMENT "MOLLM_NVFP4_ACTIVATION_CHECK_ONLY=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        endif()
        set_tests_properties(test_cuda_sigmoid_gr_reduce PROPERTIES
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        add_test(NAME test_cuda_gr_inject_tiled COMMAND bench_cuda_gr_reduce --inject-check)
        set_tests_properties(test_cuda_gr_inject_tiled PROPERTIES
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        add_mollm_test(test_cuda_backend tests/test_cuda_backend.cpp)
        add_mollm_test(test_cuda_nvfp4_logical_batch tests/test_cuda_nvfp4_logical_batch.cpp)
        set_tests_properties(test_cuda_nvfp4_logical_batch PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_HOST_HEADROOM_MB=102400;MOLLM_CUDA_DISABLE_MOE_GRAPHS=1;MOLLM_CUDA_DISABLE_DECODE_SEGMENTS=1;MOLLM_CUDA_NVFP4_MMA_PREFILL_ONLY=1;MOLLM_CUDA_NVFP4_MMA_PREFILL_BATCH=1;MOLLM_CUDA_NVFP4_MMA_PREFILL_BATCH_SMALL=1;MOLLM_CUDA_NVFP4_MMA_GATE_MIN_BATCH=1;MOLLM_CUDA_NVFP4_MMA_DOWN_MIN_BATCH=1;MOLLM_CUDA_NVFP4_MMA_BATCH_WARPS=4;MOLLM_CUDA_NVFP4_MMA_BATCH_ROUTES=8;MOLLM_CUDA_NVFP4_MMA_GATE_BATCH_ROUTES=32;MOLLM_CUDA_NVFP4_PREFILL_TILED_GROUP=1;MOLLM_CUDA_NVFP4_N32_ASYNC_WEIGHTS=1;MOLLM_CUDA_NVFP4_ACTIVATION_PREPARE_FUSION=2;MOLLM_CUDA_NVFP4_PREFILL_BF16_GATE_OUTPUT=1;MOLLM_CUDA_NVFP4_PREFILL_BF16_DOWN_OUTPUT=1;MOLLM_CUDA_NVFP4_PREFILL_EARLY_INPUTS=1;MOLLM_CUDA_NVFP4_PREFILL_VECTOR_QUANTIZE=1;MOLLM_CUDA_NVFP4_PREFILL_PARALLEL_PLAN=1;MOLLM_CUDA_NVFP4_PREFILL_MMA_ACTIVATION_FUSED=1;MOLLM_CUDA_PREFILL_GATE_SCALE_CACHE_MB=16;MOLLM_CUDA_NVFP4_PREFILL_INPUT_SCALE_KMAJOR=1;MOLLM_CUDA_NVFP4_PREFILL_COMPACT_LOGITS=1;MOLLM_CUDA_NVFP4_PREFILL_TOKEN_QUANTIZE=1;MOLLM_CUDA_NVFP4_PREFILL_TOKEN_PAIR_WORDS=1"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        add_mollm_test(test_cuda_segmented_execution tests/test_cuda_segmented_execution.cpp)
        add_test(NAME test_cuda_nvfp4_resident_gate_code_view COMMAND test_cuda_nvfp4_logical_batch)
        get_property(nvfp4_logical_test_env TEST test_cuda_nvfp4_logical_batch PROPERTY ENVIRONMENT)
        add_test(NAME test_cuda_nvfp4_logical_scale_view COMMAND test_cuda_nvfp4_logical_batch)
        set_tests_properties(test_cuda_nvfp4_logical_scale_view PROPERTIES
            ENVIRONMENT "${nvfp4_logical_test_env};MOLLM_TEST_NVFP4_RESIDENT_GATE_CODE_VIEW=1;MOLLM_TEST_NVFP4_RESIDENT_GATE_SCALE_VIEW=1"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        add_test(NAME test_cuda_nvfp4_logical_scale_view_auto_no_cache
            COMMAND ${CMAKE_COMMAND} -E env
                --unset=MOLLM_CUDA_NVFP4_PREFILL_INPUT_SCALE_KMAJOR
                --unset=MOLLM_CUDA_NVFP4_PREFILL_COMPACT_LOGITS
                --unset=MOLLM_CUDA_NVFP4_PREFILL_TOKEN_QUANTIZE
                --unset=MOLLM_CUDA_NVFP4_PREFILL_TOKEN_PAIR_WORDS
                MOLLM_CUDA_PREFILL_GATE_SCALE_CACHE_MB=0
                $<TARGET_FILE:test_cuda_nvfp4_logical_batch>)
        set_tests_properties(test_cuda_nvfp4_logical_scale_view_auto_no_cache PROPERTIES
            ENVIRONMENT "${nvfp4_logical_test_env};MOLLM_TEST_NVFP4_RESIDENT_GATE_CODE_VIEW=1;MOLLM_TEST_NVFP4_RESIDENT_GATE_SCALE_VIEW=1"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        foreach(code_view IN ITEMS 0 1)
            add_test(NAME test_cuda_nvfp4_logical_defaults_auto_${code_view}
                COMMAND ${CMAKE_COMMAND} -E env
                    --unset=MOLLM_CUDA_NVFP4_PREFILL_INPUT_SCALE_KMAJOR
                    --unset=MOLLM_CUDA_NVFP4_PREFILL_COMPACT_LOGITS
                    --unset=MOLLM_CUDA_NVFP4_PREFILL_TOKEN_QUANTIZE
                    --unset=MOLLM_CUDA_NVFP4_PREFILL_TOKEN_PAIR_WORDS
                    --unset=MOLLM_CUDA_NVFP4_PREFILL_TOKEN_CODES_KMAJOR
                    $<TARGET_FILE:test_cuda_nvfp4_logical_batch>)
            set_tests_properties(test_cuda_nvfp4_logical_defaults_auto_${code_view} PROPERTIES
                ENVIRONMENT "${nvfp4_logical_test_env};MOLLM_TEST_NVFP4_RESIDENT_GATE_CODE_VIEW=${code_view}"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        endforeach()
        foreach(option IN ITEMS INPUT_SCALE_KMAJOR COMPACT_LOGITS TOKEN_QUANTIZE TOKEN_PAIR_WORDS)
            add_test(NAME test_cuda_nvfp4_logical_${option}_off
                COMMAND ${CMAKE_COMMAND} -E env "MOLLM_CUDA_NVFP4_PREFILL_${option}=0"
                    $<TARGET_FILE:test_cuda_nvfp4_logical_batch>)
            set_tests_properties(test_cuda_nvfp4_logical_${option}_off PROPERTIES
                ENVIRONMENT "${nvfp4_logical_test_env}"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_nvfp4_logical_code_view_${option}_off
                COMMAND ${CMAKE_COMMAND} -E env "MOLLM_CUDA_NVFP4_PREFILL_${option}=0"
                    $<TARGET_FILE:test_cuda_nvfp4_logical_batch>)
            set_tests_properties(test_cuda_nvfp4_logical_code_view_${option}_off PROPERTIES
                ENVIRONMENT "${nvfp4_logical_test_env};MOLLM_TEST_NVFP4_RESIDENT_GATE_CODE_VIEW=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            add_test(NAME test_cuda_nvfp4_logical_scale_view_${option}_off
                COMMAND ${CMAKE_COMMAND} -E env "MOLLM_CUDA_NVFP4_PREFILL_${option}=0"
                    $<TARGET_FILE:test_cuda_nvfp4_logical_batch>)
            set_tests_properties(test_cuda_nvfp4_logical_scale_view_${option}_off PROPERTIES
                ENVIRONMENT "${nvfp4_logical_test_env};MOLLM_TEST_NVFP4_RESIDENT_GATE_CODE_VIEW=1;MOLLM_TEST_NVFP4_RESIDENT_GATE_SCALE_VIEW=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        endforeach()
        set_tests_properties(test_cuda_nvfp4_resident_gate_code_view PROPERTIES
            ENVIRONMENT "${nvfp4_logical_test_env};MOLLM_TEST_NVFP4_RESIDENT_GATE_CODE_VIEW=1"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        add_mollm_test(test_cuda_hc_compact_materialize tests/test_cuda_hc_compact_materialize.cpp)
        add_test(NAME test_cuda_nvfp4_resident_token_code_view COMMAND test_cuda_nvfp4_logical_batch)
        add_test(NAME test_cuda_nvfp4_resident_token_code_auto
            COMMAND ${CMAKE_COMMAND} -E env --unset=MOLLM_CUDA_NVFP4_PREFILL_TOKEN_CODES_KMAJOR
                $<TARGET_FILE:test_cuda_nvfp4_logical_batch>)
        add_test(NAME test_cuda_nvfp4_resident_token_code_off COMMAND test_cuda_nvfp4_logical_batch)
        set_tests_properties(test_cuda_nvfp4_resident_token_code_auto PROPERTIES
            ENVIRONMENT "${nvfp4_logical_test_env};MOLLM_TEST_NVFP4_RESIDENT_GATE_CODE_VIEW=1"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        set_tests_properties(test_cuda_nvfp4_resident_token_code_off PROPERTIES
            ENVIRONMENT "${nvfp4_logical_test_env};MOLLM_TEST_NVFP4_RESIDENT_GATE_CODE_VIEW=1;MOLLM_CUDA_NVFP4_PREFILL_TOKEN_CODES_KMAJOR=0"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        add_test(NAME test_cuda_nvfp4_token_code_without_weight_view COMMAND test_cuda_nvfp4_logical_batch)
        set_tests_properties(test_cuda_nvfp4_token_code_without_weight_view PROPERTIES
            ENVIRONMENT "${nvfp4_logical_test_env};MOLLM_CUDA_NVFP4_PREFILL_TOKEN_CODES_KMAJOR=1"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        set_tests_properties(test_cuda_nvfp4_resident_token_code_view PROPERTIES
            ENVIRONMENT "${nvfp4_logical_test_env};MOLLM_TEST_NVFP4_RESIDENT_GATE_CODE_VIEW=1;MOLLM_CUDA_NVFP4_PREFILL_TOKEN_CODES_KMAJOR=1"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        set_tests_properties(test_cuda_hc_compact_materialize PROPERTIES
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        add_mollm_test(test_cuda_group_rms_deferred tests/test_cuda_group_rms_deferred.cpp)
        set_tests_properties(test_cuda_group_rms_deferred PROPERTIES
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        set_tests_properties(test_cuda_segmented_execution PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_DISABLE_DECODE_SEGMENTS=1"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        add_test(NAME test_cuda_segmented_execution_gdn_quad COMMAND test_cuda_segmented_execution)
        set_tests_properties(test_cuda_segmented_execution_gdn_quad PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_DISABLE_DECODE_SEGMENTS=1;MOLLM_CUDA_GDN_PREFILL_PAIRED_VALUES=1;MOLLM_CUDA_GDN_PREFILL_QUAD_VALUES=1"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        add_test(NAME test_cuda_segmented_execution_sdpa_validated COMMAND test_cuda_segmented_execution)
        set_tests_properties(test_cuda_segmented_execution_sdpa_validated PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_DISABLE_DECODE_SEGMENTS=1;MOLLM_CUDA_SDPA_PREFILL_VALIDATED_PREFIX=1;MOLLM_CUDA_SDPA_PREFILL_SHARED_SCORE_LIMIT=1024;MOLLM_CUDA_SDPA_PREFILL_SKIP_ZERO_PV=1;MOLLM_CUDA_SDPA_PREFILL_PV_UNROLL=1;MOLLM_CUDA_SDPA_PREFILL_PV_PAIR=1"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        add_test(NAME test_cuda_segmented_execution_one_graph COMMAND test_cuda_segmented_execution)
        set_tests_properties(test_cuda_segmented_execution_one_graph PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_DISABLE_DECODE_SEGMENTS=1;MOLLM_TEST_CUDA_SEGMENTED_ONE_GRAPH=1"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        add_test(NAME test_cuda_segmented_execution_one_graph_validated COMMAND test_cuda_segmented_execution)
        set_tests_properties(test_cuda_segmented_execution_one_graph_validated PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_DISABLE_DECODE_SEGMENTS=1;MOLLM_TEST_CUDA_SEGMENTED_ONE_GRAPH=1;MOLLM_CUDA_GDN_PREFILL_PAIRED_VALUES=1;MOLLM_CUDA_GDN_PREFILL_QUAD_VALUES=1;MOLLM_CUDA_SDPA_PREFILL_VALIDATED_PREFIX=1;MOLLM_CUDA_SDPA_PREFILL_SHARED_SCORE_LIMIT=1024;MOLLM_CUDA_SDPA_PREFILL_SKIP_ZERO_PV=1;MOLLM_CUDA_SDPA_PREFILL_PV_UNROLL=1;MOLLM_CUDA_SDPA_PREFILL_PV_PAIR=1"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        if(MOLLM_CUDA_DECODE_SEGMENTS)
            target_compile_definitions(test_cuda_backend PRIVATE MOLLM_CUDA_DECODE_SEGMENTS=1)
        endif()
        add_test(NAME test_cuda_stateful_contracts COMMAND test_cuda_backend --stateful-contract-only)
        foreach(shared_cache_mode disabled enabled invalid limited partial zero compact)
            add_test(NAME test_cuda_shared_fp16_cache_${shared_cache_mode} COMMAND test_cuda_backend --shared-cache-only)
            set_tests_properties(test_cuda_shared_fp16_cache_${shared_cache_mode} PROPERTIES
                LABELS "cuda" RUN_SERIAL TRUE SKIP_RETURN_CODE 77)
        endforeach()
        set_tests_properties(test_cuda_shared_fp16_cache_disabled PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_PREFILL_FP16_CACHE_SHARED=0;MOLLM_CUDA_PREFILL_FP16_CACHE_MB=128")
        set_tests_properties(test_cuda_shared_fp16_cache_enabled PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_PREFILL_FP16_CACHE_SHARED=1;MOLLM_CUDA_PREFILL_FP16_CACHE_MB=128")
        set_tests_properties(test_cuda_shared_fp16_cache_invalid PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_PREFILL_FP16_CACHE_SHARED=bad;MOLLM_CUDA_PREFILL_FP16_CACHE_MB=128")
        set_tests_properties(test_cuda_shared_fp16_cache_limited PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_PREFILL_FP16_CACHE_SHARED=1;MOLLM_CUDA_PREFILL_FP16_CACHE_MB=1")
        set_tests_properties(test_cuda_shared_fp16_cache_partial PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_PREFILL_FP16_CACHE_SHARED=1;MOLLM_CUDA_PREFILL_FP16_CACHE_MB=5")
        set_tests_properties(test_cuda_shared_fp16_cache_zero PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_PREFILL_FP16_CACHE_SHARED=1;MOLLM_CUDA_PREFILL_FP16_CACHE_MB=0")
        set_tests_properties(test_cuda_shared_fp16_cache_compact PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_PREFILL_FP16_CACHE_SHARED=1;MOLLM_CUDA_PREFILL_FP16_CACHE_MB=128;MOLLM_CUDA_Q4_COMPACT=1")
        foreach(reshape_mode disabled enabled invalid capture)
            add_test(NAME test_cuda_strided_reshape_${reshape_mode} COMMAND test_cuda_backend --strided-reshape-only)
            set_tests_properties(test_cuda_strided_reshape_${reshape_mode} PROPERTIES
                LABELS "cuda" RUN_SERIAL TRUE SKIP_RETURN_CODE 77)
        endforeach()
        set_tests_properties(test_cuda_strided_reshape_disabled PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_STRIDED_RESHAPE_VIEWS=0;MOLLM_CUDA_DISABLE_DECODE_SEGMENTS=1")
        set_tests_properties(test_cuda_strided_reshape_enabled PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_STRIDED_RESHAPE_VIEWS=1;MOLLM_CUDA_DISABLE_DECODE_SEGMENTS=1")
        set_tests_properties(test_cuda_strided_reshape_invalid PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_STRIDED_RESHAPE_VIEWS=bad;MOLLM_CUDA_DISABLE_DECODE_SEGMENTS=1")
        set_tests_properties(test_cuda_strided_reshape_capture PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_STRIDED_RESHAPE_VIEWS=1")
        add_test(NAME test_cuda_stateful_contract_prefixes COMMAND test_cuda_backend --stateful-contract-only)
        set_tests_properties(test_cuda_stateful_contracts test_cuda_stateful_contract_prefixes PROPERTIES
            LABELS "cuda" SKIP_RETURN_CODE 77)
        set_tests_properties(test_cuda_stateful_contract_prefixes PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_MTP_STATE_PREFIXES=1;MOLLM_CUDA_DISABLE_DECODE_SEGMENTS=1")
        set_tests_properties(test_cuda_stateful_contracts PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_DISABLE_DECODE_SEGMENTS=1")
        add_test(NAME test_cuda_backend_gdn_quad_values COMMAND test_cuda_backend)
        set_tests_properties(test_cuda_backend_gdn_quad_values PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_GDN_PREFILL_QUAD_VALUES=1;MOLLM_CUDA_GDN_PREFILL_PAIRED_VALUES=1"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        add_mollm_test(test_cuda_group_rms tests/test_cuda_group_rms.cpp)
        add_mollm_test(test_cuda_recurrent_prefix tests/test_cuda_recurrent_prefix.cpp)
        set_tests_properties(test_cuda_recurrent_prefix PROPERTIES SKIP_RETURN_CODE 77 LABELS "cuda")
        set_tests_properties(test_cuda_group_rms PROPERTIES SKIP_RETURN_CODE 77 LABELS "cuda")
        add_mollm_test(test_cuda_hc_projection tests/test_cuda_hc_projection.cpp)
        set_tests_properties(test_cuda_hc_projection PROPERTIES SKIP_RETURN_CODE 77 LABELS "cuda")
        add_test(NAME test_cuda_hc_compact_backend COMMAND test_cuda_hc_projection --compact)
        set_tests_properties(test_cuda_hc_compact_backend PROPERTIES SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        add_mollm_test(test_cuda_fp16_moe tests/test_cuda_fp16_moe.cpp)
        add_mollm_test(test_cuda_q4_small_batch tests/test_cuda_q4_small_batch.cpp)
        add_executable(probe_cuda_verify_consistency tests/probe_cuda_verify_consistency.cpp)
        target_link_libraries(probe_cuda_verify_consistency PRIVATE mollm_lib)
        set_tests_properties(test_cuda_q4_small_batch PROPERTIES SKIP_RETURN_CODE 77)
        set_tests_properties(test_cuda_fp16_moe PROPERTIES SKIP_RETURN_CODE 77)
        set_tests_properties(test_cuda_backend PROPERTIES SKIP_RETURN_CODE 77)
        add_test(NAME test_cuda_backend_prefill_fp16_cache COMMAND test_cuda_backend)
        set_tests_properties(test_cuda_backend_prefill_fp16_cache PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_PREFILL_FP16_CACHE_MB=128"
            LABELS "cuda" SKIP_RETURN_CODE 77)
        add_test(NAME test_cuda_backend_prefill_fp16_cache_limited COMMAND test_cuda_backend)
        set_tests_properties(test_cuda_backend_prefill_fp16_cache_limited PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_PREFILL_FP16_CACHE_MB=1"
            LABELS "cuda" SKIP_RETURN_CODE 77)
        add_test(NAME test_cuda_backend_prefill_fp16_cache_compact COMMAND test_cuda_backend)
        set_tests_properties(test_cuda_backend_prefill_fp16_cache_compact PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_PREFILL_FP16_CACHE_MB=128;MOLLM_CUDA_Q4_COMPACT=1;MOLLM_CUDA_Q4_COMPACT_13312=1"
            LABELS "cuda" SKIP_RETURN_CODE 77)
        add_test(NAME test_cuda_backend_prefill_fp16_cache_narrow COMMAND test_cuda_backend)
        set_tests_properties(test_cuda_backend_prefill_fp16_cache_narrow PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_PREFILL_FP16_CACHE_MB=128;MOLLM_CUDA_PREFILL_FP16_CACHE_NARROW=1;MOLLM_TEST_CUDA_NARROW_CACHE=1"
            LABELS "cuda" SKIP_RETURN_CODE 77)
        add_test(NAME test_cuda_backend_prefill_fp16_cache_narrow_limited COMMAND test_cuda_backend)
        set_tests_properties(test_cuda_backend_prefill_fp16_cache_narrow_limited PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_PREFILL_FP16_CACHE_MB=1;MOLLM_CUDA_PREFILL_FP16_CACHE_NARROW=1;MOLLM_TEST_CUDA_NARROW_CACHE=1"
            LABELS "cuda" SKIP_RETURN_CODE 77)
        add_test(NAME test_cuda_backend_register_routes COMMAND test_cuda_backend)
        add_test(NAME test_cuda_backend_local16_probe COMMAND test_cuda_backend)
        set_tests_properties(test_cuda_backend_local16_probe PROPERTIES
            ENVIRONMENT "MOLLM_ROUTE_LOCAL16_PROBE=1"
            LABELS "cuda" SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
        add_test(NAME test_cuda_backend_prefill_fp16_cache_hc_down COMMAND test_cuda_backend)
        add_test(NAME test_cuda_backend_gdn_butterfly_key COMMAND test_cuda_backend)
        add_test(NAME test_cuda_backend_gdn_paired_values COMMAND test_cuda_backend)
        set_tests_properties(test_cuda_backend_gdn_paired_values PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_GDN_PREFILL_PAIRED_VALUES=1"
            LABELS "cuda" SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
        set_tests_properties(test_cuda_backend_gdn_butterfly_key PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_GDN_PREFILL_BUTTERFLY_KEY=1"
            LABELS "cuda" SKIP_RETURN_CODE 77)
        set_tests_properties(test_cuda_backend_prefill_fp16_cache_hc_down PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_PREFILL_FP16_CACHE_MB=128;MOLLM_CUDA_PREFILL_FP16_CACHE_HC_DOWN=1;MOLLM_TEST_CUDA_HC_DOWN_CACHE=1"
            LABELS "cuda" SKIP_RETURN_CODE 77)
        add_test(NAME test_cuda_backend_prefill_fp16_cache_hc_down_limited COMMAND test_cuda_backend)
        set_tests_properties(test_cuda_backend_prefill_fp16_cache_hc_down_limited PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_PREFILL_FP16_CACHE_MB=1;MOLLM_CUDA_PREFILL_FP16_CACHE_HC_DOWN=1;MOLLM_TEST_CUDA_HC_DOWN_CACHE=1"
            LABELS "cuda" SKIP_RETURN_CODE 77)
        add_test(NAME test_cuda_backend_exact_batch COMMAND test_cuda_backend)
        add_test(NAME test_cuda_backend_exact_reuse COMMAND test_cuda_backend)
        add_test(NAME test_cuda_backend_q8_verify_batch COMMAND test_cuda_backend)
        add_test(NAME test_cuda_backend_state_prefixes COMMAND test_cuda_backend)
        set_tests_properties(test_cuda_backend_state_prefixes PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_MTP_STATE_PREFIXES=1"
            LABELS "cuda" SKIP_RETURN_CODE 77)
        set_tests_properties(test_cuda_backend_q8_verify_batch PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_MTP_VERIFY_Q8_BATCH=1"
            LABELS "cuda" SKIP_RETURN_CODE 77)
        add_test(NAME test_cuda_backend_narrow_pipeline COMMAND test_cuda_backend)
        add_test(NAME test_cuda_backend_gdn_register_state COMMAND test_cuda_backend)
        set_tests_properties(test_cuda_backend_gdn_register_state PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_GDN_REGISTER_STATE=1"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        add_test(NAME test_cuda_backend_hc_projection COMMAND test_cuda_backend)
        add_test(NAME test_cuda_backend_nvfp4_pipeline COMMAND test_cuda_backend)
        add_test(NAME test_cuda_backend_nvfp4_batch_pipeline COMMAND test_cuda_backend)
        set_tests_properties(test_cuda_backend_nvfp4_batch_pipeline PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_NVFP4_BATCH_PIPELINE=1"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        set_tests_properties(test_cuda_backend_nvfp4_pipeline PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_NVFP4_SIMT_PIPELINE=1;MOLLM_CUDA_HC_PROJECTION_PAIR=1;MOLLM_CUDA_Q4_NARROW_PIPELINE=1;MOLLM_CUDA_MOE_SHARED_OVERLAP=1"
            SKIP_RETURN_CODE 77 LABELS "cuda")
        add_test(NAME test_cuda_backend_nvfp4_pipeline_direct COMMAND test_cuda_backend)
        set_tests_properties(test_cuda_backend_nvfp4_pipeline_direct PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_NVFP4_SIMT_PIPELINE=1;MOLLM_CUDA_MOE_SHARED_OVERLAP=1;MOLLM_CUDA_DISABLE_MOE_GRAPHS=1;MOLLM_CUDA_DISABLE_DECODE_SEGMENTS=1"
            SKIP_RETURN_CODE 77 LABELS "cuda")
        set_tests_properties(test_cuda_backend_hc_projection PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_HC_PROJECTION_PAIR=1;MOLLM_CUDA_Q4_NARROW_PIPELINE=1;MOLLM_CUDA_MOE_SHARED_OVERLAP=1"
            SKIP_RETURN_CODE 77 LABELS "cuda")
        set_tests_properties(test_cuda_backend_narrow_pipeline PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_Q4_NARROW_PIPELINE=1"
            LABELS "cuda")
        add_test(NAME test_cuda_backend_shared_overlap COMMAND test_cuda_backend)
        add_test(NAME test_cuda_backend_moe_prefill_half_reuse COMMAND test_cuda_backend)
        set_tests_properties(test_cuda_backend_moe_prefill_half_reuse PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_MOE_PREFILL_INPUT_HALF_REUSE=1;MOLLM_CUDA_MOE_SHARED_OVERLAP=1"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        set_tests_properties(test_cuda_backend_shared_overlap PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_MOE_SHARED_OVERLAP=1"
            SKIP_RETURN_CODE 77 LABELS "cuda")
        add_test(NAME test_cuda_backend_shared_overlap_direct COMMAND test_cuda_backend)
        set_tests_properties(test_cuda_backend_shared_overlap_direct PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_MOE_SHARED_OVERLAP=1;MOLLM_CUDA_DISABLE_MOE_GRAPHS=1;MOLLM_CUDA_DISABLE_DECODE_SEGMENTS=1"
            SKIP_RETURN_CODE 77 LABELS "cuda")
        set_tests_properties(test_cuda_backend_exact_reuse PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_Q4_SMALL_BATCH_EXACT=1;MOLLM_CUDA_Q4_SMALL_BATCH_EXACT_REUSE=1"
            SKIP_RETURN_CODE 77 LABELS "cuda")
        add_test(NAME test_cuda_backend_exact_attention COMMAND test_cuda_backend)
        add_test(NAME test_cuda_backend_validated_prefix COMMAND test_cuda_backend)
        add_test(NAME test_cuda_backend_sdpa_prefix_query_group COMMAND test_cuda_backend)
        add_test(NAME test_cuda_backend_sdpa_full_batch COMMAND test_cuda_backend --sdpa-full-batch-only)
        set_tests_properties(test_cuda_backend_sdpa_full_batch PROPERTIES
            ENVIRONMENT "MOLLM_TEST_CUDA_SDPA_PREFIX_QUERY_GROUP=2;MOLLM_CUDA_SDPA_PREFILL_VALIDATED_PREFIX=1;MOLLM_CUDA_SDPA_PREFILL_SKIP_ZERO_PV=1;MOLLM_CUDA_SDPA_PREFILL_PV_PAIR=1;MOLLM_CUDA_DISABLE_DECODE_SEGMENTS=1"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        add_test(NAME test_cuda_backend_sdpa_head_group_on COMMAND test_cuda_backend --sdpa-head-group-only)
        add_test(NAME test_cuda_backend_sdpa_native_pv_on COMMAND test_cuda_backend --sdpa-native-pv-only)
        add_test(NAME test_cuda_backend_sdpa_native_pv_off COMMAND test_cuda_backend --sdpa-native-pv-only)
        foreach(mid_mode IN ITEMS 0 1)
            add_test(NAME test_cuda_backend_sdpa_mid_pv_${mid_mode}
                COMMAND ${CMAKE_COMMAND} -E env MOLLM_TEST_CUDA_SDPA_MID_PV_GEMM=${mid_mode}
                    $<TARGET_FILE:test_cuda_backend> --sdpa-native-pv-only)
            set_tests_properties(test_cuda_backend_sdpa_mid_pv_${mid_mode} PROPERTIES
                ENVIRONMENT "MOLLM_CUDA_SDPA_PREFILL_HEAD_GROUP=4;MOLLM_TEST_CUDA_SDPA_PREFIX_QUERY_GROUP=4;MOLLM_CUDA_SDPA_PREFILL_VALIDATED_PREFIX=1;MOLLM_CUDA_SDPA_PREFILL_SKIP_ZERO_PV=1;MOLLM_CUDA_SDPA_PREFILL_PV_PAIR=1;MOLLM_CUDA_DISABLE_DECODE_SEGMENTS=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        endforeach()
        add_test(NAME test_cuda_backend_sdpa_mid_pv_auto
            COMMAND ${CMAKE_COMMAND} -E env --unset=MOLLM_TEST_CUDA_SDPA_MID_PV_GEMM
                $<TARGET_FILE:test_cuda_backend> --sdpa-native-pv-only)
        set_tests_properties(test_cuda_backend_sdpa_mid_pv_auto PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_SDPA_PREFILL_HEAD_GROUP=4;MOLLM_TEST_CUDA_SDPA_PREFIX_QUERY_GROUP=4;MOLLM_CUDA_SDPA_PREFILL_VALIDATED_PREFIX=1;MOLLM_CUDA_SDPA_PREFILL_SKIP_ZERO_PV=1;MOLLM_CUDA_SDPA_PREFILL_PV_PAIR=1;MOLLM_CUDA_DISABLE_DECODE_SEGMENTS=1"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        add_test(NAME test_cuda_backend_sdpa_native_pv_auto
            COMMAND ${CMAKE_COMMAND} -E env --unset=MOLLM_TEST_CUDA_SDPA_NATIVE_PV_GEMM $<TARGET_FILE:test_cuda_backend> --sdpa-native-pv-only)
        set_tests_properties(test_cuda_backend_sdpa_native_pv_auto PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_SDPA_PREFILL_VALIDATED_PREFIX=1;MOLLM_CUDA_SDPA_PREFILL_SKIP_ZERO_PV=1;MOLLM_CUDA_SDPA_PREFILL_PV_PAIR=1;MOLLM_CUDA_DISABLE_DECODE_SEGMENTS=1"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        foreach(attention_default IN ITEMS auto VALIDATED_PREFIX SKIP_ZERO_PV PV_PAIR)
            set(attention_override)
            if(NOT attention_default STREQUAL "auto")
                set(attention_override "MOLLM_CUDA_SDPA_PREFILL_${attention_default}=0")
            endif()
            add_test(NAME test_cuda_backend_sdpa_defaults_${attention_default}
                COMMAND ${CMAKE_COMMAND} -E env
                    --unset=MOLLM_TEST_CUDA_SDPA_NATIVE_PV_GEMM
                    --unset=MOLLM_CUDA_SDPA_PREFILL_VALIDATED_PREFIX
                    --unset=MOLLM_CUDA_SDPA_PREFILL_SKIP_ZERO_PV
                    --unset=MOLLM_CUDA_SDPA_PREFILL_PV_PAIR
                    ${attention_override} $<TARGET_FILE:test_cuda_backend> --sdpa-native-pv-only)
            set_tests_properties(test_cuda_backend_sdpa_defaults_${attention_default} PROPERTIES
                ENVIRONMENT "MOLLM_CUDA_DISABLE_DECODE_SEGMENTS=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        endforeach()
        foreach(head_lanes IN ITEMS 1 2)
            add_test(NAME test_cuda_backend_sdpa_head_lanes_${head_lanes}
                COMMAND ${CMAKE_COMMAND} -E env
                    --unset=MOLLM_TEST_CUDA_SDPA_NATIVE_PV_GEMM
                    --unset=MOLLM_CUDA_SDPA_PREFILL_VALIDATED_PREFIX
                    --unset=MOLLM_CUDA_SDPA_PREFILL_SKIP_ZERO_PV
                    --unset=MOLLM_CUDA_SDPA_PREFILL_PV_PAIR
                    MOLLM_TEST_CUDA_SDPA_PV_HEAD_LANES=${head_lanes}
                    $<TARGET_FILE:test_cuda_backend> --sdpa-native-pv-only)
            set_tests_properties(test_cuda_backend_sdpa_head_lanes_${head_lanes} PROPERTIES
                ENVIRONMENT "MOLLM_CUDA_DISABLE_DECODE_SEGMENTS=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        endforeach()
        foreach(mask_scope IN ITEMS known_mask)
            set(mask_option MOLLM_TEST_CUDA_SDPA_KNOWN_MASK)
            foreach(mask_mode IN ITEMS on off default)
                if(mask_mode STREQUAL "default")
                    set(mask_override "--unset=${mask_option}")
                elseif(mask_mode STREQUAL "on")
                    set(mask_override "${mask_option}=1")
                else()
                    set(mask_override "${mask_option}=0")
                endif()
                add_test(NAME test_cuda_backend_sdpa_${mask_scope}_${mask_mode}
                    COMMAND ${CMAKE_COMMAND} -E env
                        --unset=MOLLM_TEST_CUDA_SDPA_NATIVE_PV_GEMM
                        --unset=MOLLM_CUDA_SDPA_PREFILL_VALIDATED_PREFIX
                        --unset=MOLLM_CUDA_SDPA_PREFILL_SKIP_ZERO_PV
                        --unset=MOLLM_CUDA_SDPA_PREFILL_PV_PAIR
                        ${mask_override} MOLLM_TEST_CUDA_SDPA_PV_HEAD_LANES=2
                        $<TARGET_FILE:test_cuda_backend> --sdpa-native-pv-only)
                set_tests_properties(test_cuda_backend_sdpa_${mask_scope}_${mask_mode} PROPERTIES
                    ENVIRONMENT "MOLLM_CUDA_DISABLE_DECODE_SEGMENTS=1"
                    SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
            endforeach()
        endforeach()
        foreach(pv_mode IN ITEMS on off)
            if(pv_mode STREQUAL "on")
                set(pv_enabled 1)
            else()
                set(pv_enabled 0)
            endif()
            set_tests_properties(test_cuda_backend_sdpa_native_pv_${pv_mode} PROPERTIES
                ENVIRONMENT "MOLLM_TEST_CUDA_SDPA_NATIVE_PV_GEMM=${pv_enabled};MOLLM_CUDA_SDPA_PREFILL_VALIDATED_PREFIX=1;MOLLM_CUDA_SDPA_PREFILL_SKIP_ZERO_PV=1;MOLLM_CUDA_SDPA_PREFILL_PV_PAIR=1;MOLLM_CUDA_DISABLE_DECODE_SEGMENTS=1"
                SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        endforeach()
        add_test(NAME test_cuda_backend_sdpa_head_group_off COMMAND test_cuda_backend --sdpa-head-group-only)
        set_tests_properties(test_cuda_backend_sdpa_head_group_on PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_SDPA_PREFILL_HEAD_GROUP=4;MOLLM_TEST_CUDA_SDPA_PREFIX_QUERY_GROUP=4;MOLLM_CUDA_SDPA_PREFILL_VALIDATED_PREFIX=1;MOLLM_CUDA_SDPA_PREFILL_SKIP_ZERO_PV=1;MOLLM_CUDA_SDPA_PREFILL_PV_PAIR=1"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        set_tests_properties(test_cuda_backend_sdpa_head_group_off PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_SDPA_PREFILL_HEAD_GROUP=0;MOLLM_TEST_CUDA_SDPA_PREFIX_QUERY_GROUP=4;MOLLM_CUDA_SDPA_PREFILL_VALIDATED_PREFIX=1;MOLLM_CUDA_SDPA_PREFILL_SKIP_ZERO_PV=1;MOLLM_CUDA_SDPA_PREFILL_PV_PAIR=1"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        set_tests_properties(test_cuda_backend_sdpa_prefix_query_group PROPERTIES
            ENVIRONMENT "MOLLM_TEST_CUDA_SDPA_PREFIX_QUERY_GROUP=4;MOLLM_CUDA_SDPA_PREFILL_VALIDATED_PREFIX=1;MOLLM_CUDA_SDPA_PREFILL_SKIP_ZERO_PV=1;MOLLM_CUDA_SDPA_PREFILL_PV_PAIR=1"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        add_test(NAME test_cuda_backend_sdpa_gemm_shadow COMMAND test_cuda_backend --sdpa-shadow-only)
        set_tests_properties(test_cuda_backend_sdpa_gemm_shadow PROPERTIES
            ENVIRONMENT "MOLLM_TEST_CUDA_SDPA_GEMM_SHADOW=1;MOLLM_CUDA_SDPA_PREFILL_VALIDATED_PREFIX=1;MOLLM_CUDA_DISABLE_DECODE_SEGMENTS=1;MOLLM_CUDA_SDPA_PREFILL_SKIP_ZERO_PV=1;MOLLM_CUDA_SDPA_PREFILL_PV_PAIR=1"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        add_test(NAME test_cuda_backend_validated_prefix_zero_pv COMMAND test_cuda_backend)
        add_test(NAME test_cuda_backend_validated_prefix_pv_unroll COMMAND test_cuda_backend)
        add_test(NAME test_cuda_backend_validated_prefix_pv_pair COMMAND test_cuda_backend)
        set_tests_properties(test_cuda_backend_validated_prefix_pv_pair PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_SDPA_PREFILL_VALIDATED_PREFIX=1;MOLLM_CUDA_SDPA_PREFILL_SKIP_ZERO_PV=1;MOLLM_CUDA_SDPA_PREFILL_PV_UNROLL=1;MOLLM_CUDA_SDPA_PREFILL_PV_PAIR=1"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        set_tests_properties(test_cuda_backend_validated_prefix_pv_unroll PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_SDPA_PREFILL_VALIDATED_PREFIX=1;MOLLM_CUDA_SDPA_PREFILL_SKIP_ZERO_PV=1;MOLLM_CUDA_SDPA_PREFILL_PV_UNROLL=1"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        set_tests_properties(test_cuda_backend_validated_prefix_zero_pv PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_SDPA_PREFILL_VALIDATED_PREFIX=1;MOLLM_CUDA_SDPA_PREFILL_SKIP_ZERO_PV=1"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        add_test(NAME test_cuda_backend_validated_prefix_zero_pv_long_shared COMMAND test_cuda_backend)
        set_tests_properties(test_cuda_backend_validated_prefix_zero_pv_long_shared PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_SDPA_PREFILL_VALIDATED_PREFIX=1;MOLLM_CUDA_SDPA_PREFILL_SKIP_ZERO_PV=1;MOLLM_CUDA_SDPA_PREFILL_SHARED_SCORE_LIMIT=4096"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        add_test(NAME test_cuda_backend_validated_prefix_long_shared COMMAND test_cuda_backend)
        set_tests_properties(test_cuda_backend_validated_prefix_long_shared PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_SDPA_PREFILL_VALIDATED_PREFIX=1;MOLLM_CUDA_SDPA_PREFILL_SHARED_SCORE_LIMIT=4096"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        set_tests_properties(test_cuda_backend_validated_prefix PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_SDPA_PREFILL_VALIDATED_PREFIX=1"
            SKIP_RETURN_CODE 77 LABELS "cuda" RUN_SERIAL TRUE)
        set_tests_properties(test_cuda_backend_exact_attention PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_SDPA_SMALL_BATCH_EXACT=1;MOLLM_CUDA_GDN_SMALL_BATCH_SERIAL=1"
            SKIP_RETURN_CODE 77 LABELS "cuda")
        set_tests_properties(test_cuda_backend_exact_batch PROPERTIES
            ENVIRONMENT "MOLLM_CUDA_Q4_SMALL_BATCH_EXACT=1"
            SKIP_RETURN_CODE 77 LABELS "cuda")
        add_test(NAME test_cuda_backend_narrow_prefetch COMMAND test_cuda_backend)
        add_test(NAME test_cuda_backend_row_major COMMAND test_cuda_backend)
        add_test(NAME test_cuda_backend_row_major_extended COMMAND test_cuda_backend)
        set_tests_properties(test_cuda_backend_row_major_extended PROPERTIES
            SKIP_RETURN_CODE 77
            ENVIRONMENT "MOLLM_CUDA_Q4_ROW_MAJOR=2"
            LABELS "cuda"
            RUN_SERIAL TRUE)
        set_tests_properties(test_cuda_backend_row_major PROPERTIES
            SKIP_RETURN_CODE 77
            ENVIRONMENT "MOLLM_CUDA_Q4_ROW_MAJOR=1"
            LABELS "cuda"
            RUN_SERIAL TRUE)
        set_tests_properties(test_cuda_backend_narrow_prefetch PROPERTIES
            SKIP_RETURN_CODE 77
            ENVIRONMENT "MOLLM_CUDA_Q4_NARROW_PREFETCH=1"
            LABELS "cuda"
            RUN_SERIAL TRUE)
        set_tests_properties(test_cuda_backend_register_routes PROPERTIES
            SKIP_RETURN_CODE 77
            ENVIRONMENT "MOLLM_CUDA_REGISTER_ROUTES=1"
            LABELS "cuda"
            RUN_SERIAL TRUE)
        add_test(NAME test_cuda_backend_q4_prefetch COMMAND test_cuda_backend)
        set_tests_properties(test_cuda_backend_q4_prefetch PROPERTIES
            SKIP_RETURN_CODE 77
            ENVIRONMENT "MOLLM_CUDA_Q4_PREFETCH=1"
            LABELS "cuda"
            RUN_SERIAL TRUE)
        add_test(NAME test_cuda_backend_q4_no_prefetch COMMAND test_cuda_backend)
        set_tests_properties(test_cuda_backend_q4_no_prefetch PROPERTIES
            SKIP_RETURN_CODE 77
            ENVIRONMENT "MOLLM_CUDA_Q4_PREFETCH=0"
            LABELS "cuda"
            RUN_SERIAL TRUE)
    endif()
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        if(TARGET test_qsa_engine)
            add_test(NAME test_qsa_engine_e2e
                COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_qsa_engine_e2e.py
                    $<TARGET_FILE:test_qsa_engine>)
            set_tests_properties(test_qsa_engine_e2e PROPERTIES
                SKIP_RETURN_CODE 77 LABELS "cuda;e2e" RUN_SERIAL TRUE TIMEOUT 180)
        endif()
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
        add_mollm_python_test(test_qwen4_qsa_reference
                              test_qwen4_qsa_reference.py)
        add_mollm_python_test(test_qwen4_qsa_indexer_reference
                              test_qwen4_qsa_indexer_reference.py)
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
        add_mollm_python_test(test_nvfp4_input_calibration_audit test_nvfp4_input_calibration_audit.py)
        add_mollm_python_test(test_cuda_decode_profile test_cuda_decode_profile.py)
        add_test(NAME test_cuda_host_memory_watchdog_exit
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cuda_host_memory_watchdog_exit.py
                $<TARGET_FILE:test_cuda_host_memory_watchdog>)
        if(TARGET mollm_chat AND TARGET mollm_bench)
            add_test(NAME test_cli_diagnostics
                COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cli_diagnostics.py
                    $<TARGET_FILE:mollm_chat> $<TARGET_FILE:mollm_bench>)
        endif()
        if(TARGET mollm_ppl)
            add_test(NAME test_ppl_tokenize_only
                COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ppl_tokenize_only.py
                    $<TARGET_FILE:mollm_ppl>)
        endif()
        add_mollm_python_test(test_cuda_prefill_routes test_cuda_prefill_routes.py)
        add_mollm_python_test(test_decode_weight_traffic test_decode_weight_traffic.py)
        add_mollm_python_test(test_prefill_output_pool_audit test_prefill_output_pool_audit.py)
        add_mollm_python_test(test_gdn_chunkwise_math test_gdn_chunkwise_math.py)
        set_tests_properties(test_gdn_chunkwise_math PROPERTIES
            ENVIRONMENT "OPENBLAS_NUM_THREADS=1" SKIP_RETURN_CODE 77 LABELS "cpu" RUN_SERIAL TRUE)
        if(MOLLM_CUDA)
            add_test(NAME test_prefill_contract_selector
                COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_prefill_contract_selector.py
                    $<TARGET_FILE:mollm_bench>)
        endif()
        add_mollm_python_test(test_fp8_mtp_stream test_fp8_mtp_stream.py)
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
        add_executable(test_tiny_cpu_inference tests/test_tiny_cpu_inference.cpp)
        add_executable(test_recurrent_mtp tests/test_recurrent_mtp.cpp)
        target_link_libraries(test_recurrent_mtp PRIVATE mollm_lib)
        add_test(NAME test_recurrent_mtp_e2e
                 COMMAND ${Python3_EXECUTABLE}
                         ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_recurrent_mtp_e2e.py
                         $<TARGET_FILE:test_recurrent_mtp>)
        set_tests_properties(test_recurrent_mtp_e2e PROPERTIES SKIP_RETURN_CODE 77)
        if(MOLLM_CUDA)
            add_test(NAME test_recurrent_mtp_device_e2e
                     COMMAND ${Python3_EXECUTABLE}
                             ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_recurrent_mtp_e2e.py
                             $<TARGET_FILE:test_recurrent_mtp>)
            set_tests_properties(test_recurrent_mtp_device_e2e PROPERTIES
                ENVIRONMENT "MOLLM_CUDA_MTP_DEVICE_CHECKPOINT=1"
                SKIP_RETURN_CODE 77 LABELS "cuda")
        endif()
        target_link_libraries(test_tiny_cpu_inference PRIVATE mollm_lib)
        add_dependencies(test_tiny_cpu_inference mollm-quantize)
        add_test(
            NAME test_nvfp4_calibration_package
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_nvfp4_calibration_e2e.py
                    $<TARGET_FILE:test_tiny_cpu_inference>)
        add_test(
            NAME test_ple_offload
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_ple_offload_e2e.py
                    $<TARGET_FILE:test_tiny_cpu_inference>)
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
        set_tests_properties(test_metal_ops PROPERTIES SKIP_RETURN_CODE 77)

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
install(FILES ${PUBLIC_CORE_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mollm/core)
install(FILES ${PUBLIC_RUNTIME_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mollm/runtime)
install(FILES ${PUBLIC_CPU_BACKEND_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mollm/backends/cpu)
install(FILES ${PUBLIC_STORAGE_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mollm/storage)
install(FILES storage/ssd_expert_cache/cache.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mollm/storage/ssd_expert_cache)
install(FILES ${PUBLIC_GRAPH_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mollm/graph)

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
