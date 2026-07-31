#include "engine/cuda_backend.h"
#include "engine/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

bool copy_finite(const Tensor& tensor, std::vector<float>& values) {
    if (!tensor.data || tensor.prec != Precision::FP32)
        return false;
    values.assign(tensor.ptr<float>(),
                  tensor.ptr<float>() + tensor.nelements());
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

bool run(const char* package, const char* expected_architecture, Device device,
         std::vector<float>& prefill, std::vector<float>& decode,
         size_t stream_cache_bytes = 0, size_t device_cache_bytes = 0,
         bool require_eviction = false, bool require_device_reuse = false) {
    LLMEngine engine;
    EngineConfig config;
    config.package_path = package;
    config.device = device;
    config.n_ctx = 8;
    config.num_threads = 1;
    config.weight_loading = WeightLoadingMode::MMAP;
    if (stream_cache_bytes != 0)
        config.moe_ssd_cache_bytes = stream_cache_bytes;
    config.moe_device_cache_bytes = device_cache_bytes;
    if (!engine.load(config))
        return false;
    if (device == Device::CUDA && engine.cpu_weight_sidecar_bytes() != 0)
        return false;
    const auto& metadata = engine.package_metadata();
    const auto architecture = metadata.find("architecture");
    if (architecture == metadata.end() ||
        architecture->second != expected_architecture)
        return false;
    if (device == Device::CUDA && architecture->second == "qwen3-moe") {
        const auto layers = metadata.find("num_layers");
        const auto heads = metadata.find("num_kv_heads");
        const auto width = metadata.find("head_dim");
        if (layers == metadata.end() || heads == metadata.end() ||
            width == metadata.end())
            return false;
        const size_t layer_count = std::stoul(layers->second);
        const size_t kv_heads = std::stoul(heads->second);
        const size_t head_dim = std::stoul(width->second);
        const size_t expected_cache_bytes = layer_count *
            (2 * CacheMetadata::SIZE + 2 * head_dim * config.n_ctx *
                 kv_heads * sizeof(mollm::cpu::fp16_t));
        if (engine.kv_cache_bytes() != expected_cache_bytes) {
            std::fprintf(
                stderr,
                "CUDA Qwen3-MoE KV cache is %zu bytes, expected FP16 %zu\n",
                engine.kv_cache_bytes(), expected_cache_bytes);
            return false;
        }
    }
    Tensor prefill_tensor = engine.prefill_hidden({1, 2, 3});
    if (!copy_finite(prefill_tensor, prefill) || engine.past_len() != 3)
        return false;
    Tensor decode_tensor = engine.decode_hidden(4);
    if (!copy_finite(decode_tensor, decode) || engine.past_len() != 4)
        return false;
    if (require_device_reuse) {
        // Token 5 changes one hash-routed expert relative to token 4. This
        // forces a cross-forward eviction without evicting any entry pinned
        // by the current pointer table.
        std::vector<float> switched_decode;
        Tensor switched = engine.decode_hidden(5);
        if (!copy_finite(switched, switched_decode) || engine.past_len() != 5)
            return false;
        const auto host_before = engine.moe_ssd_stats();
        const auto device_before = engine.moe_device_cache_stats();
        std::vector<float> repeated_decode;
        Tensor repeated = engine.decode_hidden(5);
        if (!copy_finite(repeated, repeated_decode) || engine.past_len() != 6)
            return false;
        const auto host_after = engine.moe_ssd_stats();
        const auto device_after = engine.moe_device_cache_stats();
        if (host_after.bytes_read != host_before.bytes_read ||
            device_after.host_to_device_bytes !=
                device_before.host_to_device_bytes ||
            device_after.fallback_host_to_device_bytes !=
                device_before.fallback_host_to_device_bytes ||
            device_after.hits < device_before.hits + 2 ||
            device_after.direct_expert_bytes <=
                device_before.direct_expert_bytes)
            return false;
    }
    if (stream_cache_bytes != 0) {
        const auto stats = engine.moe_ssd_stats();
        if (stats.misses == 0 || stats.bytes_read == 0 ||
            (require_eviction && stats.evictions == 0))
            return false;
    }
    if (device_cache_bytes != 0) {
        const auto stats = engine.moe_device_cache_stats();
        if (stats.capacity_bytes != device_cache_bytes || stats.misses == 0 ||
            stats.host_to_device_bytes == 0 ||
            stats.direct_expert_bytes == 0 ||
            stats.resident_bytes == 0 ||
            (require_device_reuse &&
             (stats.hits < 2 || stats.evictions == 0 ||
              stats.fallback_scratch_bytes == 0 ||
              stats.fallback_scratch_bytes >= stats.peak_selected_bytes)))
            return false;
        if (require_device_reuse) {
            std::printf(
                "tiny CUDA MoE device cache: peak selected=%zu, "
                "fallback scratch=%zu, direct=%llu bytes\n",
                stats.peak_selected_bytes, stats.fallback_scratch_bytes,
                static_cast<unsigned long long>(
                    stats.direct_expert_bytes));
        }
    }
    engine.reset();
    std::vector<float> reset_prefill;
    std::vector<float> reset_decode;
    return engine.past_len() == 0 &&
        copy_finite(engine.prefill_hidden({1, 2, 3}), reset_prefill) &&
        copy_finite(engine.decode_hidden(4), reset_decode) &&
        reset_prefill == prefill && reset_decode == decode &&
        engine.past_len() == 4;
}

bool close_enough(const std::vector<float>& actual,
                  const std::vector<float>& expected, float tolerance,
                  const char* label) {
    if (actual.size() != expected.size())
        return false;
    float maximum = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i)
        maximum = std::max(maximum, std::fabs(actual[i] - expected[i]));
    std::printf("tiny MoE %s CPU/CUDA max abs error: %.7f\n", label,
                maximum);
    return maximum <= tolerance;
}

bool compare_package(const char* package, const char* architecture,
                     const char* label, size_t stream_cache_bytes = 0,
                     size_t device_cache_bytes = 0,
                     bool require_eviction = false,
                     bool require_device_reuse = false,
                     float cpu_tolerance = 4e-2f) {
    std::vector<float> cpu_prefill;
    std::vector<float> cpu_decode;
    std::vector<float> cuda_prefill;
    std::vector<float> cuda_decode;
    if (!run(package, architecture, Device::CPU, cpu_prefill, cpu_decode)) {
        std::fprintf(stderr, "tiny %s CPU reference failed\n", label);
        return false;
    }
    if (!run(package, architecture, Device::CUDA,
             cuda_prefill, cuda_decode)) {
        std::fprintf(stderr, "tiny %s CUDA inference failed\n", label);
        return false;
    }
    char prefill_label[96];
    char decode_label[96];
    std::snprintf(prefill_label, sizeof(prefill_label), "%s prefill", label);
    std::snprintf(decode_label, sizeof(decode_label), "%s decode", label);
    bool valid = close_enough(
                     cuda_prefill, cpu_prefill, cpu_tolerance,
                     prefill_label) &&
        close_enough(
            cuda_decode, cpu_decode, cpu_tolerance, decode_label);
    if (!valid || stream_cache_bytes == 0)
        return valid;

    std::vector<float> streamed_prefill;
    std::vector<float> streamed_decode;
    if (!run(package, architecture, Device::CUDA,
             streamed_prefill, streamed_decode, stream_cache_bytes,
             device_cache_bytes, require_eviction,
             require_device_reuse)) {
        std::fprintf(stderr, "tiny %s CUDA SSD inference failed\n", label);
        return false;
    }
    std::snprintf(prefill_label, sizeof(prefill_label), "%s SSD prefill", label);
    std::snprintf(decode_label, sizeof(decode_label), "%s SSD decode", label);
    if (!close_enough(
            streamed_prefill, cpu_prefill, cpu_tolerance, prefill_label) ||
        !close_enough(
            streamed_decode, cpu_decode, cpu_tolerance, decode_label))
        return false;
    std::snprintf(
        prefill_label, sizeof(prefill_label), "%s resident/SSD prefill",
        label);
    std::snprintf(
        decode_label, sizeof(decode_label), "%s resident/SSD decode",
        label);
    return close_enough(
               streamed_prefill, cuda_prefill, 5e-3f, prefill_label) &&
        close_enough(
               streamed_decode, cuda_decode, 5e-3f, decode_label);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 9) {
        std::fprintf(
            stderr,
            "usage: %s <qwen3-moe-w4g32.mollm> "
            "<qwen3.5-moe-w4g32.mollm> <qwen3-moe-w8g32.mollm> "
            "<qwen3-moe-w4g128.mollm> "
            "<deepseek-v4-fp8-mxfp4.mollm> "
            "<qwen3-moe-fp16.mollm> "
            "<deepseek-v4-fp32-routed.mollm> "
            "<deepseek-v4-fp8-routed.mollm>\n",
            argv[0]);
        return 2;
    }
    CudaBackend probe;
    if (!probe.available()) {
        std::fprintf(stderr, "CUDA device unavailable; skipping\n");
        return 77;
    }
    if (!compare_package(
            argv[1], "qwen3-moe", "Qwen3-MoE W4", 2048, 4096) ||
        !compare_package(
            argv[2], "qwen3.5-moe", "Qwen3.5-MoE W4", 2048, 4096) ||
        !compare_package(
            argv[3], "qwen3-moe", "Qwen3-MoE W8", 4096, 8192) ||
        !compare_package(
            argv[4], "qwen3-moe", "Qwen3-MoE W4G128", 32768, 65536) ||
        !compare_package(
            argv[5], "deepseek-v4",
            "DeepSeek attention/hash/HC/grouped FP8+MXFP4",
            2048, 4096, true, true) ||
        !compare_package(
            argv[6], "qwen3-moe", "Qwen3-MoE FP16 SSD scratch",
            8192) ||
        !compare_package(
            argv[6], "qwen3-moe", "Qwen3-MoE FP16 device cache",
            8192, 16384) ||
        !compare_package(
            argv[7], "deepseek-v4", "DeepSeek FP32 SSD scratch",
            32768) ||
        !compare_package(
            argv[7], "deepseek-v4", "DeepSeek FP32 device cache",
            32768, 32768) ||
        !compare_package(
            argv[8], "deepseek-v4", "DeepSeek FP8 SSD scratch",
            65536, 0, false, false, 6e-2f) ||
        !compare_package(
            argv[8], "deepseek-v4", "DeepSeek FP8 device cache",
            65536, 100000, true, true, 6e-2f))
        return 1;
    std::printf("Tiny CUDA MoE E2E tests passed\n");
    return 0;
}
