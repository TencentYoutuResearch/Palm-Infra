#include <engine/engine.h>
#include <engine/sampler.h>
#include <engine/tokenizer.h>
#include <core/activation.h>
#include <core/gdn_params.h>
#include <core/rwkv_params.h>
#include <core/attention_params.h>
#include <core/shortconv_params.h>
#include <core/bf16.h>
#include <core/prepared_weight.h>
#include <core/quant_layouts.h>
#include <runtime/host_buffer_pool.h>
#include <runtime/threading.h>
#include <runtime/trace.h>
#include <runtime/expert_provider.h>
#include <storage/mapped_file.h>
#include <storage/byte_ranges.h>
#include <storage/ssd_expert_cache/cache.h>
// Installed legacy paths remain usable alongside their canonical headers.
#include <graph/buffer_pool.h>
#include <kernels/threading.h>
#include <kernels/trace.h>
#include <graph/mmap_file.h>
#include <kernels/moe_ssd.h>

int main() {
    EngineConfig config;
    LLMEngine engine;
    Tokenizer tokenizer;
    float logits[] = {1.f, 0.f};
    unsigned int seed = config.sampling.seed;
    const int token = sample_token(logits, 2, 0.f, 1, 1.f, &seed);
    return token + tokenizer.vocab_size();
}
