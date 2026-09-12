#include <engine/engine.h>
#include <engine/sampler.h>
#include <engine/tokenizer.h>
#include <core/activation.h>
#include <core/bf16.h>
#include <core/prepared_weight.h>
#include <core/quant_layouts.h>
#include <runtime/host_buffer_pool.h>
#include <runtime/threading.h>
#include <runtime/trace.h>
// Installed legacy paths remain usable alongside their canonical headers.
#include <graph/buffer_pool.h>
#include <kernels/threading.h>
#include <kernels/trace.h>

int main() {
    EngineConfig config;
    LLMEngine engine;
    Tokenizer tokenizer;
    float logits[] = {1.f, 0.f};
    unsigned int seed = config.sampling.seed;
    const int token = sample_token(logits, 2, 0.f, 1, 1.f, &seed);
    return token + tokenizer.vocab_size();
}
