#include <engine/engine.h>
#include <engine/sampler.h>
#include <engine/tokenizer.h>

int main() {
    EngineConfig config;
    LLMEngine engine;
    Tokenizer tokenizer;
    float logits[] = {1.f, 0.f};
    unsigned int seed = config.seed;
    const int token = sample_token(logits, 2, 0.f, 1, 1.f, &seed);
    return token + tokenizer.vocab_size();
}
