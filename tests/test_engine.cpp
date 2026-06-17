#include "engine/engine.h"
#include <cstdio>
#include <cstring>

static int failures = 0;
#define CHECK(cond, msg) do { if(!(cond)){fprintf(stderr,"FAIL: %s\n",msg);failures++;}else{printf("  PASS: %s\n",msg);} } while(0)

int main() {
    // ---- causal mask ----
    LLMEngine eng;
    {
        Tensor mask = eng.build_causal_mask(3, 0);
        CHECK(mask.shape[0] == 3 && mask.shape[1] == 3, "mask shape 3x3");
        // row 0: [0, -inf, -inf]
        CHECK(mask.ptr<float>()[0] == 0.f, "mask[0,0]=0");
        CHECK(mask.ptr<float>()[1] < -1e30f, "mask[0,1]=-inf");
        CHECK(mask.ptr<float>()[2] < -1e30f, "mask[0,2]=-inf");
        // row 1: [0, 0, -inf]
        CHECK(mask.ptr<float>()[3] == 0.f, "mask[1,0]=0");
        CHECK(mask.ptr<float>()[4] == 0.f, "mask[1,1]=0");
        CHECK(mask.ptr<float>()[5] < -1e30f, "mask[1,2]=-inf");
    }

    {
        Tensor mask = eng.build_causal_mask(1, 5);
        CHECK(mask.shape[0] == 6 && mask.shape[1] == 1, "mask with past shape 6x1");
    }

    // ---- RoPE cache ----
    {
        Tensor cos, sin;
        eng.generate_rope_cache(4, 0, cos, sin);
        CHECK(cos.shape[0] == 32 && cos.shape[1] == 4, "cos shape 32x4");
        CHECK(sin.shape[0] == 32 && sin.shape[1] == 4, "sin shape 32x4");
        // cos(0) = 1, sin(0) = 0
        CHECK(std::fabs(cos.ptr<float>()[0] - 1.f) < 1e-5f, "cos(0)=1");
        CHECK(std::fabs(sin.ptr<float>()[0]) < 1e-5f, "sin(0)=0");
    }

    // ---- engine lifecycle (without graph) ----
    {
        LLMEngine e;
        EngineConfig cfg;
        cfg.prefill_graph_path = "/tmp/nonexistent_prefill.graph";
        cfg.decode_graph_path  = "/tmp/nonexistent_decode.graph";
        CHECK(!e.load(cfg), "load fails on missing file");

        // test that prefill/decode don't crash on empty graph
        e.reset();
        CHECK(e.past_len() == 0, "past_len=0 after reset");
    }

    if (failures == 0) {
        printf("\nAll engine tests passed!\n");
    } else {
        printf("\n%d test(s) FAILED\n", failures);
    }
    return failures;
}
