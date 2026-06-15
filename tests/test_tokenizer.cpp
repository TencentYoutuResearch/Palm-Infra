#include "engine/tokenizer.h"
#include <cstdio>
#include <cstring>

static int failures = 0;
#define CHECK(cond, msg) do { if(!(cond)){fprintf(stderr,"FAIL: %s\n",msg);failures++;}else{printf("  PASS: %s\n",msg);} } while(0)

int main() {
    Tokenizer tok;

    // load Youtu-LLM-2B tokenizer
    bool loaded = tok.load("/Users/molly/workspace-youtulm-ncnn/Youtu-LLM-2B/tokenizer.json");
    if (!loaded) {
        printf("SKIP: tokenizer.json not found\n");
        return 0;
    }

    CHECK(tok.vocab_size() > 100000, "vocab size > 100k");
    CHECK(tok.bos_id() >= 0, "bos_id valid");
    CHECK(tok.eos_id() >= 0, "eos_id valid");

    // encode
    auto ids = tok.encode("你好");
    CHECK(!ids.empty(), "encode '你好' not empty");
    printf("  encode('你好') → %zu tokens\n", ids.size());

    // decode
    std::string text = tok.decode(ids);
    printf("  decode → '%s'\n", text.c_str());
    CHECK(text.find("你好") != std::string::npos, "round-trip contains 你好");

    // chat template
    auto chat_ids = tok.apply_chat("你好");
    CHECK(!chat_ids.empty(), "apply_chat not empty");
    printf("  chat template → %zu tokens\n", chat_ids.size());
    CHECK(chat_ids[0] == tok.bos_id(), "chat starts with BOS");

    if (failures == 0) {
        printf("\nAll tokenizer tests passed!\n");
    } else {
        printf("\n%d test(s) FAILED\n", failures);
    }
    return failures;
}
