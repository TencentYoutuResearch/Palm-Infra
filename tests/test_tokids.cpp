#include "engine/tokenizer.h"
#include <cstdio>

int main() {
    Tokenizer tok;
    tok.load("/Users/molly/workspace-youtulm-ncnn/Youtu-LLM-2B/tokenizer.json");
    auto ids = tok.encode("Hello, world!");
    printf("Token IDs (%zu): ", ids.size());
    for (int id : ids) printf("%d ", id);
    printf("\n");
    // Also check with only first 2 tokens (shorter prompt)
    auto ids2 = tok.encode("Hello");
    printf("'Hello' IDs (%zu): ", ids2.size());
    for (int id : ids2) printf("%d ", id);
    printf("\n");
    return 0;
}
