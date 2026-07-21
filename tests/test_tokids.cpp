#include "engine/tokenizer.h"
#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s tokenizer.json\n", argv[0]);
        return 2;
    }
    Tokenizer tok;
    if (!tok.load(argv[1])) return 1;
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
