#include "engine/tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static int failures = 0;
#define CHECK(cond, msg) do { if(!(cond)){fprintf(stderr,"FAIL: %s\n",msg);failures++;}else{printf("  PASS: %s\n",msg);} } while(0)

static bool check_ids(const std::vector<int>& got,
                      const std::vector<int>& expected,
                      const char* label) {
    if (got == expected) {
        printf("  PASS: %s\n", label);
        return true;
    }
    fprintf(stderr, "FAIL: %s\n", label);
    fprintf(stderr, "  got:     ");
    for (int id : got) fprintf(stderr, "%d ", id);
    fprintf(stderr, "\n  expected:");
    for (int id : expected) fprintf(stderr, " %d", id);
    fprintf(stderr, "\n");
    failures++;
    return false;
}

static bool load_first(Tokenizer& tok, const std::vector<std::string>& paths) {
    for (const auto& path : paths) {
        std::ifstream f(path);
        if (!f.good()) continue;
        if (tok.load(path)) return true;
    }
    return false;
}

int main() {
    bool ran_any = false;

    // Loading is transactional: successful reloads replace all prior state,
    // while a failed reload leaves the current tokenizer usable.
    {
        const char* first_path = "/tmp/mollm_tokenizer_reload_first.json";
        const char* second_path = "/tmp/mollm_tokenizer_reload_second.json";
        {
            std::ofstream f(first_path);
            f << R"({"model":{"vocab":{"a":0},"merges":[]},"added_tokens":[{"id":2,"content":"<old>"},{"id":3,"content":"<|im_end|>"}]})";
        }
        {
            std::ofstream f(second_path);
            f << R"({"model":{"vocab":{"b":0},"merges":[]},"added_tokens":[]})";
        }

        Tokenizer tok;
        CHECK(tok.load(first_path), "first HF tokenizer loads");
        check_ids(tok.encode("<old>"), {2}, "first HF added token encodes");
        CHECK(tok.eos_id() == 3, "first HF tokenizer sets eos");

        CHECK(!tok.load("/tmp/mollm_tokenizer_missing.json"),
              "failed tokenizer reload is reported");
        check_ids(tok.encode("<old>"), {2},
                  "failed reload preserves tokenizer state");

        CHECK(tok.load(second_path), "second HF tokenizer loads");
        CHECK(tok.vocab_size() == 1, "successful reload replaces vocabulary");
        CHECK(tok.encode("<old>").empty(),
              "successful reload removes old added tokens");
        CHECK(tok.eos_id() == 128001,
              "successful reload resets special-token defaults");

        std::remove(first_path);
        std::remove(second_path);
        ran_any = true;
    }

    // RWKV-world vocabulary: longest byte-prefix matching over the official
    // `id python-bytes-literal byte_length` format. This is deliberately a
    // tiny fixture; the production vocabulary has the same syntax.
    {
        const char* path = "/tmp/mollm_rwkv_vocab_test.txt";
        {
            std::ofstream f(path);
            f << "0 b'<EOD>' 5\n";
            f << "1 b'a' 1\n";
            f << "2 b'b' 1\n";
            f << "3 b'ab' 2\n";
            f << "4 '\\xe4\\xbd\\xa0' 3\n";
            f << "5 '\\xe5\\xa5\\xbd' 3\n";
            f << "6 b'User: ab\\n\\nAssistant:' 20\n";
        }
        Tokenizer tok;
        CHECK(tok.load(path), "RWKV world vocabulary loads");
        check_ids(tok.encode("abab"), {3, 3}, "RWKV longest-prefix encoding");
        check_ids(tok.encode("你好"), {4, 5}, "RWKV UTF-8 literal decoding");
        CHECK(tok.decode(std::vector<int>{3, 3}) == "abab", "RWKV decode round-trip");
        CHECK(tok.apply_chat("ab") == std::vector<int>({3}), "RWKV raw prompt (no HF chat template)");
        tok.set_rwkv_legacy_chat_template(true);
        CHECK(tok.apply_chat("ab") == std::vector<int>({6}),
              "RWKV legacy chat template");
        CHECK(tok.stop_sequences() == std::vector<std::string>({"\n\n"}),
              "RWKV legacy text stop sequence");
        std::remove(path);
        ran_any = true;
    }

    // A damaged Python literal ending in a bare backslash used to advance the
    // parser past the string before dereferencing it.
    {
        const char* path = "/tmp/mollm_rwkv_vocab_trailing_slash.txt";
        {
            std::ofstream f(path);
            f << "0 b'abc\\ 4\n";
        }
        Tokenizer tok;
        CHECK(tok.load(path), "RWKV trailing-slash literal loads safely");
        check_ids(tok.encode("abc\\"), {0},
                  "RWKV trailing slash is preserved");
        CHECK(tok.decode(std::vector<int>{0}) == "abc\\",
              "RWKV trailing-slash literal decodes safely");
        std::remove(path);
        ran_any = true;
    }

    {
        Tokenizer tok;
        const char* fixture = std::getenv("MOLLM_RWKV_TOKENIZER");
        if (fixture && tok.load(fixture)) {
            ran_any = true;
            auto ids = tok.encode("你好，RWKV!");
            CHECK(!ids.empty(), "official RWKV vocabulary encodes UTF-8 prompt");
            CHECK(tok.decode(ids) == "你好，RWKV!", "official RWKV vocabulary round-trip");
        }
    }

    {
        Tokenizer tok;
        const char* fixture = std::getenv("MOLLM_YOUTU_TOKENIZER");
        bool loaded = fixture && load_first(tok, {fixture});
        if (loaded) {
            ran_any = true;

            CHECK(tok.vocab_size() > 100000, "Youtu vocab size > 100k");
            CHECK(tok.bos_id() >= 0, "Youtu bos_id valid");
            CHECK(tok.eos_id() >= 0, "Youtu eos_id valid");

            auto ids = tok.encode("你好");
            CHECK(!ids.empty(), "Youtu encode '你好' not empty");
            printf("  Youtu encode('你好') -> %zu tokens\n", ids.size());

            std::string text = tok.decode(ids);
            printf("  Youtu decode -> '%s'\n", text.c_str());
            CHECK(text.find("你好") != std::string::npos, "Youtu round-trip contains 你好");

            auto chat_ids = tok.apply_chat("你好");
            CHECK(!chat_ids.empty(), "Youtu apply_chat not empty");
            printf("  Youtu chat template -> %zu tokens\n", chat_ids.size());
            CHECK(chat_ids[0] == tok.bos_id(), "Youtu chat starts with BOS");
        }
    }

    {
        Tokenizer tok;
        const char* fixture = std::getenv("MOLLM_QWEN_TOKENIZER");
        bool loaded = fixture && load_first(tok, {fixture});
        if (loaded) {
            ran_any = true;
            CHECK(tok.vocab_size() > 248000, "Qwen vocab size > 248k");

            check_ids(tok.encode(" = ((6*n-17)*4^n - 1)/3."),
                      {283, 1718, 21, 23238, 12, 16, 22, 4653,
                       19, 83193, 471, 220, 16, 5443, 18, 13},
                      "Qwen space-prefixed punctuation matches HF");
            check_ids(tok.encode("; A072257: a(n) = ((6*n-17)*4^n - 1)/3."),
                      {26, 357, 15, 22, 17, 17, 20, 22, 25, 264,
                       1393, 8, 283, 1718, 21, 23238, 12, 16,
                       22, 4653, 19, 83193, 471, 220, 16, 5443,
                       18, 13},
                      "Qwen OEIS prefix snippet matches HF");
            check_ids(tok.encode(" hello"), {23066}, "Qwen single-space word prefix");
            check_ids(tok.encode(" =="), {606}, "Qwen single-space punctuation prefix");
            check_ids(tok.encode("  =="), {220, 606}, "Qwen double-space punctuation prefix");
            check_ids(tok.encode(" 123"), {220, 16, 17, 18}, "Qwen space before digits");
            check_ids(tok.encode("  abc"), {220, 37730}, "Qwen double-space word prefix");
            check_ids(tok.encode("你好！有什么我可以帮你的吗？"),
                      {109266, 6115, 98691, 111454, 96598, 97319, 98179, 10992},
                      "Qwen Chinese sentence matches HF");
            check_ids(tok.encode("@interface RYJViewController"),
                      {11749, 423, 56, 41, 6423},
                      "Qwen CamelCase word matches HF");
            check_ids(tok.encode("I'm sure you're fine."),
                      {40, 2688, 2617, 488, 2224, 6699, 13},
                      "Qwen contractions match HF");
            check_ids(tok.encode("HelloWorld ABCDef"),
                      {9419, 9833, 18773, 2533},
                      "Qwen mixed-case words match HF");
            check_ids(tok.encode(" \n"), {695}, "Qwen space before newline");
            check_ids(tok.encode("  \n"), {2228}, "Qwen double-space before newline");
            check_ids(tok.encode("config \nSee"), {1617, 695, 9538},
                      "Qwen newline whitespace regex matches HF");
            check_ids(tok.encode(" 밤"), {181121}, "Qwen Hangul space prefix");
            check_ids(tok.encode(" 10시경, 무라"),
                      {220, 16, 15, 28366, 63100, 11, 149285, 48650},
                      "Qwen mixed numeric Hangul segment");
            check_ids(tok.encode("いるが、実際"),
                      {148552, 27279, 5205, 156962},
                      "Qwen Japanese punctuation segment");
            check_ids(tok.encode("你好！有什么"),
                      {109266, 6115, 98691},
                      "Qwen Chinese punctuation segment");
            check_ids(tok.encode("คณะนิติศาสตร์ได้"),
                      {160043, 19571, 148673, 157210, 150060,
                       148411, 53900, 156167, 19236},
                      "Qwen Thai marks match HF");
            check_ids(tok.encode("अतिरिक्त"),
                      {168948, 76519, 161166, 150087, 196965},
                      "Qwen Devanagari marks match HF");
            check_ids(tok.encode("বাংলা"),
                      {148679, 39947, 150521, 148506, 39947},
                      "Qwen Bengali marks match HF");
            check_ids(tok.encode("عَرَبِيّ"),
                      {22500, 153458, 150765, 151003, 71263},
                      "Qwen Arabic marks match HF");
            check_ids(tok.encode("мару». Не"),
                      {173719, 3652, 71973, 152142},
                      "Qwen guillemet punctuation run matches HF");
        }
    }

    if (!ran_any) {
        printf("SKIP: tokenizer.json fixtures not found\n");
        return 0;
    }

    if (failures == 0) {
        printf("\nAll tokenizer tests passed!\n");
    } else {
        printf("\n%d test(s) FAILED\n", failures);
    }
    return failures;
}
