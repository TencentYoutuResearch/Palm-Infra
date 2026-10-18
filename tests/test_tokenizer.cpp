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

    {
        const char* path = "/tmp/mollm_hyv4_tokenizer_test.json";
        const char* chat = "/tmp/mollm_hyv4_template_test.jinja";
        {
            std::ofstream f(path);
            f << R"({"model":{"vocab":{"a":0},"merges":[]},"added_tokens":[)"
              << R"({"id":20,"content":"<｜hy_start:hash｜>"},)"
              << R"({"id":21,"content":"<｜hy_middle:hash｜>"},)"
              << R"({"id":22,"content":"<｜hy_end:hash｜>"},)"
              << R"({"id":23,"content":"<think:hash>"},)"
              << R"({"id":24,"content":"</think:hash>"},)"
              << R"({"id":25,"content":"<｜reasoning_mode:hash｜>"}]})";
        }
        {
            std::ofstream f(chat);
            f << "{% set hy_start_token = '<｜hy_start:hash｜>' %}"
                 "{% set hy_middle_token = '<｜hy_middle:hash｜>' %}"
                 "{% set hy_end_token = '<｜hy_end:hash｜>' %}";
        }
        Tokenizer tok;
        CHECK(tok.load(path, chat, "hy-v4"), "Hy4 tokenizer loads");
        CHECK(tok.bos_id() == 20 && tok.eos_id() == 22,
              "Hy4 GGUF start/end IDs detected");
        const std::string system =
            "<｜hy_start:hash｜>system<｜hy_middle:hash｜>"
            "<｜reasoning_mode:hash｜>reasoning_effort:high<｜hy_end:hash｜>";
        const std::string user =
            "<｜hy_start:hash｜>user<｜hy_middle:hash｜>a<｜hy_end:hash｜>";
        const std::string assistant =
            "<｜hy_start:hash｜>assistant<｜hy_middle:hash｜><think:hash>";
        check_ids(tok.apply_chat("a"), tok.encode(system + user + assistant),
                  "Hy4 plain high-effort prompt matches template");
        const std::string no_think_system =
            "<｜hy_start:hash｜>system<｜hy_middle:hash｜>"
            "<｜reasoning_mode:hash｜>reasoning_effort:no_think<｜hy_end:hash｜>";
        check_ids(tok.apply_chat({{"user", "a"}}, false),
                  tok.encode(no_think_system + user + assistant +
                             "</think:hash>"),
                  "Hy4 no-think prompt closes empty thinking block");
        std::remove(path);
        std::remove(chat);
        ran_any = true;
    }

    // Some HF BPE tokenizers isolate Unicode numbers in chunks of at most
    // three before their main split regex. BPE merges must not cross those
    // chunk boundaries: "28448" is "284" + "48", not five digits.
    {
        const char* path = "/tmp/mollm_digit_triples_tokenizer_test.json";
        {
            std::ofstream f(path);
            f << R"({
                "model": {
                    "vocab": {
                        "2": 0, "8": 1, "4": 2,
                        "28": 3, "284": 4, "48": 5
                    },
                    "merges": [
                        ["2", "8"], ["28", "4"], ["4", "8"]
                    ]
                },
                "added_tokens": [],
                "pre_tokenizer": {
                    "type": "Sequence",
                    "pretokenizers": [
                        {
                            "type": "Split",
                            "pattern": {"Regex": "\\p{N}{1,3}"},
                            "behavior": "Isolated",
                            "invert": false
                        },
                        {
                            "type": "Split",
                            "pattern": {"Regex": "[一-龥぀-ゟ゠-ヿ]+"},
                            "behavior": "Isolated",
                            "invert": false
                        }
                    ]
                }
            })";
        }
        Tokenizer tok;
        CHECK(tok.load(path), "digit-triples HF tokenizer loads");
        check_ids(tok.encode("28448"), {4, 5},
                  "digit triples preserve isolated BPE boundaries");
        std::remove(path);
        ran_any = true;
    }

    // DeepSeek-V4 uses full-width-bar special tokens and an explicit BOS.
    {
        const char* path = "/tmp/mollm_deepseek_v4_tokenizer_test.json";
        {
            std::ofstream f(path);
            f << R"({"model":{"vocab":{"a":0,"b":3,"c":4},"merges":[]},"added_tokens":[)"
              << R"({"id":1,"content":"<｜end▁of▁sentence｜>"},)"
              << R"({"id":2,"content":"<｜begin▁of▁sentence｜>"},)"
              << R"({"id":10,"content":"<｜User｜>"},)"
              << R"({"id":11,"content":"<｜Assistant｜>"},)"
              << R"({"id":12,"content":"</think>"}]})";
        }
        Tokenizer tok;
        CHECK(tok.load(path), "DeepSeek-V4 tokenizer loads");
        CHECK(tok.bos_id() == 2, "DeepSeek-V4 BOS is detected");
        CHECK(tok.eos_id() == 1, "DeepSeek-V4 EOS is detected");
        check_ids(tok.apply_chat("a"), {2, 10, 0, 11, 12},
                  "DeepSeek-V4 chat template");
        check_ids(
            tok.apply_chat({
                {"user", "a"},
                {"assistant", "b"},
                {"user", "c"},
            }),
            {2, 10, 0, 11, 12, 3, 1, 10, 4, 11, 12},
            "DeepSeek-V4 multi-turn chat preserves assistant prefix");
        std::remove(path);
        ran_any = true;
    }

    // Some role-based tokenizers use explicit role names separated from
    // content by a full-width `im middle` token. This is distinct from the
    // otherwise similar DeepSeek-V4 marker set.
    {
        const char* path = "/tmp/mollm_role_middle_tokenizer_test.json";
        {
            std::ofstream f(path);
            f << R"({
                "model": {
                    "vocab": {"a": 0, "b": 1},
                    "merges": []
                },
                "added_tokens": [
                    {"id": 20, "content": "<｜begin▁of▁sentence｜>"},
                    {"id": 21, "content": "<｜User｜>"},
                    {"id": 22, "content": "<｜Assistant｜>"},
                    {"id": 23, "content": "<｜System｜>"},
                    {"id": 24, "content": "<｜im▁middle｜>"},
                    {"id": 25, "content": "<｜im▁end｜>"},
                    {"id": 26, "content": "<think>"},
                    {"id": 27, "content": "</think>"},
                    {"id": 28, "content": "user"},
                    {"id": 29, "content": "assistant"},
                    {"id": 30, "content": "system"},
                    {"id": 31, "content": "\n"}
                ]
            })";
        }
        Tokenizer tok;
        CHECK(tok.load(path), "role-middle tokenizer loads");
        CHECK(tok.bos_id() == 20, "role-middle BOS is detected");
        CHECK(tok.eos_id() == 25, "role-middle EOS is detected");
        check_ids(
            tok.apply_chat("a"),
            {20, 21, 28, 24, 0, 25, 22, 29, 24, 26, 31},
            "role-middle single-turn chat template");
        check_ids(
            tok.apply_chat({{"user", "a"}}, false),
            {20, 21, 28, 24, 0, 25,
             22, 29, 24, 26, 31, 31, 27, 31, 31},
            "role-middle chat can disable thinking");
        check_ids(
            tok.apply_chat({
                {"system", "b"},
                {"user", "a"},
            }),
            {20, 23, 30, 24, 1, 25,
             21, 28, 24, 0, 25,
             22, 29, 24, 26, 31},
            "role-middle system chat template");
        check_ids(
            tok.apply_chat({
                {"user", "a"},
                {"assistant", "b"},
                {"user", "a"},
            }),
            {20,
             21, 28, 24, 0, 25,
             22, 29, 24, 26, 31, 31, 27, 31, 31, 1, 25,
             21, 28, 24, 0, 25,
             22, 29, 24, 26, 31},
            "role-middle multi-turn chat template");
        std::remove(path);
        ran_any = true;
    }

    // The packaged Jinja source disambiguates tokenizers that share the same
    // ChatML vocabulary but differ in whether generation starts with thinking
    // tags.
    {
        const char* tokenizer_path =
            "/tmp/mollm_chatml_style_tokenizer_test.json";
        const char* plain_path =
            "/tmp/mollm_chatml_plain_template_test.jinja";
        const char* optional_disable_path =
            "/tmp/mollm_chatml_optional_disable_template_test.jinja";
        const char* thinking_path =
            "/tmp/mollm_chatml_thinking_template_test.jinja";
        {
            std::ofstream f(tokenizer_path);
            f << R"({
                "model": {
                    "vocab": {"a": 0},
                    "merges": []
                },
                "added_tokens": [
                    {"id": 1, "content": "user"},
                    {"id": 2, "content": "assistant"},
                    {"id": 3, "content": "\n"},
                    {"id": 10, "content": "<|im_start|>"},
                    {"id": 11, "content": "<|im_end|>"},
                    {"id": 12, "content": "<think>"},
                    {"id": 13, "content": "</think>"}
                ]
            })";
        }
        {
            std::ofstream f(plain_path);
            f << "{{ '<|im_start|>assistant\\n' }}";
        }
        {
            std::ofstream f(thinking_path);
            f << "{{ '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}";
        }
        {
            std::ofstream f(optional_disable_path);
            f << "{{ '<|im_start|>assistant\\n' }}"
                 "{% if enable_thinking is defined and enable_thinking is false %}"
                 "{{ '<think>\\n\\n</think>\\n\\n' }}{% endif %}";
        }

        Tokenizer tok;
        CHECK(tok.load(tokenizer_path, plain_path),
              "plain ChatML template loads");
        check_ids(tok.apply_chat("a"),
                  {10, 1, 3, 0, 11, 3, 10, 2, 3},
                  "plain ChatML does not add thinking tags");
        CHECK(tok.load(tokenizer_path, optional_disable_path),
              "optional-disable ChatML template loads");
        check_ids(tok.apply_chat("a"),
                  {10, 1, 3, 0, 11, 3, 10, 2, 3},
                  "default Qwen3 chat does not disable thinking");
        CHECK(tok.load(tokenizer_path, thinking_path),
              "thinking ChatML template loads");
        check_ids(tok.apply_chat("a"),
                  {10, 1, 3, 0, 11, 3, 10, 2, 3,
                   12, 3, 3, 13, 3, 3},
                  "thinking ChatML adds empty thinking block");

        std::remove(tokenizer_path);
        std::remove(plain_path);
        std::remove(optional_disable_path);
        std::remove(thinking_path);
        ran_any = true;
    }

    // A role-token template is not equivalent to Llama-3's header template,
    // even when both tokenizers use the same BOS token.
    {
        const char* tokenizer_path =
            "/tmp/mollm_role_tokens_tokenizer_test.json";
        const char* template_path =
            "/tmp/mollm_role_tokens_template_test.jinja";
        {
            std::ofstream f(tokenizer_path);
            f << R"({
                "model": {"vocab": {"a": 0}, "merges": []},
                "added_tokens": [
                    {"id": 20, "content": "<|begin_of_text|>"},
                    {"id": 21, "content": "<|User|>"},
                    {"id": 22, "content": "<|Assistant|>"},
                    {"id": 23, "content": "<|end_of_text|>"}
                ]
            })";
        }
        {
            std::ofstream f(template_path);
            f << "{{ bos_token }}{{ '<|User|>' + content }}"
                 "{{ '<|Assistant|>' }}";
        }

        Tokenizer tok;
        CHECK(tok.load(tokenizer_path, template_path),
              "role-token template loads");
        check_ids(tok.apply_chat("a"), {20, 21, 0, 22},
                  "role-token chat does not use Llama headers");

        std::remove(tokenizer_path);
        std::remove(template_path);
        ran_any = true;
    }

    // HY-V3 uses distinct Hunyuan role tokens and injects its default
    // no-thinking mode into the system prefix.
    {
        const char* tokenizer_path =
            "/tmp/mollm_hyv3_tokenizer_test.json";
        const char* template_path =
            "/tmp/mollm_hyv3_template_test.jinja";
        {
            std::ofstream f(tokenizer_path);
            f << R"({
                "model": {"vocab": {"a": 0, "b": 1}, "merges": []},
                "added_tokens": [
                    {"id": 20, "content": "<｜hy_begin▁of▁sentence｜>"},
                    {"id": 21, "content": "<｜hy_User｜>"},
                    {"id": 22, "content": "<｜hy_Assistant｜>"},
                    {"id": 23, "content": "<eos:6124c78e>"},
                    {"id": 24, "content": "<think>"},
                    {"id": 25, "content": "</think>"},
                    {"id": 26, "content": "<｜reasoning_mode｜>"},
                    {"id": 27, "content": "reasoning_effort:no_think"}
                ]
            })";
        }
        {
            std::ofstream f(template_path);
            f << "{{ '<｜hy_begin▁of▁sentence｜><｜hy_User｜>' }}"
                 "{{ '<｜hy_Assistant｜>' }}";
        }

        Tokenizer tok;
        CHECK(tok.load(tokenizer_path, template_path),
              "HY-V3 template loads");
        CHECK(tok.bos_id() == 20, "HY-V3 BOS is detected");
        CHECK(tok.eos_id() == 23, "HY-V3 EOS is detected");
        CHECK(tok.decode(std::vector<int>{23}).empty(),
              "HY-V3 EOS is hidden from decoded text");
        check_ids(tok.apply_chat("a"),
                  {20, 26, 27, 21, 0, 22, 24, 25},
                  "HY-V3 single-turn no-think template");
        check_ids(tok.apply_chat({
                      {"system", "b"},
                      {"user", "a"},
                      {"assistant", "b"},
                      {"user", "a"},
                  }),
                  {20, 1, 26, 27, 21, 0, 22, 24, 25, 1, 23,
                   21, 0, 22, 24, 25},
                  "HY-V3 multi-turn template");

        std::remove(tokenizer_path);
        std::remove(template_path);
        ran_any = true;
    }

    // The public Hy3 checkpoint namespaces its HY-V3 control tokens with
    // `:opensource` while keeping the same template semantics.
    {
        const char* tokenizer_path =
            "/tmp/mollm_hy3_opensource_tokenizer_test.json";
        const char* template_path =
            "/tmp/mollm_hy3_opensource_template_test.jinja";
        {
            std::ofstream f(tokenizer_path);
            f << R"({
                "model": {"vocab": {"a": 0}, "merges": []},
                "added_tokens": [
                    {"id": 20, "content": "<｜hy_begin_of_sentence:opensource｜>"},
                    {"id": 21, "content": "<｜hy_User:opensource｜>"},
                    {"id": 22, "content": "<｜hy_Assistant:opensource｜>"},
                    {"id": 23, "content": "<｜hy_eos:opensource｜>"},
                    {"id": 24, "content": "<think:opensource>"},
                    {"id": 25, "content": "</think:opensource>"},
                    {"id": 26, "content": "<｜reasoning_mode:opensource｜>"},
                    {"id": 27, "content": "reasoning_effort:no_think"}
                ]
            })";
        }
        {
            std::ofstream f(template_path);
            f << "{{ '<｜hy_User{}｜>'.format(':opensource') }}"
                 "{{ '<｜hy_Assistant{}｜>'.format(':opensource') }}";
        }

        Tokenizer tok;
        CHECK(tok.load(tokenizer_path, template_path),
              "Hy3 opensource template loads");
        CHECK(tok.bos_id() == 20, "Hy3 opensource BOS is detected");
        CHECK(tok.eos_id() == 23, "Hy3 opensource EOS is detected");
        CHECK(tok.decode(std::vector<int>{23}).empty(),
              "Hy3 opensource EOS is hidden from decoded text");
        check_ids(tok.apply_chat("a"),
                  {20, 26, 27, 21, 0, 22, 24, 25},
                  "Hy3 opensource no-think template");

        std::remove(tokenizer_path);
        std::remove(template_path);
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
        const char* fixture = std::getenv("MOLLM_HYV3_TOKENIZER");
        const char* template_path = std::getenv("MOLLM_HYV3_TEMPLATE");
        if (fixture && template_path && tok.load(fixture, template_path)) {
            ran_any = true;
            CHECK(tok.bos_id() == 120000, "official HY-V3 BOS");
            CHECK(tok.eos_id() == 120025, "official HY-V3 EOS");
            check_ids(
                tok.apply_chat("今天天气真好。"),
                {120000, 120044, 70830, 277, 9002, 647, 497, 25,
                 5130, 25152, 1326, 120006, 7331, 12523, 104846, 292,
                 120007, 120029, 120030},
                "official HY-V3 chat matches Transformers");
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

    {
        Tokenizer tok;
        const char* fixture = std::getenv("MOLLM_DSV4_TOKENIZER");
        if (fixture && tok.load(fixture)) {
            ran_any = true;
            check_ids(
                tok.apply_chat("你好，请用一句话介绍你自己。"),
                {0, 128803, 30594, 303, 2788, 642, 34543, 6657,
                 36005, 320, 128804, 128822},
                "DeepSeek-V4 chat encoding matches official tokenizer");
            CHECK(tok.eos_id() == 1, "DeepSeek-V4 official EOS id");
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
