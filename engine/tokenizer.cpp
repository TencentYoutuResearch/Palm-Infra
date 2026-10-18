#include "engine/tokenizer.h"
#include <json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <limits>
#include <utility>

using json = nlohmann::json;

// --- GPT-2 byte-to-unicode table -------------------------------------------

void Tokenizer::build_byte_tables() {
    std::vector<int> bs, cs;
    for (int b = '!'; b <= '~'; b++) { bs.push_back(b); cs.push_back(b); }
    for (int b = 0xA1; b <= 0xAC; b++) { bs.push_back(b); cs.push_back(b); }
    for (int b = 0xAE; b <= 0xFF; b++) { bs.push_back(b); cs.push_back(b); }
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }
    auto to_utf8 = [](int cp) -> std::string {
        std::string s;
        if (cp < 0x80) { s += (char)cp; }
        else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
        else { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
        return s;
    };
    for (size_t i = 0; i < bs.size(); i++) {
        std::string u = to_utf8(cs[i]);
        byte_encoder_[bs[i]] = u;
        byte_decoder_[u] = (uint8_t)bs[i];
    }
}

// --- Load tokenizer.json ---------------------------------------------------

bool Tokenizer::load(const std::string& path,
                     const std::string& chat_template_path,
                     const std::string& architecture) {
    Tokenizer loaded;
    try {
        if (!loaded.load_impl(path) ||
            !loaded.configure_chat_template(chat_template_path, architecture))
            return false;
    } catch (const std::exception& e) {
        fprintf(stderr, "tokenizer: failed to load %s: %s\n", path.c_str(),
                e.what());
        return false;
    }
    *this = std::move(loaded);
    return true;
}

bool Tokenizer::configure_chat_template(const std::string& path,
                                        const std::string& architecture) {
    std::string source;
    if (!path.empty()) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            fprintf(stderr, "tokenizer: cannot open chat template %s\n",
                    path.c_str());
            return false;
        }
        file.seekg(0, std::ios::end);
        const std::streamoff size = file.tellg();
        if (size < 0) {
            fprintf(stderr, "tokenizer: cannot size chat template %s\n",
                    path.c_str());
            return false;
        }
        source.resize(static_cast<size_t>(size));
        file.seekg(0, std::ios::beg);
        if (!source.empty() &&
            !file.read(source.data(), static_cast<std::streamsize>(source.size()))) {
            fprintf(stderr, "tokenizer: cannot read chat template %s\n",
                    path.c_str());
            return false;
        }
    }

    // We intentionally implement the plain text-generation branches used by
    // the CLI rather than embedding a general-purpose Jinja interpreter.
    // Unlike special-token guessing, the packaged template source
    // unambiguously identifies templates that share the same vocabulary.
    if (source.find("<｜im▁middle｜>") != std::string::npos &&
        source.find("<｜im▁end｜>") != std::string::npos) {
        chat_template_style_ = ChatTemplateStyle::ROLE_MIDDLE;
    } else if (source.find("<|User|>") != std::string::npos &&
               source.find("<|Assistant|>") != std::string::npos) {
        chat_template_style_ = ChatTemplateStyle::ROLE_TOKENS;
    } else if (source.find("<|im_start|>") != std::string::npos) {
        // Qwen3's default template starts generation immediately after the
        // assistant header. It only emits an empty thinking block when the
        // caller explicitly sets enable_thinking=false. Do not mistake that
        // optional branch for templates (such as Qwen3.5's) that emit the
        // block by default.
        const bool explicit_disable_only =
            source.find("enable_thinking is defined and enable_thinking is false") !=
                std::string::npos &&
            source.find("enable_thinking is defined and enable_thinking is true") ==
                std::string::npos;
        chat_template_style_ =
            source.find("<think>") != std::string::npos &&
                    !explicit_disable_only
                ? ChatTemplateStyle::CHATML_THINKING
                : ChatTemplateStyle::CHATML_PLAIN;
    } else if (source.find("<|start_header_id|>") != std::string::npos) {
        chat_template_style_ = ChatTemplateStyle::LLAMA3_HEADERS;
    } else if (source.find("<｜User｜>") != std::string::npos &&
               source.find("<｜Assistant｜>") != std::string::npos) {
        chat_template_style_ = ChatTemplateStyle::DEEPSEEK_V4;
    } else if (source.find("hy_start_token") != std::string::npos &&
               source.find("hy_middle_token") != std::string::npos &&
               source.find("hy_end_token") != std::string::npos) {
        chat_template_style_ = ChatTemplateStyle::HY_V4;
    } else if (source.find("hy_User") != std::string::npos &&
               source.find("hy_Assistant") != std::string::npos) {
        chat_template_style_ = ChatTemplateStyle::HY_V3;
    } else if (!source.empty()) {
        fprintf(stderr,
                "tokenizer: unrecognized packaged chat template; using "
                "special-token compatibility fallback\n");
    }

    if (source.empty()) {
        // Compatibility for already-converted packages. New conversions
        // should embed the source template instead.
        if (architecture == "qwen3") {
            chat_template_style_ = ChatTemplateStyle::CHATML_PLAIN;
        } else if (architecture == "deepseek-v4") {
            chat_template_style_ = ChatTemplateStyle::DEEPSEEK_V4;
        } else if (architecture == "hy-v3") {
            chat_template_style_ = ChatTemplateStyle::HY_V3;
        }
    }
    return true;
}

bool Tokenizer::load_impl(const std::string& path) {
    // Packages historically extract the tokenizer payload as `tokenizer.json`
    // regardless of its source filename. Sniff the first non-whitespace byte
    // so an embedded RWKV vocabulary is still recognized after extraction.
    {
        std::ifstream probe(path);
        char c = 0;
        while (probe.get(c) && std::isspace((unsigned char)c)) {}
        if (!probe || c != '{') return load_rwkv_vocab(path);
    }
    format_ = Format::HF_BPE;
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "tokenizer: cannot open %s\n", path.c_str());
        return false;
    }

    json root;
    try {
        root = json::parse(f);
    } catch (const json::parse_error& e) {
        fprintf(stderr, "tokenizer: JSON parse error: %s\n", e.what());
        return false;
    }

    build_byte_tables();

    // Most supported BPE tokenizers use the Qwen/Llama single-regex
    // pre-tokenizer below. Some MoE checkpoints instead apply two isolated
    // splits before their main regex: Unicode numbers in groups of at most
    // three, then contiguous CJK/Kana text. Detect that tokenizer-defined
    // pipeline rather than tying the behavior to a model name.
    if (root.contains("pre_tokenizer")) {
        const auto& pre = root["pre_tokenizer"];
        if (pre.value("type", "") == "Sequence" &&
            pre.contains("pretokenizers")) {
            bool digit_triples = false;
            bool cjk_split = false;
            for (const auto& step : pre["pretokenizers"]) {
                if (step.value("type", "") != "Split" ||
                    !step.contains("pattern") ||
                    !step["pattern"].contains("Regex")) {
                    continue;
                }
                const std::string pattern =
                    step["pattern"]["Regex"].get<std::string>();
                digit_triples |= pattern == "\\p{N}{1,3}";
                cjk_split |= pattern.find("[一-龥") != std::string::npos;
            }
            if (digit_triples && cjk_split) {
                pre_tokenizer_style_ =
                    PreTokenizerStyle::DIGIT_TRIPLES_CJK;
            }
        }
    }

    const auto& vocab_obj = root["model"]["vocab"];
    int max_id = 0;
    for (auto& [piece, id_val] : vocab_obj.items()) {
        int id = id_val.get<int>();
        vocab_[piece] = id;
        if (id > max_id) max_id = id;
    }

    if (root.contains("added_tokens")) {
        for (auto& tok : root["added_tokens"]) {
            int id = tok["id"].get<int>();
            std::string content = tok["content"].get<std::string>();
            added_tokens_[content] = id;
            vocab_[content] = id;
            if (id > max_id) max_id = id;
        }
    }

    id_to_piece_.resize(max_id + 1);
    for (auto& [piece, id] : vocab_) {
        id_to_piece_[id] = piece;
    }

    for (auto& [content, id] : added_tokens_) {
        added_patterns_.push_back(content);
    }
    std::sort(added_patterns_.begin(), added_patterns_.end(),
              [](const std::string& a, const std::string& b) {
                  return a.size() > b.size();
              });

    const auto& merges_arr = root["model"]["merges"];
    for (size_t i = 0; i < merges_arr.size(); i++) {
        std::string a, b;
        if (merges_arr[i].is_array()) {
            a = merges_arr[i][0].get<std::string>();
            b = merges_arr[i][1].get<std::string>();
        } else {
            const std::string& m = merges_arr[i].get_ref<const std::string&>();
            auto sp = m.find(' ');
            if (sp == std::string::npos) continue;
            a = m.substr(0, sp);
            b = m.substr(sp + 1);
        }
        merges_[{a, b}] = (int)i;
    }

    // Auto-detect bos/eos from added_tokens.
    // Supports Llama-3, Qwen3.5, and DeepSeek-V4 conventions.
    if (added_tokens_.count("<|begin_of_text|>"))
        bos_id_ = added_tokens_["<|begin_of_text|>"];
    else if (added_tokens_.count("<|im_start|>"))
        bos_id_ = added_tokens_["<|im_start|>"];
    else if (added_tokens_.count("<｜begin▁of▁sentence｜>"))
        bos_id_ = added_tokens_["<｜begin▁of▁sentence｜>"];
    else if (added_tokens_.count("<｜hy_begin▁of▁sentence｜>"))
        bos_id_ = added_tokens_["<｜hy_begin▁of▁sentence｜>"];
    else if (added_tokens_.count("<｜hy_begin_of_sentence:opensource｜>"))
        bos_id_ = added_tokens_["<｜hy_begin_of_sentence:opensource｜>"];

    if (added_tokens_.count("<|end_of_text|>"))
        eos_id_ = added_tokens_["<|end_of_text|>"];
    else if (added_tokens_.count("<|im_end|>"))
        eos_id_ = added_tokens_["<|im_end|>"];
    else if (added_tokens_.count("<｜im▁end｜>"))
        eos_id_ = added_tokens_["<｜im▁end｜>"];
    else if (added_tokens_.count("<｜end▁of▁sentence｜>"))
        eos_id_ = added_tokens_["<｜end▁of▁sentence｜>"];
    else if (added_tokens_.count("<eos:6124c78e>"))
        eos_id_ = added_tokens_["<eos:6124c78e>"];
    else if (added_tokens_.count("<｜hy_eos:opensource｜>"))
        eos_id_ = added_tokens_["<｜hy_eos:opensource｜>"];
    for (const auto& [piece, id] : added_tokens_) {
        if (piece.rfind("<｜hy_start:", 0) == 0) bos_id_ = id;
        if (piece.rfind("<｜hy_end:", 0) == 0) eos_id_ = id;
    }

    return true;
}

// --- UTF-8 helpers ----------------------------------------------------------

static int utf8_codepoint(const char* s, int len_avail, int& len) {
    uint8_t c = (uint8_t)s[0];
    if (c < 0x80) { len = 1; return c; }
    if ((c & 0xE0) == 0xC0 && len_avail >= 2) { len = 2; return ((c & 0x1F) << 6) | (s[1] & 0x3F); }
    if ((c & 0xF0) == 0xE0 && len_avail >= 3) { len = 3; return ((c & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F); }
    if ((c & 0xF8) == 0xF0 && len_avail >= 4) { len = 4; return ((c & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F); }
    len = 1; return c;
}

static bool is_common_unicode_punctuation(int cp) {
    return cp == 0x00AB || cp == 0x00BB ||     // Guillemets
           (cp >= 0x2000 && cp <= 0x206F) ||   // General punctuation
           (cp >= 0x3000 && cp <= 0x303F) ||   // CJK symbols/punctuation
           (cp >= 0xFE30 && cp <= 0xFE4F) ||   // CJK compatibility punctuation
           (cp >= 0xFF00 && cp <= 0xFF0F) ||   // Fullwidth ASCII punctuation
           (cp >= 0xFF1A && cp <= 0xFF20) ||
           (cp >= 0xFF3B && cp <= 0xFF40) ||
           (cp >= 0xFF5B && cp <= 0xFF65);
}

static bool is_unicode_mark(int cp) {
    return (cp >= 0x0300 && cp <= 0x036F) ||
           (cp >= 0x1AB0 && cp <= 0x1AFF) ||
           (cp >= 0x1DC0 && cp <= 0x1DFF) ||
           (cp >= 0x20D0 && cp <= 0x20FF) ||
           (cp >= 0xFE20 && cp <= 0xFE2F) ||
           (cp >= 0x064B && cp <= 0x065F) ||
           (cp >= 0x0900 && cp <= 0x0903) ||
           cp == 0x093A || cp == 0x093C ||
           (cp >= 0x093E && cp <= 0x094D) ||
           (cp >= 0x0951 && cp <= 0x0957) ||
           (cp >= 0x0962 && cp <= 0x0963) ||
           (cp >= 0x0981 && cp <= 0x0983) ||
           cp == 0x09BC ||
           (cp >= 0x09BE && cp <= 0x09C4) ||
           (cp >= 0x09C7 && cp <= 0x09C8) ||
           (cp >= 0x09CB && cp <= 0x09CD) ||
           cp == 0x09D7 ||
           (cp >= 0x09E2 && cp <= 0x09E3) ||
           cp == 0x0E31 ||
           (cp >= 0x0E34 && cp <= 0x0E3A) ||
           (cp >= 0x0E47 && cp <= 0x0E4E) ||
           cp == 0xFE0F;
}

static bool is_cjk_or_kana_letter(int cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) ||   // CJK Unified
           (cp >= 0x3400 && cp <= 0x4DBF) ||   // CJK Extension A
           (cp >= 0xF900 && cp <= 0xFAFF) ||   // CJK Compatibility Ideographs
           (cp >= 0x3041 && cp <= 0x3096) ||   // Hiragana letters
           (cp >= 0x30A1 && cp <= 0x30FA) ||   // Katakana letters
           cp == 0x30FC ||                      // Katakana-Hiragana prolonged sound mark
           (cp >= 0x1100 && cp <= 0x11FF) ||   // Hangul Jamo
           (cp >= 0x3130 && cp <= 0x318F) ||   // Hangul Compatibility Jamo
           (cp >= 0xAC00 && cp <= 0xD7AF) ||   // Hangul
           (cp >= 0x3105 && cp <= 0x312F);     // Bopomofo
}

static bool is_digit_triple_number(int cp) {
    return (cp >= '0' && cp <= '9') ||
           (cp >= 0x0660 && cp <= 0x0669) ||   // Arabic-Indic
           (cp >= 0x06F0 && cp <= 0x06F9) ||   // Extended Arabic-Indic
           (cp >= 0x0966 && cp <= 0x096F) ||   // Devanagari
           (cp >= 0x09E6 && cp <= 0x09EF) ||   // Bengali
           (cp >= 0x0E50 && cp <= 0x0E59) ||   // Thai
           (cp >= 0xFF10 && cp <= 0xFF19);     // Fullwidth
}

static bool is_exact_cjk_split_letter(int cp) {
    return (cp >= 0x4E00 && cp <= 0x9FA5) ||
           (cp >= 0x3040 && cp <= 0x309F) ||
           (cp >= 0x30A0 && cp <= 0x30FF);
}

static bool is_unicode_symbol(int cp) {
    return (cp >= 0x20A0 && cp <= 0x20CF) ||   // Currency symbols
           (cp >= 0x2100 && cp <= 0x214F) ||   // Letterlike symbols
           (cp >= 0x2190 && cp <= 0x2BFF) ||   // Arrows/math/technical
           (cp >= 0x1F000 && cp <= 0x1FAFF);
}

static bool is_punctuation_or_symbol(int cp) {
    if ((cp >= '!' && cp <= '/') ||
        (cp >= ':' && cp <= '@') ||
        (cp >= '[' && cp <= '`') ||
        (cp >= '{' && cp <= '~')) {
        return true;
    }
    return is_common_unicode_punctuation(cp) || is_unicode_symbol(cp);
}

static bool is_letter(int cp) {
    if (cp >= 'A' && cp <= 'Z') return true;
    if (cp >= 'a' && cp <= 'z') return true;
    if (is_unicode_mark(cp)) return false;
    if (is_cjk_or_kana_letter(cp)) return true;
    // Approximate Unicode \p{L}/\p{M} for scripts outside the CJK ranges used
    // by our fixtures. Keep common Unicode punctuation out so it can follow
    // the punctuation branch instead of being folded into letter runs.
    if (cp > 0x7F && !is_common_unicode_punctuation(cp) &&
        !is_digit_triple_number(cp) && !is_unicode_symbol(cp)) {
        return true;
    }
    return false;
}

static bool is_digit(int cp) {
    return cp >= '0' && cp <= '9';
}

static bool is_whitespace(int cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x0B || cp == 0x0C;
}

static bool is_newline(int cp) {
    return cp == '\n' || cp == '\r';
}

// --- Pre-tokenizer ----------------------------------------------------------
//
// Implements the HuggingFace tokenizer.json Split regex followed by ByteLevel
// encoding (the ByteLevel pass is done in encode(), not here).
//
// Stage 1 regex alternatives from tokenizer.json (tried in order):
//   1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   2. [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
//   3. \p{N}                           — single digit
//   4.  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*  — punctuation with opt. space prefix
//   5. \s*[\r\n]+                      — newlines
//   6. \s+(?!\S)                       — trailing whitespace
//   7. \s+                             — whitespace

struct CharReader {
    const char* data;
    int size;
    int pos = 0;

    int peek_cp(int& len) const {
        if (pos >= size) { len = 0; return -1; }
        return utf8_codepoint(data + pos, size - pos, len);
    }
    int peek_cp() const { int l; return peek_cp(l); }
    void advance(int len) { pos += len; }
};

std::vector<std::string> Tokenizer::pre_tokenize(const std::string& text) const {
    std::vector<std::string> out;
    if (text.empty()) return out;

    const bool digit_triples_cjk =
        pre_tokenizer_style_ == PreTokenizerStyle::DIGIT_TRIPLES_CJK;

    auto emit = [&](int abs_start, int abs_end) {
        if (abs_end > abs_start)
            out.push_back(text.substr(abs_start, abs_end - abs_start));
    };

    // Try to match a contraction token ('s, 't, etc.) at current position.
    // Returns number of bytes consumed (0 if no match).
    auto try_contraction = [](const CharReader& r) -> int {
        if (r.pos >= r.size || r.data[r.pos] != '\'') return 0;
        auto lower = [](char c) -> char {
            return (c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : c;
        };
        char buf[3] = {0, 0, 0};
        int n = 0;
        for (int i = r.pos + 1; i < r.size && n < 2; i++, n++) {
            char c = r.data[i];
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))) break;
            buf[n] = lower(c);
        }
        if (n >= 1 && (buf[0] == 's' || buf[0] == 't' || buf[0] == 'm' || buf[0] == 'd'))
            return 2;
        if (n >= 2 && ((buf[0] == 'r' && buf[1] == 'e') ||
                       (buf[0] == 'v' && buf[1] == 'e') ||
                       (buf[0] == 'l' && buf[1] == 'l')))
            return 3;
        return 0;
    };

    {
        CharReader r{text.c_str(), (int)text.size()};

        while (r.pos < r.size) {
            int start = r.pos;
            int len;
            int cp = r.peek_cp(len);

            // The multi-stage tokenizer isolates number chunks and CJK/Kana
            // runs before applying its main regex.
            if (digit_triples_cjk && is_digit_triple_number(cp)) {
                int count = 0;
                while (r.pos < r.size && count < 3) {
                    int l;
                    int c = r.peek_cp(l);
                    if (!is_digit_triple_number(c)) break;
                    r.advance(l);
                    count++;
                }
                emit(start, r.pos);
                continue;
            }
            if (digit_triples_cjk && is_exact_cjk_split_letter(cp)) {
                while (r.pos < r.size) {
                    int l;
                    int c = r.peek_cp(l);
                    if (!is_exact_cjk_split_letter(c)) break;
                    r.advance(l);
                }
                emit(start, r.pos);
                continue;
            }

            // Main-regex alternative 1. The traditional profile has explicit
            // contractions. The staged profile uses ASCII punctuation plus an
            // ASCII letter run, which covers those contractions as well as
            // pieces such as "@interface" and ",world".
            if (!digit_triples_cjk) {
                int ct = try_contraction(r);
                if (ct > 0) {
                    r.pos += ct;
                    emit(start, r.pos);
                    continue;
                }
            } else if (is_punctuation_or_symbol(cp) && cp < 0x80) {
                const int punct_len = len;
                const int next_pos = r.pos + punct_len;
                if (next_pos < r.size) {
                    int next_len;
                    const int next =
                        utf8_codepoint(r.data + next_pos,
                                      r.size - next_pos, next_len);
                    if ((next >= 'A' && next <= 'Z') ||
                        (next >= 'a' && next <= 'z')) {
                        r.advance(punct_len);
                        while (r.pos < r.size) {
                            int l;
                            const int c = r.peek_cp(l);
                            if (!((c >= 'A' && c <= 'Z') ||
                                  (c >= 'a' && c <= 'z'))) {
                                break;
                            }
                            r.advance(l);
                        }
                        emit(start, r.pos);
                        continue;
                    }
                }
            }

            // --- Alt 2: optional prefix + contiguous letters/marks ---
            // The prefix is [^\r\n\p{L}\p{N}] — any char that isn't newline/letter/digit.
            // Unlike older GPT-4 regexes, this tokenizer does not split at
            // uppercase/lowercase boundaries; e.g. "ViewController" is one
            // pre-token so BPE can emit a single token.
            {
                int save = r.pos;

                // The staged regex excludes punctuation and symbols from its
                // optional prefix. It must also leave a space before an
                // already-isolated CJK run as a separate piece.
                const bool prefix_allowed =
                    digit_triples_cjk
                        ? (!is_newline(cp) && !is_letter(cp) &&
                           !is_punctuation_or_symbol(cp) && cp != -1)
                        : (!is_newline(cp) && !is_letter(cp) &&
                           !is_digit(cp) && cp != -1);
                if (prefix_allowed) {
                    int next_pos = r.pos + len;
                    if (next_pos < r.size) {
                        int l2;
                        int cp2 = utf8_codepoint(r.data + next_pos, r.size - next_pos, l2);
                        if (is_letter(cp2) &&
                            !(digit_triples_cjk &&
                              is_exact_cjk_split_letter(cp2))) {
                            r.advance(len);
                        }
                    }
                }

                int flen;
                int fcp = r.peek_cp(flen);
                if (is_letter(fcp) &&
                    !(digit_triples_cjk &&
                      is_exact_cjk_split_letter(fcp))) {
                    while (r.pos < r.size) {
                        int l; int c = r.peek_cp(l);
                        if (!is_letter(c) && !is_unicode_mark(c)) break;
                        r.advance(l);
                    }
                    emit(start, r.pos);
                    continue;
                }

                r.pos = save;  // rewind if no letter match
            }

            // Re-read current char
            cp = r.peek_cp(len);
            start = r.pos;

            // --- Alt 3: single digit (traditional profile) ---
            if (is_digit(cp)) {
                r.advance(len);
                emit(start, r.pos);
                continue;
            }

            // --- Alt 4: optional ASCII space + punctuation + trailing newlines ---
            // The tokenizer.json regex keeps a single leading space attached to
            // a punctuation run. Without this, BPE cannot form tokens such as
            // "Ġ=" or "Ġ((".
            {
                int save = r.pos;
                if (cp == ' ') {
                    int next_pos = r.pos + len;
                    if (next_pos < r.size) {
                        int l2;
                        int cp2 = utf8_codepoint(r.data + next_pos, r.size - next_pos, l2);
                        const bool punct =
                            digit_triples_cjk
                                ? is_punctuation_or_symbol(cp2)
                                : (!is_whitespace(cp2) && !is_letter(cp2) &&
                                   !is_digit(cp2) && !is_newline(cp2));
                        if (punct) {
                            r.advance(len);
                            cp = cp2;
                            len = l2;
                        }
                    }
                }
                const bool punct =
                    digit_triples_cjk
                        ? is_punctuation_or_symbol(cp)
                        : (!is_whitespace(cp) && !is_letter(cp) &&
                           !is_digit(cp) && !is_newline(cp));
                if (!(r.pos < r.size && punct)) {
                    r.pos = save;
                } else {
                    start = save;
                    while (r.pos < r.size) {
                        int l; int c = r.peek_cp(l);
                        const bool is_punct =
                            digit_triples_cjk
                                ? is_punctuation_or_symbol(c)
                                : (!is_whitespace(c) && !is_letter(c) &&
                                   !is_digit(c));
                        if (!is_punct) break;
                        r.advance(l);
                    }
                    while (r.pos < r.size && (r.data[r.pos] == '\r' || r.data[r.pos] == '\n'))
                        r.advance(1);
                    emit(start, r.pos);
                    continue;
                }
            }

            // --- Alt 5: whitespace* + newlines ---
            if (is_whitespace(cp)) {
                int save = r.pos;
                while (r.pos < r.size) {
                    int l; int c = r.peek_cp(l);
                    if (!is_whitespace(c) || is_newline(c)) break;
                    r.advance(l);
                }
                int l; int c = r.peek_cp(l);
                if (is_newline(c)) {
                    while (r.pos < r.size) {
                        int nl; int nc = r.peek_cp(nl);
                        if (!is_newline(nc)) break;
                        r.advance(nl);
                    }
                    emit(start, r.pos);
                    continue;
                }
                r.pos = save;
            }

            // --- Alt 6/7: whitespace handling ---
            // \s+(?!\S): consume whitespace greedily, then backtrack until
            // NOT followed by non-whitespace. This means: consume all spaces,
            // but leave one space if followed by non-whitespace (it will
            // become the prefix for the next word/punct match).
            if (is_whitespace(cp) && !is_newline(cp)) {
                // Consume all consecutive non-newline whitespace
                while (r.pos < r.size) {
                    int l; int c = r.peek_cp(l);
                    if (!is_whitespace(c) || is_newline(c)) break;
                    r.advance(l);
                }
                // Check what follows
                if (r.pos < r.size) {
                    int l; int c = r.peek_cp(l);
                    if (!is_whitespace(c)) {
                        // Non-whitespace follows. Backtrack: leave one space
                        // for it to use as prefix (alt 1/2/4 will pick it up).
                        // This implements \s+(?!\S) backtracking behavior.
                        int back = 1;  // ASCII space is 1 byte
                        if (r.pos - back > start) {
                            r.pos -= back;
                        }
                        // If we'd end up emitting nothing, consume at least one
                        if (r.pos <= start) {
                            r.pos = start + 1;
                        }
                    }
                }
                emit(start, r.pos);
                continue;
            }

            // Fallback: single character
            r.advance(len);
            emit(start, r.pos);
        }
    }

    return out;
}

// --- Byte-level encode/decode -----------------------------------------------

std::string Tokenizer::bytes_to_unicode(const std::string& raw) const {
    std::string out;
    for (uint8_t b : raw) out += byte_encoder_[b];
    return out;
}

std::string Tokenizer::unicode_to_bytes(const std::string& encoded) const {
    std::string out;
    const char* s = encoded.c_str();
    int n = (int)encoded.size();
    int i = 0;
    while (i < n) {
        for (int len = 3; len >= 1; len--) {
            if (i + len > n) continue;
            std::string ch(s + i, len);
            auto it = byte_decoder_.find(ch);
            if (it != byte_decoder_.end()) {
                out += (char)it->second;
                i += len;
                goto next_byte;
            }
        }
        out += s[i]; i++;
        next_byte:;
    }
    return out;
}

// --- BPE merge --------------------------------------------------------------

std::vector<std::string> Tokenizer::bpe(const std::string& token) const {
    std::vector<std::string> parts;
    const char* s = token.c_str();
    int n = (int)token.size();
    int i = 0;
    while (i < n) {
        int len;
        utf8_codepoint(s + i, n - i, len);
        parts.push_back(token.substr(i, len));
        i += len;
    }
    if (parts.size() <= 1) return parts;

    while (true) {
        int best_rank = std::numeric_limits<int>::max();
        int best_idx = -1;
        for (int j = 0; j < (int)parts.size() - 1; j++) {
            auto it = merges_.find({parts[j], parts[j + 1]});
            if (it != merges_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = j;
            }
        }
        if (best_idx < 0) break;

        // Merge all occurrences of the best pair in one pass
        std::string merged = parts[best_idx] + parts[best_idx + 1];
        std::vector<std::string> new_parts;
        for (int j = 0; j < (int)parts.size(); j++) {
            if (j < (int)parts.size() - 1 &&
                parts[j] == parts[best_idx] && parts[j + 1] == parts[best_idx + 1]) {
                new_parts.push_back(merged);
                j++;
            } else {
                new_parts.push_back(parts[j]);
            }
        }
        parts = std::move(new_parts);
        if (parts.size() <= 1) break;
    }
    return parts;
}

// --- Encode -----------------------------------------------------------------

// Split text around added/special tokens, finding the earliest match each time.
static std::vector<std::pair<std::string, bool>>
split_added_tokens(const std::string& text,
                   const std::vector<std::string>& patterns) {
    std::vector<std::pair<std::string, bool>> segments;
    std::string remaining = text;

    while (!remaining.empty()) {
        // Find the earliest occurring added token
        size_t best_pos = std::string::npos;
        size_t best_len = 0;
        for (const auto& pat : patterns) {
            auto pos = remaining.find(pat);
            if (pos != std::string::npos) {
                if (pos < best_pos || (pos == best_pos && pat.size() > best_len)) {
                    best_pos = pos;
                    best_len = pat.size();
                }
            }
        }
        if (best_pos == std::string::npos) {
            segments.push_back({remaining, false});
            break;
        }
        if (best_pos > 0)
            segments.push_back({remaining.substr(0, best_pos), false});
        segments.push_back({remaining.substr(best_pos, best_len), true});
        remaining = remaining.substr(best_pos + best_len);
    }
    return segments;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    if (format_ == Format::RWKV_WORLD) return encode_rwkv(text);
    if (text.empty()) return {};

    auto segments = split_added_tokens(text, added_patterns_);

    std::vector<int> ids;
    for (const auto& [seg_text, is_special] : segments) {
        if (is_special) {
            auto it = added_tokens_.find(seg_text);
            if (it != added_tokens_.end())
                ids.push_back(it->second);
            continue;
        }

        auto chunks = pre_tokenize(seg_text);
        for (const auto& chunk : chunks) {
            std::string encoded = bytes_to_unicode(chunk);
            auto pieces = bpe(encoded);
            for (const auto& piece : pieces) {
                auto it = vocab_.find(piece);
                if (it != vocab_.end()) {
                    ids.push_back(it->second);
                } else {
                    // Fallback: encode byte by byte
                    for (uint8_t b : chunk) {
                        auto it2 = vocab_.find(byte_encoder_[b]);
                        if (it2 != vocab_.end())
                            ids.push_back(it2->second);
                    }
                    break;
                }
            }
        }
    }
    return ids;
}

// --- Decode -----------------------------------------------------------------

std::string Tokenizer::decode(int id) const {
    if (format_ == Format::RWKV_WORLD)
        return (id >= 0 && (size_t)id < rwkv_id_to_bytes_.size()) ? rwkv_id_to_bytes_[id] : std::string();
    if (id < 0 || id >= (int)id_to_piece_.size()) return "";
    return unicode_to_bytes(id_to_piece_[id]);
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    if (format_ == Format::RWKV_WORLD) {
        std::string out;
        for (int id : ids) if (id >= 0 && (size_t)id < rwkv_id_to_bytes_.size()) out += rwkv_id_to_bytes_[id];
        return out;
    }
    std::string out;
    for (int id : ids) {
        if (id < 0 || id >= (int)id_to_piece_.size()) continue;
        const auto& piece = id_to_piece_[id];
        if (!piece.empty() && piece[0] == '<' && piece.back() == '>' &&
            (piece.find('|') != std::string::npos ||
             piece.find("｜") != std::string::npos ||
             piece.rfind("<eos:", 0) == 0))
            continue;
        out += unicode_to_bytes(piece);
    }
    return out;
}

// --- Chat template ----------------------------------------------------------

std::vector<int> Tokenizer::apply_chat(const std::string& user_message) const {
    if (format_ == Format::RWKV_WORLD) {
        if (!rwkv_legacy_chat_template_) return encode(user_message);
        return apply_chat(std::vector<ChatMessage>{{"user", user_message}});
    }
    // Single-turn: wrap one user message WITHOUT system prompt
    // (matches transformers default behavior).
    // For ChatML models (Qwen3.5), the template includes <think> tags.
    std::vector<ChatMessage> messages;
    messages.push_back({"user", user_message});
    return apply_chat(messages);
}

std::vector<int> Tokenizer::apply_chat(
    const std::vector<ChatMessage>& messages, bool enable_thinking) const {
    if (format_ == Format::RWKV_WORLD) {
        if (rwkv_legacy_chat_template_) {
            // Match rwkv-mobile's legacy template exactly. Native RWKV .pth
            // files have no tokenizer_config/chat_template, so this is set by
            // the converter as explicit package metadata rather than guessed.
            std::string prompt;
            for (size_t i = 0; i < messages.size(); ++i) {
                const auto& message = messages[i];
                std::string role = message.role == "user" ? "User" :
                                   message.role == "assistant" ? "Assistant" : "System";
                prompt += role + ": " + message.content;
                if (i + 1 < messages.size()) prompt += "\n\n";
            }
            if (!messages.empty()) {
                prompt += "\n\n";
                const auto& last = messages.back().role;
                prompt += last == "assistant" ? "User:" : "Assistant:";
            }
            return encode(prompt);
        }
        std::string prompt;
        for (const auto& message : messages) prompt += message.content;
        return encode(prompt);
    }
    // Old packages did not always embed their chat template. Retain the
    // special-token inference only as a compatibility fallback.
    // Role-middle:
    //   <｜User｜>user<｜im▁middle｜>{content}<｜im▁end｜>
    // DeepSeek-V4:
    //   <｜User｜>{content}<｜Assistant｜></think>
    // ChatML (Qwen3.5): <|im_start|>{role}\n{content}<|im_end|>\n
    // Llama-3 (Youtu-LLM-2B): <|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>
    ChatTemplateStyle style = chat_template_style_;
    if (style == ChatTemplateStyle::AUTO) {
        const bool role_middle =
            added_tokens_.count("<｜User｜>") > 0 &&
            added_tokens_.count("<｜Assistant｜>") > 0 &&
            added_tokens_.count("<｜im▁middle｜>") > 0 &&
            added_tokens_.count("<｜im▁end｜>") > 0;
        if (role_middle) {
            style = ChatTemplateStyle::ROLE_MIDDLE;
        } else if (added_tokens_.count("<｜User｜>") > 0 &&
                   added_tokens_.count("<｜Assistant｜>") > 0) {
            style = ChatTemplateStyle::DEEPSEEK_V4;
        } else if (added_tokens_.count("<|im_start|>") > 0) {
            style = ChatTemplateStyle::CHATML_THINKING;
        } else if (added_tokens_.count("<｜hy_User｜>") > 0 &&
                   added_tokens_.count("<｜hy_Assistant｜>") > 0) {
            style = ChatTemplateStyle::HY_V3;
        } else {
            style = ChatTemplateStyle::LLAMA3_HEADERS;
        }
    }

    std::string prompt;
    if (style == ChatTemplateStyle::ROLE_MIDDLE) {
        prompt = "<｜begin▁of▁sentence｜>";
        int last_user_index = -1;
        for (size_t i = 0; i < messages.size(); ++i) {
            if (messages[i].role == "user")
                last_user_index = static_cast<int>(i);
        }
        auto trim = [](const std::string& text) {
            const size_t begin = text.find_first_not_of(" \t\r\n");
            if (begin == std::string::npos) return std::string();
            const size_t end = text.find_last_not_of(" \t\r\n");
            return text.substr(begin, end - begin + 1);
        };
        for (size_t i = 0; i < messages.size(); ++i) {
            const auto& msg = messages[i];
            if (msg.role == "system") {
                prompt += "<｜System｜>system<｜im▁middle｜>" +
                          msg.content + "<｜im▁end｜>";
            } else if (msg.role == "user") {
                prompt += "<｜User｜>user<｜im▁middle｜>" +
                          msg.content + "<｜im▁end｜>";
            } else if (msg.role == "assistant") {
                std::string reasoning;
                std::string content = msg.content;
                const size_t think_end = content.rfind("</think>");
                if (think_end != std::string::npos) {
                    reasoning = content.substr(0, think_end);
                    const size_t think_begin = reasoning.rfind("<think>");
                    if (think_begin != std::string::npos)
                        reasoning.erase(0, think_begin + 7);
                    content.erase(0, think_end + 8);
                } else if (content.rfind("<think>", 0) == 0) {
                    reasoning = content.substr(7);
                    content.clear();
                }
                prompt +=
                    "<｜Assistant｜>assistant<｜im▁middle｜><think>\n";
                if (static_cast<int>(i) > last_user_index)
                    prompt += trim(reasoning);
                prompt += "\n</think>\n\n";
                prompt += trim(content);
                prompt += "<｜im▁end｜>";
            }
            // Developer/tool messages require the tool-aware branch of the
            // Jinja template and are intentionally omitted by the plain chat
            // CLI, matching the tokenizer's no-tools behavior.
        }
        prompt += "<｜Assistant｜>assistant<｜im▁middle｜>";
        prompt += enable_thinking ? "<think>\n"
                                  : "<think>\n\n</think>\n\n";
    } else if (style == ChatTemplateStyle::HY_V4) {
        // Plain text, no-tools branch of the packaged Hy4 template. The
        // checkpoint's token suffix is read from its added-token vocabulary.
        auto find_token = [&](const std::string& prefix) {
            for (const auto& [piece, id] : added_tokens_) {
                (void)id;
                if (piece.rfind(prefix, 0) == 0) return piece;
            }
            return std::string();
        };
        const std::string start = find_token("<｜hy_start:");
        const std::string middle = find_token("<｜hy_middle:");
        const std::string end = find_token("<｜hy_end:");
        const std::string think_begin = find_token("<think:");
        const std::string think_end = find_token("</think:");
        const std::string reasoning = find_token("<｜reasoning_mode:");
        if (start.empty() || middle.empty() || end.empty() ||
            think_begin.empty() || think_end.empty() || reasoning.empty())
            return {};
        const std::string effort = enable_thinking ? "high" : "no_think";
        const bool leading_system = !messages.empty() &&
                                    messages.front().role == "system";
        if (!leading_system)
            prompt = start + "system" + middle + reasoning +
                     "reasoning_effort:" + effort + end;
        for (size_t i = 0; i < messages.size(); ++i) {
            const auto& msg = messages[i];
            prompt += start + msg.role + middle;
            if (msg.role == "assistant") {
                prompt += think_begin + think_end + msg.content;
            } else {
                prompt += msg.content;
                if (msg.role == "system" && i == 0)
                    prompt += reasoning + "reasoning_effort:" + effort;
            }
            prompt += end;
        }
        if (messages.empty() || messages.back().role != "assistant") {
            prompt += start + "assistant" + middle + think_begin;
            if (!enable_thinking) prompt += think_end;
        }
    } else if (style == ChatTemplateStyle::HY_V3) {
        // Plain no-tools branch of the official HY-V3 template. Hy-MT2 uses
        // no_think by default, including its documented translation example.
        // Tool calls and reasoning_effort are intentionally outside the
        // plain chat API exposed by this runtime.
        const bool opensource =
            added_tokens_.count("<｜hy_User:opensource｜>") > 0;
        const std::string bos = opensource
            ? "<｜hy_begin_of_sentence:opensource｜>"
            : "<｜hy_begin▁of▁sentence｜>";
        const std::string user = opensource
            ? "<｜hy_User:opensource｜>" : "<｜hy_User｜>";
        const std::string assistant = opensource
            ? "<｜hy_Assistant:opensource｜>" : "<｜hy_Assistant｜>";
        const std::string think_begin = opensource
            ? "<think:opensource>" : "<think>";
        const std::string think_end = opensource
            ? "</think:opensource>" : "</think>";
        const std::string reasoning_mode = opensource
            ? "<｜reasoning_mode:opensource｜>" : "<｜reasoning_mode｜>";
        const std::string eos = opensource
            ? "<｜hy_eos:opensource｜>" : "<eos:6124c78e>";
        prompt = bos;
        for (const auto& msg : messages) {
            if (msg.role == "system") prompt += msg.content;
        }
        prompt += reasoning_mode + "reasoning_effort:no_think";
        for (size_t i = 0; i < messages.size(); ++i) {
            const auto& msg = messages[i];
            if (msg.role == "user") {
                prompt += user + msg.content;
            } else if (msg.role == "assistant") {
                prompt += assistant + think_begin + think_end + msg.content;
                if (i + 1 < messages.size()) prompt += eos;
            }
        }
        if (!messages.empty() && messages.back().role != "assistant")
            prompt += assistant + think_begin + think_end;
    } else if (style == ChatTemplateStyle::DEEPSEEK_V4) {
        prompt = "<｜begin▁of▁sentence｜>";
        for (const auto& msg : messages) {
            if (msg.role == "user" || msg.role == "developer") {
                prompt += "<｜User｜>" + msg.content;
            } else if (msg.role == "assistant") {
                // Assistant history must carry the same response prefix that
                // was used when that turn was generated. Omitting it changes
                // the token stream on the next turn and invalidates prefix
                // cache reuse.
                prompt += "<｜Assistant｜></think>" + msg.content +
                          "<｜end▁of▁sentence｜>";
            } else {
                // The official V4 encoder emits system content without a
                // role marker.
                prompt += msg.content;
            }
        }
        if (!messages.empty() && messages.back().role != "assistant")
            prompt += "<｜Assistant｜></think>";
    } else if (style == ChatTemplateStyle::CHATML_PLAIN ||
               style == ChatTemplateStyle::CHATML_THINKING) {
        // ChatML format (no BOS token).
        for (size_t i = 0; i < messages.size(); i++) {
            const auto& msg = messages[i];
            prompt += "<|im_start|>" + msg.role + "\n" + msg.content + "<|im_end|>\n";
        }
        prompt += "<|im_start|>assistant\n";
        if (style == ChatTemplateStyle::CHATML_THINKING)
            prompt += "<think>\n\n</think>\n\n";
    } else if (style == ChatTemplateStyle::ROLE_TOKENS) {
        if (bos_id_ >= 0 && bos_id_ < static_cast<int>(id_to_piece_.size()))
            prompt = id_to_piece_[bos_id_];
        for (const auto& msg : messages) {
            if (msg.role == "system") {
                prompt += msg.content;
            } else if (msg.role == "user") {
                prompt += "<|User|>" + msg.content;
            } else if (msg.role == "assistant") {
                prompt += "<|Assistant|>" + msg.content +
                          "<|end_of_text|>";
            }
        }
        if (!messages.empty() && messages.back().role != "assistant")
            prompt += "<|Assistant|>";
    } else {
        // Llama-3 format
        prompt = "<|begin_of_text|>";
        for (size_t i = 0; i < messages.size(); i++) {
            const auto& msg = messages[i];
            prompt += "<|start_header_id|>" + msg.role + "<|end_header_id|>\n\n" + msg.content;

            bool is_last = (i == messages.size() - 1);
            if (is_last && msg.role == "user") {
                prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
            } else {
                prompt += "<|eot_id|>";
            }
        }
    }

    return encode(prompt);
}
