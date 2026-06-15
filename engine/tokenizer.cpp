#include "engine/tokenizer.h"
#include <json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>

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

bool Tokenizer::load(const std::string& path) {
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

    if (added_tokens_.count("<|begin_of_text|>"))
        bos_id_ = added_tokens_["<|begin_of_text|>"];
    if (added_tokens_.count("<|end_of_text|>"))
        eos_id_ = added_tokens_["<|end_of_text|>"];

    fprintf(stderr, "Tokenizer: %d vocab + %d added tokens, %d merges\n",
           (int)vocab_obj.size(), (int)added_tokens_.size(), (int)merges_.size());
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

static bool is_cjk_or_kana(int cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) ||   // CJK Unified
           (cp >= 0x3400 && cp <= 0x4DBF) ||   // CJK Extension A
           (cp >= 0x3040 && cp <= 0x309F) ||   // Hiragana
           (cp >= 0x30A0 && cp <= 0x30FF) ||   // Katakana
           (cp >= 0xAC00 && cp <= 0xD7AF) ||   // Hangul
           (cp >= 0x3000 && cp <= 0x303F) ||   // CJK Symbols
           (cp >= 0xFF01 && cp <= 0xFF60) ||   // Fullwidth forms
           (cp >= 0xFE30 && cp <= 0xFE4F) ||   // CJK Compatibility
           (cp >= 0x3105 && cp <= 0x312F);     // Bopomofo
}

static bool is_letter(int cp) {
    if (cp >= 'A' && cp <= 'Z') return true;
    if (cp >= 'a' && cp <= 'z') return true;
    // Non-ASCII letters (not CJK/Hangul/Kana — those go through Stage 0)
    if (cp > 0x7F && !is_cjk_or_kana(cp)) return true;
    return false;
}

static bool is_upper(int cp) {
    return (cp >= 'A' && cp <= 'Z');
}

static bool is_lower(int cp) {
    if (cp >= 'a' && cp <= 'z') return true;
    // Non-ASCII non-CJK are treated as Lo (Letter, other) = lowercase-like
    if (cp > 0x7F && !is_cjk_or_kana(cp)) return true;
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
// Implements the 3-stage HuggingFace pre-tokenizer:
//   Stage 0: CJK isolation — groups consecutive CJK/Hangul/Kana characters
//   Stage 1: GPT-4 word splitting regex (simplified)
//   Stage 2: Byte-level encoding (done in encode(), not here)
//
// Stage 1 regex alternatives (tried in order):
//   1. [^\r\n\p{L}\p{N}]?\p{Lu}*\p{Ll}+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//   2. [^\r\n\p{L}\p{N}]?\p{Lu}+\p{Ll}*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//   3. \p{N}                          — single digit
//   4.  ?[^\s\p{L}\p{N}]+[\r\n/]*    — punctuation with opt. space prefix
//   5. \s*[\r\n]+                     — newlines
//   6. \s+(?!\S)                      — trailing whitespace
//   7. \s+                            — whitespace

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

    // Stage 0: Split CJK groups vs non-CJK segments
    struct Span { int start; int end; bool is_cjk; };
    std::vector<Span> spans;
    {
        CharReader r{text.c_str(), (int)text.size()};
        while (r.pos < r.size) {
            int len;
            int cp = r.peek_cp(len);
            if (is_cjk_or_kana(cp)) {
                int start = r.pos;
                while (r.pos < r.size) {
                    cp = r.peek_cp(len);
                    if (!is_cjk_or_kana(cp)) break;
                    r.advance(len);
                }
                spans.push_back({start, r.pos, true});
            } else {
                int start = r.pos;
                while (r.pos < r.size) {
                    cp = r.peek_cp(len);
                    if (is_cjk_or_kana(cp)) break;
                    r.advance(len);
                }
                spans.push_back({start, r.pos, false});
            }
        }
    }

    // Stage 1: For each span, apply GPT-4-like word splitting
    //
    // The regex tries alternatives in order at each position:
    //   1. [^\r\n\p{L}\p{N}]?\p{Lu}*\p{Ll}+(contraction)?
    //   2. [^\r\n\p{L}\p{N}]?\p{Lu}+\p{Ll}*(contraction)?
    //   3. \p{N}
    //   4.  ?[^\s\p{L}\p{N}]+[\r\n/]*
    //   5. \s*[\r\n]+
    //   6. \s+(?!\S)  — whitespace NOT followed by non-whitespace (backtracks)
    //   7. \s+

    auto emit = [&](int abs_start, int abs_end) {
        if (abs_end > abs_start)
            out.push_back(text.substr(abs_start, abs_end - abs_start));
    };

    // Try to match a contraction suffix ('s, 't, etc.) at current position.
    // Returns number of bytes consumed (0 if no match).
    auto try_contraction = [](const CharReader& r) -> int {
        if (r.pos >= r.size || r.data[r.pos] != '\'') return 0;
        int save = r.pos + 1;
        int end = save;
        while (end < r.size && r.data[end] >= 'a' && r.data[end] <= 'z') end++;
        int slen = end - save;
        const char* s = r.data + save;
        if ((slen == 1 && (*s == 's' || *s == 't' || *s == 'm' || *s == 'd')) ||
            (slen == 2 && ((s[0] == 'r' && s[1] == 'e') || (s[0] == 'v' && s[1] == 'e') || (s[0] == 'l' && s[1] == 'l')))) {
            return end - r.pos;
        }
        return 0;
    };

    for (auto& span : spans) {
        if (span.is_cjk) {
            out.push_back(text.substr(span.start, span.end - span.start));
            continue;
        }

        CharReader r{text.c_str() + span.start, span.end - span.start};

        while (r.pos < r.size) {
            int start = r.pos;
            int len;
            int cp = r.peek_cp(len);

            // --- Alt 1 & 2: optional prefix + letters + contraction ---
            // The prefix is [^\r\n\p{L}\p{N}] — any char that isn't newline/letter/digit
            {
                int save = r.pos;
                bool has_prefix = false;

                // Optional prefix: non-newline, non-letter, non-digit
                if (!is_newline(cp) && !is_letter(cp) && !is_digit(cp) && cp != -1) {
                    int next_pos = r.pos + len;
                    if (next_pos < r.size) {
                        int l2;
                        int cp2 = utf8_codepoint(r.data + next_pos, r.size - next_pos, l2);
                        if (is_letter(cp2)) {
                            has_prefix = true;
                            r.advance(len);
                        }
                    }
                }

                int flen;
                int fcp = r.peek_cp(flen);
                if (is_letter(fcp)) {
                    if (is_upper(fcp)) {
                        // Consume uppercase run
                        int upper_count = 0;
                        while (r.pos < r.size) {
                            int l; int c = r.peek_cp(l);
                            if (!is_upper(c)) break;
                            r.advance(l);
                            upper_count++;
                        }
                        // Consume lowercase run
                        while (r.pos < r.size) {
                            int l; int c = r.peek_cp(l);
                            if (!is_lower(c)) break;
                            r.advance(l);
                        }
                    } else {
                        // Lowercase start: consume all lowercase
                        while (r.pos < r.size) {
                            int l; int c = r.peek_cp(l);
                            if (!is_lower(c)) break;
                            r.advance(l);
                        }
                    }
                    // Try contraction
                    int ct = try_contraction(r);
                    r.pos += ct;
                    emit(span.start + start, span.start + r.pos);
                    continue;
                }

                r.pos = save;  // rewind if no letter match
            }

            // Re-read current char
            cp = r.peek_cp(len);
            start = r.pos;

            // --- Alt 3: single digit ---
            if (is_digit(cp)) {
                r.advance(len);
                emit(span.start + start, span.start + r.pos);
                continue;
            }

            // --- Alt 4: optional space + punctuation + trailing newlines ---
            if (!is_whitespace(cp) && !is_letter(cp) && !is_digit(cp) && !is_newline(cp)) {
                // Punctuation without space prefix
                while (r.pos < r.size) {
                    int l; int c = r.peek_cp(l);
                    if (is_whitespace(c) || is_letter(c) || is_digit(c)) break;
                    r.advance(l);
                }
                while (r.pos < r.size && (r.data[r.pos] == '\r' || r.data[r.pos] == '\n'))
                    r.advance(1);
                emit(span.start + start, span.start + r.pos);
                continue;
            }

            // --- Alt 5: whitespace* + newlines ---
            if (is_newline(cp)) {
                while (r.pos < r.size) {
                    int l; int c = r.peek_cp(l);
                    if (!is_newline(c)) break;
                    r.advance(l);
                }
                emit(span.start + start, span.start + r.pos);
                continue;
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
                emit(span.start + start, span.start + r.pos);
                continue;
            }

            // Fallback: single character
            r.advance(len);
            emit(span.start + start, span.start + r.pos);
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
                   const std::vector<std::string>& patterns,
                   const std::unordered_map<std::string, int>& added_tokens) {
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
    if (text.empty()) return {};

    auto segments = split_added_tokens(text, added_patterns_, added_tokens_);

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
    if (id < 0 || id >= (int)id_to_piece_.size()) return "";
    return unicode_to_bytes(id_to_piece_[id]);
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) {
        if (id < 0 || id >= (int)id_to_piece_.size()) continue;
        const auto& piece = id_to_piece_[id];
        if (!piece.empty() && piece[0] == '<' && piece.back() == '>' &&
            piece.find('|') != std::string::npos)
            continue;
        out += unicode_to_bytes(piece);
    }
    return out;
}

// --- Chat template ----------------------------------------------------------

std::vector<int> Tokenizer::apply_chat(const std::string& user_message) const {
    std::string prompt =
        "<|begin_of_text|>system\nYou are a helpful assistant.<|end_of_text|>\n"
        "<|begin_of_text|>user\n" + user_message + "<|end_of_text|>\n"
        "<|begin_of_text|>assistant\n";
    return encode(prompt);
}

