#include "engine/tokenizer.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

static int rwkv_hex(char c) {
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
    if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
    return -1;
}

// The official RWKV world vocabulary stores a Python bytes/string literal per
// line: `id literal byte_length`. This parser deliberately handles the escape
// forms used by that file without depending on Python at runtime.
static std::string rwkv_literal(const std::string& s) {
    size_t begin = 0;
    size_t end = s.size();
    char quote = 0;
    if (s.size() >= 3 && s[0] == 'b' &&
        (s[1] == '\'' || s[1] == '"')) {
        begin = 2;
        quote = s[1];
    } else if (s.size() >= 2 && (s[0] == '\'' || s[0] == '"')) {
        begin = 1;
        quote = s[0];
    } else {
        return s;
    }
    if (end > begin && s[end - 1] == quote)
        --end;

    std::string out;
    for (size_t i = begin; i < end; ++i) {
        if (s[i] != '\\') {
            out += s[i];
            continue;
        }
        ++i;
        if (i >= end) {
            // Preserve a trailing slash in a malformed literal instead of
            // indexing one byte past the input.
            out += '\\';
            break;
        }
        switch (s[i]) {
        case 'n':
            out += '\n';
            break;
        case 't':
            out += '\t';
            break;
        case 'r':
            out += '\r';
            break;
        case '\\':
            out += '\\';
            break;
        case '\'':
            out += '\'';
            break;
        case '"':
            out += '"';
            break;
        case 'x':
            if (i + 2 < end) {
                const int high = rwkv_hex(s[i + 1]);
                const int low = rwkv_hex(s[i + 2]);
                if (high >= 0 && low >= 0) {
                    out += (char)((high << 4) | low);
                    i += 2;
                    break;
                }
            }
            out += 'x';
            break;
        default:
            out += s[i];
            break;
        }
    }
    return out;
}

bool Tokenizer::load_rwkv_vocab(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        return false;

    format_ = Format::RWKV_WORLD;
    rwkv_trie_.assign(1, {});
    rwkv_id_to_bytes_.clear();
    std::string line;
    while (std::getline(f, line)) {
        const size_t first = line.find(' ');
        const size_t last = line.rfind(' ');
        if (first == std::string::npos || last <= first)
            continue;

        int id;
        try {
            id = std::stoi(line.substr(0, first));
        } catch (...) {
            return false;
        }
        const std::string bytes =
            rwkv_literal(line.substr(first + 1, last - first - 1));
        if (id < 0 || bytes.empty())
            continue;
        if ((size_t)id >= rwkv_id_to_bytes_.size())
            rwkv_id_to_bytes_.resize((size_t)id + 1);
        rwkv_id_to_bytes_[id] = bytes;

        int node = 0;
        for (uint8_t byte : bytes) {
            auto it = rwkv_trie_[node].next.find(byte);
            if (it == rwkv_trie_[node].next.end()) {
                const int next = (int)rwkv_trie_.size();
                rwkv_trie_[node].next[byte] = next;
                rwkv_trie_.push_back({});
                node = next;
            } else {
                node = it->second;
            }
        }
        rwkv_trie_[node].token = id;
    }
    bos_id_ = 0;
    eos_id_ = 0;
    return rwkv_trie_.size() > 1;
}

void Tokenizer::set_rwkv_legacy_chat_template(bool enabled) {
    if (format_ != Format::RWKV_WORLD) return;
    rwkv_legacy_chat_template_ = enabled;
    // rwkv-mobile uses "\n\n" as the legacy end-of-turn marker. In the
    // official world vocabulary it is token 261; only adopt it when verified
    // so an alternate vocabulary remains safe.
    if (enabled && rwkv_id_to_bytes_.size() > 261 && rwkv_id_to_bytes_[261] == "\n\n") {
        eos_id_ = 261;
    } else {
        eos_id_ = 0;
    }
}

std::vector<std::string> Tokenizer::stop_sequences() const {
    if (format_ == Format::RWKV_WORLD && rwkv_legacy_chat_template_)
        return {"\n\n"};  // rwkv-mobile's legacy end-of-turn delimiter
    return {};
}

std::vector<int> Tokenizer::encode_rwkv(const std::string& text) const {
    std::vector<int> ids;
    size_t pos = 0;
    while (pos < text.size()) {
        int node = 0;
        int best_token = -1;
        size_t cursor = pos;
        size_t best_end = pos;
        while (cursor < text.size()) {
            auto it =
                rwkv_trie_[node].next.find((uint8_t)text[cursor]);
            if (it == rwkv_trie_[node].next.end())
                break;
            node = it->second;
            ++cursor;
            if (rwkv_trie_[node].token >= 0) {
                best_token = rwkv_trie_[node].token;
                best_end = cursor;
            }
        }
        if (best_token < 0)
            return {};
        ids.push_back(best_token);
        pos = best_end;
    }
    return ids;
}
