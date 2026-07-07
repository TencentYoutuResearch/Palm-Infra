#pragma once

// Byte-level BPE tokenizer (GPT-2 / Llama-3 style)
//
// Loads a HuggingFace tokenizer.json and performs encode / decode.
// Based on the llm.ncnn tokenizer implementation.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// A single message in a chat conversation.
struct ChatMessage {
    std::string role;      // "system", "user", "assistant"
    std::string content;   // message text
};

class Tokenizer {
public:
    bool load(const std::string& tokenizer_json_path);

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;
    std::string decode(int id) const;

    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }
    int vocab_size() const { return (int)id_to_piece_.size(); }

    // Wrap a single user message in the model's chat template.
    // Convenience wrapper for single-turn prompts.
    std::vector<int> apply_chat(const std::string& user_message) const;

    // Build a multi-turn chat prompt from a list of messages.
    // Produces Llama-3 format:
    //   <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys}<|eot_id|>
    //   <|start_header_id|>user<|end_header_id|>\n\n{user1}<|eot_id|>
    //   <|start_header_id|>assistant<|end_header_id|>\n\n{asst1}<|eot_id|>
    //   ...
    //   <|start_header_id|>assistant<|end_header_id|>\n\n
    // The last message should be from "user" — the template leaves the
    // assistant header open for the model to continue.
    std::vector<int> apply_chat(const std::vector<ChatMessage>& messages) const;

private:
    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string> id_to_piece_;

    struct PairHash {
        size_t operator()(const std::pair<std::string, std::string>& p) const {
            size_t h1 = std::hash<std::string>{}(p.first);
            size_t h2 = std::hash<std::string>{}(p.second);
            return h1 ^ (h2 * 0x9e3779b97f4a7c15ULL + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
        }
    };
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> merges_;

    std::unordered_map<std::string, int> added_tokens_;
    std::vector<std::string> added_patterns_;

    std::string byte_encoder_[256];
    std::unordered_map<std::string, uint8_t> byte_decoder_;

    int bos_id_ = 128000;
    int eos_id_ = 128001;

    void build_byte_tables();
    std::vector<std::string> pre_tokenize(const std::string& text) const;
    std::string bytes_to_unicode(const std::string& raw) const;
    std::string unicode_to_bytes(const std::string& encoded) const;
    std::vector<std::string> bpe(const std::string& token) const;
};
