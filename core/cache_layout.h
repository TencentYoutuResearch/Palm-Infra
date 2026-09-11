#pragma once

#include <cstddef>
#include <cstdint>

// Layout shared by KV-cache producers and consumers.
//
// A KV-cache allocation starts with this fixed-size header, followed by the
// key/value payload. Keep this definition independent from LLMEngine so CPU
// and accelerator attention implementations can consume cache storage without
// depending on engine lifecycle or generation APIs.
struct CacheMetadata {
    uint64_t current_seq_len = 0;   // valid sequence length (past_len)
    uint64_t max_seq_len     = 0;   // buffer capacity (n_ctx)
    uint64_t num_kv_heads    = 0;
    uint64_t head_dim        = 0;
    uint64_t v_head_dim      = 0;
    uint64_t reserved[3]     = {0, 0, 0};

    static constexpr size_t SIZE = 64;
};

static_assert(sizeof(CacheMetadata) == CacheMetadata::SIZE,
              "CacheMetadata must be 64 bytes");

inline CacheMetadata* cache_meta(void* data) {
    return static_cast<CacheMetadata*>(data);
}

inline const CacheMetadata* cache_meta(const void* data) {
    return static_cast<const CacheMetadata*>(data);
}

inline void* cache_data(void* data) {
    return static_cast<char*>(data) + CacheMetadata::SIZE;
}

inline const void* cache_data(const void* data) {
    return static_cast<const char*>(data) + CacheMetadata::SIZE;
}
