#include "kernels/moe_ssd.h"

#include <cstdio>
#include <cstdint>
#include <fstream>
#include <limits>
#include <thread>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

MoeSsdTensorSpec spec(const char* name, uint64_t offset) {
    MoeSsdTensorSpec out;
    out.weight_ref = name;
    out.layer = 0;
    out.num_experts = 3;
    out.rows = 1;
    out.cols = 2;
    out.precision = Precision::FP16;
    out.data_offset = offset;
    out.data_bytes = 2 * sizeof(uint16_t);
    return out;
}

} // namespace

int main() {
    const std::string path = "/tmp/mollm_test_moe_ssd.bin";
    // Three gate expert slices, then three down slices. The values are raw
    // fp16 bit patterns; the cache must preserve their byte ordering exactly.
    const uint16_t contents[] = {
        0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600,
        0x4700, 0x4800, 0x4900, 0x4a00, 0x4b00, 0x4c00,
    };
    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(contents), sizeof(contents));
    }

    // Reject corrupt package metadata during registration, before an I/O
    // worker can turn it into a short read or an overflowing allocation.
    {
        MoeSsdCache cache;
        check(cache.open(path, 16), "open metadata-validation cache");
        auto past_end = spec("past_end", sizeof(contents));
        check(!cache.add_source(past_end), "reject expert data beyond package");
        check(cache.find_source("past_end") == nullptr,
              "rejected source is not registered");

        auto overflowing = spec("overflowing", 0);
        overflowing.data_bytes = std::numeric_limits<uint64_t>::max();
        check(!cache.add_source(overflowing), "reject overflowing expert extent");
        check(cache.find_source("overflowing") == nullptr,
              "overflowing source is not registered");
    }

    {
        MoeSsdCache cache;
        check(cache.open(path, 16), "open cache");  // holds exactly two expert pairs
        check(cache.add_source(spec("./gate", 0)), "add gate source");
        check(cache.add_source(spec("./down", 6 * sizeof(uint16_t))), "add down source");
        const MoeSsdTensorSource* gate = cache.find_source("gate");
        const MoeSsdTensorSource* down = cache.find_source("./down");
        check(gate && down, "find normalized source refs");

        Tensor gu, dw;
        check(cache.acquire(gate, down, 1, gu, dw), "load expert one");
        check(gu.prec == Precision::FP16 && gu.shape[0] == 1 && gu.shape[1] == 2,
              "gate expert tensor shape");
        const uint16_t* gu_data = static_cast<const uint16_t*>(gu.data);
        const uint16_t* dw_data = static_cast<const uint16_t*>(dw.data);
        check(gu_data[0] == 0x4200 && gu_data[1] == 0x4400, "gate bytes match expert one");
        check(dw_data[0] == 0x4900 && dw_data[1] == 0x4a00, "down bytes match expert one");

        check(cache.acquire(gate, down, 1, gu, dw), "cache hit expert one");
        check(cache.acquire(gate, down, 2, gu, dw), "load expert two");
        check(cache.acquire(gate, down, 0, gu, dw), "load expert zero and evict LRU");
        MoeSsdCache::Stats stats = cache.stats();
        check(stats.hits == 1 && stats.misses == 3, "hit/miss accounting");
        check(stats.evictions == 1, "LRU eviction accounting");
        check(stats.resident_bytes == 16, "cache capacity accounting");
        check(stats.bytes_read == 24, "pread byte accounting");
    }

    // A bounded route window can release a consumed pair and immediately make
    // room for the next expert without invalidating the still-needed entries.
    {
        MoeSsdCache cache;
        check(cache.open(path, 16, 2), "open sliding-window cache");
        check(cache.add_source(spec("gate", 0)), "add sliding-window gate source");
        check(cache.add_source(spec("down", 6 * sizeof(uint16_t))),
              "add sliding-window down source");
        const MoeSsdTensorSource* gate = cache.find_source("gate");
        const MoeSsdTensorSource* down = cache.find_source("down");
        check(gate && down, "find sliding-window sources");
        check(cache.request_many(gate, down, {0, 1, 2}), "queue sliding-window routes");
        check(cache.resident_count(gate, down, {0, 1, 2}) == 2,
              "initial window honors cache capacity");
        check(cache.contains(gate, down, 0) && cache.contains(gate, down, 1) &&
              !cache.contains(gate, down, 2), "initial window retains earliest routes");

        Tensor gu, dw;
        check(cache.acquire(gate, down, 0, gu, dw), "acquire sliding-window expert zero");
        check(cache.release(gate, down, 0), "release consumed expert zero");
        check(cache.request_many(gate, down, {1, 2}), "advance sliding-window prefetch");
        check(cache.resident_count(gate, down, {1, 2}) == 2,
              "advanced window contains remaining routes");
        check(cache.contains(gate, down, 1) && cache.contains(gate, down, 2),
              "advanced window retains ready future route");
        check(cache.acquire(gate, down, 2, gu, dw), "acquire prefetched expert two");
        check(static_cast<const uint16_t*>(gu.data)[0] == 0x4500,
              "advanced window preserves expert bytes");
    }

    // With room for all three pairs, request adjacent experts together. This
    // exercises the coalesced component reads: each component run is read as
    // one contiguous range, then scattered back to the individual tensors.
    {
        MoeSsdCache cache;
        check(cache.open(path, 24, 8), "open coalesced async cache");
        check(cache.add_source(spec("gate", 0)), "add coalesced gate source");
        check(cache.add_source(spec("down", 6 * sizeof(uint16_t))),
              "add coalesced down source");
        const MoeSsdTensorSource* gate = cache.find_source("gate");
        const MoeSsdTensorSource* down = cache.find_source("down");
        check(gate && down, "find coalesced sources");
        check(cache.request_many(gate, down, {0, 1, 2}),
              "queue adjacent coalesced expert requests");

        Tensor gu, dw;
        check(cache.acquire(gate, down, 0, gu, dw), "acquire coalesced expert zero");
        check(static_cast<const uint16_t*>(gu.data)[0] == 0x3c00 &&
              static_cast<const uint16_t*>(dw.data)[1] == 0x4800,
              "coalesced expert zero bytes match");
        check(cache.acquire(gate, down, 2, gu, dw), "acquire coalesced expert two");
        check(static_cast<const uint16_t*>(gu.data)[0] == 0x4500 &&
              static_cast<const uint16_t*>(dw.data)[1] == 0x4c00,
              "coalesced expert two bytes match");
        MoeSsdCache::Stats stats = cache.stats();
        check(stats.misses == 3 && stats.bytes_read == 24,
              "coalesced reads preserve miss and byte accounting");
    }

    // A one-entry cache cannot enqueue all requested experts at once. Verify
    // that the deferred requests are scheduled as earlier reads complete,
    // rather than blocking the caller before any compute can overlap them.
    {
        MoeSsdCache cache;
        check(cache.open(path, 8, 2), "open one-entry async cache");
        check(cache.add_source(spec("gate", 0)), "add async gate source");
        check(cache.add_source(spec("down", 6 * sizeof(uint16_t))), "add async down source");
        const MoeSsdTensorSource* gate = cache.find_source("gate");
        const MoeSsdTensorSource* down = cache.find_source("down");
        check(gate && down, "find async sources");
        check(cache.request_many(gate, down, {0, 1, 2}), "queue async expert requests");

        Tensor gu, dw;
        check(cache.acquire(gate, down, 0, gu, dw), "acquire queued expert zero");
        check(cache.acquire(gate, down, 1, gu, dw), "acquire deferred expert one");
        check(cache.acquire(gate, down, 2, gu, dw), "acquire deferred expert two");
        MoeSsdCache::Stats stats = cache.stats();
        check(stats.misses == 3, "deferred requests issue one read each");
        check(stats.evictions == 2, "one-entry cache evicts between deferred requests");
    }

    // A transient pread failure must not poison an expert entry permanently.
    // Restore the backing file and verify that the next acquire removes the
    // failed entry, queues a fresh read, and succeeds.
    {
        MoeSsdCache cache;
        check(cache.open(path, 8, 1), "open retry cache");
        check(cache.add_source(spec("retry_gate", 0)), "add retry gate source");
        check(cache.add_source(spec("retry_down", 6 * sizeof(uint16_t))),
              "add retry down source");
        const MoeSsdTensorSource* gate = cache.find_source("retry_gate");
        const MoeSsdTensorSource* down = cache.find_source("retry_down");

        {
            std::ofstream empty(path, std::ios::binary | std::ios::trunc);
        }
        Tensor gu, dw;
        check(!cache.acquire(gate, down, 0, gu, dw),
              "first acquire observes truncated-file read failure");
        check(!cache.contains(gate, down, 0),
              "failed entry is not reported as cached");

        {
            std::ofstream restored(path, std::ios::binary | std::ios::trunc);
            restored.write(reinterpret_cast<const char*>(contents),
                           sizeof(contents));
        }
        check(cache.acquire(gate, down, 0, gu, dw),
              "second acquire retries transient read failure");
        check(static_cast<const uint16_t*>(gu.data)[0] == 0x3c00 &&
                  static_cast<const uint16_t*>(dw.data)[1] == 0x4800,
              "retried expert bytes match");
    }

    // Shallow-favoring layout retains one streamable slot in every layer,
    // then gives the remaining budget to early MoE layers. With 32 bytes, two
    // layers need 8 bytes each for that baseline; layer zero gets the other
    // 16 bytes and can retain all three of its expert pairs.
    {
        MoeSsdCache cache;
        check(cache.open(path, 32, 2), "open shallow-favoring cache");
        auto gate0 = spec("gate0", 0);
        auto down0 = spec("down0", 6 * sizeof(uint16_t));
        auto gate1 = spec("gate1", 0);
        auto down1 = spec("down1", 6 * sizeof(uint16_t));
        gate1.layer = down1.layer = 1;
        check(cache.add_source(gate0) && cache.add_source(down0) &&
              cache.add_source(gate1) && cache.add_source(down1),
              "add shallow-favoring sources");
        check(cache.configure_shallow_favoring(1), "configure shallow-favoring layout");
        check(cache.layer_capacity_bytes(0) == 24 && cache.layer_capacity_bytes(1) == 8,
              "shallow layer receives the surplus cache budget");
        const MoeSsdTensorSource* g0 = cache.find_source("gate0");
        const MoeSsdTensorSource* d0 = cache.find_source("down0");
        const MoeSsdTensorSource* g1 = cache.find_source("gate1");
        const MoeSsdTensorSource* d1 = cache.find_source("down1");
        check(cache.request_many(g0, d0, {0, 1, 2}), "queue shallow layer routes");
        check(cache.request_many(g1, d1, {0, 1, 2}), "queue deep layer routes");
        check(cache.resident_count(g0, d0, {0, 1, 2}) == 3,
              "shallow layer retains all experts");
        check(cache.resident_count(g1, d1, {0, 1, 2}) == 1,
              "deep layer retains its streaming slot");
    }

    // A global pool lets one layer borrow capacity from every other layer.
    // Once layer zero fills the three-pair cache, loading a layer-one expert
    // evicts the global LRU rather than failing because layer one has no local
    // quota left.
    {
        MoeSsdCache cache;
        check(cache.open(path, 24, 2), "open global-pool cache");
        auto gate0 = spec("global_gate0", 0);
        auto down0 = spec("global_down0", 6 * sizeof(uint16_t));
        auto gate1 = spec("global_gate1", 0);
        auto down1 = spec("global_down1", 6 * sizeof(uint16_t));
        gate1.layer = down1.layer = 1;
        check(cache.add_source(gate0) && cache.add_source(down0) &&
              cache.add_source(gate1) && cache.add_source(down1),
              "add global-pool sources");
        check(cache.set_global_capacity_pool(true), "enable global capacity pool");
        const MoeSsdTensorSource* g0 = cache.find_source("global_gate0");
        const MoeSsdTensorSource* d0 = cache.find_source("global_down0");
        const MoeSsdTensorSource* g1 = cache.find_source("global_gate1");
        const MoeSsdTensorSource* d1 = cache.find_source("global_down1");
        check(cache.request_many(g0, d0, {0, 1, 2}), "fill global cache from layer zero");
        Tensor gu, dw;
        check(cache.acquire(g0, d0, 0, gu, dw) && cache.acquire(g0, d0, 1, gu, dw) &&
              cache.acquire(g0, d0, 2, gu, dw), "finish layer-zero global reads");
        check(cache.prefetch_many(g1, d1, {0}, {1.0f}),
              "submit cache-aware speculative request");
        check(!cache.contains(g1, d1, 0),
              "speculation does not evict current demand entries");
        check(cache.stats().cross_layer_rejected == 1,
              "rejected speculation is accounted");
        cache.begin_forward_pass();
        check(cache.prefetch_many(g1, d1, {0}, {1.0f}),
              "borrow stale global space for predicted layer");
        check(cache.contains(g1, d1, 0), "global pool admits the next layer");
        check(cache.resident_count(g0, d0, {0, 1, 2}) == 2,
              "global pool evicts one layer-zero LRU entry");
    }

    // Cross-layer prediction feedback trims only a consistently inaccurate
    // tail rank. Use a zero-entry prefetch window so this test exercises the
    // policy without generating irrelevant I/O.
    {
        MoeSsdCache cache;
        check(cache.open(path, 16, 1), "open adaptive-prediction cache");
        check(cache.add_source(spec("adaptive_gate", 0)) &&
              cache.add_source(spec("adaptive_down", 6 * sizeof(uint16_t))),
              "add adaptive-prediction sources");
        const MoeSsdTensorSource* gate = cache.find_source("adaptive_gate");
        const MoeSsdTensorSource* down = cache.find_source("adaptive_down");
        for (int sample = 0; sample < 128; ++sample) {
            cache.begin_forward_pass();
            check(cache.prefetch_many(gate, down, {0, 1}, {1.0f, 0.5f}, 0),
                  "record adaptive prediction");
            check(cache.request_many(gate, down, {0}),
                  "evaluate adaptive prediction");
        }
        check(cache.recommended_prefetch_count(2) == 1,
              "adaptive policy removes inaccurate tail rank");
        const auto stats = cache.stats();
        check(stats.cross_layer_rank_attempts.size() == 2 &&
              stats.cross_layer_rank_attempts[0] == 128 &&
              stats.cross_layer_rank_hits[0] == 128 &&
              stats.cross_layer_rank_hits[1] == 0,
              "adaptive policy reports per-rank accuracy");
        cache.reset_stats();
        check(cache.recommended_prefetch_count(2) == 1,
              "statistics reset preserves learned prefetch policy");
    }

    // Track cache pollution separately from useful demand residency, and flag
    // an expert which has to be reloaded immediately after eviction.
    {
        MoeSsdCache cache;
        check(cache.open(path, 16, 1), "open cache-statistics cache");
        check(cache.add_source(spec("stats_gate", 0)) &&
              cache.add_source(spec("stats_down", 6 * sizeof(uint16_t))),
              "add cache-statistics sources");
        check(cache.set_global_capacity_pool(true),
              "enable cache-statistics global pool");
        const MoeSsdTensorSource* gate = cache.find_source("stats_gate");
        const MoeSsdTensorSource* down = cache.find_source("stats_down");
        check(cache.prefetch_many(gate, down, {0}, {1.0f}),
              "queue unused speculative expert");
        for (int spin = 0; spin < 100 && cache.stats().bytes_read < 8; ++spin)
            std::this_thread::yield();

        Tensor gu, dw;
        check(cache.acquire(gate, down, 1, gu, dw),
              "fill cache beside speculative expert");
        cache.begin_forward_pass();
        check(cache.acquire(gate, down, 2, gu, dw),
              "evict stale unused speculation");
        check(cache.acquire(gate, down, 0, gu, dw),
              "reload recently evicted expert");
        const auto stats = cache.stats();
        check(!stats.layers.empty() &&
              stats.layers[0].unused_prefetch_evictions >= 1,
              "unused speculative eviction is counted");
        check(stats.layers[0].short_term_reloads >= 1,
              "short-term reload is counted");
        check(stats.layers[0].demand_acquires == 3,
              "per-layer demand acquisitions are counted");
    }

    std::remove(path.c_str());
    if (failures == 0) std::printf("All MoE SSD cache tests passed!\n");
    return failures == 0 ? 0 : 1;
}
