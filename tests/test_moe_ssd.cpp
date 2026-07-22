#include "kernels/moe_ssd.h"

#include <cstdio>
#include <cstdint>
#include <fstream>
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

    std::remove(path.c_str());
    if (failures == 0) std::printf("All MoE SSD cache tests passed!\n");
    return failures == 0 ? 0 : 1;
}
