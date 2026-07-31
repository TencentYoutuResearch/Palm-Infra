#include "graph/mmap_file.h"
#include "kernels/matmul.h"
#include "kernels/moe_ssd.h"

#include <cstdio>
#include <cmath>
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

    // Direct BG128 blocks carry their own scales. The SSD cache should accept
    // a source without a duplicate scale sidecar and expose a tensor that
    // dispatches through the embedded-scale kernels.
    const std::string bg128_path = "/tmp/mollm_test_moe_ssd_bg128.bin";
    {
        constexpr size_t block_bytes = 544;
        std::vector<uint8_t> bg128_contents(2 * block_bytes);
        {
            std::ofstream out(bg128_path, std::ios::binary);
            out.write(reinterpret_cast<const char*>(bg128_contents.data()),
                      static_cast<std::streamsize>(bg128_contents.size()));
        }
        auto bg128_spec = [&](const char* name, uint64_t offset) {
            MoeSsdTensorSpec out;
            out.weight_ref = name;
            out.layer = 0;
            out.num_experts = 1;
            out.rows = 8;
            out.cols = 128;
            out.precision = Precision::INT4;
            out.flags = MappedFile::FLAG_INT4_BG128;
            out.group_size = 128;
            out.groups_per_row = 1;
            out.data_offset = offset;
            out.data_bytes = block_bytes;
            return out;
        };

        MoeSsdCache cache;
        check(cache.open(bg128_path, 2 * block_bytes),
              "open embedded-scale BG128 cache");
        check(cache.add_source(bg128_spec("bg128_gate", 0)) &&
                  cache.add_source(bg128_spec("bg128_down", block_bytes)),
              "add BG128 sources without scale sidecars");
        const MoeSsdTensorSource* gate = cache.find_source("bg128_gate");
        const MoeSsdTensorSource* down = cache.find_source("bg128_down");
        Tensor gu, dw;
        check(cache.acquire(gate, down, 0, gu, dw),
              "load embedded-scale BG128 expert");
        check(gu.scales == nullptr && dw.scales == nullptr &&
                  gu.is_q4_g128_packed && dw.is_q4_g128_packed,
              "BG128 expert tensors omit redundant sidecars");
        check(cache.stats().bytes_read == 2 * block_bytes,
              "BG128 cache reads only embedded-scale data");
    }

    // Native MXFP4 experts keep one raw E8M0 byte per 32-value block.
    {
        const std::string mxfp4_path =
            "/tmp/mollm_test_moe_ssd_mxfp4.bin";
        std::vector<uint8_t> bytes(34, 0x22); // E2M1 value 1 in both nibbles
        bytes[32] = 127; // gate scale = 1
        bytes[33] = 127; // down scale = 1
        {
            std::ofstream out(mxfp4_path, std::ios::binary);
            out.write(reinterpret_cast<const char*>(bytes.data()),
                      static_cast<std::streamsize>(bytes.size()));
        }
        auto mx_spec = [](const char* name, uint64_t data_offset,
                          uint64_t scale_offset) {
            MoeSsdTensorSpec out;
            out.weight_ref = name;
            out.layer = 0;
            out.num_experts = 1;
            out.rows = 1;
            out.cols = 32;
            out.precision = Precision::MXFP4;
            out.group_size = 32;
            out.groups_per_row = 1;
            out.data_offset = data_offset;
            out.data_bytes = 16;
            out.scales_offset = scale_offset;
            out.scales_bytes = 1;
            return out;
        };
        MoeSsdCache cache;
        check(cache.open(mxfp4_path, bytes.size()),
              "open MXFP4 cache");
        check(cache.add_source(mx_spec("mx_gate", 0, 32)) &&
                  cache.add_source(mx_spec("mx_down", 16, 33)),
              "add MXFP4 sources");
        Tensor gate, down;
        check(cache.acquire(cache.find_source("mx_gate"),
                            cache.find_source("mx_down"), 0, gate, down),
              "load MXFP4 expert");
        check(gate.prec == Precision::MXFP4 && gate.e8m0_scales &&
                  gate.e8m0_scales[0] == 127 && gate.group_size == 32,
              "MXFP4 expert exposes raw E8M0 scale");
        float activation[32];
        for (float& value : activation) value = 1.0f;
        float output = 0.0f;
        Tensor a = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                  32, 1, 1, 1, activation);
        Tensor c = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                  1, 1, 1, 1, &output);
        kernel_matmul_fp32(a, gate, c);
        check(std::abs(output - 32.0f) < 0.25f,
              "paged MXFP4 expert dispatches through matmul");
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
        check(!cache.is_ready(gate, down, 1),
              "unrequested expert is not ready");
        check(cache.acquire(gate, down, 1, gu, dw), "load expert one");
        check(cache.is_ready(gate, down, 1),
              "acquired expert is ready");
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

    // Hash-routed layers know the next route exactly from the token id. Verify
    // that the cross-layer helper schedules only that lookup-table route and
    // that ordinary demand feedback records both ranks as useful.
    {
        MoeSsdCache cache;
        check(cache.open(path, 24, 2, true),
              "open exact hash-prefetch cache");
        check(cache.add_source(spec("hash_gate", 0)) &&
                  cache.add_source(
                      spec("hash_down", 6 * sizeof(uint16_t))),
              "add exact hash-prefetch sources");
        const MoeSsdTensorSource* gate =
            cache.find_source("hash_gate");
        const MoeSsdTensorSource* down =
            cache.find_source("hash_down");
        int32_t token_id = 1;
        const int32_t lookup[] = {
            0, 2,
            1, 2,
        };
        Tensor tokens = Tensor::create(
            Precision::INT32, MemoryType::EXTERNAL,
            1, 1, 1, 1, &token_id);
        Tensor table = Tensor::create(
            Precision::INT32, MemoryType::EXTERNAL,
            2, 2, 1, 1, const_cast<int32_t*>(lookup));
        check(schedule_moe_hash_cross_layer_prefetch(
                  tokens, table, gate, down, 3, 2),
              "schedule exact hash route");
        for (int spin = 0;
             spin < 10000 &&
             (!cache.contains(gate, down, 1) ||
              !cache.contains(gate, down, 2));
             ++spin) {
            std::this_thread::yield();
        }
        check(cache.contains(gate, down, 1) &&
                  cache.contains(gate, down, 2) &&
                  !cache.contains(gate, down, 0),
              "hash prefetch contains only the selected experts");
        check(cache.request_many(gate, down, {1, 2}),
              "evaluate exact hash prediction");
        Tensor gu, dw;
        check(cache.acquire(gate, down, 1, gu, dw) &&
                  cache.acquire(gate, down, 2, gu, dw),
              "acquire exact hash-prefetched experts");
        const auto stats = cache.stats();
        check(stats.cross_layer_tasks == 1 &&
                  stats.cross_layer_experts == 2 &&
                  stats.cross_layer_used == 2,
              "exact hash prefetch is fully useful");
        check(stats.cross_layer_rank_hits.size() == 2 &&
                  stats.cross_layer_rank_hits[0] == 1 &&
                  stats.cross_layer_rank_hits[1] == 1,
              "exact hash route reports perfect rank accuracy");
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

    // A sequential global scan should preserve stale entries from nearer
    // future layers. Otherwise an early-layer miss evicts the very next
    // layer's cached route and a cache smaller than one full forward pass
    // collapses to zero hits.
    {
        MoeSsdCache cache;
        check(cache.open(path, 24, 2),
              "open forward-aware global cache");
        auto gate0 = spec("forward_gate0", 0);
        auto down0 = spec("forward_down0", 6 * sizeof(uint16_t));
        auto gate1 = spec("forward_gate1", 0);
        auto down1 = spec("forward_down1", 6 * sizeof(uint16_t));
        auto gate2 = spec("forward_gate2", 0);
        auto down2 = spec("forward_down2", 6 * sizeof(uint16_t));
        gate1.layer = down1.layer = 1;
        gate2.layer = down2.layer = 2;
        check(cache.add_source(gate0) && cache.add_source(down0) &&
                  cache.add_source(gate1) && cache.add_source(down1) &&
                  cache.add_source(gate2) && cache.add_source(down2),
              "add forward-aware cache sources");
        check(cache.set_global_capacity_pool(true),
              "enable forward-aware global pool");
        const MoeSsdTensorSource* g0 =
            cache.find_source("forward_gate0");
        const MoeSsdTensorSource* d0 =
            cache.find_source("forward_down0");
        const MoeSsdTensorSource* g1 =
            cache.find_source("forward_gate1");
        const MoeSsdTensorSource* d1 =
            cache.find_source("forward_down1");
        const MoeSsdTensorSource* g2 =
            cache.find_source("forward_gate2");
        const MoeSsdTensorSource* d2 =
            cache.find_source("forward_down2");
        Tensor gu, dw;
        check(cache.acquire(g1, d1, 0, gu, dw) &&
                  cache.acquire(g2, d2, 0, gu, dw) &&
                  cache.acquire(g2, d2, 1, gu, dw),
              "populate next-forward future routes");
        cache.begin_forward_pass();
        check(cache.request_many(g0, d0, {0}),
              "admit an early-layer miss");
        check(cache.acquire(g0, d0, 0, gu, dw),
              "acquire the early-layer miss");
        check(cache.contains(g1, d1, 0),
              "retain the nearest future-layer route");
        check(cache.request_many(g1, d1, {0}) &&
                  cache.acquire(g1, d1, 0, gu, dw),
              "reuse the retained future-layer route");
        check(cache.stats().hits == 1,
              "forward-aware eviction converts the next layer to a hit");
    }

    // Protect high-confidence routes across forward boundaries. This tiny
    // pool gives each layer room for only one retained route, so admitting a
    // changed early-layer route should sacrifice the farthest future layer,
    // not the immediately following layer.
    {
        MoeSsdCache cache;
        check(cache.open(path, 24, 2),
              "open retained-route global cache");
        auto gate0 = spec("retained_gate0", 0);
        auto down0 = spec("retained_down0", 6 * sizeof(uint16_t));
        auto gate1 = spec("retained_gate1", 0);
        auto down1 = spec("retained_down1", 6 * sizeof(uint16_t));
        auto gate2 = spec("retained_gate2", 0);
        auto down2 = spec("retained_down2", 6 * sizeof(uint16_t));
        gate1.layer = down1.layer = 1;
        gate2.layer = down2.layer = 2;
        check(cache.add_source(gate0) && cache.add_source(down0) &&
                  cache.add_source(gate1) && cache.add_source(down1) &&
                  cache.add_source(gate2) && cache.add_source(down2),
              "add retained-route cache sources");
        check(cache.set_global_capacity_pool(true),
              "enable retained-route global pool");
        const MoeSsdTensorSource* g0 =
            cache.find_source("retained_gate0");
        const MoeSsdTensorSource* d0 =
            cache.find_source("retained_down0");
        const MoeSsdTensorSource* g1 =
            cache.find_source("retained_gate1");
        const MoeSsdTensorSource* d1 =
            cache.find_source("retained_down1");
        const MoeSsdTensorSource* g2 =
            cache.find_source("retained_gate2");
        const MoeSsdTensorSource* d2 =
            cache.find_source("retained_down2");
        Tensor gu, dw;
        check(cache.retain_for_next_forward(g0, d0, {0, 1}) &&
                  cache.retain_for_next_forward(g1, d1, {0, 1}) &&
                  cache.retain_for_next_forward(g2, d2, {0, 1}),
              "record one retained route per layer");
        check(cache.acquire(g0, d0, 0, gu, dw) &&
                  cache.acquire(g1, d1, 0, gu, dw) &&
                  cache.acquire(g2, d2, 0, gu, dw),
              "populate retained routes");
        cache.begin_forward_pass();
        check(cache.retain_for_next_forward(g0, d0, {1, 0}),
              "replace the early layer retained route");
        check(cache.request_many(g0, d0, {1}) &&
                  cache.acquire(g0, d0, 1, gu, dw),
              "admit the changed retained route");
        check(cache.contains(g1, d1, 0),
              "retained policy preserves the nearest future layer");
        check(cache.request_many(g1, d1, {0}) &&
                  cache.acquire(g1, d1, 0, gu, dw),
              "reuse the retained next-layer route");
        check(cache.stats().hits == 1,
              "retained route produces a next-forward hit");
    }

    // Switch the global victim order according to cache pressure. When one
    // layer's fair share cannot hold its complete route, recycle a finished
    // layer before a stale future entry. Once the route fits, preserve the
    // current left-layer entry and evict stale residency instead.
    {
        auto run_pressure_case = [&](size_t capacity, bool expect_left_eviction,
                                     const char* label) {
            MoeSsdCache cache;
            check(cache.open(path, capacity, 2), label);
            auto gate0 = spec("pressure_gate0", 0);
            auto down0 = spec("pressure_down0", 6 * sizeof(uint16_t));
            auto gate1 = spec("pressure_gate1", 0);
            auto down1 = spec("pressure_down1", 6 * sizeof(uint16_t));
            auto gate2 = spec("pressure_gate2", 0);
            auto down2 = spec("pressure_down2", 6 * sizeof(uint16_t));
            gate1.layer = down1.layer = 1;
            gate2.layer = down2.layer = 2;
            check(cache.add_source(gate0) && cache.add_source(down0) &&
                      cache.add_source(gate1) && cache.add_source(down1) &&
                      cache.add_source(gate2) && cache.add_source(down2),
                  "add pressure-adaptive cache sources");
            check(cache.set_global_capacity_pool(true),
                  "enable pressure-adaptive global pool");
            const MoeSsdTensorSource* g0 =
                cache.find_source("pressure_gate0");
            const MoeSsdTensorSource* d0 =
                cache.find_source("pressure_down0");
            const MoeSsdTensorSource* g1 =
                cache.find_source("pressure_gate1");
            const MoeSsdTensorSource* d1 =
                cache.find_source("pressure_down1");
            const MoeSsdTensorSource* g2 =
                cache.find_source("pressure_gate2");
            const MoeSsdTensorSource* d2 =
                cache.find_source("pressure_down2");
            Tensor gu, dw;
            check(cache.acquire(g0, d0, 0, gu, dw) &&
                      cache.acquire(g1, d1, 0, gu, dw) &&
                      cache.acquire(g2, d2, 0, gu, dw),
                  "populate pressure-adaptive cache");
            if (capacity >= 48) {
                check(cache.acquire(g0, d0, 1, gu, dw) &&
                          cache.acquire(g1, d1, 1, gu, dw) &&
                          cache.acquire(g2, d2, 1, gu, dw),
                      "fill roomy pressure-adaptive cache");
            }
            cache.begin_forward_pass();
            check(cache.request_many(g0, d0, {0}) &&
                      cache.acquire(g0, d0, 0, gu, dw),
                  "mark current early-layer route");
            check(cache.request_many(g1, d1, {0, 2}) &&
                      cache.acquire(g1, d1, 0, gu, dw) &&
                      cache.acquire(g1, d1, 2, gu, dw),
                  "admit changed middle-layer route");
            check(cache.contains(g0, d0, 0) != expect_left_eviction,
                  expect_left_eviction
                      ? "tight cache recycles the finished layer"
                      : "roomy cache preserves the current finished layer");
            check(cache.contains(g2, d2, 0) == expect_left_eviction,
                  expect_left_eviction
                      ? "tight cache preserves stale future residency"
                      : "roomy cache recycles stale future residency");
        };
        run_pressure_case(24, true, "open tight pressure-adaptive cache");
        run_pressure_case(48, false, "open roomy pressure-adaptive cache");
    }

    // Protect all resident members of a route before allocating its misses.
    // Otherwise the leading miss can evict expert zero before request_many()
    // reaches the trailing hit.
    {
        MoeSsdCache cache;
        check(cache.open(path, 16, 2), "open route-protection cache");
        check(cache.add_source(spec("route_gate", 0)) &&
                  cache.add_source(
                      spec("route_down", 6 * sizeof(uint16_t))),
              "add route-protection sources");
        check(cache.set_global_capacity_pool(true),
              "enable route-protection global pool");
        const MoeSsdTensorSource* gate =
            cache.find_source("route_gate");
        const MoeSsdTensorSource* down =
            cache.find_source("route_down");
        Tensor gu, dw;
        check(cache.acquire(gate, down, 0, gu, dw) &&
                  cache.acquire(gate, down, 1, gu, dw),
              "populate route-protection cache");
        cache.begin_forward_pass();
        check(cache.request_many(gate, down, {2, 0}),
              "request leading miss and trailing hit");
        check(cache.contains(gate, down, 0),
              "route admission retains its trailing cached hit");
        check(!cache.contains(gate, down, 1),
              "route admission evicts an expert outside the route");
    }

    // A backend handoff may discard CPU-resident payloads while retaining the
    // registered package sources and I/O workers for later prefill calls.
    {
        MoeSsdCache cache;
        check(cache.open(path, 16, 1), "open clear-resident cache");
        check(cache.add_source(spec("clear_gate", 0)) &&
              cache.add_source(spec("clear_down", 6 * sizeof(uint16_t))),
              "add clear-resident sources");
        const MoeSsdTensorSource* gate = cache.find_source("clear_gate");
        const MoeSsdTensorSource* down = cache.find_source("clear_down");
        Tensor gu, dw;
        check(cache.acquire(gate, down, 0, gu, dw),
              "populate clear-resident cache");
        check(cache.stats().resident_bytes == 8,
              "clear-resident cache is populated");
        check(cache.clear_resident(), "clear completed expert payloads");
        check(cache.stats().resident_bytes == 0 &&
              !cache.contains(gate, down, 0),
              "clear removes residency but keeps source lookup valid");
        check(cache.acquire(gate, down, 0, gu, dw),
              "reload after clear-resident");
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
    std::remove(bg128_path.c_str());
    if (failures == 0) std::printf("All MoE SSD cache tests passed!\n");
    return failures == 0 ? 0 : 1;
}
