// Ranged-read transports and an expert cache served from memory instead of a file.
#include "storage/range_transport.h"
#include "storage/ssd_expert_cache/cache.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

std::vector<uint8_t> pattern(size_t bytes) {
    std::vector<uint8_t> data(bytes);
    for (size_t i = 0; i < bytes; ++i) data[i] = static_cast<uint8_t>((i * 131 + (i >> 8) * 17 + 7) & 255);
    return data;
}

// Every transport must return exactly the object's bytes for any range.
void check_reads(mollm::RangeTransport& transport, const std::vector<uint8_t>& expected, const char* what) {
    check(transport.size() == expected.size(), what);
    std::vector<uint8_t> out(expected.size(), 0);
    check(transport.read(0, out.data(), out.size()) && out == expected, what);
    for (const auto range : {std::pair<size_t, size_t>{0, 1}, {1, 4095}, {4096, 4096}, {5000, 12345},
                             {expected.size() - 1, 1}, {expected.size(), 0}}) {
        std::vector<uint8_t> part(range.second, 0xee);
        check(transport.read(range.first, part.data(), part.size()) &&
              !std::memcmp(part.data(), expected.data() + range.first, range.second), what);
    }
    // One contiguous range delivered into three buffers, one of them empty.
    std::vector<uint8_t> a(100), b, c(3000);
    check(transport.read_segments(777, {{a.data(), a.size()}, {b.data(), 0}, {c.data(), c.size()}}) &&
          !std::memcmp(a.data(), expected.data() + 777, a.size()) &&
          !std::memcmp(c.data(), expected.data() + 877, c.size()), what);
    check(transport.read_segments(0, {}), what);
    // Concurrent readers see the same bytes.
    std::vector<std::thread> readers;
    std::vector<int> ok(8, 0);
    for (int t = 0; t < 8; ++t)
        readers.emplace_back([&, t] {
            std::vector<uint8_t> local(2048);
            bool good = true;
            for (int i = 0; i < 64; ++i) {
                const size_t offset = (static_cast<size_t>(t) * 4099 + static_cast<size_t>(i) * 257) %
                    (expected.size() - local.size());
                good = good && transport.read(offset, local.data(), local.size()) &&
                    !std::memcmp(local.data(), expected.data() + offset, local.size());
            }
            ok[t] = good;
        });
    for (auto& reader : readers) reader.join();
    for (int good : ok) check(good != 0, what);
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

// The cache hands out the same expert payloads whichever transport serves the package.
bool borrow_all(MoeSsdCache& cache, const uint16_t* package) {
    auto* gate = cache.find_source("g");
    auto* down = cache.find_source("d");
    if (!gate || !down) return false;
    bool good = true;
    for (int expert = 0; expert < 3; ++expert) {
        ExpertLease lease;
        good = good && cache.borrow(gate, down, expert, lease) &&
            !std::memcmp(lease.gate_up.data, package + expert * 2, 2 * sizeof(uint16_t)) &&
            !std::memcmp(lease.down.data, package + 6 + expert * 2, 2 * sizeof(uint16_t));
    }
    return good;
}

}  // namespace

int main() {
    const std::vector<uint8_t> object = pattern(64 * 1024 + 123);
    const std::string path = "/tmp/mollm_test_range_transport_" + std::to_string(getpid()) + ".bin";
    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(object.data()), static_cast<std::streamsize>(object.size()));
    }

    auto file = mollm::FileRangeTransport::open(path);
    check(file != nullptr, "open file transport");
    if (file) {
        check(!std::strcmp(file->kind(), "file"), "file transport kind");
        check_reads(*file, object, "file transport reads");
        uint8_t byte = 0;
        check(!file->read(object.size(), &byte, 1), "file read past the end fails");
        check(!file->read_segments(object.size() - 1, {{&byte, 1}, {&byte, 1}}), "file segments past the end fail");
    }
    check(mollm::FileRangeTransport::open(path + ".missing") == nullptr, "missing file is rejected");
    if (file) check(file->direct_alignment() == 0, "default file transport has no direct path");

    // Direct reads need a filesystem that supports them, so use the working
    // directory and not /tmp, which is often tmpfs. The object's size is not
    // a multiple of the block size, which exercises the short last block.
    {
        const std::string local = "mollm_test_range_transport_direct_" + std::to_string(getpid()) + ".bin";
        const std::vector<uint8_t> large = pattern((size_t{3} << 20) + 1234);
        {
            std::ofstream out(local, std::ios::binary);
            out.write(reinterpret_cast<const char*>(large.data()), static_cast<std::streamsize>(large.size()));
        }
        auto direct = mollm::FileRangeTransport::open(local, mollm::FileReadMode::Direct);
        if (!direct || !direct->direct_alignment()) {
            std::printf("direct reads are not available here; skipping that part\n");
        } else {
            const size_t unit = direct->direct_alignment();
            check(!std::strcmp(direct->kind(), "file-direct"), "direct transport kind");
            // Unaligned callers still get correct bytes, through the page cache.
            check_reads(*direct, large, "direct transport serves unaligned reads");
            check(direct->direct_reads() == 0 && direct->cached_reads() > 0, "unaligned reads use the page cache");
            // Aligned offset, length and destination go to the device.
            void* raw = nullptr;
            check(posix_memalign(&raw, unit, 64 * unit) == 0, "aligned buffer");
            auto* aligned = static_cast<uint8_t*>(raw);
            check(direct->read(8 * unit, aligned, 16 * unit) && direct->direct_reads() == 1 &&
                  !std::memcmp(aligned, large.data() + 8 * unit, 16 * unit), "aligned read takes the direct path");
            // An aligned read may run past the end by less than one unit.
            const size_t last = large.size() / unit * unit;
            std::memset(aligned, 0xee, 2 * unit);
            check(direct->read(last, aligned, unit) && direct->direct_reads() == 2 &&
                  !std::memcmp(aligned, large.data() + last, large.size() - last),
                  "aligned read of the short last block");
            check(direct->read(last - unit, aligned, 2 * unit) &&
                  !std::memcmp(aligned, large.data() + last - unit, large.size() - last + unit),
                  "aligned read ending past the end");
            const uint64_t before = direct->direct_reads();
            check(direct->read(8 * unit, aligned + 1, unit) && direct->read(8 * unit + 1, aligned, unit) &&
                  direct->read(8 * unit, aligned, unit - 1) && direct->direct_reads() == before,
                  "any misalignment falls back to the page cache");
            std::free(raw);

            // An expert cache on a direct transport reads every payload from
            // the device into its final buffer, at offsets inside blocks.
            MoeSsdTensorSpec gate = spec("g", 5000);
            MoeSsdTensorSpec down = spec("d", 5000 + 3 * 70000);
            gate.cols = down.cols = 35000;
            gate.data_bytes = down.data_bytes = 35000 * sizeof(uint16_t);
            MoeSsdCache cache;
            check(cache.open(direct, 16 * 70000, 2, false, false) && cache.add_source(gate) &&
                  cache.add_source(down), "open a cache on a direct transport");
            const uint64_t cached_before = direct->cached_reads();
            const uint64_t direct_before = direct->direct_reads();
            bool good = true;
            for (int round = 0; round < 2; ++round)
                for (int expert = 0; expert < 3; ++expert) {
                    ExpertLease lease;
                    good = good && cache.borrow(cache.find_source("g"), cache.find_source("d"), expert, lease) &&
                        !std::memcmp(lease.gate_up.data, large.data() + 5000 + expert * 70000, 70000) &&
                        !std::memcmp(lease.down.data, large.data() + 5000 + (3 + expert) * 70000, 70000);
                }
            check(good, "direct cache hands out the package bytes");
            check(direct->direct_reads() == direct_before + 6 && direct->cached_reads() == cached_before,
                  "every cache read took the direct path exactly once");

            // NVFP4 keeps a tensor's scales right after its data. Each tensor
            // is then one direct read into one allocation, not two.
            MoeSsdTensorSpec fp4_gate, fp4_down;
            for (MoeSsdTensorSpec* tensor : {&fp4_gate, &fp4_down}) {
                tensor->layer = 1;
                tensor->num_experts = 2;
                tensor->rows = 16;
                tensor->cols = 64;
                tensor->precision = Precision::NVFP4;
                tensor->group_size = 16;
                tensor->groups_per_row = 4;
                tensor->data_bytes = 16 * 64 / 2;
                tensor->scales_bytes = 16 * 64 / 16 + 16 * 4;
                tensor->expert_stride = tensor->data_bytes + tensor->scales_bytes;
            }
            fp4_gate.weight_ref = "fg";
            fp4_down.weight_ref = "fd";
            fp4_gate.data_offset = 123457;
            fp4_gate.scales_offset = fp4_gate.data_offset + fp4_gate.data_bytes;
            fp4_down.data_offset = 777001;
            fp4_down.scales_offset = fp4_down.data_offset + fp4_down.data_bytes;
            const bool added = cache.add_source(fp4_gate) && cache.add_source(fp4_down);
            if (!added) {
                std::printf("this build rejects the NVFP4 fixture geometry; skipping the shared-read part\n");
            } else {
                const uint64_t before_fp4 = direct->direct_reads();
                bool fp4_good = true;
                for (int expert = 0; expert < 2; ++expert) {
                    ExpertLease lease;
                    const size_t stride = fp4_gate.expert_stride;
                    fp4_good = fp4_good &&
                        cache.borrow(cache.find_source("fg"), cache.find_source("fd"), expert, lease) &&
                        !std::memcmp(lease.gate_up.data, large.data() + 123457 + expert * stride, fp4_gate.data_bytes) &&
                        !std::memcmp(lease.gate_up.nvfp4_scales,
                                     large.data() + 123457 + expert * stride + fp4_gate.data_bytes, 16 * 4) &&
                        !std::memcmp(lease.down.data, large.data() + 777001 + expert * stride, fp4_down.data_bytes) &&
                        !std::memcmp(lease.down.nvfp4_scales,
                                     large.data() + 777001 + expert * stride + fp4_down.data_bytes, 16 * 4);
                }
                check(fp4_good, "direct cache hands out NVFP4 data and scales from one read");
                check(direct->direct_reads() == before_fp4 + 4 && direct->cached_reads() == cached_before,
                      "an NVFP4 expert costs two direct reads");
            }
        }
        unlink(local.c_str());
    }

    mollm::MemoryRangeTransport memory(object.data(), object.size());
    check(!std::strcmp(memory.kind(), "memory"), "memory transport kind");
    check_reads(memory, object, "memory transport reads");
    uint8_t byte = 0;
    check(!memory.read(object.size(), &byte, 1) && !memory.read(UINT64_MAX, &byte, 1) &&
          !memory.read(1, &byte, SIZE_MAX), "memory read past the end fails");
    check(memory.profile().planes == 1, "default profile has one plane");

    // The owner keeps the bytes alive for as long as the transport exists.
    auto owned = std::make_shared<std::vector<uint8_t>>(object);
    std::weak_ptr<std::vector<uint8_t>> watch = owned;
    {
        mollm::MemoryRangeTransport holder(owned->data(), owned->size(), owned);
        owned.reset();
        check(!watch.expired(), "memory transport holds its owner");
        std::vector<uint8_t> out(16);
        check(holder.read(100, out.data(), out.size()) && !std::memcmp(out.data(), object.data() + 100, 16),
              "memory transport reads through its owner");
    }
    check(watch.expired(), "memory transport releases its owner");

    // Expert cache: gate payloads at byte 0, down payloads at byte 12.
    uint16_t package[12];
    for (int i = 0; i < 12; ++i) package[i] = static_cast<uint16_t>(0x3c00 + i * 257);
    const std::string package_path = path + ".package";
    {
        std::ofstream out(package_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(package), sizeof(package));
    }
    {
        MoeSsdCache from_file;
        check(from_file.open(package_path, 24, 1, false, false) &&
              from_file.add_source(spec("g", 0)) && from_file.add_source(spec("d", 12)) &&
              borrow_all(from_file, package), "cache served from a file");
    }
    {
        MoeSsdCache from_memory;
        check(from_memory.open(std::make_shared<mollm::MemoryRangeTransport>(package, sizeof(package)),
                               24, 1, false, false) &&
              from_memory.add_source(spec("g", 0)) && from_memory.add_source(spec("d", 12)) &&
              borrow_all(from_memory, package), "cache served from memory");
        // A source that reaches past the transport's object is rejected.
        check(!from_memory.add_source(spec("past", 24)), "source beyond the transport is rejected");
    }
    {
        MoeSsdCache closed;
        check(!closed.open(std::shared_ptr<mollm::RangeTransport>(), 24, 1, false, false),
              "null transport is rejected");
        check(!closed.open(package_path + ".missing", 24, 1, false, false), "missing package is rejected");
        check(!closed.add_source(spec("g", 0)), "sources need an open cache");
    }

    unlink(path.c_str());
    unlink(package_path.c_str());
    if (failures) return 1;
    std::printf("range transports and transport-backed expert cache pass\n");
    return 0;
}
