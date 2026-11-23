// Ranged-read transports and an expert cache served from memory instead of a file.
#include "storage/range_transport.h"
#include "storage/ssd_expert_cache/cache.h"

#include <cstdint>
#include <cstdio>
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
