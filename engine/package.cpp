#include "engine/engine.h"
#include "kernels/moe_ssd.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <json.hpp>
#include <limits>
#include <new>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

using json = nlohmann::json;

static constexpr uint32_t PACKAGE_MAGIC = 0x4D4C4F4D; // "MOLM"
static constexpr uint32_t PACKAGE_VERSION = 1;
static constexpr uint64_t PACKAGE_HEADER_SIZE = 128;
static constexpr uint64_t MAX_METADATA_SIZE = 256ull * 1024 * 1024;
static constexpr uint64_t MAX_TOKENIZER_SIZE = 2ull * 1024 * 1024 * 1024;
static constexpr uint64_t MAX_TEMPLATE_SIZE = 64ull * 1024 * 1024;
static constexpr uint64_t MAX_GRAPH_SIZE = 1ull * 1024 * 1024 * 1024;

namespace {

class ScopedFd {
public:
    explicit ScopedFd(int fd = -1) : fd_(fd) {}
    ~ScopedFd() {
        if (fd_ >= 0)
            close(fd_);
    }

    ScopedFd(const ScopedFd&) = delete;
    ScopedFd& operator=(const ScopedFd&) = delete;

    int get() const { return fd_; }

    bool close_now() {
        if (fd_ < 0)
            return true;
        const int fd = fd_;
        fd_ = -1;
        return close(fd) == 0;
    }

private:
    int fd_;
};

class ScopedMapping {
public:
    ScopedMapping(void* address, size_t size)
        : address_(address), size_(size) {}
    ~ScopedMapping() {
        if (address_)
            munmap(address_, size_);
    }

    ScopedMapping(const ScopedMapping&) = delete;
    ScopedMapping& operator=(const ScopedMapping&) = delete;

    const uint8_t* bytes() const {
        return static_cast<const uint8_t*>(address_);
    }

    void* release() {
        void* address = address_;
        address_ = nullptr;
        size_ = 0;
        return address;
    }

private:
    void* address_ = nullptr;
    size_t size_ = 0;
};

struct PackageHeaderInfo {
    uint64_t meta_off = 0, meta_len = 0;
    uint64_t tok_off = 0, tok_len = 0;
    uint64_t jin_off = 0, jin_len = 0;
    uint64_t pf_off = 0, pf_len = 0;
    uint64_t dc_off = 0, dc_len = 0;
    uint64_t w_off = 0, w_len = 0;
};

bool read_exact_at(int fd, uint64_t offset, void* dst, size_t len,
                   const char* label) {
    uint8_t* out = static_cast<uint8_t*>(dst);
    size_t done = 0;
    while (done < len) {
        size_t chunk = std::min<size_t>(len - done, 64 * 1024 * 1024);
        ssize_t n =
            pread(fd, out + done, chunk, static_cast<off_t>(offset + done));
        if (n < 0) {
            if (errno == EINTR)
                continue;
            fprintf(stderr, "Engine: failed to read %s: %s\n", label,
                    strerror(errno));
            return false;
        }
        if (n == 0) {
            fprintf(stderr, "Engine: short read while reading %s\n", label);
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

bool write_exact(int fd, const void* src, size_t len, const char* label) {
    const uint8_t* data = static_cast<const uint8_t*>(src);
    size_t done = 0;
    while (done < len) {
        ssize_t written = write(fd, data + done, len - done);
        if (written < 0) {
            if (errno == EINTR)
                continue;
            fprintf(stderr, "Engine: failed to write %s: %s\n", label,
                    strerror(errno));
            return false;
        }
        if (written == 0) {
            fprintf(stderr, "Engine: short write while writing %s\n", label);
            return false;
        }
        done += static_cast<size_t>(written);
    }
    return true;
}

bool extract_temp_section(int source_fd, const uint8_t* mapped,
                          uint64_t offset, uint64_t length, const char* label,
                          std::string& out_path,
                          std::vector<std::string>& temp_files) {
    if (length == 0)
        return true;

    char path[] = "/tmp/mollm_pkg_XXXXXX";
    ScopedFd output(mkstemp(path));
    if (output.get() < 0) {
        fprintf(stderr, "Engine: failed to create temporary %s: %s\n", label,
                strerror(errno));
        return false;
    }

    std::vector<uint8_t> buffer(1024 * 1024);
    uint64_t done = 0;
    bool ok = true;
    while (ok && done < length) {
        const size_t chunk = static_cast<size_t>(
            std::min<uint64_t>(buffer.size(), length - done));
        const void* source = nullptr;
        if (mapped) {
            source = mapped + offset + done;
        } else {
            ok = read_exact_at(source_fd, offset + done, buffer.data(), chunk,
                               label);
            source = buffer.data();
        }
        if (ok)
            ok = write_exact(output.get(), source, chunk, label);
        done += chunk;
    }
    if (!output.close_now())
        ok = false;
    if (!ok) {
        std::remove(path);
        return false;
    }

    out_path = path;
    temp_files.push_back(out_path);
    return true;
}

bool section_in_file(uint64_t off, uint64_t len, size_t file_size,
                     const char* label) {
    uint64_t size = static_cast<uint64_t>(file_size);
    if (off > size || len > size - off) {
        fprintf(stderr, "Engine: package %s section extends beyond file\n",
                label);
        return false;
    }
    if (len > static_cast<uint64_t>(SIZE_MAX)) {
        fprintf(stderr,
                "Engine: package %s section is too large for this platform\n",
                label);
        return false;
    }
    return true;
}

bool checked_multiply(uint64_t a, uint64_t b, uint64_t& result) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a)
        return false;
    result = a * b;
    return true;
}

bool parse_package_header(const uint8_t* header, size_t file_size,
                          PackageHeaderInfo& out) {
    if (file_size < PACKAGE_HEADER_SIZE) {
        fprintf(stderr, "Engine: package too small\n");
        return false;
    }

    uint32_t magic = 0, version = 0;
    std::memcpy(&magic, header + 0, 4);
    std::memcpy(&version, header + 4, 4);
    if (magic != PACKAGE_MAGIC) {
        fprintf(stderr, "Engine: bad package magic 0x%08x\n", magic);
        return false;
    }
    if (version != PACKAGE_VERSION) {
        fprintf(stderr,
                "Engine: unsupported package version %u (expected %u)\n",
                version, PACKAGE_VERSION);
        return false;
    }

    std::memcpy(&out.meta_off, header + 8, 8);
    std::memcpy(&out.meta_len, header + 16, 8);
    std::memcpy(&out.tok_off, header + 24, 8);
    std::memcpy(&out.tok_len, header + 32, 8);
    std::memcpy(&out.jin_off, header + 40, 8);
    std::memcpy(&out.jin_len, header + 48, 8);
    std::memcpy(&out.pf_off, header + 56, 8);
    std::memcpy(&out.pf_len, header + 64, 8);
    std::memcpy(&out.dc_off, header + 72, 8);
    std::memcpy(&out.dc_len, header + 80, 8);
    std::memcpy(&out.w_off, header + 88, 8);
    std::memcpy(&out.w_len, header + 96, 8);

    if (!section_in_file(out.meta_off, out.meta_len, file_size, "metadata") ||
        !section_in_file(out.tok_off, out.tok_len, file_size, "tokenizer") ||
        !section_in_file(out.jin_off, out.jin_len, file_size,
                         "chat template") ||
        !section_in_file(out.pf_off, out.pf_len, file_size,
                         "prefill graph") ||
        !section_in_file(out.dc_off, out.dc_len, file_size, "decode graph") ||
        !section_in_file(out.w_off, out.w_len, file_size, "weights")) {
        return false;
    }

    if (out.meta_len == 0 || out.meta_len > MAX_METADATA_SIZE ||
        out.tok_len > MAX_TOKENIZER_SIZE ||
        out.jin_len > MAX_TEMPLATE_SIZE ||
        out.pf_len == 0 || out.pf_len > MAX_GRAPH_SIZE ||
        out.dc_len == 0 || out.dc_len > MAX_GRAPH_SIZE) {
        fprintf(stderr, "Engine: package contains an empty or oversized section\n");
        return false;
    }

    struct Section {
        uint64_t offset;
        uint64_t length;
        const char* label;
    };
    std::vector<Section> sections = {
        {out.meta_off, out.meta_len, "metadata"},
        {out.tok_off, out.tok_len, "tokenizer"},
        {out.jin_off, out.jin_len, "chat template"},
        {out.pf_off, out.pf_len, "prefill graph"},
        {out.dc_off, out.dc_len, "decode graph"},
        {out.w_off, out.w_len, "weights"},
    };
    sections.erase(
        std::remove_if(sections.begin(), sections.end(),
                       [](const Section& section) {
                           return section.length == 0;
                       }),
        sections.end());
    std::sort(sections.begin(), sections.end(),
              [](const Section& a, const Section& b) {
                  return a.offset < b.offset;
              });
    uint64_t previous_end = PACKAGE_HEADER_SIZE;
    for (const Section& section : sections) {
        if (section.offset < previous_end) {
            fprintf(stderr, "Engine: package %s section overlaps header or "
                            "another section\n",
                    section.label);
            return false;
        }
        previous_end = section.offset + section.length;
    }
    return true;
}

} // namespace

bool LLMEngine::load_package(const std::string& path, std::string& pf_path,
                             std::string& dc_path, std::string& tok_path) {
    ScopedFd package_file(open(path.c_str(), O_RDONLY));
    if (package_file.get() < 0) {
        fprintf(stderr, "Engine: failed to open package %s\n", path.c_str());
        return false;
    }
    struct stat st;
    if (fstat(package_file.get(), &st) != 0) {
        return false;
    }
    if (st.st_size < 0 ||
        static_cast<uint64_t>(st.st_size) >
            static_cast<uint64_t>(SIZE_MAX)) {
        fprintf(stderr, "Engine: package size is unsupported\n");
        return false;
    }
    size_t file_size = st.st_size;

    uint8_t header[128];
    if (!read_exact_at(package_file.get(), 0, header, sizeof(header),
                       "package header")) {
        return false;
    }

    PackageHeaderInfo ph;
    if (!parse_package_header(header, file_size, ph)) {
        return false;
    }

    int prefill_seq_len = 256;
    auto parse_metadata = [&](const std::string& meta_str) -> bool {
        try {
            auto meta = json::parse(meta_str);
            if (meta.contains("prefill_seq_len")) {
                prefill_seq_len = meta["prefill_seq_len"].get<int>();
                if (prefill_seq_len <= 0) {
                    fprintf(stderr,
                            "Engine: package prefill_seq_len must be positive\n");
                    return false;
                }
            }
            if (meta.contains("weights")) {
                for (auto& [name, info] : meta["weights"].items()) {
                    if (!info.is_array() || info.size() < 2) {
                        fprintf(stderr,
                                "Engine: bad package metadata for weight %s\n",
                                name.c_str());
                        return false;
                    }
                    uint64_t off = info[0].get<uint64_t>();
                    uint64_t sz = info[1].get<uint64_t>();
                    if (off > ph.w_len || sz > ph.w_len - off) {
                        fprintf(stderr,
                                "Engine: package weight %s extends beyond "
                                "weights section\n",
                                name.c_str());
                        return false;
                    }
                    package_weight_map_[name] = {off, sz};
                }
            }
            if (cfg_.moe_ssd_cache_bytes != 0) {
                auto storage_it = meta.find("moe_expert_storage");
                if (storage_it == meta.end() || !storage_it->is_object() ||
                    !storage_it->contains("layers") ||
                    !(*storage_it)["layers"].is_array()) {
                    fprintf(stderr, "Engine: --ssd-cache-mb requires package "
                                    "MoE expert storage metadata\n");
                    return false;
                }
                auto cache = std::make_unique<MoeSsdCache>();
                if (!cache->open(path, cfg_.moe_ssd_cache_bytes,
                                 cfg_.moe_ssd_io_workers,
                                 cfg_.moe_ssd_global_cache &&
                                     cfg_.moe_ssd_cross_layer_prefetch,
                                 cfg_.lock_moe_ssd_cache))
                    return false;
                if (cfg_.lock_moe_ssd_cache) {
                    std::fprintf(stderr,
                                 "Engine: expert RAM cache pages will be locked on demand\n");
                }
                size_t source_count = 0;
                for (const auto& layer : (*storage_it)["layers"]) {
                    if (!layer.is_object()) {
                        fprintf(stderr,
                                "Engine: bad MoE expert layer metadata\n");
                        return false;
                    }
                    int layer_index = layer.value("layer", -1);
                    int num_experts = layer.value(
                        "num_experts", storage_it->value("num_experts", 0));
                    for (const char* name : {"gate_up", "down"}) {
                        if (!layer.contains(name) || !layer[name].is_object()) {
                            fprintf(stderr,
                                    "Engine: missing MoE %s storage metadata\n",
                                    name);
                            return false;
                        }
                        const auto& item = layer[name];
                        MoeSsdTensorSpec spec;
                        spec.weight_ref = item.value("weight", std::string());
                        spec.layer = layer_index;
                        spec.num_experts = num_experts;
                        spec.rows = item.value("rows_per_expert", 0);
                        spec.cols = item.value("cols", 0);
                        spec.precision = static_cast<Precision>(
                            item.value("precision", 99u));
                        spec.flags = item.value("flags", 0u);
                        spec.group_size = item.value("group_size", 0u);
                        spec.groups_per_row = item.value("groups_per_row", 0u);
                        uint64_t weight_offset =
                            item.value("weight_offset", uint64_t{0});
                        uint64_t data_offset =
                            item.value("data_offset", uint64_t{0});
                        uint64_t data_bytes =
                            item.value("expert_data_bytes", uint64_t{0});
                        uint64_t scales_offset =
                            item.value("scales_offset", uint64_t{0});
                        uint64_t scales_bytes =
                            item.value("expert_scales_bytes", uint64_t{0});
                        uint64_t all_data = 0;
                        uint64_t all_scales = 0;
                        if (layer_index < 0 || num_experts <= 0 ||
                            spec.rows <= 0 || spec.cols <= 0 ||
                            static_cast<uint32_t>(spec.precision) >
                                static_cast<uint32_t>(Precision::INT4) ||
                            !checked_multiply(
                                data_bytes,
                                static_cast<uint64_t>(num_experts),
                                all_data) ||
                            !checked_multiply(
                                scales_bytes,
                                static_cast<uint64_t>(num_experts),
                                all_scales) ||
                            weight_offset > ph.w_len ||
                            data_offset > ph.w_len - weight_offset ||
                            all_data > ph.w_len - weight_offset - data_offset ||
                            (scales_bytes &&
                             (scales_offset > ph.w_len - weight_offset ||
                              all_scales >
                                  ph.w_len - weight_offset - scales_offset))) {
                            fprintf(stderr,
                                    "Engine: MoE expert range out of package "
                                    "bounds for %s\n",
                                    spec.weight_ref.c_str());
                            return false;
                        }
                        moe_ssd_expert_ranges_.push_back(
                            {weight_offset + data_offset, all_data});
                        if (all_scales != 0) {
                            moe_ssd_expert_ranges_.push_back(
                                {weight_offset + scales_offset, all_scales});
                        }
                        spec.data_offset =
                            ph.w_off + weight_offset + data_offset;
                        spec.data_bytes = data_bytes;
                        spec.scales_offset =
                            scales_bytes
                                ? ph.w_off + weight_offset + scales_offset
                                : 0;
                        spec.scales_bytes = scales_bytes;
                        if (!cache->add_source(spec))
                            return false;
                        ++source_count;
                    }
                }
                if (source_count == 0) {
                    fprintf(
                        stderr,
                        "Engine: package has no MoE expert storage entries\n");
                    return false;
                }
                if (!cache->set_global_capacity_pool(cfg_.moe_ssd_global_cache))
                    return false;
                if (!cache->configure_shallow_favoring(
                        cfg_.moe_ssd_shallow_cache_layers)) {
                    return false;
                }
                if (cfg_.moe_ssd_global_cache) {
                    std::fprintf(stderr, "Engine: SSD cache uses a shared "
                                         "global capacity pool\n");
                }
                if (cfg_.moe_ssd_shallow_cache_layers > 0) {
                    std::fprintf(
                        stderr,
                        "Engine: SSD cache favors the first %d MoE layers\n",
                        cfg_.moe_ssd_shallow_cache_layers);
                }
                moe_ssd_cache_ = std::move(cache);
            }
            // Retain all top-level scalar fields for CLI banner / display.
            // Skip "weights" (object) and any non-scalar. ints/bools are
            // stringified.
            for (auto it = meta.begin(); it != meta.end(); ++it) {
                const std::string& key = it.key();
                if (key == "weights")
                    continue;
                if (it->is_string()) {
                    package_metadata_[key] = it->get<std::string>();
                } else if (it->is_number_integer()) {
                    package_metadata_[key] = std::to_string(it->get<int64_t>());
                } else if (it->is_number_unsigned()) {
                    package_metadata_[key] =
                        std::to_string(it->get<uint64_t>());
                } else if (it->is_boolean()) {
                    package_metadata_[key] = it->get<bool>() ? "true" : "false";
                }
                // arrays / objects / layer_types list are skipped (not needed
                // for banner)
            }
        } catch (std::exception& e) {
            fprintf(stderr, "Engine: failed to parse package metadata: %s\n",
                    e.what());
            return false;
        }
        return true;
    };

    if (cfg_.weight_loading == WeightLoadingMode::RESIDENT) {
        std::string meta_str(static_cast<size_t>(ph.meta_len), '\0');
        if (!read_exact_at(package_file.get(), ph.meta_off, meta_str.data(),
                           meta_str.size(), "package metadata")) {
            return false;
        }
        if (!parse_metadata(meta_str)) {
            return false;
        }

        try {
            package_weights_storage_.resize(static_cast<size_t>(ph.w_len));
        } catch (const std::bad_alloc&) {
            fprintf(stderr,
                    "Engine: failed to allocate %.1f MB for resident package "
                    "weights; try --mmap\n",
                    ph.w_len / 1e6);
            return false;
        }
        if (ph.w_len > 0 &&
            !read_exact_at(package_file.get(), ph.w_off,
                           package_weights_storage_.data(),
                           package_weights_storage_.size(), "package weights")) {
            return false;
        }
        package_weights_base_ = package_weights_storage_.empty()
                                    ? nullptr
                                    : package_weights_storage_.data();
        package_weights_size_ = package_weights_storage_.size();
        package_weights_resident_ = true;

        bool ok =
            extract_temp_section(package_file.get(), nullptr, ph.pf_off,
                                 ph.pf_len, "prefill graph", pf_path,
                                 temp_files_) &&
            extract_temp_section(package_file.get(), nullptr, ph.dc_off,
                                 ph.dc_len, "decode graph", dc_path,
                                 temp_files_) &&
            extract_temp_section(package_file.get(), nullptr, ph.tok_off,
                                 ph.tok_len, "tokenizer", tok_path,
                                 temp_files_);
        if (!ok)
            return false;

        fprintf(stderr,
                "Engine: loaded package %s (%.1f MB, %zu weights, "
                "prefill_seq=%d, weights=resident)\n",
                path.c_str(), file_size / 1e6, package_weight_map_.size(),
                prefill_seq_len);
        return true;
    }

    void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE,
                        package_file.get(), 0);
    package_file.close_now();
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "Engine: mmap failed for %s\n", path.c_str());
        return false;
    }
    ScopedMapping mapping(mapped, file_size);
    const uint8_t* base = mapping.bytes();

    std::string meta_str(reinterpret_cast<const char*>(base + ph.meta_off),
                         static_cast<size_t>(ph.meta_len));
    if (!parse_metadata(meta_str)) {
        return false;
    }

    if (!extract_temp_section(-1, base, ph.pf_off, ph.pf_len,
                              "prefill graph", pf_path, temp_files_) ||
        !extract_temp_section(-1, base, ph.dc_off, ph.dc_len, "decode graph",
                              dc_path, temp_files_) ||
        !extract_temp_section(-1, base, ph.tok_off, ph.tok_len, "tokenizer",
                              tok_path, temp_files_)) {
        return false;
    }

    package_mmap_ = mapping.release();
    package_mmap_size_ = file_size;
    package_weights_base_ = base + ph.w_off;
    package_weights_size_ = static_cast<size_t>(ph.w_len);
    package_weights_resident_ = false;

    fprintf(stderr,
            "Engine: loaded package %s (%.1f MB, %zu weights, prefill_seq=%d, "
            "weights=mmap)\n",
            path.c_str(), file_size / 1e6, package_weight_map_.size(),
            prefill_seq_len);
    return true;
}
