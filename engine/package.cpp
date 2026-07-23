#include "engine/engine.h"
#include "kernels/moe_ssd.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <json.hpp>
#include <new>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

using json = nlohmann::json;

static constexpr uint32_t PACKAGE_MAGIC = 0x4D4C4F4D; // "MOLM"

namespace {

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

bool parse_package_header(const uint8_t* header, size_t file_size,
                          PackageHeaderInfo& out) {
    if (file_size < 128) {
        fprintf(stderr, "Engine: package too small\n");
        return false;
    }

    uint32_t magic = 0, version = 0;
    std::memcpy(&magic, header + 0, 4);
    std::memcpy(&version, header + 4, 4);
    (void)version;
    if (magic != PACKAGE_MAGIC) {
        fprintf(stderr, "Engine: bad package magic 0x%08x\n", magic);
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

    return section_in_file(out.meta_off, out.meta_len, file_size, "metadata") &&
           section_in_file(out.tok_off, out.tok_len, file_size, "tokenizer") &&
           section_in_file(out.jin_off, out.jin_len, file_size,
                           "chat template") &&
           section_in_file(out.pf_off, out.pf_len, file_size,
                           "prefill graph") &&
           section_in_file(out.dc_off, out.dc_len, file_size, "decode graph") &&
           section_in_file(out.w_off, out.w_len, file_size, "weights");
}

} // namespace

bool LLMEngine::load_package(const std::string& path, std::string& pf_path,
                             std::string& dc_path, std::string& tok_path,
                             std::string& jin_path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Engine: failed to open package %s\n", path.c_str());
        return false;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return false;
    }
    size_t file_size = st.st_size;

    uint8_t header[128];
    if (!read_exact_at(fd, 0, header, sizeof(header), "package header")) {
        close(fd);
        return false;
    }

    PackageHeaderInfo ph;
    if (!parse_package_header(header, file_size, ph)) {
        close(fd);
        return false;
    }

    auto parse_metadata = [&](const std::string& meta_str) -> bool {
        try {
            auto meta = json::parse(meta_str);
            if (meta.contains("prefill_seq_len")) {
                package_prefill_seq_len_ = meta["prefill_seq_len"].get<int>();
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
                                     cfg_.moe_ssd_cross_layer_prefetch))
                    return false;
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
                        uint64_t all_data =
                            data_bytes * static_cast<uint64_t>(num_experts);
                        uint64_t all_scales =
                            scales_bytes * static_cast<uint64_t>(num_experts);
                        if (weight_offset > ph.w_len ||
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

    // Extract graphs + tokenizer + jinja to temp files
    pid_t pid = getpid();
    auto write_tmp_from_memory = [&](const uint8_t* base, const char* label,
                                     uint64_t off, uint64_t len,
                                     std::string& out_path) -> bool {
        if (len == 0)
            return true;
        out_path = "/tmp/mollm_pkg_" + std::to_string(pid) + "_" + label;
        std::ofstream f(out_path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(base + off),
                static_cast<std::streamsize>(len));
        if (!f) {
            fprintf(stderr,
                    "Engine: failed to write extracted package section %s\n",
                    label);
            return false;
        }
        temp_files_.push_back(out_path);
        return true;
    };
    auto write_tmp_from_fd = [&](const char* label, uint64_t off, uint64_t len,
                                 std::string& out_path) -> bool {
        if (len == 0)
            return true;
        out_path = "/tmp/mollm_pkg_" + std::to_string(pid) + "_" + label;
        std::ofstream f(out_path, std::ios::binary);
        if (!f) {
            fprintf(stderr,
                    "Engine: failed to create extracted package section %s\n",
                    label);
            return false;
        }
        std::vector<uint8_t> buf(1024 * 1024);
        uint64_t done = 0;
        while (done < len) {
            size_t chunk =
                static_cast<size_t>(std::min<uint64_t>(buf.size(), len - done));
            if (!read_exact_at(fd, off + done, buf.data(), chunk, label)) {
                return false;
            }
            f.write(reinterpret_cast<const char*>(buf.data()), chunk);
            if (!f) {
                fprintf(
                    stderr,
                    "Engine: failed to write extracted package section %s\n",
                    label);
                return false;
            }
            done += chunk;
        }
        temp_files_.push_back(out_path);
        return true;
    };

    if (cfg_.weight_loading == WeightLoadingMode::RESIDENT) {
        std::string meta_str(static_cast<size_t>(ph.meta_len), '\0');
        if (!read_exact_at(fd, ph.meta_off, meta_str.data(), meta_str.size(),
                           "package metadata")) {
            close(fd);
            return false;
        }
        if (!parse_metadata(meta_str)) {
            close(fd);
            return false;
        }

        try {
            package_weights_storage_.resize(static_cast<size_t>(ph.w_len));
        } catch (const std::bad_alloc&) {
            fprintf(stderr,
                    "Engine: failed to allocate %.1f MB for resident package "
                    "weights; try --mmap\n",
                    ph.w_len / 1e6);
            close(fd);
            return false;
        }
        if (ph.w_len > 0 &&
            !read_exact_at(fd, ph.w_off, package_weights_storage_.data(),
                           package_weights_storage_.size(),
                           "package weights")) {
            close(fd);
            return false;
        }
        package_weights_base_ = package_weights_storage_.empty()
                                    ? nullptr
                                    : package_weights_storage_.data();
        package_weights_size_ = package_weights_storage_.size();
        package_weights_resident_ = true;

        bool ok =
            write_tmp_from_fd("prefill.graph", ph.pf_off, ph.pf_len, pf_path) &&
            write_tmp_from_fd("decode.graph", ph.dc_off, ph.dc_len, dc_path) &&
            write_tmp_from_fd("tokenizer.json", ph.tok_off, ph.tok_len,
                              tok_path) &&
            write_tmp_from_fd("chat_template.jinja", ph.jin_off, ph.jin_len,
                              jin_path);
        close(fd);
        if (!ok)
            return false;

        fprintf(stderr,
                "Engine: loaded package %s (%.1f MB, %zu weights, "
                "prefill_seq=%d, weights=resident)\n",
                path.c_str(), file_size / 1e6, package_weight_map_.size(),
                package_prefill_seq_len_);
        return true;
    }

    void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "Engine: mmap failed for %s\n", path.c_str());
        return false;
    }
    package_mmap_ = mapped;
    package_mmap_size_ = file_size;

    const uint8_t* base = static_cast<const uint8_t*>(mapped);
    package_weights_base_ = base + ph.w_off;
    package_weights_size_ = static_cast<size_t>(ph.w_len);
    package_weights_resident_ = false;

    std::string meta_str(reinterpret_cast<const char*>(base + ph.meta_off),
                         static_cast<size_t>(ph.meta_len));
    if (!parse_metadata(meta_str)) {
        return false;
    }

    if (!write_tmp_from_memory(base, "prefill.graph", ph.pf_off, ph.pf_len,
                               pf_path) ||
        !write_tmp_from_memory(base, "decode.graph", ph.dc_off, ph.dc_len,
                               dc_path) ||
        !write_tmp_from_memory(base, "tokenizer.json", ph.tok_off, ph.tok_len,
                               tok_path) ||
        !write_tmp_from_memory(base, "chat_template.jinja", ph.jin_off,
                               ph.jin_len, jin_path)) {
        return false;
    }

    fprintf(stderr,
            "Engine: loaded package %s (%.1f MB, %zu weights, prefill_seq=%d, "
            "weights=mmap)\n",
            path.c_str(), file_size / 1e6, package_weight_map_.size(),
            package_prefill_seq_len_);
    return true;
}
