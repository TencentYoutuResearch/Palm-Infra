#include "graph/mmap_file.h"
#include "kernels/tensor.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int INT4_BG128_BLOCK_BYTES = 544;

float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int exp_unbiased = -14;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp_unbiased--;
            }
            mant &= 0x03ffu;
            uint32_t exp32 = (uint32_t)(exp_unbiased + 127);
            bits = sign | (exp32 << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        uint32_t exp32 = exp + (127 - 15);
        bits = sign | (exp32 << 23) | (mant << 13);
    }

    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

uint16_t load_u16_le(const uint8_t* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

float load_value(const uint8_t* raw, const std::string& dtype, size_t idx) {
    if (dtype == "f16") {
        return half_to_float(load_u16_le(raw + idx * 2));
    }
    if (dtype == "f32") {
        float v;
        std::memcpy(&v, raw + idx * 4, sizeof(v));
        return v;
    }
    throw std::runtime_error("unsupported dtype: " + dtype);
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open input: " + path);
    }
    in.seekg(0, std::ios::end);
    std::streamoff size = in.tellg();
    if (size < 0) {
        throw std::runtime_error("failed to stat input: " + path);
    }
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> data((size_t)size);
    if (!data.empty()) {
        in.read((char*)data.data(), size);
        if (!in) {
            throw std::runtime_error("failed to read input: " + path);
        }
    }
    return data;
}

void write_weight_file(const std::string& path,
                       const std::vector<uint8_t>& data,
                       const std::vector<float>& scales,
                       int n, int k,
                       Precision precision,
                       uint32_t flags,
                       int group_size,
                       int num_groups) {
    MappedFile::Header hdr = {};
    hdr.magic = MappedFile::MAGIC;
    hdr.flags = flags;
    hdr.ndim = 2;
    hdr.precision = (uint32_t)precision;
    hdr.shape[0] = (uint64_t)n;
    hdr.shape[1] = (uint64_t)k;
    hdr.shape[2] = 1;
    hdr.shape[3] = 1;
    hdr.data_offset = sizeof(MappedFile::Header);
    hdr.data_size = data.size();
    hdr.scales_offset = sizeof(MappedFile::Header) + data.size();
    hdr.scales_size = scales.size() * sizeof(float);
    hdr.group_size = (uint32_t)group_size;
    hdr.num_groups = (uint32_t)num_groups;

    if (scales.empty()) {
        hdr.scales_offset = 0;
        hdr.scales_size = 0;
        hdr.group_size = 0;
        hdr.num_groups = 0;
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open output: " + path);
    }
    out.write((const char*)&hdr, sizeof(hdr));
    if (!data.empty()) {
        out.write((const char*)data.data(), (std::streamsize)data.size());
    }
    if (!scales.empty()) {
        out.write((const char*)scales.data(), (std::streamsize)(scales.size() * sizeof(float)));
    }
    if (!out) {
        throw std::runtime_error("failed to write output: " + path);
    }
}

int thread_count_from_env() {
    const char* env = std::getenv("MOLLM_QUANT_THREADS");
    if (env && *env) {
        int v = std::atoi(env);
        if (v > 0) return v;
    }
    unsigned hc = std::thread::hardware_concurrency();
    return hc > 0 ? (int)hc : 4;
}

template <typename Fn>
void parallel_rows(int n, int threads, Fn fn) {
    threads = std::max(1, std::min(threads, n));
    std::vector<std::thread> workers;
    workers.reserve((size_t)threads);
    for (int t = 0; t < threads; t++) {
        int begin = (int)((int64_t)n * t / threads);
        int end = (int)((int64_t)n * (t + 1) / threads);
        workers.emplace_back([=, &fn]() { fn(begin, end); });
    }
    for (auto& th : workers) th.join();
}

void quantize_w8(const uint8_t* raw, const std::string& dtype,
                 int n, int k, int group_size, int threads,
                 std::vector<uint8_t>& out, std::vector<float>& scales) {
    int groups_per_row = (k + group_size - 1) / group_size;
    out.resize((size_t)n * k);
    scales.resize((size_t)n * groups_per_row);

    parallel_rows(n, threads, [&](int r0, int r1) {
        for (int row = r0; row < r1; row++) {
            for (int g = 0; g < groups_per_row; g++) {
                int begin = g * group_size;
                int end = std::min(begin + group_size, k);
                float max_abs = 0.0f;
                for (int col = begin; col < end; col++) {
                    max_abs = std::max(max_abs, std::fabs(load_value(raw, dtype, (size_t)row * k + col)));
                }
                float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
                scales[(size_t)row * groups_per_row + g] = scale;
                for (int col = begin; col < end; col++) {
                    float x = load_value(raw, dtype, (size_t)row * k + col) / scale;
                    int q = (int)std::nearbyint(x);
                    q = std::max(-127, std::min(127, q));
                    out[(size_t)row * k + col] = (uint8_t)(int8_t)q;
                }
            }
        }
    });
}

void quantize_w4(const uint8_t* raw, const std::string& dtype,
                 int n, int k, int group_size, bool q4dot, bool bg128, int threads,
                 std::vector<uint8_t>& out, std::vector<float>& scales, uint32_t& flags) {
    int groups_per_row = (k + group_size - 1) / group_size;
    int row_stride = (k + 1) / 2;
    int k_blocks = k / 32;
    int n_padded = ((n + 7) / 8) * 8;

    flags = 0;
    if (q4dot) {
        if (k % 32 != 0) {
            throw std::runtime_error("q4dot W4 requires K multiple of 32");
        }
        out.assign((size_t)(n_padded / 8) * k_blocks * 8 * 16, 0);
        flags = MappedFile::FLAG_INT4_Q4DOT;
    } else if (bg128) {
        if (group_size != 128 || k % 128 != 0) {
            throw std::runtime_error("BG128 W4 requires group_size=128 and K multiple of 128");
        }
        out.assign((size_t)(n_padded / 8) * groups_per_row * INT4_BG128_BLOCK_BYTES, 0);
        flags = MappedFile::FLAG_INT4_BG128;
    } else {
        out.assign((size_t)n * row_stride, 0);
    }
    scales.resize((size_t)n * groups_per_row);

    parallel_rows(n, threads, [&](int r0, int r1) {
        for (int row = r0; row < r1; row++) {
            for (int g = 0; g < groups_per_row; g++) {
                int begin = g * group_size;
                int end = std::min(begin + group_size, k);
                float max_abs = 0.0f;
                for (int col = begin; col < end; col++) {
                    max_abs = std::max(max_abs, std::fabs(load_value(raw, dtype, (size_t)row * k + col)));
                }
                float scale = max_abs > 0.0f ? max_abs / 7.0f : 1.0f;
                scales[(size_t)row * groups_per_row + g] = scale;
                size_t bg128_block_base = 0;
                if (bg128) {
                    bg128_block_base =
                        ((size_t)(row / 8) * groups_per_row + (size_t)g) *
                        INT4_BG128_BLOCK_BYTES;
                    std::memcpy(out.data() + bg128_block_base + (size_t)(row % 8) * sizeof(float),
                                &scale, sizeof(scale));
                }

                for (int col = begin; col < end; col++) {
                    float x = load_value(raw, dtype, (size_t)row * k + col) / scale;
                    int q = (int)std::nearbyint(x);
                    q = std::max(-7, std::min(7, q));
                    uint8_t nibble = (uint8_t)(q & 0x0f);

                    if (q4dot) {
                        size_t idx = (((size_t)(row / 8) * k_blocks + (col / 32)) * 8 + (row % 8)) * 16
                                     + (size_t)((col % 32) / 2);
                        if (col & 1) out[idx] |= (uint8_t)(nibble << 4);
                        else out[idx] |= nibble;
                    } else if (bg128) {
                        int qgi = (col - begin) / 32;
                        size_t idx = bg128_block_base + 32 +
                                     ((size_t)qgi * 8 + (size_t)(row % 8)) * 16 +
                                     (size_t)((col % 32) / 2);
                        if (col & 1) out[idx] |= (uint8_t)(nibble << 4);
                        else out[idx] |= nibble;
                    } else {
                        size_t idx = (size_t)row * row_stride + (size_t)(col / 2);
                        if (col & 1) out[idx] |= (uint8_t)(nibble << 4);
                        else out[idx] |= nibble;
                    }
                }
            }
        }
    });
}

int parse_int(const char* s, const char* name) {
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (!s || *s == '\0' || *end != '\0' || v <= 0 || v > INT32_MAX) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + (s ? s : ""));
    }
    return (int)v;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 8 || argc > 9) {
        std::cerr
            << "Usage: " << argv[0]
            << " <input.raw> <output.weights> <f16|f32> <N> <K> <w8|w4> <group_size> [threads]\n";
        return 2;
    }

    try {
        std::string input = argv[1];
        std::string output = argv[2];
        std::string dtype = argv[3];
        int n = parse_int(argv[4], "N");
        int k = parse_int(argv[5], "K");
        std::string kind = argv[6];
        int group_size = parse_int(argv[7], "group_size");
        int threads = argc >= 9 ? parse_int(argv[8], "threads") : thread_count_from_env();

        if (dtype != "f16" && dtype != "f32") {
            throw std::runtime_error("dtype must be f16 or f32");
        }
        if (kind != "w8" && kind != "w4") {
            throw std::runtime_error("kind must be w8 or w4");
        }
        size_t elem_size = dtype == "f16" ? 2 : 4;
        std::vector<uint8_t> raw = read_file(input);
        size_t expected = (size_t)n * k * elem_size;
        if (raw.size() != expected) {
            throw std::runtime_error("input size mismatch: expected " + std::to_string(expected) +
                                     " bytes, got " + std::to_string(raw.size()));
        }

        std::vector<uint8_t> qdata;
        std::vector<float> scales;
        uint32_t flags = 0;
        if (kind == "w8") {
            quantize_w8(raw.data(), dtype, n, k, group_size, threads, qdata, scales);
            write_weight_file(output, qdata, scales, n, k, Precision::INT8,
                              0, group_size, (int)scales.size());
        } else {
            bool bg128 = (group_size == 128) && (k % 128 == 0);
            bool q4dot = !bg128 && (k % 32 == 0) && (group_size % 32 == 0);
            quantize_w4(raw.data(), dtype, n, k, group_size, q4dot, bg128,
                        threads, qdata, scales, flags);
            write_weight_file(output, qdata, scales, n, k, Precision::INT4,
                              flags, group_size, (int)scales.size());
        }
    } catch (const std::exception& e) {
        std::cerr << "mollm-quantize: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
