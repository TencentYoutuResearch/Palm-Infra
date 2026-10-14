#pragma once
#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace mollm::detail {
// Small streaming SHA-256 for companion integrity, without a crypto dependency.
inline std::string sha256_file(const std::string& path) {
    constexpr uint32_t k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,
                     0x9b05688c,0x1f83d9ab,0x5be0cd19};
    auto r = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    auto block = [&](const uint8_t* data) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = uint32_t(data[4*i]) << 24 | uint32_t(data[4*i+1]) << 16 |
                   uint32_t(data[4*i+2]) << 8 | data[4*i+3];
        for (int i = 16; i < 64; ++i)
            w[i] = w[i-16] + (r(w[i-15],7) ^ r(w[i-15],18) ^ (w[i-15] >> 3)) +
                   w[i-7] + (r(w[i-2],17) ^ r(w[i-2],19) ^ (w[i-2] >> 10));
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],v=h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = v + (r(e,6)^r(e,11)^r(e,25)) + ((e&f)^(~e&g)) + k[i] + w[i];
            const uint32_t t2 = (r(a,2)^r(a,13)^r(a,22)) + ((a&b)^(a&c)^(b&c));
            v=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=v;
    };
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot hash companion file");
    std::array<uint8_t, 65536> buffer{};
    std::array<uint8_t, 128> tail{};
    uint64_t total = 0; size_t remaining = 0;
    while (input.read(reinterpret_cast<char*>(buffer.data()), buffer.size()) || input.gcount()) {
        const size_t count = static_cast<size_t>(input.gcount()); total += count;
        const size_t full = count / 64 * 64;
        for (size_t i = 0; i < full; i += 64) block(buffer.data() + i);
        remaining = count - full;
        for (size_t i = 0; i < remaining; ++i) tail[i] = buffer[full+i];
    }
    if (input.bad()) throw std::runtime_error("companion hash read failed");
    tail[remaining] = 0x80;
    const size_t final_bytes = remaining < 56 ? 64 : 128;
    for (int i = 0; i < 8; ++i) tail[final_bytes-1-i] = static_cast<uint8_t>((total*8) >> (8*i));
    block(tail.data()); if (final_bytes == 128) block(tail.data()+64);
    const char* hex = "0123456789abcdef";
    std::string digest; digest.reserve(64);
    for (uint32_t word : h) for (int shift = 28; shift >= 0; shift -= 4) digest.push_back(hex[(word >> shift)&15]);
    return digest;
}
}
