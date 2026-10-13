#include "storage/mapped_file.h"
#include "kernels/cpu/matmul/matmul.h"
#include "storage/ssd_expert_cache/cache.h"

#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <thread>
#include <string>
#include <vector>
#if defined(MOLLM_CUDA)
#include <cuda_runtime_api.h>
#endif

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

template <typename Predicate>
bool wait_until(Predicate predicate,
                std::chrono::milliseconds timeout =
                    std::chrono::milliseconds(1000)) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline)
            return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
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

    // Resident loading uses the same expert layouts but sizes itself for the
    // complete model. Every later route must be a hit, without disk reads.
    {
        MoeSsdCache cache;
        check(cache.open(path, 1), "open resident expert provider");
        check(cache.add_source(spec("resident_gate", 0)) &&
                  cache.add_source(spec("resident_down", 12)),
              "register resident expert sources");
        const auto* gate = cache.find_source("resident_gate");
        const auto* down = cache.find_source("resident_down");
        std::vector<ExpertLease> layer_leases;
        check(!cache.borrow_resident_layer(gate, down, layer_leases),
              "demand providers do not opt into permanent layer pins");
        check(!cache.load_resident({}), "reject empty resident source set");
        check(!cache.load_resident({{gate, down}, {gate, down}}),
              "reject duplicate resident pairs");
        check(cache.load_resident({{gate, down}}), "preload and lock every expert");
        check(cache.stats().resident_bytes == sizeof(contents),
              "resident provider owns exactly one copy of every expert");
        for (int pass = 0; pass < 2; ++pass) {
            for (int expert = 0; expert < 3; ++expert) {
                Tensor gu, dw;
                ExpertLease host_lease;
                const bool acquired = gate->resident_device_views
                    ? cache.borrow(gate, down, expert, host_lease)
                    : cache.acquire(gate, down, expert, gu, dw);
                if (host_lease) { gu = host_lease.gate_up; dw = host_lease.down; }
                check(acquired, "resident expert remains available");
                if (!acquired) continue;
                const char* arena_option = std::getenv("MOLLM_RESIDENT_EXPERT_ARENA");
                if (arena_option && std::strcmp(arena_option, "1") == 0)
                    check(reinterpret_cast<uintptr_t>(gu.ptr<uint16_t>()) % 64 == 0 &&
                              reinterpret_cast<uintptr_t>(dw.ptr<uint16_t>()) % 64 == 0,
                          "resident arena components are 64-byte aligned");
                check(gu.ptr<uint16_t>()[0] == contents[2 * expert] &&
                          dw.ptr<uint16_t>()[0] == contents[6 + 2 * expert],
                      "resident expert data is preserved");
            }
        }
        check(cache.stats().bytes_read == 0 && cache.stats().evictions == 0 &&
                  cache.stats().misses == 0,
              "resident routes do no I/O or eviction after startup");
        check(cache.borrow_resident_layer(gate, down, layer_leases) && layer_leases.size() == 3,
              "resident provider lends all expert IDs without I/O");
        for (size_t i = 0; i < layer_leases.size(); ++i) {
            uint16_t first = 0;
#if defined(MOLLM_CUDA)
            if (gate->resident_device_views)
                check(cudaMemcpy(&first, layer_leases[i].gate_up.data, sizeof(first),
                    cudaMemcpyDeviceToHost) == cudaSuccess, "read device layer lease");
            else
#endif
                first = layer_leases[i].gate_up.ptr<uint16_t>()[0];
            check(first == contents[2 * i],
                  "resident layer leases preserve expert-ID order");
        }
        check(!cache.clear_resident() && !cache.release(gate, down, 1),
              "long-lived layer leases protect asynchronous GPU reads");
        layer_leases.clear();
        check(cache.clear_resident(), "resident layer teardown returns all pins");
        check(!cache.borrow_resident_layer(gate, down, layer_leases),
              "clearing the provider revokes its resident capability");
    }

#if defined(MOLLM_CUDA)
    int cuda_devices = 0;
    if (cudaGetDeviceCount(&cuda_devices) == cudaSuccess && cuda_devices > 0) {
        const char* old_option = std::getenv("MOLLM_RESIDENT_EXPERT_ARENA");
        const bool had_option = old_option != nullptr;
        const std::string saved_option = old_option ? old_option : "";
        setenv("MOLLM_RESIDENT_EXPERT_ARENA", "cuda-device", 1);
        {
            const std::string native_path = "/tmp/mollm_test_moe_native_layout.bin";
            std::vector<uint8_t> bytes(1280);
            for (int projection = 0; projection < 2; ++projection) {
                for (int i = 0; i < 512; ++i) bytes[projection * 640 + i] = (i * 37 + projection * 19) & 255;
                for (int i = 0; i < 64; ++i) bytes[projection * 640 + 512 + i] = 48 + i % 16;
                for (int i = 0; i < 16; ++i) {
                    const float global = (i + 1) / 64.f;
                    std::memcpy(bytes.data() + projection * 640 + 576 + i * 4, &global, 4);
                }
            }
            {
                std::ofstream file(native_path, std::ios::binary);
                file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            }
            const char* previous = std::getenv("MOLLM_RESIDENT_NVFP4_MMA_LAYOUT");
            const bool had_previous = previous != nullptr;
            const std::string saved_native = previous ? previous : "";
            setenv("MOLLM_RESIDENT_NVFP4_MMA_LAYOUT", "1", 1);
            {
                MoeSsdCache cache;
                auto native_spec = spec("native_gate", 0);
                native_spec.num_experts = 1; native_spec.rows = 16; native_spec.cols = 64;
                native_spec.precision = Precision::NVFP4; native_spec.group_size = 16;
                native_spec.groups_per_row = 4; native_spec.data_bytes = 512;
                native_spec.scales_offset = 512; native_spec.scales_bytes = 128;
                native_spec.expert_stride = 640;
                native_spec.flags = MappedFile::FLAG_EXPERT_INTERLEAVED | MappedFile::FLAG_NVFP4_Q8_PAIR;
                auto down_spec = native_spec;
                down_spec.weight_ref = "native_down"; down_spec.data_offset = 640;
                down_spec.scales_offset = 1152;
                check(cache.open(native_path, 1) && cache.add_source(native_spec) && cache.add_source(down_spec),
                    "create native-layout resident fixture");
                const auto* gate = cache.find_source("native_gate");
                const auto* down = cache.find_source("native_down");
                const bool loaded = cache.load_resident({{gate, down}});
                check(loaded, "prepare single-copy native resident layout");
                if (loaded) {
                    std::vector<ExpertLease> layer;
                    check(cache.borrow_resident_layer(gate, down, layer) && layer.size() == 1 &&
                        layer[0].gate_up.nvfp4_mma_layout && layer[0].down.nvfp4_mma_layout,
                        "whole-layer views explicitly identify native layout");
                    layer.clear();
                    ExpertLease host;
                    check(cache.borrow(gate, down, 0, host), "restore CPU views from native layout");
                    if (host) {
                        check(!host.gate_up.nvfp4_mma_layout && !host.down.nvfp4_mma_layout &&
                            !std::memcmp(host.gate_up.data, bytes.data(), 512) &&
                            !std::memcmp(host.gate_up.nvfp4_scales, bytes.data() + 512, 128) &&
                            !std::memcmp(host.down.data, bytes.data() + 640, 512) &&
                            !std::memcmp(host.down.nvfp4_scales, bytes.data() + 1152, 128),
                            "CPU fallback restores every weight/scale byte");
                    }
                    host.reset();
                    check(cache.stats().resident_bytes == bytes.size(), "native layout retains one payload");
                    check(cache.clear_resident() && !gate->resident_nvfp4_mma_layout,
                        "clear resets native layout capability");
                }
            }
            if (had_previous) setenv("MOLLM_RESIDENT_NVFP4_MMA_LAYOUT", saved_native.c_str(), 1);
            else unsetenv("MOLLM_RESIDENT_NVFP4_MMA_LAYOUT");
            std::remove(native_path.c_str());
        }
        {
            const char* names[]={"MOLLM_RESIDENT_NVFP4_GATE_CODES_KMAJOR",
                "MOLLM_CUDA_NVFP4_MMA_PREFILL_ONLY","MOLLM_RESIDENT_NVFP4_MMA_LAYOUT",
                "MOLLM_TEST_RESIDENT_NVFP4_GATE_SCALES_KMAJOR"};
            std::vector<std::string> saved;
            std::vector<bool> present;
            for(const char* name:names){const char* value=std::getenv(name);present.push_back(value!=nullptr);saved.emplace_back(value?value:"");}
            setenv(names[0],"1",1);setenv(names[1],"1",1);setenv(names[2],"0",1);
            const std::string code_path="/tmp/mollm_test_moe_gate_codes.bin";
            for(const char* scale_mode:{"0","1",static_cast<const char*>(nullptr)})for(bool pair:{false,true}) {
                const bool scale_view=!scale_mode||!std::strcmp(scale_mode,"1");
                if(scale_mode)setenv(names[3],scale_mode,1);else unsetenv(names[3]);
                auto gate_spec=spec("code_gate",0);
                gate_spec.num_experts=1;gate_spec.rows=1280;gate_spec.cols=2560;
                gate_spec.precision=Precision::NVFP4;gate_spec.group_size=16;
                gate_spec.groups_per_row=160;gate_spec.data_bytes=1280*2560/2;
                gate_spec.scales_offset=gate_spec.data_bytes;
                gate_spec.scales_bytes=1280*160+1280*sizeof(float);
                gate_spec.expert_stride=gate_spec.data_bytes+gate_spec.scales_bytes;
                gate_spec.flags=MappedFile::FLAG_EXPERT_INTERLEAVED|(pair?MappedFile::FLAG_NVFP4_Q8_PAIR:0);
                gate_spec.input_scales={1.f,1.f};gate_spec.input_scales_per_expert=2;
                auto down_spec=gate_spec;
                down_spec.weight_ref="code_down";down_spec.rows=2560;down_spec.cols=640;
                down_spec.groups_per_row=40;down_spec.data_bytes=2560*640/2;
                down_spec.data_offset=gate_spec.expert_stride;
                down_spec.scales_offset=down_spec.data_offset+down_spec.data_bytes;
                down_spec.scales_bytes=2560*40+2560*sizeof(float);
                down_spec.expert_stride=down_spec.data_bytes+down_spec.scales_bytes;
                down_spec.input_scales={1.f};down_spec.input_scales_per_expert=1;
                std::vector<uint8_t> bytes(gate_spec.expert_stride+down_spec.expert_stride);
                for(size_t i=0;i<bytes.size();++i){uint32_t v=uint32_t(i)*0x9e3779b9u;bytes[i]=(v^(v>>16))&255;}
                for(const auto* projection:{&gate_spec,&down_spec}) {
                    const size_t block_bytes=size_t(projection->rows)*projection->groups_per_row;
                    for(int row=0;row<projection->rows;++row) {
                        const float scale=(row%7+1)/2560.f;
                        std::memcpy(bytes.data()+projection->scales_offset+block_bytes+row*sizeof(float),&scale,sizeof(float));
                    }
                }
                {std::ofstream file(code_path,std::ios::binary);file.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());}
                MoeSsdCache cache;
                check(cache.open(code_path,1)&&cache.add_source(gate_spec)&&cache.add_source(down_spec),"create bounded gate-code cache fixture");
                const auto* gate=cache.find_source("code_gate");const auto* down=cache.find_source("code_down");
                setenv(names[2],"1",1);
                check(!cache.load_resident({{gate,down}}),"reject conflicting native-MMA and K64-major layouts");
                setenv(names[2],"0",1);
                const bool loaded=cache.load_resident({{gate,down}});
                check(loaded,"load gate-code view into existing arena");
                if(loaded) {
                    check(gate->resident_nvfp4_gate_codes_kmajor&&!down->resident_nvfp4_gate_codes_kmajor,"mark only gate source layout");
                    std::vector<ExpertLease> layer;
                    check(cache.borrow_resident_layer(gate,down,layer)&&layer.size()==1,"borrow gate-code device view");
                    if(!layer.empty()) {
                        check(layer[0].gate_up.nvfp4_gate_codes_kmajor&&!layer[0].down.nvfp4_gate_codes_kmajor,"propagate tensor layout flags");
                        std::vector<uint8_t> actual(gate_spec.data_bytes),expected(actual.size());
                        const size_t blocks=pair?320:1280,tile_bytes=pair?128:32;
                        for(size_t q=0;q<40;++q)for(size_t row=0;row<blocks;++row)
                            std::memcpy(expected.data()+(q*blocks+row)*tile_bytes,bytes.data()+(row*40+q)*tile_bytes,tile_bytes);
                        check(cudaMemcpy(actual.data(),layer[0].gate_up.data,actual.size(),cudaMemcpyDeviceToHost)==cudaSuccess&&actual==expected,"resident gate matches independent byte transpose");
                        check(layer[0].gate_up.nvfp4_gate_scales_kmajor==scale_view&&
                            !layer[0].down.nvfp4_gate_scales_kmajor,"propagate gate-only scale layout flag");
                        actual.resize(gate_spec.scales_bytes);expected.resize(gate_spec.scales_bytes);
                        std::memcpy(expected.data(),bytes.data()+gate_spec.scales_offset,expected.size());
                        if(scale_view)for(int row=0;row<1280;++row)for(int group=0;group<160;++group)
                            expected[(size_t(group/4)*1280+row)*4+group%4]=
                                bytes[gate_spec.scales_offset+size_t(row)*160+group];
                        check(cudaMemcpy(actual.data(),layer[0].gate_up.nvfp4_scales,actual.size(),cudaMemcpyDeviceToHost)==cudaSuccess&&actual==expected,
                            "resident gate scale bytes match transpose and preserve row-global tail");
                    }
                    layer.clear();
                    ExpertLease host;
                    check(cache.borrow(gate,down,0,host),"borrow canonical CPU fallback from code view");
                    if(host) {
                        check(!host.gate_up.nvfp4_gate_codes_kmajor&&!host.down.nvfp4_gate_codes_kmajor&&
                            !host.gate_up.nvfp4_gate_scales_kmajor&&!host.down.nvfp4_gate_scales_kmajor&&
                            !std::memcmp(host.gate_up.data,bytes.data(),gate_spec.data_bytes)&&
                            !std::memcmp(host.gate_up.nvfp4_scales,bytes.data()+gate_spec.scales_offset,gate_spec.scales_bytes)&&
                            !std::memcmp(host.down.data,bytes.data()+down_spec.data_offset,down_spec.data_bytes)&&
                            !std::memcmp(host.down.nvfp4_scales,bytes.data()+down_spec.scales_offset,down_spec.scales_bytes),"canonical CPU fallback restores all gate/down bytes and flags");
                    }
                    host.reset();
                    check(cache.stats().resident_bytes==bytes.size(),"gate code view retains one resident payload");
                    check(cache.clear_resident()&&!gate->resident_nvfp4_gate_codes_kmajor&&!down->resident_nvfp4_gate_codes_kmajor&&
                        !gate->resident_nvfp4_gate_scales_kmajor&&!down->resident_nvfp4_gate_scales_kmajor,"clear gate code/scale capability after pins released");
                }
            }
            for(size_t i=0;i<saved.size();++i){if(present[i])setenv(names[i],saved[i].c_str(),1);else unsetenv(names[i]);}
            std::remove(code_path.c_str());
        }
        {
            MoeSsdCache cache;
            check(cache.open(path, 1) && cache.add_source(spec("device_gate", 0)) &&
                cache.add_source(spec("device_down", 12)), "create device-resident cache");
            const auto* gate = cache.find_source("device_gate");
            const auto* down = cache.find_source("device_down");
            const bool loaded = cache.load_resident({{gate, down}});
            check(loaded, "upload all experts to single device allocation");
            if (loaded) {
                check(gate->resident_device_views && down->resident_device_views,
                    "resident sources identify device-only views");
                Tensor gu, dw;
                check(!cache.acquire(gate, down, 0, gu, dw),
                    "legacy host acquire never exposes device-only addresses");
                std::vector<ExpertLease> resident;
                check(cache.borrow_resident_layer(gate, down, resident) && resident.size() == 3,
                    "borrow immutable device layer without host copies");
                for (int expert = 0; expert < static_cast<int>(resident.size()); ++expert) {
                    uint16_t bytes[2]{};
                    check(cudaMemcpy(bytes, resident[expert].gate_up.data, sizeof(bytes),
                            cudaMemcpyDeviceToHost) == cudaSuccess &&
                        bytes[0] == contents[expert * 2] && bytes[1] == contents[expert * 2 + 1],
                        "device-resident gate payload preserves bytes and expert-ID order");
                }
                resident.clear();
                ExpertLease host;
                check(cache.borrow(gate, down, 1, host), "CPU fallback obtains owned host views");
                if (host) {
                    check(host.owned_payload && host.gate_up.ptr<uint16_t>()[0] == contents[2] &&
                        host.down.ptr<uint16_t>()[0] == contents[8], "CPU fallback copy is exact");
                    ExpertLease moved(std::move(host));
                    check(!host && moved.owned_payload && moved.gate_up.ptr<uint16_t>()[1] == contents[3],
                        "moving fallback lease preserves owned payload and pin");
                    check(!cache.clear_resident(), "fallback lease pins the device source");
                    moved.reset();
                    check(!moved.owned_payload, "fallback reset releases temporary host storage");
                }
                check(cache.stats().bytes_read == 0 && cache.stats().resident_bytes == sizeof(contents),
                    "device routes and CPU fallback do no post-startup file reads");
                check(cache.clear_resident() && !gate->resident_device_views,
                    "clearing device cache resets source capability");
            }
        }
        if (had_option) setenv("MOLLM_RESIDENT_EXPERT_ARENA", saved_option.c_str(), 1);
        else unsetenv("MOLLM_RESIDENT_EXPERT_ARENA");
    }
#endif

    // Leases protect resident payloads independently of cache-retention policy.
    {
        MoeSsdCache cache;
        check(cache.open(path, 16), "open two-pair lease cache");
        check(cache.add_source(spec("lease_gate", 0)) &&
                  cache.add_source(spec("lease_down", 12)), "register lease sources");
        const auto* gate = cache.find_source("lease_gate");
        const auto* down = cache.find_source("lease_down");
        ExpertProvider& provider = cache;
        ExpertLease first, second, missing;
        check(provider.borrow(gate, down, 0, first) &&
                  provider.borrow(gate, down, 1, second), "borrow two expert pairs");
        check(!provider.evict(gate, down, 0) && !cache.clear_resident(),
              "eviction and cache clearing reject active leases");
        check(!cache.open(path, 16), "reopening cannot invalidate active leases");
        check(provider.request_many(gate, down, {2}), "request while both slots are pinned");
        check(provider.contains(gate, down, 0) && provider.contains(gate, down, 1),
              "new demand cannot recycle pinned expert data");
        check(!provider.borrow(gate, down, 2, missing, false) && !missing,
              "nonblocking borrow does not wait or allocate on a miss");
        ExpertLease moved = std::move(first);
        check(!first && moved && !provider.evict(gate, down, 0),
              "moving a lease transfers its pin");
        check(moved.gate_up.data &&
                  moved.gate_up.ptr<uint16_t>()[0] == contents[0],
              "pinned bytes survive competing demand");
        moved.reset();
        check(provider.request_many(gate, down, {2}) &&
                  wait_until([&] { return provider.is_ready(gate, down, 2); }),
              "freed pin lets the request window advance");
        check(provider.borrow(gate, down, 2, missing, false), "borrow newly-ready pair");
        missing = std::move(second);
        check(!second && provider.evict(gate, down, 2) &&
                  !provider.evict(gate, down, 1),
              "move assignment releases the old pin and retains the new one");
        ExpertSource foreign;
        foreign.provider = &provider;
        check(!provider.borrow(&foreign, down, 0, missing) && !missing,
              "failed replacement clears the old lease and rejects foreign metadata");
        check(cache.clear_resident(), "all lease pins are returned on reset and failure");
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

        auto short_stride = spec("short_stride", 0);
        short_stride.expert_stride = short_stride.data_bytes - 1;
        check(!cache.add_source(short_stride),
              "reject expert stride shorter than its data");
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

        auto duplicate_scales = bg128_spec("bg128_duplicate_scales", 0);
        duplicate_scales.scales_offset = 0;
        duplicate_scales.scales_bytes = 8 * sizeof(float);
        check(!cache.add_source(duplicate_scales),
              "reject BG128 expert with duplicate scale sidecar");

        auto legacy = bg128_spec("legacy_q4dot", 0);
        legacy.flags = MappedFile::FLAG_INT4_Q4DOT_LEGACY;
        check(!cache.add_source(legacy),
              "reject legacy Q4DOT expert storage");
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

    // Offline-pair-packed NVFP4 experts must be accepted directly from the
    // package and exposed to every CPU dispatch without an online transpose.
    {
        constexpr int rows = 4;
        constexpr int cols = 32;
        constexpr size_t data_bytes = rows * cols / 2;
        constexpr size_t scale_bytes = rows * (cols / 16) +
                                       rows * sizeof(float);
        std::vector<uint8_t> rowmajor(data_bytes);
        for (size_t i = 0; i < rowmajor.size(); ++i)
            rowmajor[i] = static_cast<uint8_t>(i * 13 + 7);
        std::vector<uint8_t> pair_packed(data_bytes);
        check(pack_nvfp4_q8_pairs(
                  rowmajor.data(), pair_packed.data(), rows, cols),
              "pack NVFP4 SSD fixture");
        std::vector<uint8_t> scales(scale_bytes, 0x38);
        float* row_scales = reinterpret_cast<float*>(
            scales.data() + rows * (cols / 16));
        for (int row = 0; row < rows; ++row) row_scales[row] = 1.0f;
        const std::string nvfp4_path =
            "/tmp/mollm_test_moe_ssd_nvfp4_pair.bin";
        {
            std::ofstream out(nvfp4_path, std::ios::binary);
            out.write(reinterpret_cast<const char*>(pair_packed.data()),
                      static_cast<std::streamsize>(pair_packed.size()));
            out.write(reinterpret_cast<const char*>(pair_packed.data()),
                      static_cast<std::streamsize>(pair_packed.size()));
            out.write(reinterpret_cast<const char*>(scales.data()),
                      static_cast<std::streamsize>(scales.size()));
            out.write(reinterpret_cast<const char*>(scales.data()),
                      static_cast<std::streamsize>(scales.size()));
        }
        auto nv_spec = [&](const char* name, uint64_t data_offset,
                           uint64_t scales_offset) {
            MoeSsdTensorSpec out;
            out.weight_ref = name;
            out.layer = 0;
            out.num_experts = 1;
            out.rows = rows;
            out.cols = cols;
            out.precision = Precision::NVFP4;
            out.flags = MappedFile::FLAG_NVFP4_Q8_PAIR;
            out.group_size = 16;
            out.groups_per_row = cols / 16;
            out.data_offset = data_offset;
            out.data_bytes = data_bytes;
            out.scales_offset = scales_offset;
            out.scales_bytes = scale_bytes;
            return out;
        };

        MoeSsdCache cache;
        check(cache.open(nvfp4_path, 2 * (data_bytes + scale_bytes)),
              "open pair-packed NVFP4 cache");
        auto calibrated_gate = nv_spec("nvfp4_gate", 0, 2 * data_bytes);
        calibrated_gate.input_scales_per_expert = 2;
        calibrated_gate.input_scales = {0.0013892765f, 0.0031854538f};
        auto invalid_calibration = calibrated_gate;
        invalid_calibration.weight_ref = "invalid_calibration";
        invalid_calibration.input_scales.pop_back();
        check(!cache.add_source(invalid_calibration), "reject incomplete activation calibration");
        for (float invalid : {0.f, -1.f, INFINITY, NAN}) {
            invalid_calibration = calibrated_gate;
            invalid_calibration.weight_ref = "invalid_calibration";
            invalid_calibration.input_scales[0] = invalid;
            check(!cache.add_source(invalid_calibration), "reject invalid activation calibration value");
        }
        check(cache.add_source(calibrated_gate) &&
              cache.add_source(nv_spec(
                  "nvfp4_down", data_bytes,
                  2 * data_bytes + scale_bytes)),
              "add pair-packed NVFP4 sources");
        const auto* calibrated_source = cache.find_source("nvfp4_gate");
        check(calibrated_source && calibrated_source->nvfp4_input_scales_per_expert == 2 &&
                  calibrated_source->nvfp4_input_scales == calibrated_gate.input_scales,
              "preserve distinct gate/up checkpoint activation scales");
        check(cache.find_source("nvfp4_down")->nvfp4_input_scales.empty(),
              "old sources remain explicitly uncalibrated");
        Tensor gate, down;
        check(cache.acquire(cache.find_source("nvfp4_gate"),
                            cache.find_source("nvfp4_down"),
                            0, gate, down),
              "load pair-packed NVFP4 expert");
        check(gate.nvfp4_q8_pair_data == gate.data &&
                  down.nvfp4_q8_pair_data == down.data,
              "pair-packed NVFP4 aliases cached package bytes");

        std::vector<float> activation(cols);
        for (int k = 0; k < cols; ++k)
            activation[k] = static_cast<float>(k - 15) / 16.0f;
        std::vector<float> expected(rows), actual(rows);
        Tensor input = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            cols, 1, 1, 1, activation.data());
        Tensor reference = Tensor::create(
            Precision::NVFP4, MemoryType::EXTERNAL,
            rows, cols, 1, 1, rowmajor.data());
        reference.nvfp4_scales = scales.data();
        reference.nvfp4_row_scales = row_scales;
        reference.group_size = 16;
        reference.groups_per_row = cols / 16;
        reference.num_groups = rows * (cols / 16);
        Tensor expected_tensor = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            rows, 1, 1, 1, expected.data());
        Tensor actual_tensor = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            rows, 1, 1, 1, actual.data());
        kernel_matmul_fp32(input, reference, expected_tensor);
        kernel_matmul_fp32(input, gate, actual_tensor);
        for (int row = 0; row < rows; ++row) {
            if (std::abs(expected[row] - actual[row]) >= 1e-4f) {
                std::fprintf(stderr,
                             "NVFP4 SSD mismatch row=%d expected=%g actual=%g\n",
                             row, expected[row], actual[row]);
                check(false,
                      "pair-packed NVFP4 SSD matmul matches row-major");
            }
        }
        std::remove(nvfp4_path.c_str());
    }

    // Storage tensors may place each expert's data and scale sidecar in one
    // chunk. Verify that the explicit stride reaches the next expert rather
    // than treating the intervening sidecar bytes as matrix data.
    {
        const std::string interleaved_path =
            "/tmp/mollm_test_moe_ssd_interleaved.bin";
        std::vector<uint16_t> interleaved(24, 0xffff);
        for (int expert = 0; expert < 3; ++expert) {
            interleaved[expert * 4] =
                static_cast<uint16_t>(0x3c00 + expert);
            interleaved[expert * 4 + 1] =
                static_cast<uint16_t>(0x4000 + expert);
            interleaved[12 + expert * 4] =
                static_cast<uint16_t>(0x4400 + expert);
            interleaved[12 + expert * 4 + 1] =
                static_cast<uint16_t>(0x4800 + expert);
        }
        {
            std::ofstream out(interleaved_path, std::ios::binary);
            out.write(reinterpret_cast<const char*>(interleaved.data()),
                      static_cast<std::streamsize>(
                          interleaved.size() * sizeof(uint16_t)));
        }
        auto gate_spec = spec("interleaved_gate", 0);
        auto down_spec = spec(
            "interleaved_down", 12 * sizeof(uint16_t));
        gate_spec.expert_stride = 4 * sizeof(uint16_t);
        down_spec.expert_stride = 4 * sizeof(uint16_t);
        MoeSsdCache cache;
        check(cache.open(interleaved_path, 8),
              "open expert-interleaved cache");
        check(cache.add_source(gate_spec) && cache.add_source(down_spec),
              "add expert-interleaved sources");
        Tensor gate, down;
        check(cache.acquire(cache.find_source("interleaved_gate"),
                            cache.find_source("interleaved_down"),
                            2, gate, down),
              "load strided expert");
        check(static_cast<const uint16_t*>(gate.data)[0] == 0x3c02 &&
                  static_cast<const uint16_t*>(down.data)[1] == 0x4802,
              "expert stride skips interleaved sidecar bytes");
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
        check(stats.demand_load_bytes == 24 &&
                  stats.prefetch_load_bytes == 0,
              "demand and prefetch load bytes are separated");
        check(stats.expert_bytes_acquired == 32,
              "expert acquire bytes include cache hits");

        cache.reset_stats();
        check(cache.acquire(gate, down, 0, gu, dw),
              "acquire resident expert after statistics reset");
        stats = cache.stats();
        check(stats.bytes_read == 0 && stats.expert_bytes_acquired == 8 &&
                  stats.hits == 1,
              "statistics interval counts resident weight use without a load");
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

    // With room for all three pairs, request adjacent experts together and
    // verify independent expert-pair jobs preserve every tensor slice.
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
        check(cache.acquire(gate, down, 1, gu, dw),
              "finish adjacent expert-pair requests");
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
        check(wait_until([&] {
                  return cache.contains(gate, down, 1) &&
                      cache.contains(gate, down, 2);
              }),
              "wait for exact hash prefetch");
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
        ExpertLease failed;
        check(!cache.borrow(gate, down, 0, failed) && !failed,
              "failed borrow observes truncated-file read failure and releases its pin");
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

    // Exact hash routes are known before their layer starts, so they should
    // not reserve the previous token's route. Clearing cross-token retention
    // must make the oldest former route entry evictable again.
    {
        MoeSsdCache cache;
        check(cache.open(path, 24, 2),
              "open exact-route retention cache");
        auto gate0 = spec("exact_retain_gate0", 0);
        auto down0 = spec("exact_retain_down0", 6 * sizeof(uint16_t));
        auto gate1 = spec("exact_retain_gate1", 0);
        auto down1 = spec("exact_retain_down1", 6 * sizeof(uint16_t));
        gate1.layer = down1.layer = 1;
        check(cache.add_source(gate0) && cache.add_source(down0) &&
                  cache.add_source(gate1) && cache.add_source(down1) &&
                  cache.set_global_capacity_pool(true),
              "configure exact-route retention cache");
        const MoeSsdTensorSource* g0 =
            cache.find_source("exact_retain_gate0");
        const MoeSsdTensorSource* d0 =
            cache.find_source("exact_retain_down0");
        const MoeSsdTensorSource* g1 =
            cache.find_source("exact_retain_gate1");
        const MoeSsdTensorSource* d1 =
            cache.find_source("exact_retain_down1");
        Tensor gu, dw;
        check(cache.retain_for_next_forward(g0, d0, {0, 1}) &&
                  cache.acquire(g0, d0, 0, gu, dw) &&
                  cache.acquire(g0, d0, 1, gu, dw) &&
                  cache.acquire(g1, d1, 0, gu, dw),
              "populate one retained and one ordinary route");
        cache.begin_forward_pass();
        check(cache.retain_for_next_forward(g0, d0, {0, 1}, false),
              "clear exact-route cross-token retention");
        check(cache.acquire(g1, d1, 0, gu, dw) &&
                  cache.acquire(g1, d1, 1, gu, dw),
              "admit a current-layer route after retention clear");
        check(!cache.contains(g0, d0, 0) &&
                  cache.contains(g1, d1, 0) &&
                  cache.contains(g1, d1, 1),
              "exact-route retention clear restores LRU eviction");
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

    // Logical load origin and eventual usefulness are independent events: a
    // prefill-interval prefetch can be consumed during a later decode interval.
    {
        MoeSsdCache cache;
        check(cache.open(path, 8, 1), "open useful-prefetch cache");
        check(cache.add_source(spec("useful_gate", 0)) &&
                  cache.add_source(spec("useful_down", 6 * sizeof(uint16_t))),
              "add useful-prefetch sources");
        const MoeSsdTensorSource* gate = cache.find_source("useful_gate");
        const MoeSsdTensorSource* down = cache.find_source("useful_down");
        check(cache.prefetch_many(gate, down, {0}, {1.0f}),
              "queue useful speculative expert");
        check(wait_until([&] { return cache.stats().bytes_read >= 8; }),
              "wait for useful speculative expert");
        Tensor gu, dw;
        check(cache.acquire(gate, down, 0, gu, dw),
              "consume useful speculative expert");
        const auto stats = cache.stats();
        check(stats.prefetch_load_bytes == 8 &&
                  stats.useful_prefetch_bytes == 8 &&
                  stats.expert_bytes_acquired == 8,
              "useful prefetch bytes are counted on first consumption");
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
        check(wait_until([&] { return cache.stats().bytes_read >= 8; }),
              "wait for unused speculative expert");

        Tensor gu, dw;
        check(cache.acquire(gate, down, 1, gu, dw),
              "fill cache beside speculative expert");
        cache.begin_forward_pass();
        check(cache.acquire(gate, down, 2, gu, dw),
              "evict stale unused speculation");
        check(cache.acquire(gate, down, 0, gu, dw),
              "reload recently evicted expert");
        const auto stats = cache.stats();
        check(stats.prefetch_load_bytes >= 8,
              "speculative-origin load bytes are counted separately");
        check(stats.unused_prefetch_bytes >= 8,
              "unused speculative bytes are counted on eviction");
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
