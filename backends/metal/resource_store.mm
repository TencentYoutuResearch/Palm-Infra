#include "backends/metal/resource_store.h"

#include "backends/metal/weight_layout.h"
#include "core/quant_layouts.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <unistd.h>

struct MetalResourceStore::Impl {
    explicit Impl(void* device_handle)
        : device((__bridge id<MTLDevice>)device_handle) {}

    struct WeightRegion {
        id<MTLBuffer> buffer = nil;
        char* base = nullptr;
        size_t size = 0;
    };

    id<MTLDevice> device = nil;
    std::vector<WeightRegion> weight_regions;
    bool copy_weights = false;
    std::vector<id<MTLBuffer>> persistent;
    std::vector<id<MTLBuffer>> weight_copies;
    std::unordered_map<const void*, id<MTLBuffer>> copied_weights;
    std::unordered_map<const void*, id<MTLBuffer>> decoded_q4_weights;
    std::unordered_map<std::string, id<MTLBuffer>> input_buffers;
    std::unordered_map<std::string, size_t> input_capacity;
    std::unordered_map<std::string, bool> input_is_zero;

    const WeightRegion* find_weight_region(const void* pointer, size_t bytes,
                                           size_t& offset) const {
        if (!pointer) return nullptr;
        const uintptr_t value = reinterpret_cast<uintptr_t>(pointer);
        for (const auto& region : weight_regions) {
            const uintptr_t base = reinterpret_cast<uintptr_t>(region.base);
            if (value < base) continue;
            const size_t relative = static_cast<size_t>(value - base);
            if (relative <= region.size && bytes <= region.size - relative) {
                offset = relative;
                return &region;
            }
        }
        return nullptr;
    }
};

MetalResourceStore::MetalResourceStore(void* device)
    : impl_(new Impl(device)) {}

MetalResourceStore::~MetalResourceStore() = default;

bool MetalResourceStore::register_weight_region(void* base, size_t size) {
    if (!impl_->device || !base || size == 0) return false;
    @autoreleasepool {
        impl_->weight_regions.clear();
        const size_t page_size = static_cast<size_t>(getpagesize());
        const size_t max_buffer =
            static_cast<size_t>(impl_->device.maxBufferLength);
        const size_t chunk_limit = (max_buffer / page_size) * page_size;
        if (chunk_limit == 0) return false;
        size_t offset = 0;
        while (offset < size) {
            const size_t length = std::min(chunk_limit, size - offset);
            char* chunk_base = static_cast<char*>(base) + offset;
            id<MTLBuffer> buffer =
                [impl_->device newBufferWithBytesNoCopy:chunk_base
                                                  length:length
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];
            if (!buffer) {
                std::fprintf(stderr,
                    "MetalBackend: newBufferWithBytesNoCopy(%zu) failed "
                    "at package offset %zu (maxBufferLength=%llu)\n",
                    length, offset,
                    static_cast<unsigned long long>(
                        impl_->device.maxBufferLength));
                impl_->weight_regions.clear();
                return false;
            }
            impl_->weight_regions.push_back({buffer, chunk_base, length});
            offset += length;
        }
        if (impl_->weight_regions.size() > 1) {
            std::fprintf(stderr,
                "MetalBackend: split %.1f MB weight region across %zu "
                "zero-copy buffers\n",
                size / 1e6, impl_->weight_regions.size());
        }
    }
    return true;
}

void MetalResourceStore::enable_weight_copy_mode() {
    impl_->copy_weights = true;
}

bool MetalResourceStore::has_weight_copies() const {
    return !impl_->weight_copies.empty();
}

bool MetalResourceStore::locate_weight(const void* pointer, size_t bytes,
                                       void*& buffer, size_t& offset) const {
    const auto* region = impl_->find_weight_region(pointer, bytes, offset);
    if (!region) {
        buffer = nullptr;
        return false;
    }
    buffer = (__bridge void*)region->buffer;
    return true;
}

void MetalResourceStore::wrap_weight(Tensor& tensor) {
    if (!tensor.data) return;
    if (impl_->weight_regions.empty()) {
        if (!impl_->copy_weights) return;
        if (tensor.prec == Precision::FP16 || tensor.prec == Precision::FP32) {
            void* source = tensor.data;
            const size_t bytes = tensor.nbytes();
            auto found = impl_->copied_weights.find(source);
            if (found != impl_->copied_weights.end()) {
                tensor.device_data = (__bridge void*)found->second;
                tensor.device_offset = 0;
            } else {
                @autoreleasepool {
                    id<MTLBuffer> buffer =
                        [impl_->device newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];
                    std::memcpy([buffer contents], source, bytes);
                    impl_->weight_copies.push_back(buffer);
                    impl_->copied_weights[source] = buffer;
                    tensor.device_data = (__bridge void*)buffer;
                    tensor.device_offset = 0;
                }
            }
        }
        return;
    }

    size_t storage_bytes = tensor.nbytes();
    if (tensor.prec == Precision::INT4 &&
        ((tensor.is_q4_g128_packed && tensor.q4_g128_data) ||
         (tensor.is_q4_g32_packed && tensor.q4_g32_data))) {
        int last = 3;
        while (last > 1 && tensor.shape[last] == 1) --last;
        int64_t rows = 1;
        for (int dimension = 0; dimension < last; ++dimension)
            rows *= tensor.shape[dimension];
        const size_t block_bytes = tensor.is_q4_g128_packed
            ? sizeof(Q4B8G128Block)
            : sizeof(Q4B8G32Block);
        storage_bytes = static_cast<size_t>((rows + 7) / 8) *
            static_cast<size_t>(tensor.groups_per_row) * block_bytes;
    }

    size_t offset = 0;
    void* buffer = nullptr;
    void* source = tensor.data;
    if (!locate_weight(source, storage_bytes, buffer, offset)) {
        // Copy tensors that straddle a maxBufferLength package boundary.
        alloc_persistent(tensor, storage_bytes);
        std::memcpy(tensor.data, source, storage_bytes);
        if (tensor.prec == Precision::INT8) wrap_weight_int8(tensor);
        return;
    }
    tensor.device_data = buffer;
    tensor.device_offset = offset;
    if (tensor.prec == Precision::INT8) wrap_weight_int8(tensor);
}

void MetalResourceStore::wrap_weight_int8(Tensor& tensor) {
    if (tensor.prec != Precision::INT8 || !tensor.scales ||
        tensor.num_groups == 0)
        return;
    const size_t scale_bytes =
        static_cast<size_t>(tensor.num_groups) * sizeof(float);
    size_t offset = 0;
    void* buffer_handle = nullptr;
    if (locate_weight(tensor.scales, scale_bytes, buffer_handle, offset)) {
        tensor.scales_device_data = buffer_handle;
        tensor.scales_device_offset = offset;
        return;
    }
    @autoreleasepool {
        id<MTLBuffer> buffer =
            [impl_->device newBufferWithLength:scale_bytes
                                       options:MTLResourceStorageModeShared];
        if (!buffer) return;
        std::memcpy([buffer contents], tensor.scales, scale_bytes);
        impl_->persistent.push_back(buffer);
        tensor.scales_device_data = (__bridge void*)buffer;
        tensor.scales_device_offset = 0;
    }
}

void MetalResourceStore::wrap_weight_int4(Tensor& tensor,
                                          bool keep_native_experts) {
    const bool bg32 = tensor.prec == Precision::INT4 &&
        tensor.is_q4_g32_packed && tensor.q4_g32_data;
    const bool bg128 = tensor.prec == Precision::INT4 &&
        tensor.is_q4_g128_packed && tensor.q4_g128_data;
    if (!bg32 && !bg128) return;

    int last = 3;
    while (last > 1 && tensor.shape[last] == 1) --last;
    const int columns = static_cast<int>(tensor.shape[last]);
    int64_t rows64 = 1;
    for (int dimension = 0; dimension < last; ++dimension)
        rows64 *= tensor.shape[dimension];
    const int rows = static_cast<int>(rows64);
    if (keep_native_experts || last >= 2) {
        wrap_weight(tensor);
        return;
    }

    const int groups_per_row = static_cast<int>(tensor.groups_per_row);
    const size_t nibble_bytes = static_cast<size_t>(rows) * (columns / 2);
    const size_t scale_bytes =
        static_cast<size_t>(rows) * groups_per_row * sizeof(float);
    const void* packed = bg32 ? tensor.q4_g32_data : tensor.q4_g128_data;
    auto cached = impl_->decoded_q4_weights.find(packed);
    if (cached != impl_->decoded_q4_weights.end()) {
        tensor.device_data = (__bridge void*)cached->second;
        tensor.device_offset = 0;
        return;
    }

    @autoreleasepool {
        id<MTLBuffer> buffer =
            [impl_->device newBufferWithLength:nibble_bytes + scale_bytes
                                       options:MTLResourceStorageModeShared];
        if (!buffer) return;
        uint8_t* nibbles = static_cast<uint8_t*>([buffer contents]);
        float* scales = reinterpret_cast<float*>(nibbles + nibble_bytes);
        if (!mollm::metal::decode_q4_weight(
                packed,
                bg32 ? mollm::metal::PackedQ4Layout::BG32
                     : mollm::metal::PackedQ4Layout::BG128,
                rows, columns, groups_per_row, nibbles, scales)) {
            std::fprintf(stderr,
                         "MetalBackend: invalid packed Q4 weight layout\n");
            return;
        }
        if (impl_->copy_weights)
            impl_->weight_copies.push_back(buffer);
        else
            impl_->persistent.push_back(buffer);
        tensor.device_data = (__bridge void*)buffer;
        tensor.device_offset = 0;
        impl_->decoded_q4_weights[packed] = buffer;
    }
}

void MetalResourceStore::alloc_persistent(Tensor& tensor, size_t bytes) {
    @autoreleasepool {
        id<MTLBuffer> buffer =
            [impl_->device newBufferWithLength:bytes
                                       options:MTLResourceStorageModeShared];
        impl_->persistent.push_back(buffer);
        tensor.device_data = (__bridge void*)buffer;
        tensor.device_offset = 0;
        tensor.data = [buffer contents];
    }
}

void MetalResourceStore::upload_input(Tensor& tensor, const std::string& key,
                                      const void* source, size_t bytes) {
    id<MTLBuffer> buffer = nil;
    auto found = impl_->input_buffers.find(key);
    if (found != impl_->input_buffers.end() &&
        impl_->input_capacity[key] >= bytes) {
        buffer = found->second;
    } else {
        buffer = [impl_->device newBufferWithLength:bytes
                                            options:MTLResourceStorageModeShared];
        impl_->input_buffers[key] = buffer;
        impl_->input_capacity[key] = bytes;
    }
    if (source) std::memcpy([buffer contents], source, bytes);
    impl_->input_is_zero[key] = false;
    tensor.device_data = (__bridge void*)buffer;
    tensor.device_offset = 0;
}

void MetalResourceStore::upload_zero_input(Tensor& tensor,
                                           const std::string& key,
                                           size_t bytes) {
    id<MTLBuffer> buffer = nil;
    auto found = impl_->input_buffers.find(key);
    const bool grow = found == impl_->input_buffers.end() ||
        impl_->input_capacity[key] < bytes;
    if (grow) {
        buffer = [impl_->device newBufferWithLength:bytes
                                            options:MTLResourceStorageModeShared];
        impl_->input_buffers[key] = buffer;
        impl_->input_capacity[key] = bytes;
        impl_->input_is_zero[key] = false;
    } else {
        buffer = found->second;
    }
    if (!impl_->input_is_zero[key]) {
        std::memset([buffer contents], 0, impl_->input_capacity[key]);
        impl_->input_is_zero[key] = true;
    }
    tensor.device_data = (__bridge void*)buffer;
    tensor.device_offset = 0;
}
