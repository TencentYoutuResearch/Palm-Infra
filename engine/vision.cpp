#include "engine/engine.h"
#include "engine/vision.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>
#else
#define STBI_FAILURE_USERMSG
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"
#endif

namespace {

bool fail(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
}

int metadata_int(const std::unordered_map<std::string, std::string>& metadata,
                 const char* key, int fallback = 0) {
    auto it = metadata.find(key);
    if (it == metadata.end()) return fallback;
    try {
        return std::stoi(it->second);
    } catch (...) {
        return fallback;
    }
}

float load_weight_value(const Tensor& tensor, size_t index) {
    const void* raw = tensor.rowmajor_data ? tensor.rowmajor_data : tensor.data;
    if (tensor.prec == Precision::FP16)
        return static_cast<float>(static_cast<const __fp16*>(raw)[index]);
    return static_cast<const float*>(raw)[index];
}

bool image_dimensions_allowed(int width, int height) {
    constexpr int64_t max_source_pixels = int64_t{64} * 1024 * 1024;
    return width > 0 && height > 0 &&
        static_cast<int64_t>(width) * height <= max_source_pixels;
}

float bicubic_filter(float x) {
    x = std::fabs(x);
    if (x < 1.0f)
        return ((1.5f * x - 2.5f) * x) * x + 1.0f;
    if (x < 2.0f)
        return (((-0.5f * x + 2.5f) * x - 4.0f) * x) + 2.0f;
    return 0.0f;
}

std::vector<uint8_t> resize_rgba_bicubic(
    const std::vector<uint8_t>& source, int source_w, int source_h,
    int target_w, int target_h) {
    if (source_w == target_w && source_h == target_h)
        return source;

    std::vector<uint8_t> horizontal(
        static_cast<size_t>(target_w) * source_h * 4);
    const float scale_x =
        static_cast<float>(source_w) / static_cast<float>(target_w);
    const float filter_scale_x = std::max(1.0f, scale_x);
    const float support_x = 2.0f * filter_scale_x;
    for (int y = 0; y < source_h; ++y) {
        for (int dx = 0; dx < target_w; ++dx) {
            const float center = (dx + 0.5f) * scale_x;
            const int begin = std::max(
                0, static_cast<int>(std::floor(center - support_x + 0.5f)));
            const int end = std::min(
                source_w,
                static_cast<int>(std::floor(center + support_x + 0.5f)));
            float weight_sum = 0.0f;
            for (int sx = begin; sx < end; ++sx)
                weight_sum += bicubic_filter(
                    (sx + 0.5f - center) / filter_scale_x);
            for (int channel = 0; channel < 4; ++channel) {
                float value = 0.0f;
                for (int sx = begin; sx < end; ++sx) {
                    const float weight = bicubic_filter(
                        (sx + 0.5f - center) / filter_scale_x) /
                        weight_sum;
                    value += weight * source[
                        (static_cast<size_t>(y) * source_w + sx) * 4 +
                        channel];
                }
                horizontal[
                    (static_cast<size_t>(y) * target_w + dx) * 4 +
                    channel] = static_cast<uint8_t>(
                        std::clamp(std::lround(value), 0l, 255l));
            }
        }
    }

    std::vector<uint8_t> output(
        static_cast<size_t>(target_w) * target_h * 4);
    const float scale_y =
        static_cast<float>(source_h) / static_cast<float>(target_h);
    const float filter_scale_y = std::max(1.0f, scale_y);
    const float support_y = 2.0f * filter_scale_y;
    for (int dy = 0; dy < target_h; ++dy) {
        const float center = (dy + 0.5f) * scale_y;
        const int begin = std::max(
            0, static_cast<int>(std::floor(center - support_y + 0.5f)));
        const int end = std::min(
            source_h,
            static_cast<int>(std::floor(center + support_y + 0.5f)));
        float weight_sum = 0.0f;
        for (int sy = begin; sy < end; ++sy)
            weight_sum += bicubic_filter(
                (sy + 0.5f - center) / filter_scale_y);
        for (int x = 0; x < target_w; ++x) {
            for (int channel = 0; channel < 4; ++channel) {
                float value = 0.0f;
                for (int sy = begin; sy < end; ++sy) {
                    const float weight = bicubic_filter(
                        (sy + 0.5f - center) / filter_scale_y) /
                        weight_sum;
                    value += weight * horizontal[
                        (static_cast<size_t>(sy) * target_w + x) * 4 +
                        channel];
                }
                output[
                    (static_cast<size_t>(dy) * target_w + x) * 4 +
                    channel] = static_cast<uint8_t>(
                        std::clamp(std::lround(value), 0l, 255l));
            }
        }
    }
    return output;
}

#ifdef __APPLE__
bool decode_image_rgba(CGImageRef image, int width, int height,
                       std::vector<uint8_t>& decoded) {
    decoded.resize(static_cast<size_t>(height) * width * 4);

    // ImageIO commonly exposes PNG/JPEG samples as 8-bit RGBA/RGBX. Reading
    // those samples directly is important for transparent PNGs: drawing into
    // a premultiplied CGContext destroys the hidden RGB values at alpha=0,
    // while Pillow's RGB conversion (used by the reference processor) keeps
    // them.
    const CGBitmapInfo bitmap_info = CGImageGetBitmapInfo(image);
    const CGImageAlphaInfo alpha_info = CGImageGetAlphaInfo(image);
    const CGBitmapInfo byte_order =
        bitmap_info & kCGBitmapByteOrderMask;
    const bool direct_rgba =
        CGImageGetBitsPerComponent(image) == 8 &&
        CGImageGetBitsPerPixel(image) == 32 &&
        CGImageGetBytesPerRow(image) >= static_cast<size_t>(width) * 4 &&
        (alpha_info == kCGImageAlphaLast ||
         alpha_info == kCGImageAlphaNoneSkipLast) &&
        (byte_order == kCGBitmapByteOrderDefault ||
         byte_order == kCGBitmapByteOrder32Big);
    if (direct_rgba) {
        CGDataProviderRef provider = CGImageGetDataProvider(image);
        CFDataRef data = provider ? CGDataProviderCopyData(provider) : nullptr;
        if (data) {
            const UInt8* source = CFDataGetBytePtr(data);
            const CFIndex length = CFDataGetLength(data);
            const size_t source_stride = CGImageGetBytesPerRow(image);
            const size_t required =
                source_stride * static_cast<size_t>(height);
            if (source && length >= 0 &&
                static_cast<size_t>(length) >= required) {
                for (int y = 0; y < height; ++y) {
                    std::memcpy(
                        decoded.data() +
                            static_cast<size_t>(y) * width * 4,
                        source + static_cast<size_t>(y) * source_stride,
                        static_cast<size_t>(width) * 4);
                }
                CFRelease(data);
                return true;
            }
            CFRelease(data);
        }
    }

    CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(
        decoded.data(), width, height, 8,
        static_cast<size_t>(width) * 4, color_space,
        kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(color_space);
    if (!context)
        return false;
    CGContextDrawImage(
        context, CGRectMake(0, 0, width, height), image);
    CGContextRelease(context);
    return true;
}
#endif

} // namespace

void mollm::detail::smart_resize_dimensions(
    int source_h, int source_w, int factor, int min_pixels, int max_pixels,
    int& target_h, int& target_w) {
    target_h = std::max(
        factor,
        static_cast<int>(std::nearbyint(
            static_cast<double>(source_h) / factor)) * factor);
    target_w = std::max(
        factor,
        static_cast<int>(std::nearbyint(
            static_cast<double>(source_w) / factor)) * factor);
    const int64_t rounded_pixels =
        static_cast<int64_t>(target_h) * target_w;
    if (rounded_pixels > max_pixels) {
        const double beta = std::sqrt(
            static_cast<double>(source_h) * source_w / max_pixels);
        target_h = std::max(
            factor,
            static_cast<int>(std::floor(
                source_h / beta / factor)) * factor);
        target_w = std::max(
            factor,
            static_cast<int>(std::floor(
                source_w / beta / factor)) * factor);
    } else if (rounded_pixels < min_pixels) {
        const double beta = std::sqrt(
            static_cast<double>(min_pixels) /
            (static_cast<double>(source_h) * source_w));
        target_h = static_cast<int>(
            std::ceil(source_h * beta / factor)) * factor;
        target_w = static_cast<int>(
            std::ceil(source_w * beta / factor)) * factor;
    }
}

bool mollm::detail::decode_image_file_rgba(
    const std::string& path, std::vector<unsigned char>& rgba,
    int& width, int& height) {
    rgba.clear();
    width = 0;
    height = 0;
#if !defined(__APPLE__)
    int channels = 0;
    if (!stbi_info(path.c_str(), &width, &height, &channels) ||
        !image_dimensions_allowed(width, height)) {
        width = 0;
        height = 0;
        return false;
    }
    stbi_uc* decoded = stbi_load(
        path.c_str(), &width, &height, &channels, STBI_rgb_alpha);
    if (!decoded || !image_dimensions_allowed(width, height)) {
        stbi_image_free(decoded);
        width = 0;
        height = 0;
        return false;
    }
    const size_t count = static_cast<size_t>(width) * height * 4;
    rgba.assign(decoded, decoded + count);
    stbi_image_free(decoded);
    return true;
#else
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(
        kCFAllocatorDefault,
        reinterpret_cast<const UInt8*>(path.data()),
        static_cast<CFIndex>(path.size()), false);
    if (!url) return false;
    CGImageSourceRef source = CGImageSourceCreateWithURL(url, nullptr);
    CFRelease(url);
    if (!source) return false;
    CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, nullptr);
    CFRelease(source);
    if (!image) return false;
    width = static_cast<int>(CGImageGetWidth(image));
    height = static_cast<int>(CGImageGetHeight(image));
    const bool ok =
        image_dimensions_allowed(width, height) &&
        decode_image_rgba(image, width, height, rgba);
    CGImageRelease(image);
    if (!ok) {
        rgba.clear();
        width = 0;
        height = 0;
    }
    return ok;
#endif
}

void mollm::detail::fill_multimodal_rope_cache(
    float* cos_cache, float* sin_cache, int seq_len, int token_offset,
    const std::vector<int>& position_ids, int rope_dim, float rope_theta,
    int section_t, int section_h, int section_w) {
    (void)section_t; // Temporal is the default axis, including tail slots.
    const int half = rope_dim / 2;
    const int prompt_tokens =
        static_cast<int>(position_ids.size() / 3);
    for (int token = 0; token < seq_len; ++token) {
        for (int i = 0; i < half; ++i) {
            int axis = 0;
            if (i % 3 == 1 && i < section_h * 3)
                axis = 1;
            else if (i % 3 == 2 && i < section_w * 3)
                axis = 2;
            const int position = position_ids[
                axis * prompt_tokens + token_offset + token];
            const float exponent =
                static_cast<float>(2 * i) / static_cast<float>(rope_dim);
            const float angle =
                position / std::pow(rope_theta, exponent);
            cos_cache[static_cast<size_t>(token) * half + i] =
                std::cos(angle);
            sin_cache[static_cast<size_t>(token) * half + i] =
                std::sin(angle);
        }
    }
}

bool mollm::detail::build_multimodal_position_ids(
    const std::vector<int>& token_ids, int image_token_id,
    int grid_t, int grid_h, int grid_w, int vision_tokens, int merge,
    int base_position, std::vector<int>& position_ids,
    int& rope_position_delta, std::string* error) {
    const int n = static_cast<int>(token_ids.size());
    if (grid_t != 1 || merge <= 0 ||
        grid_h % merge != 0 || grid_w % merge != 0) {
        return fail(error,
                    "only one static image with a valid merged grid is "
                    "supported");
    }
    const int image_count = static_cast<int>(
        std::count(token_ids.begin(), token_ids.end(), image_token_id));
    if (image_count != vision_tokens) {
        return fail(
            error, "image placeholder count " +
                       std::to_string(image_count) +
                       " does not match vision token count " +
                       std::to_string(vision_tokens));
    }

    position_ids.assign(static_cast<size_t>(3) * n, 0);
    int current = base_position;
    int i = 0;
    bool consumed_image = false;
    while (i < n) {
        if (token_ids[i] != image_token_id) {
            for (int axis = 0; axis < 3; ++axis)
                position_ids[axis * n + i] = current;
            ++current;
            ++i;
            continue;
        }
        if (consumed_image)
            return fail(error, "multiple images are not supported yet");
        consumed_image = true;
        const int llm_h = grid_h / merge;
        const int llm_w = grid_w / merge;
        for (int token = 0; token < vision_tokens; ++token) {
            const int index = i + token;
            if (index >= n || token_ids[index] != image_token_id) {
                return fail(
                    error, "image placeholder tokens must be contiguous");
            }
            const int row = (token / llm_w) % llm_h;
            const int col = token % llm_w;
            position_ids[index] = current;
            position_ids[n + index] = current + row;
            position_ids[2 * n + index] = current + col;
        }
        current += std::max(llm_h, llm_w);
        i += vision_tokens;
    }
    rope_position_delta = current - (base_position + n);
    if (error) error->clear();
    return true;
}

bool LLMEngine::prepare_multimodal_positions(
    const std::vector<int>& token_ids, int image_token_id,
    const VisionEmbedding& vision, std::string* error) {
    const int merge = metadata_int(
        package_metadata_, "vision_spatial_merge_size", 2);
    return mollm::detail::build_multimodal_position_ids(
        token_ids, image_token_id,
        vision.grid_t, vision.grid_h, vision.grid_w,
        vision.tokens, merge, multimodal_base_past_,
        multimodal_position_ids_, rope_position_delta_, error);
}

void LLMEngine::generate_multimodal_rope_cache(
    int seq_len, int token_offset, Tensor& cos, Tensor& sin) {
    const int half = cfg_.rope_dim / 2;
    const int prompt_tokens =
        static_cast<int>(multimodal_position_ids_.size() / 3);
    if (token_offset < 0 || token_offset + seq_len > prompt_tokens) {
        std::fprintf(
            stderr,
            "multimodal RoPE range [%d, %d) exceeds prompt length %d\n",
            token_offset, token_offset + seq_len, prompt_tokens);
        return;
    }

    void* cb =
        graph_prefill_.runtime.pool.acquire(half * seq_len * sizeof(float));
    void* sb =
        graph_prefill_.runtime.pool.acquire(half * seq_len * sizeof(float));
    cos = Tensor::create(
        Precision::FP32, MemoryType::POOLED, half, seq_len, 1, 1, cb);
    sin = Tensor::create(
        Precision::FP32, MemoryType::POOLED, half, seq_len, 1, 1, sb);
    cos.owner_id = graph_prefill_.runtime.pool.id();
    sin.owner_id = graph_prefill_.runtime.pool.id();
    cos.storage_id = graph_prefill_.runtime.pool.storage_id(cb);
    sin.storage_id = graph_prefill_.runtime.pool.storage_id(sb);

    int section_t = metadata_int(package_metadata_, "mrope_section_t", 11);
    int section_h = metadata_int(package_metadata_, "mrope_section_h", 11);
    int section_w = metadata_int(package_metadata_, "mrope_section_w", 10);
    if (section_t <= 0 || section_h <= 0 || section_w <= 0 ||
        section_t + section_h + section_w != half) {
        section_t = 11;
        section_h = 11;
        section_w = 10;
    }
    mollm::detail::fill_multimodal_rope_cache(
        cos.ptr<float>(), sin.ptr<float>(), seq_len, token_offset,
        multimodal_position_ids_, cfg_.rope_dim, cfg_.rope_theta,
        section_t, section_h, section_w);
}

int LLMEngine::prefill_with_image(const std::vector<int>& token_ids,
                                  int image_token_id,
                                  const VisionEmbedding& vision,
                                  std::string* error) {
    const auto reject = [error](const char* message) {
        if (error) *error = message;
        return -1;
    };
    if (!has_vision_encoder())
        return reject("package does not contain a vision encoder");
    if (past_len_ != 0)
        return reject("image input currently requires an empty context");
    if (!embed_weight_ ||
        vision.hidden_size != static_cast<int>(embed_weight_->shape[1])) {
        return reject("vision embedding size does not match the text model");
    }
    if (vision.tokens <= 0 || vision.hidden_size <= 0 ||
        vision.values.size() !=
            static_cast<size_t>(vision.tokens) *
                static_cast<size_t>(vision.hidden_size)) {
        return reject(
            "vision embedding values do not match its declared shape");
    }
    if (cfg_.static_padded)
        return reject("multimodal prefill does not support --static-padded");

    multimodal_base_past_ = past_len_;
    if (!prepare_multimodal_positions(
            token_ids, image_token_id, vision, error)) {
        multimodal_position_ids_.clear();
        return -1;
    }
    active_vision_ = &vision;
    active_image_token_id_ = image_token_id;
    active_vision_cursor_ = 0;
    const int result = prefill(token_ids);
    const bool complete = active_vision_cursor_ == vision.tokens;
    active_vision_ = nullptr;
    active_image_token_id_ = -1;
    multimodal_position_ids_.clear();
    if (result < 0 || !complete) {
        rope_position_delta_ = 0;
        if (error && error->empty())
            *error = complete ? "multimodal prefill failed"
                              : "not all vision embeddings were consumed";
        return -1;
    }
    if (error) error->clear();
    return result;
}

bool LLMEngine::encode_image_file(const std::string& path,
                                  VisionEmbedding& output,
                                  std::string* error) {
    std::vector<uint8_t> decoded;
    int source_w = 0;
    int source_h = 0;
    if (!mollm::detail::decode_image_file_rgba(
            path, decoded, source_w, source_h)) {
        return fail(error, "unsupported or unreadable image");
    }
    const int patch = metadata_int(
        package_metadata_, "vision_patch_size", 16);
    const int temporal = metadata_int(
        package_metadata_, "vision_temporal_patch_size", 2);
    const int merge = metadata_int(
        package_metadata_, "vision_spatial_merge_size", 2);
    const int min_pixels = metadata_int(
        package_metadata_, "vision_min_pixels", 256 * 256);
    const int processor_max_pixels = metadata_int(
        package_metadata_, "vision_max_pixels", 4096 * 4096);
    const int max_pixels =
        std::min(processor_max_pixels, cfg_.image_max_pixels);
    if (source_w <= 0 || source_h <= 0 || patch <= 0 || temporal <= 0 ||
        merge <= 0 || min_pixels <= 0 || max_pixels <= 0) {
        return fail(error, "invalid image or vision processor metadata");
    }
    const double aspect =
        static_cast<double>(std::max(source_w, source_h)) /
        std::min(source_w, source_h);
    if (aspect > 200.0) {
        return fail(error, "image aspect ratio exceeds 200");
    }

    const int factor = patch * merge;
    int target_h = 0;
    int target_w = 0;
    mollm::detail::smart_resize_dimensions(
        source_h, source_w, factor, min_pixels, max_pixels,
        target_h, target_w);

    std::vector<uint8_t> rgba = resize_rgba_bicubic(
        decoded, source_w, source_h, target_w, target_h);

    const int grid_h = target_h / patch;
    const int grid_w = target_w / patch;
    const int patch_dim = 3 * temporal * patch * patch;
    std::vector<float> pixels(
        static_cast<size_t>(grid_h) * grid_w * patch_dim);
    size_t out_index = 0;
    // Match Qwen2VLImageProcessor's flattened order:
    // [block_h, block_w, inner_h, inner_w, channel, temporal, y, x].
    for (int block_h = 0; block_h < grid_h; block_h += merge) {
        for (int block_w = 0; block_w < grid_w; block_w += merge) {
            for (int inner_h = 0; inner_h < merge; ++inner_h) {
                for (int inner_w = 0; inner_w < merge; ++inner_w) {
                    const int patch_y = (block_h + inner_h) * patch;
                    const int patch_x = (block_w + inner_w) * patch;
                    for (int channel = 0; channel < 3; ++channel) {
                        for (int t = 0; t < temporal; ++t) {
                            (void)t; // A static image is repeated in time.
                            for (int y = 0; y < patch; ++y) {
                                for (int x = 0; x < patch; ++x) {
                                    const size_t source_index =
                                        (static_cast<size_t>(patch_y + y) *
                                             target_w +
                                         patch_x + x) *
                                            4 +
                                        channel;
                                    pixels[out_index++] =
                                        rgba[source_index] / 127.5f - 1.0f;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return encode_vision_patches(
        pixels, 1, grid_h, grid_w, output, error);
}

bool LLMEngine::encode_vision_patches(
    const std::vector<float>& pixel_values, int grid_t, int grid_h, int grid_w,
    VisionEmbedding& output, std::string* error) {
    output = VisionEmbedding{};
    if (graph_vision_.nodes.empty())
        return fail(error, "package does not contain a vision encoder");
    if (!vision_pos_embed_ || !vision_pos_embed_->rowmajor_data)
        return fail(error, "vision position embedding is unavailable");
    if (grid_t <= 0 || grid_h <= 0 || grid_w <= 0)
        return fail(error, "vision grid dimensions must be positive");

    const int merge = metadata_int(
        package_metadata_, "vision_spatial_merge_size", 2);
    const int hidden = metadata_int(
        package_metadata_, "vision_hidden_size", 0);
    const int heads = metadata_int(
        package_metadata_, "vision_num_heads", 0);
    const int positions = metadata_int(
        package_metadata_, "vision_num_position_embeddings", 0);
    if (merge <= 0 || grid_h % merge != 0 || grid_w % merge != 0)
        return fail(error, "vision grid is not divisible by spatial merge");
    if (hidden <= 0 || heads <= 0 || hidden % heads != 0 ||
        positions <= 0)
        return fail(error, "invalid vision metadata");

    const int64_t patch_count64 =
        static_cast<int64_t>(grid_t) * grid_h * grid_w;
    if (patch_count64 > std::numeric_limits<int>::max())
        return fail(error, "vision grid is too large");
    const int patch_count = static_cast<int>(patch_count64);
    int patch_dim = 0;
    for (const auto& node : graph_vision_.nodes) {
        if (node.op_type == OpType::INPUT && !node.params.str.empty() &&
            node.params.str[0] == "pixel_values") {
            patch_dim = static_cast<int>(node.out_shape[0]);
            break;
        }
    }
    if (patch_dim <= 0 ||
        pixel_values.size() !=
            static_cast<size_t>(patch_count) *
                static_cast<size_t>(patch_dim)) {
        return fail(error, "pixel_values shape does not match vision grid");
    }

    const int side =
        static_cast<int>(std::llround(std::sqrt(static_cast<double>(positions))));
    if (side * side != positions)
        return fail(error, "vision position table is not a square grid");
    const int head_dim = hidden / heads;
    const int rope_half = head_dim / 2;
    if (rope_half % 2 != 0)
        return fail(error, "unsupported vision RoPE dimension");

    // Processor patch order is [t, block_h, block_w, inner_h, inner_w].
    std::vector<int> rows;
    std::vector<int> cols;
    rows.reserve(patch_count);
    cols.reserve(patch_count);
    for (int t = 0; t < grid_t; ++t) {
        (void)t;
        for (int bh = 0; bh < grid_h; bh += merge) {
            for (int bw = 0; bw < grid_w; bw += merge) {
                for (int ih = 0; ih < merge; ++ih) {
                    for (int iw = 0; iw < merge; ++iw) {
                        rows.push_back(bh + ih);
                        cols.push_back(bw + iw);
                    }
                }
            }
        }
    }

    std::vector<float> position_embeds(
        static_cast<size_t>(patch_count) * hidden);
    const Tensor& table = *vision_pos_embed_;
    for (int token = 0; token < patch_count; ++token) {
        const float fy = grid_h == 1
                             ? 0.0f
                             : (side - 1.0f) * rows[token] / (grid_h - 1.0f);
        const float fx = grid_w == 1
                             ? 0.0f
                             : (side - 1.0f) * cols[token] / (grid_w - 1.0f);
        const int y0 = static_cast<int>(fy);
        const int x0 = static_cast<int>(fx);
        const int y1 = std::min(y0 + 1, side - 1);
        const int x1 = std::min(x0 + 1, side - 1);
        const float wy = fy - y0;
        const float wx = fx - x0;
        const int indices[4] = {
            y0 * side + x0, y0 * side + x1,
            y1 * side + x0, y1 * side + x1};
        const float weights[4] = {
            (1.0f - wy) * (1.0f - wx), (1.0f - wy) * wx,
            wy * (1.0f - wx), wy * wx};
        float* dst = position_embeds.data() +
                     static_cast<size_t>(token) * hidden;
        for (int d = 0; d < hidden; ++d) {
            float value = 0.0f;
            for (int corner = 0; corner < 4; ++corner) {
                value += weights[corner] *
                         load_weight_value(
                             table,
                             static_cast<size_t>(indices[corner]) * hidden + d);
            }
            dst[d] = value;
        }
    }

    std::vector<float> cos_cache(
        static_cast<size_t>(patch_count) * rope_half);
    std::vector<float> sin_cache(cos_cache.size());
    const int axis_freqs = rope_half / 2;
    for (int token = 0; token < patch_count; ++token) {
        const int axis_pos[2] = {rows[token], cols[token]};
        for (int axis = 0; axis < 2; ++axis) {
            for (int i = 0; i < axis_freqs; ++i) {
                const float exponent = static_cast<float>(2 * i) / rope_half;
                const float angle =
                    axis_pos[axis] / std::pow(10000.0f, exponent);
                const size_t index =
                    static_cast<size_t>(token) * rope_half +
                    axis * axis_freqs + i;
                cos_cache[index] = std::cos(angle);
                sin_cache[index] = std::sin(angle);
            }
        }
    }

    Tensor pixels = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, patch_dim, patch_count, 1, 1,
        const_cast<float*>(pixel_values.data()));
    Tensor pos = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, hidden, patch_count, 1, 1,
        position_embeds.data());
    Tensor cos = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, rope_half, patch_count, 1, 1,
        cos_cache.data());
    Tensor sin = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, rope_half, patch_count, 1, 1,
        sin_cache.data());

    for (const auto& node : graph_vision_.nodes) {
        if (node.op_type != OpType::INPUT || node.params.str.empty()) continue;
        Tensor& input = graph_vision_.runtime.tensors[node.id];
        const std::string& name = node.params.str[0];
        if (name == "pixel_values")
            input = pixels;
        else if (name == "position_embeds")
            input = pos;
        else if (name == "vision_cos")
            input = cos;
        else if (name == "vision_sin")
            input = sin;
        if (accelerator_backend_ &&
            exec_ctx_vision_.backend == accelerator_backend_.get() &&
            input.data) {
            accelerator_backend_->upload_input(
                input, "vision_" + name, input.data, input.nbytes());
        }
    }

    exec_ctx_vision_.runtime_seq_len = patch_count;
    exec_ctx_vision_.runtime_batch = 1;
    exec_ctx_vision_.static_padded = false;
    exec_ctx_vision_.padded_seq_len = -1;
    inject_runtime_shapes(exec_ctx_vision_);
    if (accelerator_backend_ &&
        exec_ctx_vision_.backend == accelerator_backend_.get())
        accelerator_backend_->begin_graph();
    execute_graph(exec_ctx_vision_);
    if (accelerator_backend_ &&
        exec_ctx_vision_.backend == accelerator_backend_.get())
        accelerator_backend_->end_graph();
    if (exec_ctx_vision_.execution_failed) {
        release_vision_buffers();
        return fail(error, "vision graph execution failed");
    }
    if (graph_vision_.graph_outputs.size() != 1) {
        release_vision_buffers();
        return fail(error, "vision graph must have exactly one output");
    }
    const Tensor& result = graph_vision_.runtime.tensors[
        graph_vision_.graph_outputs[0]];
    const int output_tokens = patch_count / (merge * merge);
    if (!result.data || result.prec != Precision::FP32 ||
        result.shape[0] <= 0 || result.shape[1] != output_tokens) {
        release_vision_buffers();
        return fail(error, "vision graph produced an invalid output");
    }

    output.tokens = output_tokens;
    output.hidden_size = static_cast<int>(result.shape[0]);
    output.grid_t = grid_t;
    output.grid_h = grid_h;
    output.grid_w = grid_w;
    output.values.resize(
        static_cast<size_t>(output.tokens) * output.hidden_size);
    if (!exec_ctx_vision_.backend->copy_to_host(
            result, output.values.data(),
            output.values.size() * sizeof(float))) {
        release_vision_buffers();
        return fail(error, "vision output readback failed");
    }
    // Vision runs once per attached image. Its SDPA graph conservatively keeps
    // intermediates alive through the call, so return both CPU pool storage
    // and accelerator allocations immediately instead of retaining hundreds
    // of MB during text decode.
    release_vision_buffers();
    if (error) error->clear();
    return true;
}
