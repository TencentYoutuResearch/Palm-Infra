#pragma once

#include <string>
#include <vector>

namespace mollm::detail {

/// Match Qwen's smart-resize dimension calculation for already validated
/// positive inputs. The returned dimensions are multiples of `factor`.
void smart_resize_dimensions(int source_h, int source_w, int factor,
                             int min_pixels, int max_pixels,
                             int& target_h, int& target_w);

/// Decode an image to row-major RGBA while preserving hidden RGB values in
/// fully transparent PNG pixels. Apple uses ImageIO; other platforms use the
/// vendored PNG/JPEG decoder.
bool decode_image_file_rgba(const std::string& path,
                            std::vector<unsigned char>& rgba,
                            int& width, int& height);

/// Fill Qwen3.5's interleaved temporal/height/width RoPE cache from
/// axis-major [3, prompt_tokens] position IDs.
void fill_multimodal_rope_cache(
    float* cos_cache, float* sin_cache, int seq_len, int token_offset,
    const std::vector<int>& position_ids, int rope_dim, float rope_theta,
    int section_t, int section_h, int section_w);

/// Build Qwen3.5 axis-major [3, prompt_tokens] position IDs for one image.
bool build_multimodal_position_ids(
    const std::vector<int>& token_ids, int image_token_id,
    int grid_t, int grid_h, int grid_w, int vision_tokens, int merge,
    int base_position, std::vector<int>& position_ids,
    int& rope_position_delta, std::string* error);

} // namespace mollm::detail
