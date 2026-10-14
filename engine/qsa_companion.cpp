#include "engine/qsa_companion.h"
#include "storage/sha256.h"
#include <json.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

namespace {
void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
GraphNode node(OpType op, Precision precision, int64_t width, int64_t rows = 1) {
    GraphNode n; n.op_type = op; n.out_prec = precision;
    n.out_shape[0] = width; n.out_shape[1] = rows;
    return n;
}
}

bool attach_qsa_companion(Graph& g, const std::string& directory,
                          int context, std::string& error) {
    namespace fs = std::filesystem;
    using json = nlohmann::json;
    try {
        check(context > 0 && context <= 32768, "QSA context must be 1..32768");
        check(g.runtime.weights.empty(), "QSA rewrite requires an unloaded graph");
        const fs::path root = fs::canonical(directory);
        check(fs::file_size(root / "manifest.json") <= 1024 * 1024, "oversized QSA manifest");
        std::ifstream input(root / "manifest.json");
        const json manifest = json::parse(input);
        check(manifest.at("format") == "mollm-qsa-weight-bundle-v1" &&
              manifest.at("standalone_model") == false &&
              manifest.at("activation_precision") == "FP32" &&
              manifest.at("norm_weight_format") == "centered", "unsupported QSA manifest");
        const auto& geometry = manifest.at("geometry");
        const int heads = geometry.at("heads"), dim = geometry.at("dimension");
        const int ratio = geometry.at("ratio"), budget = geometry.at("budget");
        const int rope = geometry.at("rope_dim"), hidden = geometry.at("hidden_size");
        const int main_heads = geometry.at("main_heads"), kv_heads = geometry.at("main_kv_heads");
        const int main_dim = geometry.at("main_dimension");
        const double theta = geometry.at("rope_theta"), eps = geometry.at("eps");
        check(heads >= 1 && heads <= 16 && dim >= rope && dim <= 256 &&
              rope >= 2 && rope % 2 == 0 && ratio >= 1 && ratio <= 16 &&
              budget > 0 && budget <= 4096 && budget % ratio == 0 &&
              budget + ratio - 1 <= 4096 && hidden > 0 && hidden <= 1048576 &&
              main_heads > 0 && main_heads <= 4096 && kv_heads > 0 && kv_heads <= 4096 && main_heads % kv_heads == 0 &&
              main_dim >= rope && main_dim <= 4096 && std::isfinite(theta) && theta > 0 &&
              std::isfinite(eps) && eps > 0 && std::isfinite(static_cast<float>(eps)) &&
              static_cast<float>(eps) > 0, "unsupported QSA geometry");
        check(g.metadata.at("rope_dim") == std::to_string(rope) &&
              std::abs(std::stod(g.metadata.at("rope_theta")) / theta - 1.) < 1e-6,
              "QSA/main graph RoPE mismatch");
        std::set<int> layers;
        for (const auto& value : geometry.at("layers")) {
            const int layer = value.get<int>();
            check(layer >= 0 && layer < 100000 && layers.insert(layer).second,
                  "invalid/duplicate QSA layer");
        }
        check(!layers.empty() && manifest.at("weights").is_object() &&
              manifest.at("weights").size() == 3 * layers.size(), "QSA weight set mismatch");
        std::unordered_map<std::string, std::string> files;
        for (int layer : layers) {
            const std::string prefix = "model_language_model_layers_" + std::to_string(layer) +
                                       "_self_attn_indexer_";
            for (const std::string suffix : {"index_qk_proj", "q_layernorm", "k_layernorm"}) {
                const std::string basename = prefix + suffix + "_weight.weights";
                const auto& weight = manifest.at("weights").at("./" + basename);
                check(weight.at("file") == basename, "noncanonical QSA filename");
                const fs::path path = fs::canonical(root / basename);
                check(path.parent_path() == root && fs::is_regular_file(path), "QSA weight escapes bundle");
                const bool projection = suffix == "index_qk_proj";
                const json shape = projection ? json::array({(heads + 1) * dim, hidden}) : json::array({dim});
                check(weight.at("shape") == shape && weight.at("precision") == (projection ? "FP16" : "FP32"),
                      "QSA weight manifest shape/precision mismatch");
                const auto bytes = fs::file_size(path);
                check(bytes == weight.at("bytes").get<uint64_t>() && bytes <= 512 * 1024 * 1024,
                      "QSA weight size mismatch");
                check(mollm::detail::sha256_file(path.string()) == weight.at("sha256").get<std::string>(),
                      "QSA weight SHA-256 mismatch");
                MappedFile mapped;
                check(mapped.open(path.c_str()), "invalid QSA weight header");
                const auto& header = mapped.header();
                check(header.flags == 0 && header.scales_size == 0 &&
                      header.precision == static_cast<uint32_t>(projection ? Precision::FP16 : Precision::FP32) &&
                      header.shape[0] == static_cast<uint64_t>(projection ? (heads + 1) * dim : dim) &&
                      header.shape[1] == static_cast<uint64_t>(projection ? hidden : 1) &&
                      header.shape[2] == 1 && header.shape[3] == 1 &&
                      header.data_size == uint64_t(projection ? (heads + 1) * dim : dim) *
                                          (projection ? uint64_t(hidden) * 2 : 4), "QSA weight header mismatch");
                if (projection) {
                    const auto* values = static_cast<const uint16_t*>(mapped.data());
                    for (size_t i = 0; i < header.data_size / 2; ++i)
                        check((values[i] & 0x7c00) != 0x7c00, "nonfinite QSA projection");
                } else {
                    const auto* values = static_cast<const float*>(mapped.data());
                    for (int i = 0; i < dim; ++i) check(std::isfinite(values[i]), "nonfinite QSA norm");
                }
                files.emplace("./" + basename, path.string());
            }
        }
        // Resolve all producers first. Any unsupported graph fails without mutation.
        std::unordered_map<int, uint32_t> projections;
        int next_aux = 0;
        for (const auto& n : g.nodes) {
            check(n.id < g.nodes.size() && &g.nodes[n.id] == &n, "noncanonical graph IDs");
            for (auto id : n.inputs) check(id < n.id, "graph is not topologically ordered");
            if (n.op_type == OpType::INPUT && !n.params.str.empty() &&
                n.params.str[0].compare(0, 9, "aux_state") == 0) {
                size_t end = 0; const int value = std::stoi(n.params.str[0].substr(9), &end);
                check(end == n.params.str[0].size() - 9 && value >= 0 && value < 100000,
                      "invalid auxiliary state name");
                next_aux = std::max(next_aux, value + 1);
            }
            check(n.op_type != OpType::QSA_INDEXER && n.op_type != OpType::QSA_SDPA,
                  "graph already contains QSA");
            if (n.op_type != OpType::MATMUL || n.inputs.size() != 2) continue;
            const auto& w = g.nodes[n.inputs[1]];
            if (w.op_type != OpType::CONSTANT || w.params.str.empty()) continue;
            for (int layer : layers) {
                const std::string ref = "./model_language_model_layers_" + std::to_string(layer) +
                                        "_self_attn_qkv_proj_weight.weights";
                if (w.params.str[0] == ref) {
                    check(projections.emplace(layer, n.id).second, "duplicate main QKV projection");
                    check(w.out_shape[0] == int64_t(2 * main_heads + 2 * kv_heads) * main_dim &&
                          w.out_shape[1] == hidden, "main QKV weight geometry mismatch");
                    const auto& x = g.nodes[n.inputs[0]];
                    check(x.out_prec == Precision::FP32 && x.out_shape[0] == hidden &&
                          x.out_shape[2] == 1 && x.out_shape[3] == 1, "QSA projection input mismatch");
                }
            }
        }
        std::vector<GraphNode> nodes;
        std::vector<uint32_t> remap(g.nodes.size()), extra_inputs;
        std::set<int> attached;
        auto add = [&](GraphNode n) {
            n.id = static_cast<uint32_t>(nodes.size()); nodes.push_back(std::move(n));
            return nodes.back().id;
        };
        for (const auto& original : g.nodes) {
            GraphNode n = original;
            for (auto& id : n.inputs) id = remap[id];
            if (original.op_type == OpType::SDPA) {
                check(original.inputs.size() == 6 && original.params.i32.size() >= 6,
                      "unsupported main attention signature");
                const auto& cache = g.nodes[original.inputs[4]];
                check(cache.op_type == OpType::INPUT && !cache.params.str.empty() &&
                      cache.params.str[0].compare(0, 7, "cache_k") == 0, "missing main K cache");
                size_t end = 0; const int layer = std::stoi(cache.params.str[0].substr(7), &end);
                check(end == cache.params.str[0].size() - 7 && layers.count(layer) &&
                      attached.insert(layer).second && projections.count(layer), "QSA attention layer mismatch");
                const auto& p = original.params.i32;
                check(p[0] == 2 && p[1] == 1 && p[2] == main_heads && p[3] == kv_heads &&
                      p[4] == main_dim && p[5] == main_dim, "main attention geometry mismatch");
                const auto& value_cache = g.nodes[original.inputs[5]];
                check(value_cache.op_type == OpType::INPUT && !value_cache.params.str.empty() &&
                      value_cache.params.str[0] == "cache_v" + std::to_string(layer) &&
                      cache.out_shape[0] == main_dim && cache.out_shape[2] == kv_heads &&
                      value_cache.out_shape[0] == main_dim && value_cache.out_shape[2] == kv_heads,
                      "main KV cache geometry/layer mismatch");
                const uint32_t producer = projections.at(layer);
                std::vector<uint32_t> pending{original.inputs[0]}; std::set<uint32_t> visited;
                bool ancestor = false;
                while (!pending.empty()) {
                    const auto id = pending.back(); pending.pop_back();
                    if (!visited.insert(id).second) continue;
                    if (id == producer) { ancestor = true; break; }
                    for (auto input_id : g.nodes[id].inputs) pending.push_back(input_id);
                }
                check(ancestor, "main query does not use matching QKV producer");
                const auto& x = g.nodes[g.nodes[producer].inputs[0]];
                auto weight = [&](const std::string& suffix, bool projection) {
                    auto w = node(OpType::CONSTANT, projection ? Precision::FP16 : Precision::FP32,
                                  projection ? (heads + 1) * dim : dim, projection ? hidden : 1);
                    w.params.str = {files.at("./model_language_model_layers_" + std::to_string(layer) +
                                             "_self_attn_indexer_" + suffix + "_weight.weights")};
                    return add(std::move(w));
                };
                const auto pw = weight("index_qk_proj", true);
                const auto qw = weight("q_layernorm", false), kw = weight("k_layernorm", false);
                auto frequency = node(OpType::CONSTANT, Precision::FP32, rope / 2);
                frequency.params.str = {"__qsa_frequency__"};
                for (int i = 0; i < rope / 2; ++i)
                    frequency.params.f32.push_back(static_cast<float>(std::pow(theta, -2. * i / rope)));
                const auto fw = add(std::move(frequency));
                auto projection = node(OpType::MATMUL, Precision::FP32, (heads + 1) * dim, x.out_shape[1]);
                projection.dim_expr[1] = x.dim_expr[1];
                projection.inputs = {remap[g.nodes[producer].inputs[0]], pw}; projection.params.i32 = {0, 0, -1};
                const auto projected = add(std::move(projection));
                auto state = [&](int64_t rows) {
                    auto s = node(OpType::INPUT, Precision::FP32, dim, rows);
                    s.params.str = {"aux_state" + std::to_string(next_aux++)};
                    const auto id = add(std::move(s)); extra_inputs.push_back(id); return id;
                };
                const auto raw = state(context), pooled = state(std::max(1, context / ratio));
                auto indexer = node(OpType::QSA_INDEXER, Precision::INT32, budget + ratio - 1, x.out_shape[1]);
                indexer.dim_expr[1] = x.dim_expr[1];
                indexer.params.i32 = {heads, dim, rope, ratio, budget}; indexer.params.f32 = {static_cast<float>(eps)};
                indexer.inputs = {projected, qw, kw, fw, raw, pooled, n.inputs[4], n.inputs[3]};
                n.inputs.push_back(add(std::move(indexer))); n.op_type = OpType::QSA_SDPA;
            }
            remap[original.id] = add(std::move(n));
        }
        check(attached == layers, "QSA/full-attention layer set mismatch");
        auto inputs = g.graph_inputs, outputs = g.graph_outputs;
        for (auto& id : inputs) id = remap.at(id);
        for (auto& id : outputs) id = remap.at(id);
        inputs.insert(inputs.end(), extra_inputs.begin(), extra_inputs.end());
        auto metadata = g.metadata;
        if (metadata.count("mtp_state_output_node"))
            metadata["mtp_state_output_node"] = std::to_string(remap.at(std::stoul(metadata.at("mtp_state_output_node"))));
        metadata["qsa"] = "contiguous-causal-fp32-v1";
        std::vector<Tensor> tensors(nodes.size());
        g.nodes = std::move(nodes); g.graph_inputs = std::move(inputs); g.graph_outputs = std::move(outputs);
        g.metadata = std::move(metadata); g.runtime.tensors = std::move(tensors);
        return true;
    } catch (const std::exception& e) { error = e.what(); return false; }
}
