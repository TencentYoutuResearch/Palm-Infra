#include "backends/cuda/backend.h"
#include "graph/execute.h"
#include "core/cache_layout.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
void require(bool ok, const char* label) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
bool outer_graph_active = false;
struct OuterGraph {
    CudaBackend& backend;
    bool active = false;
    explicit OuterGraph(CudaBackend& b) : backend(b) {
        const char* flag = std::getenv("MOLLM_TEST_CUDA_SEGMENTED_ONE_GRAPH");
        active = flag && !std::strcmp(flag, "1");
        if (active) {
            require(!outer_graph_active, "no nested single-graph fixture");
            outer_graph_active = true; backend.begin_graph();
        }
    }
    void finish() {
        if (active) {
            backend.end_graph(); active = outer_graph_active = false;
            require(!backend.dispatch_failed(), "outer graph final synchronization succeeds");
        }
    }
    ~OuterGraph() { finish(); }
};
Tensor allocate(CudaBackend& backend, int width, int rows, int heads = 1) {
    Tensor t = Tensor::create(Precision::FP32, MemoryType::NONE, width, rows, heads, 1);
    require(backend.alloc_output(t, t.nbytes(), nullptr) != nullptr, "allocate device tensor");
    return t;
}
std::vector<float> read(CudaBackend& backend, const Tensor& t) {
    std::vector<float> result(t.nelements());
    require(t.data && backend.copy_to_host(t, result.data(), t.nbytes()), "download device tensor");
    return result;
}
void write(CudaBackend& backend, Tensor& t, const std::vector<float>& values) {
    require(t.nbytes() == values.size() * sizeof(float) &&
        backend.copy_from_host(values.data(), t, t.nbytes()), "upload device tensor");
}
void equal(const std::vector<float>& a, const std::vector<float>& b, const char* label) {
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i)
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]) ||
            std::memcmp(&a[i], &b[i], sizeof(float))) {
            std::fprintf(stderr, "%s index=%zu actual=%g expected=%g\n", label, i, a[i], b[i]);
            break;
        }
    require(a.size() == b.size() &&
        std::all_of(a.begin(), a.end(), [](float x) { return std::isfinite(x); }) &&
        std::all_of(b.begin(), b.end(), [](float x) { return std::isfinite(x); }) &&
        std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0, label);
}

struct Frame {
    Graph graph;
    BufferPool pool;
    ExecContext ctx;
    CudaBackend& backend;
    Frame(CudaBackend& b, int width, Tensor weight, Tensor first_state, Tensor second_state,
          int liveness, bool workspace, const Graph* source = nullptr) : backend(b) {
        if (source) {
            require(initialize_execution_frame(*source, graph, {0}), "clone independent CUDA runtime frame");
        } else {
        graph.nodes.resize(12); graph.runtime.tensors.resize(12);
        for (int i = 0; i < 12; ++i) {
            auto& n = graph.nodes[i]; n.id = i; n.out_prec = Precision::FP32;
            n.out_shape[0] = width; n.out_shape[1] = 17;
            n.dim_expr[1].kind = DIM_SEQ;
        }
        for (int i : {0, 1, 2, 8}) graph.nodes[i].op_type = OpType::INPUT;
        for (int i : {1, 2, 8}) graph.nodes[i].dim_expr[1].kind = DIM_CONST;
        graph.nodes[3].op_type = OpType::ADD; graph.nodes[3].inputs = {0, 0};
        graph.nodes[4].op_type = OpType::RESHAPE; graph.nodes[4].inputs = {3};
        graph.nodes[5].op_type = OpType::SHORTCONV; graph.nodes[5].inputs = {4, 1, 2};
        graph.nodes[5].params.i32 = {4, 17};
        graph.nodes[6].op_type = OpType::ADD; graph.nodes[6].inputs = {5, 5};
        graph.nodes[7].op_type = OpType::RESHAPE; graph.nodes[7].inputs = {6};
        graph.nodes[9].op_type = OpType::SHORTCONV; graph.nodes[9].inputs = {7, 1, 8};
        graph.nodes[9].params.i32 = {4, 17};
        graph.nodes[10].op_type = OpType::ADD; graph.nodes[10].inputs = {9, 9};
        graph.nodes[11].op_type = OpType::RESHAPE; graph.nodes[11].inputs = {10};
        graph.graph_inputs = {0, 1, 2, 8}; graph.graph_outputs = {11};
        graph.runtime.tensors[1] = weight; graph.runtime.tensors[2] = first_state;
        graph.runtime.tensors[8] = second_state;
        }
        ctx.graph = &graph; ctx.pool = &pool; ctx.backend = &backend;
        ctx.conservative_stateful_liveness = liveness != 0;
        ctx.stateful_readonly_last_use = liveness == 2;
        ctx.reuse_same_shape_workspace = workspace;
        prepare_execution(ctx);
    }
    void input(int rows, int seed) {
        auto& t = graph.runtime.tensors[0];
        if (t.data) backend.free_output(t, nullptr);
        t = allocate(backend, graph.nodes[0].out_shape[0], rows);
        std::vector<float> values(t.nelements());
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = float(int((i * 13 + seed * 17) % 97) - 48) / 128.f;
        write(backend, t, values);
        ctx.runtime_seq_len = rows; inject_runtime_shapes(ctx);
    }
    void full() {
        backend.begin_graph(); execute_graph(ctx); backend.end_graph();
        require(!ctx.execution_failed && !backend.dispatch_failed(), "full reference execution");
    }
    void segment(int first, int end) {
        if (!outer_graph_active) backend.begin_graph();
        execute_graph_segment(ctx, first, end);
        if (!outer_graph_active) backend.end_graph();
        require(!ctx.execution_failed && !backend.dispatch_failed(), "segmented execution");
    }
    std::vector<float> output() { return read(backend, graph.runtime.tensors[11]); }
    ~Frame() {
        // Inputs 1/2/8 are shared; views 4/7/11 borrow their storage owner.
        for (int i : {0, 3, 5, 6, 9, 10}) {
            auto& t = graph.runtime.tensors[i];
            if (t.data) backend.free_output(t, nullptr);
        }
    }
};

void run(CudaBackend& backend, int width, int liveness, bool workspace, int cut) {
    Tensor weight = allocate(backend, 4, width);
    Tensor state0 = allocate(backend, 3, width), state1 = allocate(backend, 3, width);
    std::vector<float> weights(weight.nelements()), initial0(state0.nelements()), initial1(state1.nelements());
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = float(int(i % 13) - 6) / 32.f;
    for (size_t i = 0; i < initial0.size(); ++i) {
        initial0[i] = float(int(i % 17) - 8) / 128.f;
        initial1[i] = float(int(i % 19) - 9) / 64.f;
    }
    write(backend, weight, weights);
    {
        Frame a(backend, width, weight, state0, state1, liveness, workspace);
        Frame b(backend, width, weight, state0, state1, liveness, workspace, &a.graph);
        require(a.graph.runtime.tensors.data() != b.graph.runtime.tensors.data() &&
            a.graph.runtime.tensors[2].device.buffer == b.graph.runtime.tensors[2].device.buffer &&
            a.graph.runtime.tensors[8].device.buffer == b.graph.runtime.tensors[8].device.buffer,
            "frame storage independent but both layer states shared");
        // Both windows share each layer's persistent state, as in a real
        // layer-major causal schedule. Independent runtime intermediates only.
        for (int pass = 0; pass < 3; ++pass) {
            a.input(pass == 1 ? 19 : 17, pass + 1);
            b.input(pass == 1 ? 7 : 11, pass + 9);
            write(backend, state0, initial0); write(backend, state1, initial1);
            a.full(); const auto expected_a = a.output();
            b.full(); const auto expected_b = b.output();
            const auto expected_state0 = read(backend, state0), expected_state1 = read(backend, state1);
            write(backend, state0, initial0); write(backend, state1, initial1);
            OuterGraph outer(backend);
            a.segment(0, cut); b.segment(0, cut);
            a.segment(cut, 12); b.segment(cut, 12);
            outer.finish();
            equal(a.output(), expected_a, "interleaved A output byte exact");
            equal(b.output(), expected_b, "interleaved B output byte exact");
            equal(read(backend, state0), expected_state0, "shared layer 0 causal state byte exact");
            equal(read(backend, state1), expected_state1, "shared layer 1 causal state byte exact");
            // Carry final state into the next changed-input/shape pass.
            initial0 = expected_state0; initial1 = expected_state1;
        }
    }
    for (Tensor* t : {&weight, &state0, &state1}) backend.free_output(*t, nullptr);
}

void test_private_boundaries(CudaBackend& backend) {
    Graph source, a, b;
    source.nodes.resize(7); source.runtime.tensors.resize(7);
    const char* names[] = {"hidden", "mask", "cos", "sin"};
    for (int i = 0; i < 7; ++i) {
        auto& n = source.nodes[i]; n.id = i; n.out_shape[0] = 16; n.out_shape[1] = 1;
        if (i < 4) { n.op_type = OpType::INPUT; n.params.str = {names[i]}; }
        else n.op_type = OpType::ADD;
    }
    source.nodes[4].inputs = {0, 1}; source.nodes[5].inputs = {2, 3};
    source.nodes[6].inputs = {4, 5}; source.graph_inputs = {0, 1, 2, 3}; source.graph_outputs = {6};
    std::vector<std::vector<float>> host(4, std::vector<float>(16, 100.f));
    for (int i = 0; i < 4; ++i) {
        auto& t = source.runtime.tensors[i];
        t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 16, 1, 1, 1, host[i].data());
        backend.upload_input(t, names[i], host[i].data(), t.nbytes());
    }
    require(initialize_execution_frame(source, a, {0, 1, 2, 3}) &&
        initialize_execution_frame(source, b, {0, 1, 2, 3}), "clone named boundary frames");
    ExecContext ca, cb;
    ca.graph = &a; ca.pool = &a.runtime.pool; ca.backend = &backend;
    cb.graph = &b; cb.pool = &b.runtime.pool; cb.backend = &backend;
    prepare_execution(ca); prepare_execution(cb);
    for (int i = 0; i < 4; ++i) {
        a.runtime.tensors[i] = allocate(backend, 16, 1);
        b.runtime.tensors[i] = allocate(backend, 16, 1);
        write(backend, a.runtime.tensors[i], std::vector<float>(16, float(i + 1)));
    }
    backend.begin_graph(); execute_graph_segment(ca, 0, 5); backend.end_graph();
    // Upload B only after A suspends. Also overwrite all shared named slots
    // deliberately, demonstrating that neither private frame uses those slots.
    for (int i = 0; i < 4; ++i) {
        write(backend, b.runtime.tensors[i], std::vector<float>(16, float(i + 11)));
        std::fill(host[i].begin(), host[i].end(), -99.f);
        backend.upload_input(source.runtime.tensors[i], names[i], host[i].data(), host[i].size() * sizeof(float));
        require(a.runtime.tensors[i].device.buffer != b.runtime.tensors[i].device.buffer &&
            a.runtime.tensors[i].device.buffer != source.runtime.tensors[i].device.buffer,
            "private boundary allocations disjoint from other frame and name cache");
    }
    backend.begin_graph(); execute_graph_segment(cb, 0, 5); backend.end_graph();
    backend.begin_graph(); execute_graph_segment(ca, 5, 7); backend.end_graph();
    backend.begin_graph(); execute_graph_segment(cb, 5, 7); backend.end_graph();
    require(!ca.execution_failed && !cb.execution_failed && !backend.dispatch_failed(), "boundary interleaving succeeds");
    equal(read(backend, a.runtime.tensors[6]), std::vector<float>(16, 10.f), "A suspended boundary bytes preserved");
    equal(read(backend, b.runtime.tensors[6]), std::vector<float>(16, 50.f), "B private boundary bytes preserved");
    for (Graph* frame : {&a, &b}) for (auto& t : frame->runtime.tensors)
        if (t.data) backend.free_output(t, nullptr);
    // Source boundary_buffers are backend-owned, not pooled outputs.
}

void test_gdn_frames(CudaBackend& backend, int keys, int values, int liveness,
                     bool workspace, bool padded, int cut) {
    std::printf("gdn_case keys=%d values=%d liveness=%d workspace=%d padded=%d cut=%d\n",
        keys, values, liveness, workspace, padded, cut);
    const int qkv_width = (2 * keys + values) * 128, out_width = values * 128;
    Graph source, a, b;
    source.nodes.resize(15); source.runtime.tensors.resize(15);
    for (int i = 0; i < 15; ++i) {
        auto& n = source.nodes[i]; n.id = i; n.out_prec = Precision::FP32;
        n.out_shape[0] = i == 0 || i == 8 || i == 9 ? qkv_width : out_width;
        n.out_shape[1] = 24; n.dim_expr[1].kind = DIM_SEQ;
        n.op_type = OpType::INPUT;
    }
    for (int i : {1, 2}) source.nodes[i].out_shape[0] = values;
    for (int i : {4, 5, 6, 7, 11}) {
        auto& n = source.nodes[i]; n.dim_expr[1].kind = DIM_CONST;
        if (i == 4 || i == 5 || i == 6) n.op_type = OpType::CONSTANT;
    }
    source.nodes[8].op_type = OpType::ADD; source.nodes[8].inputs = {0, 0};
    source.nodes[9].op_type = OpType::RESHAPE; source.nodes[9].inputs = {8};
    for (int i : {10, 12}) {
        auto& n = source.nodes[i]; n.op_type = OpType::GATED_DELTANET_PREFILL;
        n.inputs = {9, 1, 2, uint32_t(i == 10 ? 3 : 10), 4, 5, 6, uint32_t(i == 10 ? 7 : 11)};
        n.params.i32 = {keys, 128, 128, 24, 3, 4, 17, values};
        n.params.f32 = {1e-6f, 1e-6f, .08838834764831845f};
    }
    source.nodes[13].op_type = OpType::ADD; source.nodes[13].inputs = {12, 12};
    source.nodes[14].op_type = OpType::RESHAPE; source.nodes[14].inputs = {13};
    source.graph_inputs = {0, 1, 2, 3, 7, 11}; source.graph_outputs = {14};
    for (int i : {4, 5, 6, 7, 11}) {
        auto& t = source.runtime.tensors[i];
        t = allocate(backend, i == 6 || i == 7 || i == 11 ? 128 : values,
                     i == 7 || i == 11 ? 128 * values : 1);
        std::vector<float> data(t.nelements());
        for (size_t j = 0; j < data.size(); ++j)
            data[j] = i == 4 ? -.9f : i == 5 ? -.2f : i == 6 ? .75f :
                i == 7 ? float(int(j % 19) - 9) / 128.f : float(int(j % 23) - 11) / 96.f;
        write(backend, t, data);
    }
    require(initialize_execution_frame(source, a, {0, 1, 2, 3}) &&
            initialize_execution_frame(source, b, {0, 1, 2, 3}), "clone GDN runtime frames");
    ExecContext ca, cb;
    ca.graph = &a; ca.pool = &a.runtime.pool; ca.backend = &backend;
    cb.graph = &b; cb.pool = &b.runtime.pool; cb.backend = &backend;
    for (ExecContext* c : {&ca, &cb}) {
        c->conservative_stateful_liveness = liveness != 0;
        c->stateful_readonly_last_use = liveness == 2;
        c->reuse_same_shape_workspace = workspace;
        prepare_execution(*c);
    }
    auto execute = [&](ExecContext& c, int first, int end, bool full) {
        if (!outer_graph_active) backend.begin_graph();
        if (full) execute_graph(c); else execute_graph_segment(c, first, end);
        if (!outer_graph_active) backend.end_graph();
        require(!c.execution_failed && !backend.dispatch_failed(), "native GDN frame execution");
    };
    auto state0 = read(backend, source.runtime.tensors[7]);
    auto state1 = read(backend, source.runtime.tensors[11]);
    for (int pass = 0; pass < 3; ++pass) {
        for (ExecContext* c : {&ca, &cb}) {
            const int real = c == &ca ? (pass == 1 ? 19 : 17) : (pass == 1 ? 7 : 11);
            const int rows = padded ? 24 : real;
            c->runtime_seq_len = real; c->static_padded = padded; c->padded_seq_len = padded ? rows : -1;
            for (int i = 0; i < 4; ++i) {
                auto& t = c->graph->runtime.tensors[i];
                if (t.data) backend.free_output(t, nullptr);
                t = allocate(backend, source.nodes[i].out_shape[0], rows);
                std::vector<float> data(t.nelements());
                for (size_t j = 0; j < data.size(); ++j)
                    data[j] = float(int((j * 7 + pass * 13 + i * 3 + (c == &ca ? 1 : 5)) % 37) - 18) / 128.f;
                write(backend, t, data);
            }
            inject_runtime_shapes(*c);
        }
        write(backend, source.runtime.tensors[7], state0); write(backend, source.runtime.tensors[11], state1);
        execute(ca, 0, 15, true); const auto expected_a = read(backend, a.runtime.tensors[14]);
        execute(cb, 0, 15, true); const auto expected_b = read(backend, b.runtime.tensors[14]);
        const auto expected_state0 = read(backend, source.runtime.tensors[7]);
        const auto expected_state1 = read(backend, source.runtime.tensors[11]);
        write(backend, source.runtime.tensors[7], state0); write(backend, source.runtime.tensors[11], state1);
        OuterGraph outer(backend);
        execute(ca, 0, cut, false); execute(cb, 0, cut, false);
        execute(ca, cut, 15, false); execute(cb, cut, 15, false);
        outer.finish();
        equal(read(backend, a.runtime.tensors[14]), expected_a, "GDN window A output byte exact");
        equal(read(backend, b.runtime.tensors[14]), expected_b, "GDN window B output byte exact");
        equal(read(backend, source.runtime.tensors[7]), expected_state0, "GDN layer 0 shared state byte exact");
        equal(read(backend, source.runtime.tensors[11]), expected_state1, "GDN layer 1 shared state byte exact");
        state0 = expected_state0; state1 = expected_state1;
    }
    for (Graph* frame : {&a, &b}) for (int i : {0, 1, 2, 3, 8, 10, 12, 13}) {
        auto& t = frame->runtime.tensors[i]; if (t.data) backend.free_output(t, nullptr);
    }
    for (int i : {4, 5, 6, 7, 11}) backend.free_output(source.runtime.tensors[i], nullptr);
}
void test_sdpa_frames(CudaBackend& backend, int dim, int heads, bool half,
                      int liveness, bool workspace, int cut) {
    constexpr int capacity = 128;
    const size_t element_bytes = half ? 2 : 4;
    const size_t payload_bytes = size_t(dim) * capacity * element_bytes;
    Graph source, a, b;
    source.nodes.resize(15); source.runtime.tensors.resize(15);
    for (int i = 0; i < 15; ++i) {
        auto& n = source.nodes[i]; n.id = i; n.out_prec = Precision::FP32;
        n.out_shape[0] = dim; n.out_shape[1] = 17;
        n.out_shape[2] = i == 0 || i == 6 || i == 7 || i == 10 || i >= 11 ? heads : 1;
        n.dim_expr[1].kind = DIM_SEQ; n.op_type = OpType::INPUT;
    }
    for (int i : {3, 4, 5, 8, 9}) source.nodes[i].dim_expr[1].kind = DIM_CONST;
    source.nodes[6].op_type = OpType::ADD; source.nodes[6].inputs = {0, 0};
    source.nodes[7].op_type = OpType::RESHAPE; source.nodes[7].inputs = {6};
    for (int i : {10, 12}) {
        auto& n = source.nodes[i]; n.op_type = OpType::SDPA;
        n.inputs = {uint32_t(i == 10 ? 7 : 10), 1, 2, 3,
                    uint32_t(i == 10 ? 4 : 8), uint32_t(i == 10 ? 5 : 9)};
        n.params.i32 = {2, 1, heads, 1, dim, dim}; n.params.f32 = {1.f / std::sqrt(float(dim))};
    }
    source.nodes[11].op_type = OpType::ADD; source.nodes[11].inputs = {0, 0};
    source.nodes[13].op_type = OpType::ADD; source.nodes[13].inputs = {12, 12};
    source.nodes[14].op_type = OpType::RESHAPE; source.nodes[14].inputs = {13};
    source.graph_inputs = {0, 1, 2, 3, 4, 5, 8, 9}; source.graph_outputs = {14};
    std::vector<std::vector<uint8_t>> initial(4, std::vector<uint8_t>(payload_bytes, 0));
    const int cache_ids[] = {4, 5, 8, 9};
    for (int index = 0; index < 4; ++index) {
        auto& t = source.runtime.tensors[cache_ids[index]];
        const size_t bytes = CacheMetadata::SIZE + payload_bytes;
        t = Tensor::create(half ? Precision::FP16 : Precision::FP32, MemoryType::EXTERNAL, bytes / element_bytes);
        backend.alloc_persistent(t, bytes, PersistentHostAccess::HOST_AUTHORITATIVE_PREFIX, CacheMetadata::SIZE);
        require(t.data != nullptr, "allocate shared KV cache");
        // Nonempty, distinct layer/K/V prefixes; untouched capacity is zero.
        for (int j = 0; j < 3 * dim; ++j) {
            if (half) {
                const uint16_t value = uint16_t(0x2800 + index * 0x400);
                std::memcpy(initial[index].data() + j * element_bytes, &value, element_bytes);
            } else {
                const float value = std::ldexp(.03125f, index);
                std::memcpy(initial[index].data() + j * element_bytes, &value, element_bytes);
            }
        }
    }
    require(initialize_execution_frame(source, a, {0, 1, 2, 3}) &&
            initialize_execution_frame(source, b, {0, 1, 2, 3}), "clone SDPA runtime frames");
    ExecContext ca, cb;
    ca.graph = &a; ca.pool = &a.runtime.pool; ca.backend = &backend;
    cb.graph = &b; cb.pool = &b.runtime.pool; cb.backend = &backend;
    for (ExecContext* c : {&ca, &cb}) {
        c->conservative_stateful_liveness = liveness != 0;
        c->stateful_readonly_last_use = liveness == 2;
        c->reuse_same_shape_workspace = workspace; prepare_execution(*c);
    }
    if (liveness == 2) {
        require(backend.independent_stateful_mutable_inputs(a.nodes[10]) ==
                ((uint64_t(1) << 4) | (uint64_t(1) << 5)), "SDPA declares both mutable caches");
        auto releases = [&](uint32_t id) {
            for (const auto& batch : ca.release_queue)
                if (std::find(batch.begin(), batch.end(), id) != batch.end()) return true;
            return false;
        };
        require(releases(6) && releases(10) && releases(12),
                "SDPA last-use planner releases Q owner and independent layer outputs");
        for (int id : cache_ids) require(!releases(uint32_t(id)), "SDPA planner preserves both K/V states");
    }
    auto metadata = [&](int past) {
        CacheMetadata m{}; m.current_seq_len = past; m.max_seq_len = capacity;
        m.num_kv_heads = 1; m.head_dim = dim; m.v_head_dim = dim;
        for (int id : cache_ids)
            require(backend.copy_from_host(&m, source.runtime.tensors[id], sizeof(m)), "restore window KV metadata");
    };
    auto restore = [&] {
        for (int i = 0; i < 4; ++i)
            require(backend.copy_from_host(initial[i].data(), source.runtime.tensors[cache_ids[i]],
                payload_bytes, CacheMetadata::SIZE), "restore initial KV payload");
    };
    auto payloads = [&] {
        std::vector<std::vector<uint8_t>> result(4, std::vector<uint8_t>(payload_bytes));
        for (int i = 0; i < 4; ++i)
            require(backend.copy_to_host(source.runtime.tensors[cache_ids[i]], result[i].data(),
                payload_bytes, CacheMetadata::SIZE), "read complete KV payload including untouched tail");
        return result;
    };
    auto execute = [&](ExecContext& c, int past, int first, int end, bool full) {
        // KV payload is shared, but its current_seq_len describes THIS
        // window, not whichever window last used this backend.
        metadata(past); if (!outer_graph_active) backend.begin_graph();
        if (full) execute_graph(c); else execute_graph_segment(c, first, end);
        if (!outer_graph_active) backend.end_graph();
        require(!c.execution_failed && !backend.dispatch_failed(), "native causal SDPA frame execution");
    };
    int past = 3;
    for (int pass = 0; pass < 3; ++pass) {
        const int rows_a = pass == 1 ? 19 : 17, rows_b = pass == 1 ? 7 : 11;
        for (ExecContext* c : {&ca, &cb}) {
            const int rows = c == &ca ? rows_a : rows_b;
            for (int i = 0; i < 3; ++i) {
                auto& t = c->graph->runtime.tensors[i];
                if (t.data) backend.free_output(t, nullptr);
                t = allocate(backend, dim, rows, i == 0 ? heads : 1);
                std::vector<float> data(t.nelements());
                for (size_t j = 0; j < data.size(); ++j)
                    data[j] = float(int((j * 7 + pass * 13 + i * 3 + (c == &ca ? 1 : 11)) % 43) - 21) / 64.f;
                write(backend, t, data);
            }
            auto& mask = c->graph->runtime.tensors[3];
            if (mask.data) backend.free_output(mask, nullptr);
            const int window_past = c == &ca ? past : past + rows_a;
            const int keys = window_past + rows;
            mask = allocate(backend, keys, rows);
            std::vector<float> mask_data(size_t(keys) * rows);
            for (int q = 0; q < rows; ++q) for (int k = 0; k < keys; ++k)
                mask_data[size_t(q) * keys + k] = k <= window_past + q ? 0.f : -INFINITY;
            write(backend, mask, mask_data);
            c->runtime_seq_len = rows; inject_runtime_shapes(*c);
        }
        restore(); execute(ca, past, 0, 15, true); const auto expected_a = read(backend, a.runtime.tensors[14]);
        execute(cb, past + rows_a, 0, 15, true); const auto expected_b = read(backend, b.runtime.tensors[14]);
        const auto expected_payload = payloads();
        restore(); OuterGraph outer(backend);
        execute(ca, past, 0, cut, false); execute(cb, past + rows_a, 0, cut, false);
        execute(ca, past, cut, 15, false); execute(cb, past + rows_a, cut, 15, false);
        outer.finish();
        equal(read(backend, a.runtime.tensors[14]), expected_a, "SDPA A output byte exact");
        equal(read(backend, b.runtime.tensors[14]), expected_b, "SDPA B output byte exact");
        require(payloads() == expected_payload, "both layers full K/V payload byte exact");
        past += rows_a + rows_b; metadata(past);
        for (int id : cache_ids) {
            CacheMetadata actual{};
            require(backend.copy_to_host(source.runtime.tensors[id], &actual, sizeof(actual)) &&
                actual.current_seq_len == uint64_t(past) && actual.max_seq_len == capacity,
                "final cache metadata advances to both windows' end");
        }
        initial = expected_payload;
    }
    for (Graph* frame : {&a, &b}) for (int i : {0, 1, 2, 3, 6, 10, 11, 12, 13}) {
        auto& t = frame->runtime.tensors[i]; if (t.data) backend.free_output(t, nullptr);
    }
    // Persistent KV allocations and host metadata mirrors are backend-owned.
}
} // namespace

void test_qk_rope_frames(CudaBackend& backend, int dim, int heads, int liveness, bool workspace) {
    Graph source, a, b;
    source.nodes.resize(10); source.runtime.tensors.resize(10);
    for (int i = 0; i < 10; ++i) {
        auto& n = source.nodes[i]; n.id = i; n.out_prec = Precision::FP32;
        n.op_type = i < 6 ? OpType::INPUT : OpType::ADD;
        n.out_shape[0] = dim; n.out_shape[1] = 17;
        n.dim_expr[1].kind = DIM_SEQ;
    }
    for (int i : {0, 6}) {
        source.nodes[i].out_shape[1] *= heads;
        source.nodes[i].dim_expr[1].kind = DIM_MUL;
        source.nodes[i].dim_expr[1].coeff = heads;
    }
    for (int i : {2, 3}) {
        source.nodes[i].out_shape[1] = 1; source.nodes[i].dim_expr[1].kind = DIM_CONST;
        source.runtime.tensors[i] = allocate(backend, dim, 1);
        write(backend, source.runtime.tensors[i], std::vector<float>(dim, 1.f));
    }
    for (int i : {4, 5}) source.nodes[i].out_shape[0] = dim / 2;
    source.nodes[6].inputs = {0, 0};
    source.nodes[7].op_type = OpType::QK_RMS_NORM_ROPE;
    source.nodes[7].inputs = {6, 1, 2, 3, 4, 5};
    source.nodes[7].params.i32 = {dim, 0, heads}; source.nodes[7].params.f32 = {1e-6f};
    source.nodes[8].inputs = {7, 7}; source.nodes[9].op_type = OpType::RESHAPE;
    source.nodes[9].inputs = {8};
    for (int i : {7, 8, 9}) source.nodes[i].out_shape[2] = heads + 1;
    source.graph_inputs = {0, 1, 2, 3, 4, 5}; source.graph_outputs = {9};
    require(initialize_execution_frame(source, a, {0, 1, 4, 5}) &&
            initialize_execution_frame(source, b, {0, 1, 4, 5}), "clone QK norm/rope frames");
    ExecContext ca, cb;
    ca.graph = &a; ca.pool = &a.runtime.pool; ca.backend = &backend;
    cb.graph = &b; cb.pool = &b.runtime.pool; cb.backend = &backend;
    for (ExecContext* c : {&ca, &cb}) {
        c->conservative_stateful_liveness = liveness != 0;
        c->stateful_readonly_last_use = liveness == 2;
        c->reuse_same_shape_workspace = workspace; prepare_execution(*c, false);
        if (liveness == 2) {
            require(backend.independent_stateful_mutable_inputs(c->graph->nodes[7]) == 0,
                    "QK norm/rope explicitly read-only");
            require(std::find(c->release_queue[7].begin(), c->release_queue[7].end(), 6) !=
                    c->release_queue[7].end(), "QK query owner released after normalization");
            require(std::find(c->release_queue[8].begin(), c->release_queue[8].end(), 7) !=
                    c->release_queue[8].end(), "QK independent output released after consumer");
        }
    }
    auto input = [&](ExecContext& c, int rows, int seed) {
        for (int id : {0, 1, 4, 5}) {
            Tensor& t = c.graph->runtime.tensors[id];
            if (t.data) backend.free_output(t, nullptr);
            const int width = id >= 4 ? dim / 2 : dim;
            t = allocate(backend, width, id == 0 ? rows * heads : rows);
            std::vector<float> values(t.nelements());
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = id == 4 ? .8f : id == 5 ? .6f : float(int((i * 13 + seed) % 97) - 48) / 128.f;
            write(backend, t, values);
        }
        c.runtime_seq_len = rows; inject_runtime_shapes(c);
    };
    auto execute = [&](ExecContext& c, int first, int end) {
        if (!outer_graph_active) backend.begin_graph();
        if (first < 0) execute_graph(c); else execute_graph_segment(c, first, end);
        if (!outer_graph_active) backend.end_graph();
        require(!c.execution_failed && !backend.dispatch_failed(), "native QK norm/rope execution");
    };
    for (int pass = 0; pass < 3; ++pass) {
        input(ca, pass == 1 ? 19 : 17, pass + 1);
        input(cb, pass == 1 ? 7 : 11, pass + 9);
        execute(ca, -1, 0); const auto expected_a = read(backend, a.runtime.tensors[9]);
        execute(cb, -1, 0); const auto expected_b = read(backend, b.runtime.tensors[9]);
        OuterGraph outer(backend);
        execute(ca, 0, 8); execute(cb, 0, 8);
        execute(ca, 8, 10); execute(cb, 8, 10);
        outer.finish();
        equal(read(backend, a.runtime.tensors[9]), expected_a, "QK A output byte exact");
        equal(read(backend, b.runtime.tensors[9]), expected_b, "QK B output byte exact");
    }
    for (Graph* g : {&a, &b}) {
        for (int id : {6, 7, 8})
            if (g->runtime.tensors[id].data) backend.free_output(g->runtime.tensors[id], nullptr);
        for (int id : {0, 1, 4, 5}) backend.free_output(g->runtime.tensors[id], nullptr);
    }
    for (int id : {2, 3}) backend.free_output(source.runtime.tensors[id], nullptr);
}

int main() {
    CudaBackend backend;
    if (!backend.available()) return 77;
    require(backend.set_operator_fallback_policy(OperatorFallbackPolicy::REQUIRE_NATIVE), "native-only policy");
    test_private_boundaries(backend);
    for (int dim : {64, 256}) for (int heads : {2, 24}) for (int liveness : {0, 1, 2})
        for (bool workspace : {false, true}) test_qk_rope_frames(backend, dim, heads, liveness, workspace);
    std::puts("CUDA QK RMSNorm/RoPE:72 paired private-window passes, read-only operands, byte-exact outputs and owner last-use");
    for (int width : {19, 2560}) for (int liveness : {0, 1, 2})
        for (bool workspace : {false, true}) for (int cut : {4, 6, 8})
            run(backend, width, liveness, workspace, cut);
    for (int keys : {1, 3, 16}) for (int liveness : {0, 1, 2})
        for (bool workspace : {false, true}) for (bool padded : {false, true})
            for (int cut : {9, 11, 12})
                test_gdn_frames(backend, keys, keys == 16 ? 48 : keys * 2, liveness, workspace, padded, cut);
    std::puts("CUDA GDN segmented execution: 324 paired two-layer passes; cloned frames, shared causal states, byte-exact output/state, dynamic and padded");
    for (int dim : {4, 128, 256}) for (int heads : {1, 2}) for (bool half : {false, true})
        for (int liveness : {0, 1, 2}) for (bool workspace : {false, true})
            for (int cut : {8, 11}) test_sdpa_frames(backend, dim, heads, half, liveness, workspace, cut);
    std::puts("CUDA SDPA segmented execution: 432 paired two-layer passes; per-window past metadata and private causal masks, nonempty prefix, GQA, FP16/FP32 KV, byte-exact outputs/full payload");
    std::puts("CUDA segmented execution: 108 paired window passes, native-only; outputs and both shared layer states byte exact");
    return 0;
}
