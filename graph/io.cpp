#include "graph/graph.h"

#include <cstdio>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// binary format (little-endian)
//
//   Offset  Size   Field
//   ------  ----   -----
//   0       4      magic: 0x4D4C4C47 ("GLLM")
//   4       4      version (current = 3)
//                      v1: initial
//                      v2: added metadata in header
//                      v3: added per-node dim_expr[4] (DimExpr)
//   8       4      node_count
//   12      4      input_count
//   16      4*N    graph_inputs[node_id...]
//   *       4      output_count
//   *       4*N    graph_outputs[node_id...]
//   --- per node ---
//   *       4      node_id
//   *       4      op_type (uint32)
//   *       4      input_count
//   *       4*N    input_ids[]
//   *       4*4    out_shape[4] (int64)
//   *       8*4    dim_expr[4] (8 bytes each: kind + 3 pad + coeff int32)  [v3+]
//   *       4      out_prec (uint32)
//   *       4      param_i32_count
//   *       4*N    param_i32[] (int32)
//   *       4      param_f32_count
//   *       4*N    param_f32[] (float)
//   *       4      param_str_count
//   *       per-str: 4(len) + data
//
// All integers are stored in native byte order (little-endian on all
// supported platforms).
//
// Not forward-compatible with v2. Old .mollm/.graph files must be re-
// transpiled.
// ---------------------------------------------------------------------------

namespace {

static constexpr uint32_t GRAPH_MAGIC   = 0x4D4C4C47;  // "GLLM"
static constexpr uint32_t GRAPH_VERSION = 3;  // v3: added per-node dynamic[4]

// ---- low-level I/O helpers ----

static bool read_u32(FILE* f, uint32_t* v) { return fread(v, 4, 1, f) == 1; }
static bool read_i64(FILE* f, int64_t* v)  { return fread(v, 8, 1, f) == 1; }

static bool write_u32(FILE* f, uint32_t v) { return fwrite(&v, 4, 1, f) == 1; }
static bool write_i64(FILE* f, int64_t v)  { return fwrite(&v, 8, 1, f) == 1; }

static bool read_vec_u32(FILE* f, std::vector<uint32_t>& v, uint32_t count) {
    v.resize(count);
    return fread(v.data(), 4, count, f) == count;
}

static bool write_vec_u32(FILE* f, const std::vector<uint32_t>& v) {
    return fwrite(v.data(), 4, v.size(), f) == v.size();
}

static bool read_str(FILE* f, std::string& s) {
    uint32_t len = 0;
    if (!read_u32(f, &len)) return false;
    s.resize(len);
    if (len > 0 && fread(&s[0], 1, len, f) != len) return false;
    return true;
}

static bool write_str(FILE* f, const std::string& s) {
    uint32_t len = (uint32_t)s.size();
    if (!write_u32(f, len)) return false;
    if (len > 0 && fwrite(s.data(), 1, len, f) != len) return false;
    return true;
}

} // anonymous namespace

// =========================================================================
// graph_load
// =========================================================================

bool graph_load(Graph& g, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "graph_load: cannot open %s\n", path);
        return false;
    }

    // header
    uint32_t magic = 0, version = 0, node_count = 0;
    if (!read_u32(f, &magic) || magic != GRAPH_MAGIC) {
        fprintf(stderr, "graph_load: bad magic 0x%08x\n", magic);
        fclose(f); return false;
    }
    if (!read_u32(f, &version) || version != GRAPH_VERSION) {
        fprintf(stderr, "graph_load: unsupported version %u (expected %u)\n", version, GRAPH_VERSION);
        fclose(f); return false;
    }
    if (!read_u32(f, &node_count)) { fclose(f); return false; }

    // metadata (key=value string pairs)
    uint32_t meta_count = 0;
    if (!read_u32(f, &meta_count)) { fclose(f); return false; }
    for (uint32_t i = 0; i < meta_count; i++) {
        std::string key, val;
        if (!read_str(f, key) || !read_str(f, val)) { fclose(f); return false; }
        g.metadata[key] = val;
    }

    // graph inputs
    uint32_t input_count = 0;
    if (!read_u32(f, &input_count)) { fclose(f); return false; }
    if (!read_vec_u32(f, g.graph_inputs, input_count)) { fclose(f); return false; }

    // graph outputs
    uint32_t output_count = 0;
    if (!read_u32(f, &output_count)) { fclose(f); return false; }
    if (!read_vec_u32(f, g.graph_outputs, output_count)) { fclose(f); return false; }

    // nodes
    g.nodes.resize(node_count);
    for (uint32_t i = 0; i < node_count; i++) {
        auto& n = g.nodes[i];

        if (!read_u32(f, &n.id))       { fclose(f); return false; }
        uint32_t op = 0;
        if (!read_u32(f, &op))         { fclose(f); return false; }
        n.op_type = static_cast<OpType>(op);

        // inputs
        uint32_t ic = 0;
        if (!read_u32(f, &ic))         { fclose(f); return false; }
        if (!read_vec_u32(f, n.inputs, ic)) { fclose(f); return false; }

        // output shape (always 4 int64s)
        for (int d = 0; d < 4; d++) {
            if (!read_i64(f, &n.out_shape[d])) { fclose(f); return false; }
        }

        // dim_expr[4] (4 × 8 bytes: kind + coeff + pad) — v3+
        // Format per dim: 1 byte kind, 3 bytes padding, 4 bytes coeff (int32)
        for (int d = 0; d < 4; d++) {
            uint8_t buf[8];
            if (fread(buf, 1, 8, f) != 8) { fclose(f); return false; }
            n.dim_expr[d].kind  = static_cast<int8_t>(buf[0]);
            int32_t coeff;
            std::memcpy(&coeff, buf + 4, 4);
            n.dim_expr[d].coeff = coeff;
        }

        // output precision
        uint32_t prec = 0;
        if (!read_u32(f, &prec))       { fclose(f); return false; }
        n.out_prec = static_cast<Precision>(prec);

        // param i32
        uint32_t pi32_count = 0;
        if (!read_u32(f, &pi32_count)) { fclose(f); return false; }
        n.params.i32.resize(pi32_count);
        if (pi32_count > 0 && fread(n.params.i32.data(), 4, pi32_count, f) != pi32_count) {
            fclose(f); return false;
        }

        // param f32
        uint32_t pf32_count = 0;
        if (!read_u32(f, &pf32_count)) { fclose(f); return false; }
        n.params.f32.resize(pf32_count);
        if (pf32_count > 0 && fread(n.params.f32.data(), 4, pf32_count, f) != pf32_count) {
            fclose(f); return false;
        }

        // param str
        uint32_t ps_count = 0;
        if (!read_u32(f, &ps_count))   { fclose(f); return false; }
        n.params.str.resize(ps_count);
        for (uint32_t s = 0; s < ps_count; s++) {
            if (!read_str(f, n.params.str[s])) { fclose(f); return false; }
        }
    }

    // initialise runtime tensors
    g.runtime.tensors.resize(node_count);

    fclose(f);
    return true;
}

// =========================================================================
// graph_save
// =========================================================================

bool graph_save(const Graph& g, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "graph_save: cannot open %s\n", path);
        return false;
    }

    uint32_t node_count = (uint32_t)g.nodes.size();
    uint32_t input_count = (uint32_t)g.graph_inputs.size();
    uint32_t output_count = (uint32_t)g.graph_outputs.size();

    // header
    write_u32(f, GRAPH_MAGIC);
    write_u32(f, GRAPH_VERSION);
    write_u32(f, node_count);

    // metadata
    uint32_t meta_count = (uint32_t)g.metadata.size();
    write_u32(f, meta_count);
    for (const auto& [key, val] : g.metadata) {
        write_str(f, key);
        write_str(f, val);
    }

    write_u32(f, input_count);
    write_vec_u32(f, g.graph_inputs);
    write_u32(f, output_count);
    write_vec_u32(f, g.graph_outputs);

    // nodes
    for (const auto& n : g.nodes) {
        write_u32(f, n.id);
        write_u32(f, static_cast<uint32_t>(n.op_type));

        uint32_t ic = (uint32_t)n.inputs.size();
        write_u32(f, ic);
        write_vec_u32(f, n.inputs);

        // output shape (4 int64s)
        for (int d = 0; d < 4; d++) write_i64(f, n.out_shape[d]);

        // dim_expr[4] (4 × 8 bytes: kind + 3 pad + coeff int32) — v3+
        for (int d = 0; d < 4; d++) {
            uint8_t buf[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            buf[0] = static_cast<uint8_t>(n.dim_expr[d].kind);
            std::memcpy(buf + 4, &n.dim_expr[d].coeff, 4);
            fwrite(buf, 1, 8, f);
        }

        write_u32(f, static_cast<uint32_t>(n.out_prec));

        // param i32
        uint32_t pi32_count = (uint32_t)n.params.i32.size();
        write_u32(f, pi32_count);
        if (pi32_count > 0) fwrite(n.params.i32.data(), 4, pi32_count, f);

        // param f32
        uint32_t pf32_count = (uint32_t)n.params.f32.size();
        write_u32(f, pf32_count);
        if (pf32_count > 0) fwrite(n.params.f32.data(), 4, pf32_count, f);

        // param str
        uint32_t ps_count = (uint32_t)n.params.str.size();
        write_u32(f, ps_count);
        for (uint32_t s = 0; s < ps_count; s++) {
            write_str(f, n.params.str[s]);
        }
    }

    fclose(f);
    return true;
}
