#include "graph/graph.h"

#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Binary graph format (native little-endian).
//
// Header:
//   magic, version, node_count, metadata_count, metadata pairs,
//   graph_input_count + ids, graph_output_count + ids.
//
// Each node stores its id, op, input ids, four int64 dimensions, four
// serialized DimExpr values, precision, and three parameter arrays.
// ---------------------------------------------------------------------------

namespace {

constexpr uint32_t GRAPH_MAGIC = 0x4D4C4C47;  // "GLLM"
constexpr uint32_t GRAPH_VERSION = 3;

constexpr uint32_t MAX_GRAPH_NODES = 1'000'000;
constexpr uint32_t MAX_COLLECTION_ITEMS = 16'000'000;
constexpr uint32_t MAX_METADATA_ITEMS = 1'000'000;
constexpr uint32_t MAX_STRING_BYTES = 64 * 1024 * 1024;
constexpr uint32_t MIN_SERIALIZED_NODE_BYTES = 92;

class BinaryReader {
public:
    explicit BinaryReader(const char* path) : file_(std::fopen(path, "rb")) {
        if (!file_)
            return;
        if (std::fseek(file_, 0, SEEK_END) != 0) {
            close();
            return;
        }
        const long size = std::ftell(file_);
        if (size < 0 || std::fseek(file_, 0, SEEK_SET) != 0) {
            close();
            return;
        }
        remaining_ = static_cast<uint64_t>(size);
    }

    ~BinaryReader() { close(); }

    BinaryReader(const BinaryReader&) = delete;
    BinaryReader& operator=(const BinaryReader&) = delete;

    bool is_open() const { return file_ != nullptr; }
    uint64_t remaining() const { return remaining_; }

    template <typename T>
    bool read(T& value) {
        return read_bytes(&value, sizeof(value));
    }

    bool read_bytes(void* dst, size_t size) {
        if (!file_ || size > remaining_)
            return false;
        if (size != 0 && std::fread(dst, 1, size, file_) != size)
            return false;
        remaining_ -= size;
        return true;
    }

    template <typename T>
    bool read_vector(std::vector<T>& values, uint32_t count) {
        if (count > MAX_COLLECTION_ITEMS ||
            count > remaining_ / sizeof(T)) {
            return false;
        }
        values.resize(count);
        return read_bytes(values.data(), static_cast<size_t>(count) *
                                             sizeof(T));
    }

    bool read_string(std::string& value) {
        uint32_t size = 0;
        if (!read(size) || size > MAX_STRING_BYTES || size > remaining_)
            return false;
        value.resize(size);
        return read_bytes(value.data(), size);
    }

private:
    void close() {
        if (file_) {
            std::fclose(file_);
            file_ = nullptr;
        }
        remaining_ = 0;
    }

    FILE* file_ = nullptr;
    uint64_t remaining_ = 0;
};

class BinaryWriter {
public:
    explicit BinaryWriter(const char* path) : file_(std::fopen(path, "wb")) {}

    ~BinaryWriter() {
        if (file_)
            std::fclose(file_);
    }

    BinaryWriter(const BinaryWriter&) = delete;
    BinaryWriter& operator=(const BinaryWriter&) = delete;

    bool is_open() const { return file_ != nullptr; }

    template <typename T>
    void write(const T& value) {
        write_bytes(&value, sizeof(value));
    }

    void write_bytes(const void* data, size_t size) {
        if (!ok_ || !file_)
            return;
        if (size != 0 && std::fwrite(data, 1, size, file_) != size)
            ok_ = false;
    }

    template <typename T>
    void write_vector(const std::vector<T>& values) {
        write_bytes(values.data(), values.size() * sizeof(T));
    }

    void write_string(const std::string& value) {
        if (value.size() > std::numeric_limits<uint32_t>::max()) {
            ok_ = false;
            return;
        }
        const uint32_t size = static_cast<uint32_t>(value.size());
        write(size);
        write_bytes(value.data(), value.size());
    }

    bool finish() {
        if (!file_)
            return false;
        const bool success = ok_ && std::fclose(file_) == 0;
        file_ = nullptr;
        return success;
    }

private:
    FILE* file_ = nullptr;
    bool ok_ = true;
};

bool valid_precision(uint32_t value) {
    return value <= static_cast<uint32_t>(Precision::INT4);
}

bool valid_dim_expr(int8_t kind) {
    return kind >= DIM_CONST && kind <= DIM_BATCH;
}

bool validate_graph(const Graph& graph, std::string& error) {
    const size_t node_count = graph.nodes.size();
    if (node_count > MAX_GRAPH_NODES) {
        error = "too many nodes";
        return false;
    }

    for (size_t index = 0; index < node_count; ++index) {
        const GraphNode& node = graph.nodes[index];
        if (node.id != index) {
            error = "node ids must match their serialized index";
            return false;
        }
        if (std::strcmp(op_type_name(node.op_type), "UNKNOWN") == 0) {
            error = "unknown op type";
            return false;
        }
        if (!valid_precision(static_cast<uint32_t>(node.out_prec))) {
            error = "unknown tensor precision";
            return false;
        }
        for (const DimExpr& expr : node.dim_expr) {
            if (!valid_dim_expr(expr.kind)) {
                error = "unknown dimension expression";
                return false;
            }
        }
        for (uint32_t input : node.inputs) {
            if (input >= index) {
                error = "node input is missing or not topologically earlier";
                return false;
            }
        }
    }

    for (uint32_t input : graph.graph_inputs) {
        if (input >= node_count) {
            error = "graph input id is out of range";
            return false;
        }
    }
    for (uint32_t output : graph.graph_outputs) {
        if (output >= node_count) {
            error = "graph output id is out of range";
            return false;
        }
    }
    return true;
}

bool read_count(BinaryReader& reader, uint32_t& count, uint32_t maximum) {
    return reader.read(count) && count <= maximum;
}

bool read_graph(BinaryReader& reader, Graph& graph) {
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t node_count = 0;
    if (!reader.read(magic) || magic != GRAPH_MAGIC ||
        !reader.read(version) || version != GRAPH_VERSION ||
        !read_count(reader, node_count, MAX_GRAPH_NODES)) {
        return false;
    }
    if (node_count > reader.remaining() / MIN_SERIALIZED_NODE_BYTES)
        return false;

    uint32_t metadata_count = 0;
    if (!read_count(reader, metadata_count, MAX_METADATA_ITEMS) ||
        metadata_count > reader.remaining() / (2 * sizeof(uint32_t))) {
        return false;
    }
    for (uint32_t i = 0; i < metadata_count; ++i) {
        std::string key;
        std::string value;
        if (!reader.read_string(key) || !reader.read_string(value))
            return false;
        graph.metadata[std::move(key)] = std::move(value);
    }

    uint32_t input_count = 0;
    if (!read_count(reader, input_count, MAX_COLLECTION_ITEMS) ||
        !reader.read_vector(graph.graph_inputs, input_count)) {
        return false;
    }

    uint32_t output_count = 0;
    if (!read_count(reader, output_count, MAX_COLLECTION_ITEMS) ||
        !reader.read_vector(graph.graph_outputs, output_count)) {
        return false;
    }

    graph.nodes.resize(node_count);
    for (GraphNode& node : graph.nodes) {
        uint32_t op = 0;
        uint32_t input_size = 0;
        if (!reader.read(node.id) || !reader.read(op) ||
            !read_count(reader, input_size, MAX_COLLECTION_ITEMS) ||
            !reader.read_vector(node.inputs, input_size)) {
            return false;
        }
        node.op_type = static_cast<OpType>(op);

        for (int64_t& dimension : node.out_shape) {
            if (!reader.read(dimension))
                return false;
        }

        for (DimExpr& expression : node.dim_expr) {
            uint8_t serialized[8];
            if (!reader.read_bytes(serialized, sizeof(serialized)))
                return false;
            expression.kind = static_cast<int8_t>(serialized[0]);
            std::memcpy(&expression.coeff, serialized + 4,
                        sizeof(expression.coeff));
        }

        uint32_t precision = 0;
        if (!reader.read(precision))
            return false;
        node.out_prec = static_cast<Precision>(precision);

        uint32_t count = 0;
        if (!read_count(reader, count, MAX_COLLECTION_ITEMS) ||
            !reader.read_vector(node.params.i32, count) ||
            !read_count(reader, count, MAX_COLLECTION_ITEMS) ||
            !reader.read_vector(node.params.f32, count) ||
            !read_count(reader, count, MAX_COLLECTION_ITEMS) ||
            count > reader.remaining() / sizeof(uint32_t)) {
            return false;
        }
        node.params.str.resize(count);
        for (std::string& value : node.params.str) {
            if (!reader.read_string(value))
                return false;
        }
    }

    std::string error;
    return validate_graph(graph, error);
}

bool fits_u32(size_t value) {
    return value <= std::numeric_limits<uint32_t>::max();
}

bool write_graph(BinaryWriter& writer, const Graph& graph) {
    if (!fits_u32(graph.nodes.size()) ||
        !fits_u32(graph.metadata.size()) ||
        !fits_u32(graph.graph_inputs.size()) ||
        !fits_u32(graph.graph_outputs.size())) {
        return false;
    }

    writer.write(GRAPH_MAGIC);
    writer.write(GRAPH_VERSION);
    writer.write(static_cast<uint32_t>(graph.nodes.size()));

    writer.write(static_cast<uint32_t>(graph.metadata.size()));
    for (const auto& [key, value] : graph.metadata) {
        writer.write_string(key);
        writer.write_string(value);
    }

    writer.write(static_cast<uint32_t>(graph.graph_inputs.size()));
    writer.write_vector(graph.graph_inputs);
    writer.write(static_cast<uint32_t>(graph.graph_outputs.size()));
    writer.write_vector(graph.graph_outputs);

    for (const GraphNode& node : graph.nodes) {
        if (!fits_u32(node.inputs.size()) ||
            !fits_u32(node.params.i32.size()) ||
            !fits_u32(node.params.f32.size()) ||
            !fits_u32(node.params.str.size())) {
            return false;
        }

        writer.write(node.id);
        writer.write(static_cast<uint32_t>(node.op_type));
        writer.write(static_cast<uint32_t>(node.inputs.size()));
        writer.write_vector(node.inputs);
        for (int64_t dimension : node.out_shape)
            writer.write(dimension);

        for (const DimExpr& expression : node.dim_expr) {
            uint8_t serialized[8] = {};
            serialized[0] = static_cast<uint8_t>(expression.kind);
            std::memcpy(serialized + 4, &expression.coeff,
                        sizeof(expression.coeff));
            writer.write_bytes(serialized, sizeof(serialized));
        }

        writer.write(static_cast<uint32_t>(node.out_prec));
        writer.write(static_cast<uint32_t>(node.params.i32.size()));
        writer.write_vector(node.params.i32);
        writer.write(static_cast<uint32_t>(node.params.f32.size()));
        writer.write_vector(node.params.f32);
        writer.write(static_cast<uint32_t>(node.params.str.size()));
        for (const std::string& value : node.params.str)
            writer.write_string(value);
    }
    return true;
}

void replace_serialized_graph(Graph& destination, Graph& source) {
    destination.runtime.pool.clear();
    destination.runtime.weights.clear();
    destination.runtime.tensors.clear();

    destination.nodes = std::move(source.nodes);
    destination.graph_inputs = std::move(source.graph_inputs);
    destination.graph_outputs = std::move(source.graph_outputs);
    destination.metadata = std::move(source.metadata);
    destination.runtime.tensors.resize(destination.nodes.size());
}

}  // namespace

bool graph_load(Graph& graph, const char* path) {
    BinaryReader reader(path);
    if (!reader.is_open()) {
        std::fprintf(stderr, "graph_load: cannot open %s\n", path);
        return false;
    }

    Graph loaded;
    if (!read_graph(reader, loaded)) {
        std::fprintf(stderr, "graph_load: invalid or truncated graph %s\n",
                     path);
        return false;
    }

    replace_serialized_graph(graph, loaded);
    return true;
}

bool graph_save(const Graph& graph, const char* path) {
    std::string error;
    if (!validate_graph(graph, error)) {
        std::fprintf(stderr, "graph_save: invalid graph: %s\n", error.c_str());
        return false;
    }

    BinaryWriter writer(path);
    if (!writer.is_open()) {
        std::fprintf(stderr, "graph_save: cannot open %s\n", path);
        return false;
    }

    if (!write_graph(writer, graph) || !writer.finish()) {
        std::fprintf(stderr, "graph_save: failed while writing %s\n", path);
        std::remove(path);
        return false;
    }
    return true;
}
