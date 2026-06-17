#pragma once

#include "graph/graph.h"
#include <vector>

class ThreadPool;

// ---------------------------------------------------------------------------
// PROJECT_NAME — Execution context
//
// prepare_execution() computes liveness information from the graph.
// execute_graph() runs one full pass.
// ---------------------------------------------------------------------------

struct NodeProfileStat {
    uint32_t node_id = 0;
    OpType op_type = OpType::INPUT;
    uint64_t calls = 0;
    uint64_t total_ns = 0;
};

struct ExecContext {
    Graph*        graph   = nullptr;
    BufferPool*   pool    = nullptr;
    ThreadPool*   thread_pool = nullptr;
    bool          profile_enabled = false;

    // liveness: release_queue[i] = nodes that can be freed after node i
    std::vector<std::vector<uint32_t>> release_queue;
    std::vector<NodeProfileStat> profile_stats;
};

void reset_profile_stats(ExecContext& ctx);

/// Compute use_count + last_use + release_queue from the graph topology.
/// Must be called once after graph_load().
void prepare_execution(ExecContext& ctx);

/// Execute the graph once.  Input tensors must already have data set.
/// Allocates intermediate tensors from the BufferPool.
void execute_graph(ExecContext& ctx);
