#pragma once

#include "graph/graph.h"
#include <vector>

// ---------------------------------------------------------------------------
// PROJECT_NAME — Execution context
//
// prepare_execution() computes liveness information from the graph.
// execute_graph() runs one full pass.
// ---------------------------------------------------------------------------

struct ExecContext {
    Graph*        graph   = nullptr;
    BufferPool*   pool    = nullptr;

    // liveness: release_queue[i] = nodes that can be freed after node i
    std::vector<std::vector<uint32_t>> release_queue;
};

/// Compute use_count + last_use + release_queue from the graph topology.
/// Must be called once after graph_load().
void prepare_execution(ExecContext& ctx);

/// Execute the graph once.  Input tensors must already have data set.
/// Allocates intermediate tensors from the BufferPool.
void execute_graph(ExecContext& ctx);
