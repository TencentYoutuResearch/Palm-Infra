#pragma once
#include "core/qsa_limits.h"
#include "graph/graph.h"
#include <string>

// Transactional rewrite of an unloaded Qwen4Exp text graph. No package writes.
// Call before loading constants or preparing the execution plan.
bool attach_qsa_companion(Graph& graph, const std::string& directory,
                          int context, std::string& error);
