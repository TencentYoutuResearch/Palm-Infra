#pragma once

namespace mollm::hc {
// Shared capacity contract for CUDA HC producers, consumers, and restoration.
// This does not authorize larger engine chunks or attention query lengths.
inline constexpr int prefill_max_rows = 2048;
}
