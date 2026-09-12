#pragma once

#include <cstdint>
#include <string>

// Lightweight Chrome Trace / Perfetto recorder.  It is process-global because
// the I/O workers and CPU path need to contribute to the same timeline.  All
// hot-path operations are no-ops unless start() has been called.
namespace mollm_trace {

void start(const std::string& path);
bool enabled();
uint64_t now_ns();
void set_thread_name(const char* name);
void record_duration(const char* category, const char* name,
                     uint64_t start_ns, uint64_t end_ns,
                     const std::string& args_json = {},
                     const char* color_name = nullptr);
void record_flow(const char* category, const char* name, uint64_t timestamp_ns,
                 uint64_t flow_id, bool start, const std::string& args_json = {});
bool write();

class ScopedEvent {
public:
    ScopedEvent(const char* category, const char* name,
                const std::string& args_json = {},
                const char* color_name = nullptr);
    ~ScopedEvent();
    ScopedEvent(const ScopedEvent&) = delete;
    ScopedEvent& operator=(const ScopedEvent&) = delete;

private:
    const char* category_ = nullptr;
    const char* name_ = nullptr;
    std::string args_json_;
    const char* color_name_ = nullptr;
    uint64_t start_ns_ = 0;
};

} // namespace mollm_trace
