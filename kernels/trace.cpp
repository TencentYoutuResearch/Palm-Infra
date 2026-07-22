#include "kernels/trace.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace mollm_trace {
namespace {

struct Event {
    uint64_t ts_ns = 0;
    uint64_t dur_ns = 0;
    uint64_t flow_id = 0;
    bool flow_start = false;
    bool is_flow = false;
    std::string category;
    std::string name;
    std::string args_json;
};

struct ThreadBuffer {
    uint64_t tid = 0;
    std::string name;
    std::vector<Event> events;
};

struct LocalBuffer {
    uint64_t generation = 0;
    ThreadBuffer* buffer = nullptr;
};

thread_local LocalBuffer g_local_buffer;

class Recorder {
public:
    void start(const std::string& path) {
        enabled_.store(false, std::memory_order_release);
        wait_for_writers();
        std::lock_guard<std::mutex> lock(mutex_);
        path_ = path;
        buffers_.clear();
        ++generation_;
        epoch_ = std::chrono::steady_clock::now();
        enabled_.store(!path.empty(), std::memory_order_release);
    }

    bool enabled() const { return enabled_.load(std::memory_order_acquire); }

    uint64_t now_ns() const {
        if (!enabled()) return 0;
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - epoch_).count());
    }

    void set_thread_name(const char* name) {
        if (!enabled() || !name) return;
        thread_buffer().name = name;
    }

    void record(const char* category, const char* name, uint64_t start_ns,
                uint64_t end_ns, const std::string& args_json) {
        if (!enabled() || !category || !name || end_ns < start_ns) return;
        active_writers_.fetch_add(1, std::memory_order_acq_rel);
        if (!enabled()) {
            active_writers_.fetch_sub(1, std::memory_order_release);
            return;
        }
        Event event;
        event.ts_ns = start_ns;
        event.dur_ns = end_ns - start_ns;
        event.category = category;
        event.name = name;
        event.args_json = args_json;
        thread_buffer().events.push_back(std::move(event));
        active_writers_.fetch_sub(1, std::memory_order_release);
    }

    void record_flow(const char* category, const char* name, uint64_t timestamp_ns,
                     uint64_t flow_id, bool start, const std::string& args_json) {
        if (!enabled() || !category || !name || flow_id == 0) return;
        active_writers_.fetch_add(1, std::memory_order_acq_rel);
        if (!enabled()) {
            active_writers_.fetch_sub(1, std::memory_order_release);
            return;
        }
        Event event;
        event.ts_ns = timestamp_ns;
        event.flow_id = flow_id;
        event.flow_start = start;
        event.is_flow = true;
        event.category = category;
        event.name = name;
        event.args_json = args_json;
        thread_buffer().events.push_back(std::move(event));
        active_writers_.fetch_sub(1, std::memory_order_release);
    }

    bool write() {
        std::vector<std::pair<uint64_t, std::string>> thread_names;
        std::vector<std::pair<uint64_t, Event>> events;
        std::string path;
        enabled_.store(false, std::memory_order_release);
        wait_for_writers();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            path = path_;
            for (const auto& buffer : buffers_) {
                if (!buffer->name.empty()) thread_names.emplace_back(buffer->tid, buffer->name);
                for (const Event& event : buffer->events) {
                    events.emplace_back(buffer->tid, event);
                }
            }
        }
        if (path.empty()) return true;
        std::sort(events.begin(), events.end(), [](const auto& a, const auto& b) {
            return a.second.ts_ns < b.second.ts_ns;
        });
        std::FILE* file = std::fopen(path.c_str(), "wb");
        if (!file) {
            std::fprintf(stderr, "Trace: could not write %s\n", path.c_str());
            return false;
        }
        std::fputs("{\"traceEvents\":[", file);
        bool first = true;
        const auto comma = [&] {
            if (!first) std::fputc(',', file);
            first = false;
        };
        for (const auto& item : thread_names) {
            comma();
            std::fprintf(file,
                         "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":%llu,\"args\":{\"name\":\"",
                         static_cast<unsigned long long>(item.first));
            write_json_string(file, item.second);
            std::fputs("\"}}", file);
        }
        for (const auto& item : events) {
            const uint64_t tid = item.first;
            const Event& event = item.second;
            comma();
            std::fputs("{\"name\":\"", file);
            write_json_string(file, event.name);
            std::fputs("\",\"cat\":\"", file);
            write_json_string(file, event.category);
            std::fprintf(file, "\",\"ph\":\"%c\",\"pid\":1,\"tid\":%llu,\"ts\":",
                         event.is_flow ? (event.flow_start ? 's' : 'f') : 'X',
                         static_cast<unsigned long long>(tid));
            write_time_us(file, event.ts_ns);
            if (event.is_flow) {
                std::fprintf(file, ",\"id\":%llu", static_cast<unsigned long long>(event.flow_id));
            } else {
                std::fputs(",\"dur\":", file);
                write_time_us(file, event.dur_ns);
            }
            if (!event.args_json.empty()) {
                std::fputs(",\"args\":", file);
                std::fputs(event.args_json.c_str(), file);
            }
            std::fputc('}', file);
        }
        std::fputs("]}\n", file);
        const bool ok = std::fclose(file) == 0;
        if (ok) {
            std::fprintf(stderr, "Trace: wrote %zu events to %s\n", events.size(), path.c_str());
        }
        return ok;
    }

private:
    static uint64_t thread_id() {
        return static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    }

    ThreadBuffer& thread_buffer() {
        const uint64_t generation = generation_.load(std::memory_order_acquire);
        if (g_local_buffer.buffer && g_local_buffer.generation == generation) {
            return *g_local_buffer.buffer;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        auto buffer = std::make_unique<ThreadBuffer>();
        buffer->tid = thread_id();
        ThreadBuffer* raw = buffer.get();
        buffers_.push_back(std::move(buffer));
        g_local_buffer = {generation, raw};
        return *raw;
    }

    void wait_for_writers() const {
        while (active_writers_.load(std::memory_order_acquire) != 0) {
            std::this_thread::yield();
        }
    }

    static void write_json_string(std::FILE* file, const std::string& text) {
        for (unsigned char ch : text) {
            switch (ch) {
            case '"': std::fputs("\\\"", file); break;
            case '\\': std::fputs("\\\\", file); break;
            case '\n': std::fputs("\\n", file); break;
            case '\r': std::fputs("\\r", file); break;
            case '\t': std::fputs("\\t", file); break;
            default:
                if (ch < 0x20) std::fprintf(file, "\\u%04x", static_cast<unsigned>(ch));
                else std::fputc(ch, file);
            }
        }
    }

    // Chrome Trace timestamps are microseconds. JSON permits a fractional
    // number, so preserve our nanosecond clock rather than collapsing short
    // cache-hit and gather intervals onto the same pixel.
    static void write_time_us(std::FILE* file, uint64_t nanoseconds) {
        std::fprintf(file, "%llu.%03llu",
                     static_cast<unsigned long long>(nanoseconds / 1000),
                     static_cast<unsigned long long>(nanoseconds % 1000));
    }

    mutable std::mutex mutex_;
    std::atomic<bool> enabled_{false};
    std::atomic<uint64_t> generation_{1};
    std::atomic<uint64_t> active_writers_{0};
    std::chrono::steady_clock::time_point epoch_;
    std::string path_;
    std::vector<std::unique_ptr<ThreadBuffer>> buffers_;
};

Recorder& recorder() {
    static Recorder instance;
    return instance;
}

} // namespace

void start(const std::string& path) { recorder().start(path); }
bool enabled() { return recorder().enabled(); }
uint64_t now_ns() { return recorder().now_ns(); }
void set_thread_name(const char* name) { recorder().set_thread_name(name); }
void record_duration(const char* category, const char* name,
                     uint64_t start_ns, uint64_t end_ns,
                     const std::string& args_json) {
    recorder().record(category, name, start_ns, end_ns, args_json);
}
void record_flow(const char* category, const char* name, uint64_t timestamp_ns,
                 uint64_t flow_id, bool start, const std::string& args_json) {
    recorder().record_flow(category, name, timestamp_ns, flow_id, start, args_json);
}
bool write() { return recorder().write(); }

ScopedEvent::ScopedEvent(const char* category, const char* name,
                         const std::string& args_json)
    : category_(category), name_(name), args_json_(args_json), start_ns_(now_ns()) {}

ScopedEvent::~ScopedEvent() {
    if (start_ns_ != 0) record_duration(category_, name_, start_ns_, now_ns(), args_json_);
}

} // namespace mollm_trace
