#include "kernels/moe_ssd_internal.h"

#include "runtime/trace.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <sys/uio.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

namespace {

bool preadv_exact(int fd, uint64_t offset, std::vector<iovec> vectors) {
    size_t index = 0;
    while (index < vectors.size()) {
        const ssize_t bytes = preadv(
            fd, vectors.data() + static_cast<ptrdiff_t>(index),
            static_cast<int>(vectors.size() - index),
            static_cast<off_t>(offset));
        if (bytes < 0 && errno == EINTR)
            continue;
        if (bytes <= 0) {
            std::fprintf(stderr,
                         "MoE SSD: preadv failed at offset %llu: %s\n",
                         static_cast<unsigned long long>(offset),
                         bytes == 0 ? "unexpected EOF" : std::strerror(errno));
            return false;
        }
        offset += static_cast<uint64_t>(bytes);
        size_t consumed = static_cast<size_t>(bytes);
        while (index < vectors.size() && consumed >= vectors[index].iov_len) {
            consumed -= vectors[index].iov_len;
            ++index;
        }
        if (index < vectors.size() && consumed != 0) {
            vectors[index].iov_base =
                static_cast<uint8_t*>(vectors[index].iov_base) + consumed;
            vectors[index].iov_len -= consumed;
        }
    }
    return true;
}

}  // namespace

bool MoeSsdCache::read_exact(uint64_t offset, void* dst, size_t bytes) const {
    uint8_t* out = static_cast<uint8_t*>(dst);
    size_t done = 0;
    while (done < bytes) {
        ssize_t n = pread(fd_, out + done, bytes - done,
                          static_cast<off_t>(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            std::fprintf(stderr, "MoE SSD: pread failed at offset %llu: %s\n",
                         (unsigned long long)(offset + done),
                         n == 0 ? "unexpected EOF" : std::strerror(errno));
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

const MoeSsdCache::ByteBuffer& MoeSsdCache::component_buffer(const Entry& entry,
                                                              uint8_t component) {
    switch (component) {
    case 0: return entry.gate_up_data;
    case 1: return entry.gate_up_scales;
    case 2: return entry.down_data;
    default: return entry.down_scales;
    }
}

MoeSsdCache::ByteBuffer& MoeSsdCache::component_buffer(Entry& entry, uint8_t component) {
    return const_cast<MoeSsdCache::ByteBuffer&>(component_buffer(
        static_cast<const Entry&>(entry), component));
}

uint64_t MoeSsdCache::component_offset(const Entry& entry, uint8_t component) {
    switch (component) {
    case 0:
        return entry.gate_up->spec.data_file_offset(entry.expert);
    case 1:
        return entry.gate_up->spec.scales_file_offset(entry.expert);
    case 2:
        return entry.down->spec.data_file_offset(entry.expert);
    default:
        return entry.down->spec.scales_file_offset(entry.expert);
    }
}

void MoeSsdCache::enqueue_entry_reads_locked(const std::vector<Entry*>& entries,
                                              bool low_priority) {
    if (entries.empty()) return;
    std::vector<Entry*> sorted = entries;
    // Exact route computation consumes experts in id order. Speculative
    // entries arrive in router-confidence order instead; preserve that order
    // so the most likely next-layer experts finish first.
    if (!low_priority) {
        std::sort(sorted.begin(), sorted.end(),
                  [](const Entry* a, const Entry* b) {
                      return a->expert < b->expert;
                  });
    }

    // Fill a small expert window completely before moving to the next one.
    // This makes the first routed pairs consumable while later reads continue
    // instead of maximizing the number of half-loaded entries.
    constexpr size_t kWindow = 1;
    for (size_t begin = 0; begin < sorted.size(); begin += kWindow) {
        const size_t end = std::min(sorted.size(), begin + kWindow);
        for (size_t index = begin; index < end; ++index) {
            Entry* entry = sorted[index];
            const bool is_nvfp4 =
                entry->gate_up->spec.precision == Precision::NVFP4 &&
                entry->down->spec.precision == Precision::NVFP4;
            const bool adjacent_gate =
                component_offset(*entry, 1) ==
                component_offset(*entry, 0) +
                    component_buffer(*entry, 0).size();
            const bool adjacent_down =
                component_offset(*entry, 3) ==
                component_offset(*entry, 2) +
                    component_buffer(*entry, 2).size();
            // NVFP4 stores each expert's data and scales contiguously. One
            // preadv per tensor halves foreground I/O jobs without changing
            // the storage assumptions of the other quantization formats.
            const bool combine =
                is_nvfp4 && adjacent_gate && adjacent_down;
            const uint8_t first_component = combine ? 4 : 0;
            const uint8_t component_count = combine ? 2 : 4;
            for (uint8_t component_index = 0;
                 component_index < component_count; ++component_index) {
                const uint8_t component = first_component + component_index;
                IoJob job;
                job.component = component;
                job.trace_id = next_trace_id_++;
                job.speculative = low_priority;
                job.entries.push_back(entry);
                ++entry->pending_reads;
                if (mollm_trace::enabled()) {
                    mollm_trace::record_flow(
                        "ssd.io", "queued_read", mollm_trace::now_ns(), job.trace_id, true,
                        "{\"layer\":" + std::to_string(entry->gate_up->spec.layer) +
                        ",\"first_expert\":" + std::to_string(entry->expert) +
                        ",\"experts\":1}");
                }
                if (low_priority)
                    low_priority_io_jobs_.push_back(std::move(job));
                else
                    io_jobs_.push_back(std::move(job));
            }
        }
    }
}

bool MoeSsdCache::read_job(const IoJob& job) {
    if (job.entries.empty()) return false;
    auto lock_buffer = [&](ByteBuffer& buffer) {
        if (!lock_expert_pages_ || buffer.lock()) return;
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            std::fprintf(stderr,
                         "MoE SSD: warning: could not lock expert cache pages: %s\n",
                         std::strerror(errno));
        }
    };
    if (job.component == 4 || job.component == 5) {
        Entry& entry = *job.entries.front();
        auto read_tensor = [&](uint8_t data_component,
                               uint8_t scales_component) {
            ByteBuffer& data = component_buffer(entry, data_component);
            ByteBuffer& scales = component_buffer(entry, scales_component);
            const uint64_t data_offset =
                component_offset(entry, data_component);
            const uint64_t scales_offset =
                component_offset(entry, scales_component);
            if (scales.empty() || scales_offset != data_offset + data.size()) {
                if (!data.empty() &&
                    !read_exact(data_offset, data.data(), data.size()))
                    return false;
                if (!scales.empty() &&
                    !read_exact(scales_offset, scales.data(), scales.size()))
                    return false;
            } else {
                std::vector<iovec> vectors = {
                    {data.data(), data.size()},
                    {scales.data(), scales.size()},
                };
                if (!preadv_exact(fd_, data_offset, std::move(vectors)))
                    return false;
            }
            lock_buffer(data);
            lock_buffer(scales);
            return true;
        };
        return job.component == 4 ? read_tensor(0, 1)
                                  : read_tensor(2, 3);
    }
    const size_t bytes_per_entry = component_buffer(*job.entries.front(), job.component).size();
    if (bytes_per_entry == 0) return true;
    const uint64_t offset = component_offset(*job.entries.front(), job.component);
    if (job.entries.size() == 1) {
        ByteBuffer& dst = component_buffer(*job.entries.front(), job.component);
        if (!read_exact(offset, dst.data(), dst.size())) return false;
        lock_buffer(dst);
        return true;
    }

    // These entries are adjacent in the package. Scatter one contiguous file
    // read directly into their final cache buffers instead of allocating a
    // temporary merged buffer and copying several megabytes a second time.
    std::vector<iovec> vectors;
    vectors.reserve(job.entries.size());
    for (Entry* entry : job.entries) {
        ByteBuffer& dst = component_buffer(*entry, job.component);
        if (dst.size() != bytes_per_entry) return false;
        vectors.push_back({dst.data(), dst.size()});
    }
    if (!preadv_exact(fd_, offset, std::move(vectors)))
        return false;
    for (Entry* entry : job.entries)
        lock_buffer(component_buffer(*entry, job.component));
    return true;
}

void MoeSsdCache::io_worker_main(int worker_index) {
#if defined(__APPLE__)
    // These threads unblock the foreground decode path. Explicitly keep them
    // out of the utility/background QoS classes when the compute pool is
    // continuously busy on heterogeneous Apple Silicon cores.
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
#endif
    if (mollm_trace::enabled()) {
        const std::string name = "ssd-io-" + std::to_string(worker_index);
        mollm_trace::set_thread_name(name.c_str());
    }
    for (;;) {
        IoJob job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            io_cv_.wait(lock, [&] {
                return stop_io_ || !io_jobs_.empty() ||
                       !low_priority_io_jobs_.empty();
            });
            if (stop_io_ && io_jobs_.empty() && low_priority_io_jobs_.empty()) return;
            if (!io_jobs_.empty()) {
                job = std::move(io_jobs_.front());
                io_jobs_.pop_front();
            } else {
                job = std::move(low_priority_io_jobs_.front());
                low_priority_io_jobs_.pop_front();
            }
        }
        std::string trace_args;
        if (mollm_trace::enabled() && !job.entries.empty()) {
            const Entry& first = *job.entries.front();
            const char* component = job.component == 4 ? "gate_tensor"
                                  : job.component == 5 ? "down_tensor"
                                  : job.component == 0 ? "gate_data"
                                  : job.component == 1 ? "gate_scales"
                                  : job.component == 2 ? "down_data" : "down_scales";
            trace_args = "{\"layer\":" + std::to_string(first.gate_up->spec.layer) +
                         ",\"first_expert\":" + std::to_string(first.expert) +
                         ",\"experts\":" + std::to_string(job.entries.size()) +
                         ",\"component\":\"" + component + "\""
                         ",\"kind\":\"" + (job.speculative ? "prefetch" : "route") + "\"}";
        }
        bool ok = false;
        {
            mollm_trace::ScopedEvent trace_event(
                "ssd.io", job.speculative ? "pread.prefetch" : "pread.route", trace_args,
                job.speculative ? "yellow" : "good");
            ok = read_job(job);
        }
        if (job.trace_id != 0) {
            mollm_trace::record_flow("ssd.io", "queued_read", mollm_trace::now_ns(),
                                     job.trace_id, false, trace_args);
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (Entry* entry : job.entries) {
                if (entry->pending_reads > 0) --entry->pending_reads;
                if (!ok && entry->state == Entry::State::Loading)
                    entry->state = Entry::State::LoadingFailed;
                if (entry->pending_reads == 0) {
                    entry->state = entry->state == Entry::State::LoadingFailed
                                       ? Entry::State::Failed
                                       : Entry::State::Ready;
                    if (entry->is_ready()) {
                        const uint64_t bytes = entry->bytes();
                        bytes_read_ += bytes;
                        if (entry->load_origin_speculative)
                            prefetch_load_bytes_ += bytes;
                        else
                            demand_load_bytes_ += bytes;
                    }
                }
            }
        }
        ready_cv_.notify_all();
    }
}

void MoeSsdCache::cross_layer_worker_main() {
    if (mollm_trace::enabled()) mollm_trace::set_thread_name("ssd-predict");
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cross_layer_cv_.wait(lock, [&] { return stop_io_ || !cross_layer_tasks_.empty(); });
            if (stop_io_) return;
            task = std::move(cross_layer_tasks_.front());
            cross_layer_tasks_.pop_front();
        }
        mollm_trace::ScopedEvent trace_event("ssd.predict", "cross_layer_gate");
        task();
    }
}
