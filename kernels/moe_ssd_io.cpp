#include "kernels/moe_ssd_internal.h"

#include "kernels/trace.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include <unistd.h>

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
    const uint64_t expert = static_cast<uint64_t>(entry.expert);
    switch (component) {
    case 0:
        return entry.gate_up->spec.data_offset + expert * entry.gate_up->spec.data_bytes;
    case 1:
        return entry.gate_up->spec.scales_offset + expert * entry.gate_up->spec.scales_bytes;
    case 2:
        return entry.down->spec.data_offset + expert * entry.down->spec.data_bytes;
    default:
        return entry.down->spec.scales_offset + expert * entry.down->spec.scales_bytes;
    }
}

void MoeSsdCache::enqueue_entry_reads_locked(const std::vector<Entry*>& entries,
                                              bool low_priority) {
    if (entries.empty()) return;
    std::vector<Entry*> sorted = entries;
    std::sort(sorted.begin(), sorted.end(),
              [](const Entry* a, const Entry* b) { return a->expert < b->expert; });

    // Four independent file regions make up one expert pair. Keep enough
    // component jobs in flight to fill the I/O workers, but queue only a
    // small expert batch at a time so the first pairs become ready while the
    // main thread is already computing them. A single all-expert batch made
    // every pair wait behind the final down/scales reads and lost that overlap.
    const size_t batch_entries = std::max<size_t>(1, (size_t)io_workers_count_ / 4);
    for (size_t batch_begin = 0; batch_begin < sorted.size(); batch_begin += batch_entries) {
        const size_t batch_end = std::min(sorted.size(), batch_begin + batch_entries);
        // Group only truly adjacent expert ids. Their slices are contiguous in
        // the package, so one larger pread can feed every entry in the run.
        for (uint8_t component = 0; component < 4; component++) {
            size_t begin = batch_begin;
            while (begin < batch_end) {
                if (component_buffer(*sorted[begin], component).empty()) {
                    ++begin;
                    continue;
                }
                size_t end = begin + 1;
                while (end < batch_end &&
                       !component_buffer(*sorted[end], component).empty() &&
                       sorted[end]->expert == sorted[end - 1]->expert + 1) {
                    ++end;
                }
                IoJob job;
                job.component = component;
                job.trace_id = next_trace_id_++;
                job.speculative = low_priority;
                job.entries.assign(sorted.begin() + static_cast<ptrdiff_t>(begin),
                                   sorted.begin() + static_cast<ptrdiff_t>(end));
                for (Entry* entry : job.entries) ++entry->pending_reads;
                if (mollm_trace::enabled()) {
                    const Entry& first = *job.entries.front();
                    mollm_trace::record_flow(
                        "ssd.io", "queued_read", mollm_trace::now_ns(), job.trace_id, true,
                        "{\"layer\":" + std::to_string(first.gate_up->spec.layer) +
                        ",\"first_expert\":" + std::to_string(first.expert) +
                        ",\"experts\":" + std::to_string(job.entries.size()) + "}");
                }
                if (low_priority) low_priority_io_jobs_.push_back(std::move(job));
                else io_jobs_.push_back(std::move(job));
                begin = end;
            }
        }
    }
}

bool MoeSsdCache::read_job(const IoJob& job) {
    if (job.entries.empty()) return false;
    const size_t bytes_per_entry = component_buffer(*job.entries.front(), job.component).size();
    if (bytes_per_entry == 0) return true;
    const uint64_t offset = component_offset(*job.entries.front(), job.component);
    if (job.entries.size() == 1) {
        ByteBuffer& dst = component_buffer(*job.entries.front(), job.component);
        return read_exact(offset, dst.data(), dst.size());
    }

    const size_t merged_bytes = bytes_per_entry * job.entries.size();
    std::unique_ptr<uint8_t[]> merged(new uint8_t[merged_bytes]);
    if (!read_exact(offset, merged.get(), merged_bytes)) return false;
    for (size_t i = 0; i < job.entries.size(); i++) {
        ByteBuffer& dst = component_buffer(*job.entries[i], job.component);
        std::memcpy(dst.data(), merged.get() + i * bytes_per_entry, bytes_per_entry);
    }
    return true;
}

void MoeSsdCache::io_worker_main(int worker_index) {
    if (mollm_trace::enabled()) {
        const std::string name = "ssd-io-" + std::to_string(worker_index);
        mollm_trace::set_thread_name(name.c_str());
    }
    for (;;) {
        IoJob job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            io_cv_.wait(lock, [&] {
                return stop_io_ || !io_jobs_.empty() || !low_priority_io_jobs_.empty();
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
            const char* component = job.component == 0 ? "gate_data"
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
                    if (entry->is_ready())
                        bytes_read_ += entry->bytes();
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
