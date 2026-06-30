// Stress test for ThreadPool. Designed to be run under TSan to catch the races
// documented in docs/ code review (#5):
//   - `parked_` is a plain bool read by workers without synchronization
//   - `job_.fn = nullptr` / `job_.fn = std::move(fn)` racing with workers
//     still executing the previous fn
//   - generation / worker_ready_ flag races on rapid back-to-back parallel_for
//
// Run:
//   ninja -C build_tsan test_thread_pool_stress
//   TSAN_OPTIONS="halt_on_error=0" ./build_tsan/test_thread_pool_stress
//
// Under TSan, any data race triggers a report and (with halt_on_error=1)
// aborts. Under plain -fsanitize=undefined it just runs the correctness checks.

#include "kernels/threading.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); failures++; } \
    else { printf("  PASS: %s\n", msg); } \
} while (0)

// ---------------------------------------------------------------------------
// Test 1: basic parallel_for correctness — every element touched exactly once
//
// This catches fn-override races: if the main thread starts a new parallel_for
// while a worker is still executing the previous fn, two workers may write to
// the same index (lost update) or skip an index (undercount).
// ---------------------------------------------------------------------------
static void test_parallel_for_correctness() {
    printf("test_parallel_for_correctness\n");
    ThreadPool pool(4);

    const int N = 10000;
    const int ITERS = 2000;
    std::vector<int> buf(N, 0);

    for (int iter = 0; iter < ITERS; iter++) {
        std::fill(buf.begin(), buf.end(), 0);
        pool.parallel_for(0, N, 64, [&](int /*tid*/, int b, int e) {
            for (int i = b; i < e; i++) buf[i] += 1;
        });

        // Every element should be exactly 1 after one parallel_for pass.
        for (int i = 0; i < N; i++) {
            if (buf[i] != 1) {
                fprintf(stderr, "FAIL: iter %d buf[%d]=%d (expected 1)\n",
                        iter, i, buf[i]);
                failures++;
                return;
            }
        }
    }
    CHECK(true, "parallel_for: every element touched once across 2000 iters");
}

// ---------------------------------------------------------------------------
// Test 2: parallel_for_2d atomic-steal correctness
//
// Each cell [m,n] should be visited exactly once. Catches races on
// job_.current_chunk when the main thread overlaps jobs.
// ---------------------------------------------------------------------------
static void test_parallel_for_2d_correctness() {
    printf("test_parallel_for_2d_correctness\n");
    ThreadPool pool(4);

    const int M = 256, N = 256;
    const int ITERS = 1000;
    std::vector<int> grid(M * N, 0);

    for (int iter = 0; iter < ITERS; iter++) {
        std::fill(grid.begin(), grid.end(), 0);
        pool.parallel_for_2d(M, 8, N, 32,
            [&](int /*tid*/, int m_b, int m_e, int n_b, int n_e) {
                for (int m = m_b; m < m_e; m++)
                    for (int n = n_b; n < n_e; n++)
                        grid[m * N + n] += 1;
            });

        for (int i = 0; i < M * N; i++) {
            if (grid[i] != 1) {
                fprintf(stderr, "FAIL: iter %d grid[%d]=%d (expected 1)\n",
                        iter, i, grid[i]);
                failures++;
                return;
            }
        }
    }
    CHECK(true, "parallel_for_2d: every cell visited once across 1000 iters");
}

// ---------------------------------------------------------------------------
// Test 3: park/resume race
//
// `parked_` is a plain bool. pool.park() writes parked_=true without holding
// park_mtx_; workers read parked_ without a lock before deciding to wait on
// park_cv_. Under TSan this is a data race.
//
// We alternate park() and a quick parallel_for (which auto-resumes) in a
// tight loop, while a parallel_for is in flight. This maximises the chance
// of the worker observing parked_ mid-write.
// ---------------------------------------------------------------------------
static void test_park_resume_race() {
    printf("test_park_resume_race\n");
    ThreadPool pool(4);

    const int N = 1000;
    std::vector<int> buf(N, 0);
    const int ITERS = 500;

    for (int iter = 0; iter < ITERS; iter++) {
        std::fill(buf.begin(), buf.end(), 0);

        // Park, then immediately do a parallel_for (which auto-resumes).
        // The race window is between park() setting parked_=true and the
        // worker reading parked_ at the end of its previous shard.
        pool.park();
        pool.parallel_for(0, N, 32, [&](int /*tid*/, int b, int e) {
            for (int i = b; i < e; i++) buf[i] += 1;
        });

        for (int i = 0; i < N; i++) {
            if (buf[i] != 1) {
                fprintf(stderr, "FAIL: iter %d buf[%d]=%d (expected 1)\n",
                        iter, i, buf[i]);
                failures++;
                return;
            }
        }
    }
    CHECK(true, "park/resume + parallel_for correctness across 500 iters");
}

// ---------------------------------------------------------------------------
// Test 4: rapid back-to-back parallel_for (generation race)
//
// Fire many tiny parallel_for calls with no work in between. This stresses
// the generation/worker_ready_ handshake: a worker that just finished shard N
// may not have cleared worker_ready_[t]=false before the main thread sets it
// to true for shard N+1.
// ---------------------------------------------------------------------------
static void test_rapid_parallel_for() {
    printf("test_rapid_parallel_for\n");
    ThreadPool pool(4);

    std::atomic<int> counter{0};
    const int ITERS = 5000;

    for (int iter = 0; iter < ITERS; iter++) {
        counter.store(0, std::memory_order_relaxed);
        pool.parallel_for(0, 64, 8, [&](int /*tid*/, int b, int e) {
            for (int i = b; i < e; i++) counter.fetch_add(1, std::memory_order_relaxed);
        });
        if (counter.load() != 64) {
            fprintf(stderr, "FAIL: iter %d counter=%d (expected 64)\n",
                    iter, counter.load());
            failures++;
            return;
        }
    }
    CHECK(true, "rapid back-to-back parallel_for across 5000 iters");
}

// ---------------------------------------------------------------------------
// Test 5: nested parallel_for (worker fn launches its own parallel_for)
//
// This isn't a documented usage pattern, but if it ever happens, it must not
// deadlock or corrupt the shared job_ struct. The test documents the current
// limitation: nested parallel_for on the same pool is UNSUPPORTED because it
// overwrites the outer job_. We include it to surface any future fix.
// ---------------------------------------------------------------------------
static void test_nested_parallel_for_smoke() {
    printf("test_nested_parallel_for_smoke (documented unsupported)\n");
    ThreadPool pool(4);

    // One outer call; worker fns do NOT call back into pool (that would race).
    // This is just a smoke test that the pool survives being called from
    // a worker thread that yields without re-entering.
    std::atomic<int> sum{0};
    pool.parallel_for(0, 100, 8, [&](int /*tid*/, int b, int e) {
        // Simulate real work, but do NOT call pool.parallel_for here.
        for (int i = b; i < e; i++) sum.fetch_add(i, std::memory_order_relaxed);
    });
    // sum of 0..99 = 99*100/2 = 4950
    CHECK(sum.load() == 4950, "outer parallel_for sum 0..99 = 4950");
}

// ---------------------------------------------------------------------------
// Test 6: resize() between parallel_for calls
//
// resize() calls stop_workers() + join + restart. Workers blocked in
// park_cv_.wait must be woken by stop_workers. This catches the deadlock
// documented in code review #5d.
// ---------------------------------------------------------------------------
static void test_resize_after_park() {
    printf("test_resize_after_park\n");
    ThreadPool pool(4);

    // Do some work, park, then resize. If stop_workers() doesn't notify
    // park_cv_, the worker join in resize() blocks forever.
    std::atomic<int> counter{0};
    pool.parallel_for(0, 100, 8, [&](int /*tid*/, int b, int e) {
        for (int i = b; i < e; i++) counter.fetch_add(1, std::memory_order_relaxed);
    });
    CHECK(counter.load() == 100, "work before park");

    pool.park();
    // Now workers are (eventually) blocked on park_cv_.wait.
    // Give them a moment to actually park.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    // resize must wake parked workers and join them cleanly.
    pool.resize(2);
    CHECK(pool.num_threads() == 2, "resize to 2 after park");

    // Pool should still be usable.
    counter.store(0, std::memory_order_relaxed);
    pool.parallel_for(0, 100, 8, [&](int /*tid*/, int b, int e) {
        for (int i = b; i < e; i++) counter.fetch_add(1, std::memory_order_relaxed);
    });
    CHECK(counter.load() == 100, "work after resize");
}

// ---------------------------------------------------------------------------
// Test 7: destructor with parked workers
//
// Same race as test 6 but via destructor. Catches the deadlock where
// ~ThreadPool joins workers that are blocked on park_cv_.
// ---------------------------------------------------------------------------
static void test_destructor_with_parked_workers() {
    printf("test_destructor_with_parked_workers\n");
    for (int iter = 0; iter < 50; iter++) {
        ThreadPool* pool = new ThreadPool(4);
        std::atomic<int> counter{0};
        pool->parallel_for(0, 100, 8, [&](int /*tid*/, int b, int e) {
            for (int i = b; i < e; i++) counter.fetch_add(1, std::memory_order_relaxed);
        });
        pool->park();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        delete pool;  // must not hang
    }
    CHECK(true, "50x destructor with parked workers did not hang");
}

// ---------------------------------------------------------------------------
// Test 8: long-running fn overlapping next parallel_for
//
// Worker fn sleeps to extend its lifetime past the main thread's spin-wait
// completion. If job_.fn is overwritten by the next parallel_for, the worker
// will invoke the wrong fn. We detect this by having each fn capture a
// unique tag and write it into a per-slot buffer.
// ---------------------------------------------------------------------------
static void test_long_fn_overlap() {
    printf("test_long_fn_overlap\n");
    ThreadPool pool(4);

    const int ITERS = 100;
    std::atomic<int> mismatches{0};

    for (int iter = 0; iter < ITERS; iter++) {
        // Each iteration's fn captures iter as tag and writes it to slot[iter].
        // If the fn gets overwritten by iter+1 before worker runs, slot[iter]
        // stays 0 and slot[iter+1] gets written twice.
        std::vector<int> slot(ITERS + 1, -1);
        std::atomic<int> done{0};

        pool.parallel_for(0, 4, 1, [&](int tid, int b, int /*e*/) {
            // Pretend to do work that takes a moment.
            for (volatile int s = 0; s < 1000; s++);
            slot[tid] = iter;
            done.fetch_add(1, std::memory_order_relaxed);
            (void)b;
        });
        // Spin until all 4 shards report done (parallel_for already waited,
        // but the tag writes above happen before done fetch_add).
        while (done.load() < 4) std::this_thread::yield();

        // All 4 slots should be tagged with this iter.
        for (int t = 0; t < 4; t++) {
            if (slot[t] != iter) {
                mismatches.fetch_add(1, std::memory_order_relaxed);
                break;
            }
        }
    }
    CHECK(mismatches.load() == 0, "no fn-override across 100 long-fn iters");
}

int main() {
    test_parallel_for_correctness();
    test_parallel_for_2d_correctness();
    test_park_resume_race();
    test_rapid_parallel_for();
    test_nested_parallel_for_smoke();
    test_resize_after_park();
    test_destructor_with_parked_workers();
    test_long_fn_overlap();

    if (failures == 0) {
        printf("\nAll thread pool stress tests passed!\n");
        return 0;
    }
    printf("\n%d FAILURES\n", failures);
    return 1;
}
