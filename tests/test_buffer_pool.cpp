#include "graph/buffer_pool.h"
#include <cstdio>

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); failures++; } \
    else { printf("  PASS: %s\n", msg); } \
} while(0)

int main() {
    BufferPool pool;

    // ---- basic acquire / release ----
    void* p1 = pool.acquire(100);   // → 1024 bucket
    CHECK(p1 != nullptr, "acquire 100 bytes");
    CHECK(pool.active_bytes() >= 1024, "active >= 1024");

    void* p2 = pool.acquire(2000);  // → 2048 bucket
    CHECK(p2 != nullptr, "acquire 2000 bytes");
    CHECK(pool.active_bytes() >= 3072, "active >= 1024+2048");

    // ---- release + reuse ----
    pool.release(p1, 100);
    CHECK(pool.active_bytes() < 3072, "active dropped after release");
    void* p1b = pool.acquire(50);   // should reuse the 1024 bucket
    CHECK(p1b == p1, "reused same pointer from freelist");

    pool.release(p1b, 50);
    pool.release(p2, 2000);

    // ---- clear ----
    size_t peak_before = pool.peak_bytes();
    pool.clear();
    CHECK(pool.active_bytes() == 0, "clear → active=0");
    CHECK(pool.peak_bytes() == peak_before, "peak preserved after clear");

    // ---- round-up edge cases ----
    BufferPool pool2;
    void* p3 = pool2.acquire(1);          // → MIN_BUCKET
    CHECK(p3 != nullptr, "acquire 1 byte");
    CHECK(pool2.active_bytes() == BufferPool::MIN_BUCKET, "1 byte → MIN_BUCKET");

    void* p4 = pool2.acquire(1023);       // → MIN_BUCKET
    CHECK(p4 != nullptr, "acquire 1023 bytes");
    CHECK(pool2.active_bytes() == 2 * BufferPool::MIN_BUCKET, "1023 → 1024");

    void* p5 = pool2.acquire(1025);       // → 2048
    CHECK(pool2.active_bytes() == 2 * BufferPool::MIN_BUCKET + 2048, "1025 → 2048");

    void* p6 = pool2.acquire(4096);       // → 4096 (exact power of 2)
    CHECK(pool2.active_bytes() == 2 * BufferPool::MIN_BUCKET + 2048 + 4096, "4096 → 4096");

    pool2.release(p3, 1);
    pool2.release(p4, 1023);
    pool2.release(p5, 1025);
    pool2.release(p6, 4096);
    CHECK(pool2.pool_bytes() == 2 * BufferPool::MIN_BUCKET + 2048 + 4096, "pool_bytes correct");

    // ---- reset ----
    // After releasing all, reset clears the pool
    pool2.reset();
    CHECK(pool2.active_bytes() == 0, "reset → active=0");
    CHECK(pool2.pool_bytes() == 0, "reset → pool_bytes=0");

    // ---- alignment ----
    BufferPool pool3;
    void* p7 = pool3.acquire(65);   // → 128 after round_up?  MIN_BUCKET=1024, so → 1024
    CHECK(p7 != nullptr, "acquire 65 → MIN_BUCKET");
    CHECK(reinterpret_cast<uintptr_t>(p7) % BufferPool::ALIGNMENT == 0, "aligned pointer");
    pool3.release(p7, 65);

    if (failures == 0) {
        printf("\nAll buffer_pool tests passed!\n");
    } else {
        printf("\n%d test(s) FAILED\n", failures);
    }
    return failures;
}
