#include "graph/buffer_pool.h"
#include <cstdio>
#include <utility>

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); failures++; } \
    else { printf("  PASS: %s\n", msg); } \
} while(0)

int main() {
    BufferPool pool;
    BufferPool other_pool;
    CHECK(pool.id() != 0, "pool id is non-zero");
    CHECK(other_pool.id() != 0, "second pool id is non-zero");
    CHECK(pool.id() != other_pool.id(), "pool ids are unique");

    // ---- basic acquire / release ----
    void* p1 = pool.acquire(100);   // → 1024 bucket
    CHECK(p1 != nullptr, "acquire 100 bytes");
    CHECK(pool.active_bytes() >= 1024, "active >= 1024");
    uint64_t p1_sid = pool.storage_id(p1);
    CHECK(p1_sid != 0, "acquired buffer has storage id");

    void* p2 = pool.acquire(2000);  // → 2048 bucket
    CHECK(p2 != nullptr, "acquire 2000 bytes");
    CHECK(pool.active_bytes() >= 3072, "active >= 1024+2048");
    CHECK(pool.storage_id(p2) != 0 && pool.storage_id(p2) != p1_sid,
          "different buffers have different storage ids");

    // ---- release + reuse ----
    pool.release(p1, 100);
    CHECK(pool.active_bytes() < 3072, "active dropped after release");
    void* p1b = pool.acquire(50);   // should reuse the 1024 bucket
    CHECK(p1b == p1, "reused same pointer from freelist");
    CHECK(pool.storage_id(p1b) == p1_sid, "reused buffer keeps storage id");

    pool.release(p1b, 50);
    pool.release(p2, 2000);

    // ---- clear ----
    size_t peak_before = pool.peak_bytes();
    pool.clear();
    CHECK(pool.active_bytes() == 0, "clear → active=0");
    CHECK(pool.pool_bytes() == 0, "clear → freelist=0");
    CHECK(pool.peak_bytes() == peak_before, "peak preserved after clear");

    // ---- clear active allocations ----
    void* p_active1 = pool.acquire(333);
    void* p_active2 = pool.acquire(4097);
    CHECK(p_active1 != nullptr && p_active2 != nullptr, "active allocations acquired");
    CHECK(pool.active_bytes() >= BufferPool::MIN_BUCKET + 8192, "active includes unreleased allocations");
    pool.clear();
    CHECK(pool.active_bytes() == 0, "clear releases active allocations");
    CHECK(pool.pool_bytes() == 0, "clear active leaves no freelist");

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

    // ---- move ownership ----
    BufferPool move_source;
    void* moved_active = move_source.acquire(1500);
    uint32_t source_id = move_source.id();
    uint64_t moved_storage_id = move_source.storage_id(moved_active);

    BufferPool move_target;
    move_target.acquire(3000);  // move-assignment must release this allocation
    move_target = std::move(move_source);
    CHECK(move_target.id() == source_id, "move assignment transfers pool id");
    CHECK(move_target.active_bytes() == 2048,
          "move assignment transfers active allocation");
    CHECK(move_target.storage_id(moved_active) == moved_storage_id,
          "move assignment preserves storage identity");
    CHECK(move_source.id() != source_id &&
              move_source.active_bytes() == 0 &&
              move_source.pool_bytes() == 0,
          "moved-from pool is reusable and empty");
    move_target.release(moved_active, 1500);

    BufferPool move_constructed(std::move(move_target));
    CHECK(move_constructed.id() == source_id,
          "move construction transfers pool id");
    CHECK(move_constructed.pool_bytes() == 2048,
          "move construction transfers freelist");

    if (failures == 0) {
        printf("\nAll buffer_pool tests passed!\n");
    } else {
        printf("\n%d test(s) FAILED\n", failures);
    }
    return failures;
}
