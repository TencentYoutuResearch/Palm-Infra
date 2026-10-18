#include "engine/recurrent_checkpoint.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#ifdef MOLLM_CUDA
#include "backends/cuda/backend.h"
#endif

class TestBackend : public Backend {
public:
    ShapeMode shape_mode() const override { return ShapeMode::DYNAMIC; }
    void dispatch(const GraphNode&, const std::vector<const Tensor*>&,
                  Tensor*, ThreadPool*) override {}
    bool fail_read = false;
    bool native_requested = false;
    bool supports_state_snapshot(const std::vector<Tensor*>&) const override {
        return native_requested;
    }
    bool opaque = false;
    int reads = 0, writes = 0;
    std::vector<float> device_state;
    bool copy_to_host(const Tensor& t, void* out, size_t n,
                      size_t offset = 0) override {
        ++reads;
        if (fail_read) return false;
        if (!opaque) return Backend::copy_to_host(t, out, n, offset);
        if (offset > device_state.size() * sizeof(float) ||
            n > device_state.size() * sizeof(float) - offset) return false;
        std::memcpy(out, reinterpret_cast<const uint8_t*>(device_state.data()) + offset, n);
        return true;
    }
    bool copy_from_host(const void* in, Tensor& t, size_t n,
                        size_t offset = 0) override {
        ++writes;
        if (!opaque) return Backend::copy_from_host(in, t, n, offset);
        if (offset > device_state.size() * sizeof(float) ||
            n > device_state.size() * sizeof(float) - offset) return false;
        std::memcpy(reinterpret_cast<uint8_t*>(device_state.data()) + offset, in, n);
        return true;
    }
};

int main() {
    int failed = 0;
    auto check = [&](bool ok, const char* message) {
        if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); ++failed; }
    };
    TestBackend backend;
    std::vector<float> state{1, 2, 3, 4}, original = state;
    Tensor tensor = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                  4, 1, 1, 1, state.data());
    mollm::detail::RecurrentCheckpoint snapshot;
    check(!snapshot.restore(backend), "uncaptured checkpoint cannot restore");
    check(snapshot.capture(backend, {nullptr, &tensor, &tensor}), "capture");
    check(backend.reads == 1 && snapshot.bytes() == 16,
          "duplicate states saved once");
    // Simulate recurrent verification of three inputs, rejecting after one.
    auto advance = [&](float input) { for (float& v : state) v = v * 2 + input; };
    advance(5); advance(9); advance(7);
    check(snapshot.restore(backend) && state == original, "restore entire state");
    advance(5);
    for (size_t i = 0; i < state.size(); ++i)
        check(state[i] == original[i] * 2 + 5, "accepted-only replay");
    tensor.shape[0] = 3;
    tensor.compute_strides();
    int writes = backend.writes;
    check(!snapshot.restore(backend) && writes == backend.writes,
          "changed shape rejected before writes");
    backend.fail_read = true;
    check(!snapshot.capture(backend, {&tensor}) && !snapshot.restore(backend),
          "failed capture invalidates old checkpoint");
    backend.fail_read = false;
    backend.native_requested = true;
    const int reads_before_native_failure = backend.reads;
    check(!snapshot.capture(backend,{&tensor}) && !snapshot.restore(backend) &&
          backend.reads==reads_before_native_failure,
          "failed native capture invalidates checkpoint without silent host fallback");
    backend.native_requested = false;
    check(snapshot.capture(backend, {}) && snapshot.empty() &&
          snapshot.restore(backend), "stateless checkpoint is a no-op");

    backend.opaque = true;
    backend.device_state = original;
    tensor = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                            4, 1, 1, 1, nullptr);
    check(snapshot.capture(backend, {&tensor}), "opaque device uses transfer hook");
    backend.device_state.assign(4, -99);
    check(snapshot.restore(backend) && backend.device_state == original,
          "opaque device restored without dereferencing tensor data");
#ifdef MOLLM_CUDA
    CudaBackend cuda;
    if (cuda.available()) {
        Tensor device_tensor = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, 4);
        cuda.alloc_persistent(device_tensor, 16, PersistentHostAccess::NONE);
        check(device_tensor.device.buffer &&
              cuda.copy_from_host(original.data(), device_tensor, 16),
              "CUDA checkpoint initial state upload");
        check(snapshot.capture(cuda, {&device_tensor}), "CUDA checkpoint capture");
        const char* device_option = std::getenv("MOLLM_CUDA_MTP_DEVICE_CHECKPOINT");
        check(snapshot.device_resident() == (device_option && !std::strcmp(device_option,"1")),
              "CUDA checkpoint option selects device storage");
        check(cuda.zero_tensor(device_tensor, 16) && snapshot.restore(cuda),
              "CUDA checkpoint restores overwritten device state");
        std::vector<float> restored(4);
        check(cuda.copy_to_host(device_tensor, restored.data(), 16) &&
              restored == original, "CUDA checkpoint roundtrip is bit exact");
        check(snapshot.capture(cuda, {&device_tensor, &device_tensor}) && snapshot.bytes()==16,
              "CUDA duplicate device states captured once");
        // Native device snapshots validate all destination spans before writes.
        if (snapshot.device_resident()) {
            check(!snapshot.restore(backend), "device checkpoint rejects a different backend");
            device_tensor.shape[0]=3;device_tensor.compute_strides();
            check(!snapshot.restore(cuda), "device checkpoint rejects changed span");
            device_tensor.shape[0]=4;device_tensor.compute_strides();
            check(snapshot.restore(cuda), "device checkpoint remains immutable after rejected restore");
            device_tensor.device.offset+=sizeof(float);
            check(!snapshot.restore(cuda), "device checkpoint rejects changed destination address");
            device_tensor.device.offset-=sizeof(float);
            Tensor second=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,4);
            cuda.alloc_persistent(second,16,PersistentHostAccess::NONE);
            check(cuda.copy_from_host(original.data(),second,16) &&
                  snapshot.capture(cuda,{&device_tensor,&second}) && snapshot.bytes()==32,
                  "multiple state buffers share one snapshot");
            check(cuda.zero_tensor(device_tensor,16) && cuda.zero_tensor(second,16), "overwrite both states");
            second.shape[0]=3;second.compute_strides();
            check(!snapshot.restore(cuda) && cuda.copy_to_host(device_tensor,restored.data(),16) &&
                  restored==std::vector<float>(4,0), "validate every span before the first restore write");
            second.shape[0]=4;second.compute_strides();
            check(snapshot.restore(cuda) && cuda.copy_to_host(device_tensor,restored.data(),16) && restored==original,
                  "restore multiple buffers after validation failure");
            {
                mollm::detail::RecurrentCheckpoint other;
                const std::vector<float> changed(4,-99);
                check(cuda.copy_from_host(changed.data(),device_tensor,16) &&
                      other.capture(cuda,{&device_tensor}), "second immutable device snapshot");
                check(snapshot.restore(cuda) && cuda.copy_to_host(device_tensor,restored.data(),16) && restored==original,
                      "live snapshots do not alias their pooled storage");
                check(other.restore(cuda) && cuda.copy_to_host(device_tensor,restored.data(),16) && restored==changed,
                      "second pooled snapshot retains its own state");
                check(snapshot.restore(cuda), "restore original after second snapshot");
            }
            Tensor view=device_tensor;view.shape[0]=2;view.compute_strides();
            view.device.offset+=sizeof(float);
            check(snapshot.capture(cuda,{&view}) && snapshot.bytes()==8 &&
                  cuda.zero_tensor(view,8) && snapshot.restore(cuda) &&
                  cuda.copy_to_host(device_tensor,restored.data(),16) && restored==original,
                  "offset view restores only its own storage span");
            Tensor metadata=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,4);
            cuda.alloc_persistent(metadata,16,PersistentHostAccess::HOST_AUTHORITATIVE_PREFIX,8);
            check(!cuda.supports_state_snapshot({&metadata}), "host metadata stays on transfer hooks");
            check(snapshot.capture(cuda,{&device_tensor,&metadata}) && !snapshot.device_resident() && snapshot.bytes()==32,
                  "mixed metadata collection falls back to host storage");
        }
    } else {
        std::printf("CUDA checkpoint test skipped: device unavailable\n");
    }
#endif
    std::printf("recurrent checkpoint: %d failures\n", failed);
    return failed ? 1 : 0;
}
