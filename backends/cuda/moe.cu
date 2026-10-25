#include "backends/cuda/backend_internal.h"
#include "backends/cuda/prefill_batch_policy.h"
#include "kernels/cuda/moe.h"
#include "kernels/cpu/moe/moe_routing.h"
#include "runtime/expert_provider.h"
#include <cmath>

size_t CudaBackend::nvfp4_activation_fused_calls() const {return impl_->moe_mma_activation_fused_prefill_calls;}
size_t CudaBackend::nvfp4_gate_scale_cache_bytes() const {return impl_->prefill_gate_scale_cache_bytes;}
size_t CudaBackend::nvfp4_input_kmajor_calls() const {return impl_->prefill_input_scale_kmajor_calls;}
size_t CudaBackend::nvfp4_token_pair_word_calls() const {return impl_->prefill_token_pair_word_calls;}
size_t CudaBackend::nvfp4_token_code_kmajor_calls() const {return impl_->prefill_token_code_kmajor_calls;}

using mollm::cuda_backend_detail::device_pointer;
using mollm::cuda_backend_detail::device_pointer_const;
using mollm::cuda_backend_detail::fp32_contiguous;
using mollm::cuda_backend_detail::same_shape;

bool CudaBackend::try_fp16_moe(const GraphNode& node,
    const std::vector<const Tensor*>& inputs, Tensor* output) {
    if (inputs.size() != 8 || !output ||
        !std::all_of(inputs.begin(), inputs.end(), [](const Tensor* t) { return t; }) ||
        inputs[2]->prec != Precision::FP16 || inputs[3]->prec != Precision::FP16)
        return false;
    const auto& p = node.params;
    const int hidden = graph_params::get_i32(p, 0, 0);
    const int experts = graph_params::get_i32(p, 1, 0);
    const int top_k = graph_params::get_i32(p, 2, 0);
    const int inter = graph_params::get_i32(p, 3, 0);
    const int shared_inter = graph_params::get_i32(p, 4, inter);
    const int score_func = graph_params::get_i32(p, 5, 0);
    const int bias = graph_params::get_i32(p, 11, 8);
    if (hidden <= 0 || inter <= 0 || shared_inter <= 0 || experts <= 0 || experts > 1024 ||
        top_k <= 0 || top_k > std::min(experts, 32) || score_func < 0 || score_func > 2 ||
        graph_params::get_i32(p, 7, 1) != 1 || graph_params::get_i32(p, 8, 1) != 1 ||
        graph_params::get_i32(p, 9, 1) != 1 || graph_params::get_i32(p, 10, 1) != 1 || (bias >= 0 && bias < 8) ||
        graph_params::get_i32(p, 12, -1) >= 0 || graph_params::get_i32(p, 13, -1) >= 0 ||
        graph_params::get_f32(p, 1, 0) != 0 || !fp32_contiguous(*inputs[0]) ||
        !fp32_contiguous(*output) || !same_shape(*inputs[0], *output) ||
        inputs[0]->shape[0] != hidden || inputs[0]->shape[2] != 1 || inputs[0]->shape[3] != 1 ||
        inputs[0]->shape[1] <= 0 || inputs[0]->shape[1] > 65535 / top_k ||
        static_cast<int64_t>(experts) * 2 * inter > INT32_MAX ||
        static_cast<int64_t>(experts) * hidden > INT32_MAX)
        return false;
    const auto* gate_weight = impl_->find_weight(*inputs[2]);
    const auto* down_weight = impl_->find_weight(*inputs[3]);
    if (!gate_weight || !down_weight || gate_weight->type != CUDA_R_16F || down_weight->type != CUDA_R_16F ||
        gate_weight->layout != Impl::WeightLayout::Dense || down_weight->layout != Impl::WeightLayout::Dense ||
        gate_weight->n != experts * 2 * inter || gate_weight->k != hidden ||
        down_weight->n != experts * hidden || down_weight->k != inter ||
        inputs[2]->device.offset || inputs[3]->device.offset) return false;
    for (int index : {1, 4, 5, 6, 7}) {
        const auto* weight = impl_->find_weight(*inputs[index]);
        const int n = index == 1 ? experts : index == 6 ? hidden : index == 7 ? 1 : shared_inter;
        const int k = index == 6 ? shared_inter : hidden;
        if (!weight || weight->n != n || weight->k != k || inputs[index]->device.offset) return false;
    }
    const float* x = device_pointer_const<float>(*inputs[0]);
    float* destination = device_pointer<float>(*output);
    if (!x || !destination) return false;
    auto fail = [&]() { impl_->failed = true; return true; };
    const int rows = static_cast<int>(inputs[0]->shape[1]);
    const int routes = rows * top_k;
    const size_t count = static_cast<size_t>(rows) *
        (static_cast<size_t>(experts) + 3 * static_cast<size_t>(shared_inter) + hidden + 1) +
        static_cast<size_t>(routes) * (3 * static_cast<size_t>(inter) + hidden + 1);
    if (!impl_->reserve(impl_->moe_scratch, impl_->moe_scratch_bytes, count * sizeof(float)) ||
        !impl_->reserve(impl_->moe_jobs, impl_->moe_jobs_bytes,
                        static_cast<size_t>(routes) * 2 * sizeof(mollm_cuda::Nvfp4Projection))) return fail();
    auto found = std::find_if(impl_->fp16_moe_tables.begin(), impl_->fp16_moe_tables.end(),
        [&](const auto& t) { return t.gate == gate_weight->data && t.down == down_weight->data &&
            t.experts == experts && t.hidden == hidden && t.inter == inter; });
    if (found == impl_->fp16_moe_tables.end()) {
        std::vector<mollm_cuda::Nvfp4Projection> tables(2 * experts);
        for (int e = 0; e < experts; ++e) {
            tables[e].weight = static_cast<const uint8_t*>(gate_weight->data) +
                static_cast<size_t>(e) * 2 * inter * hidden * sizeof(__half);
            tables[experts + e].weight = static_cast<const uint8_t*>(down_weight->data) +
                static_cast<size_t>(e) * hidden * inter * sizeof(__half);
        }
        void* device_tables = nullptr;
        if (!mollm_cuda::malloc_device(&device_tables, tables.size() * sizeof(tables[0]), "FP16 expert tables")) return fail();
        impl_->device_allocations.push_back(device_tables);
        if (!mollm_cuda::copy_memory(device_tables, tables.data(), tables.size() * sizeof(tables[0]),
                                    cudaMemcpyHostToDevice, "FP16 expert tables")) return fail();
        impl_->fp16_moe_tables.push_back({gate_weight->data, down_weight->data, experts, hidden, inter,
                                        static_cast<mollm_cuda::Nvfp4Projection*>(device_tables)});
        found = impl_->fp16_moe_tables.end() - 1;
    }
    float* logits = static_cast<float*>(impl_->moe_scratch);
    float* gate_up = logits + static_cast<size_t>(rows) * experts;
    float* activated = gate_up + static_cast<size_t>(routes) * 2 * inter;
    float* down = activated + static_cast<size_t>(routes) * inter;
    float* shared_g = down + static_cast<size_t>(routes) * hidden;
    float* shared_u = shared_g + static_cast<size_t>(rows) * shared_inter;
    float* shared_a = shared_u + static_cast<size_t>(rows) * shared_inter;
    float* shared_d = shared_a + static_cast<size_t>(rows) * shared_inter;
    float* shared_scale = shared_d + static_cast<size_t>(rows) * hidden;
    float* route_weights = shared_scale + rows;
    auto* jobs = static_cast<mollm_cuda::Nvfp4Projection*>(impl_->moe_jobs);
    auto matmul = [&](const float* input, int index, float* out, int n, int k) {
        return impl_->run_matmul_device(input, k, *inputs[index], out, n, rows, n, k,
                                       Activation::NONE, 0, -1);
    };
    if (!matmul(x, 1, logits, experts, hidden)) return fail();
    mollm_cuda::launch_moe_resident_routes(logits, rows, experts, top_k, score_func,
        graph_params::get_i32(p, 6, 1) != 0, graph_params::get_f32(p, 0, 1),
        found->tables, x, gate_up, activated, down, hidden, inter,
        jobs, jobs + routes, route_weights);
    if (!mollm_cuda::report_cuda(mollm_cuda::launch_fp16_expert_projections(jobs, routes, 2 * inter, hidden), "FP16 expert gate/up")) return fail();
    mollm_cuda::launch_moe_activation(gate_up, gate_up + inter, route_weights,
                                     activated, routes, inter, 2 * inter);
    if (!mollm_cuda::report_cuda(mollm_cuda::launch_fp16_expert_projections(jobs + routes, routes, hidden, inter), "FP16 expert down")) return fail();
    if (!matmul(x, 4, shared_g, shared_inter, hidden) || !matmul(x, 5, shared_u, shared_inter, hidden)) return fail();
    mollm_cuda::launch_moe_activation(shared_g, shared_u, nullptr, shared_a, rows, shared_inter, shared_inter);
    if (!matmul(shared_a, 6, shared_d, hidden, shared_inter) || !matmul(x, 7, shared_scale, 1, hidden)) return fail();
    mollm_cuda::launch_moe_combine(down, shared_d, shared_scale, destination, rows, hidden, top_k);
    if (!mollm_cuda::report_cuda(cudaGetLastError(), "FP16 MoE combine")) return fail();
    return true;
}

bool CudaBackend::dispatch_moe_pair(const GraphNode& node,
    const std::vector<const Tensor*>& a, const std::vector<const Tensor*>& b,
    Tensor* output_a, Tensor* output_b) {
    if (impl_->segment_enabled ||
        node.op_type != OpType::MOE || a.size() != 8 || b.size() != 8 ||
        !output_a || !output_b || !a[0] || !b[0] ||
        !fp32_contiguous(*a[0]) || !fp32_contiguous(*b[0]) ||
        !fp32_contiguous(*output_a) || !fp32_contiguous(*output_b) ||
        !same_shape(*a[0], *output_a) || !same_shape(*b[0], *output_b) ||
        a[0]->shape[0] != 2560 || b[0]->shape[0] != 2560 ||
        a[0]->shape[2] != 1 || a[0]->shape[3] != 1 ||
        b[0]->shape[2] != 1 || b[0]->shape[3] != 1 ||
        a[0]->shape[1] < 1 || a[0]->shape[1] > 1024 ||
        b[0]->shape[1] < 1 || b[0]->shape[1] > 1024) return false;
    for (int i = 1; i < 8; ++i) {
        if (!a[i] || !b[i] || a[i]->data != b[i]->data ||
            a[i]->device.buffer != b[i]->device.buffer ||
            a[i]->device.offset != b[i]->device.offset ||
            a[i]->moe_ssd_source != b[i]->moe_ssd_source ||
            a[i]->prec != b[i]->prec || !same_shape(*a[i], *b[i])) return false;
    }
    if (a[2]->prec != Precision::NVFP4 || a[3]->prec != Precision::NVFP4 ||
        graph_params::get_i32(node.params, 0, 0) != 2560 ||
        graph_params::get_i32(node.params, 1, 0) != 512 ||
        graph_params::get_i32(node.params, 2, 0) != 10 ||
        !device_pointer_const<float>(*a[0]) || !device_pointer_const<float>(*b[0]) ||
        !device_pointer<float>(*output_a) || !device_pointer<float>(*output_b)) return false;
    for(const auto* group:{&a,&b})for(const auto* input:*group) {
        if(impl_->deferred_rms_overlaps(device_pointer_const<uint8_t>(*input),input->view_span_bytes(),true)&&
            !impl_->materialize_deferred_rms())return true;
        if(!impl_->restore_hc_compact(device_pointer_const<uint8_t>(*input),input->view_span_bytes()))return true;
    }
    for(auto* output:{output_a,output_b}) {
        if(impl_->deferred_rms_overlaps(device_pointer_const<uint8_t>(*output),output->view_span_bytes())&&
            !impl_->materialize_deferred_rms())return true;
        if(!impl_->restore_hc_compact(device_pointer_const<uint8_t>(*output),output->view_span_bytes()))return true;
    }
    struct Temporary {
        CudaBackend& backend; Tensor input, output;
        ~Temporary() {
            if (input.device.buffer) backend.free_output(input, nullptr);
            if (output.device.buffer) backend.free_output(output, nullptr);
        }
    } temp{*this, Tensor::create(Precision::FP32, MemoryType::NONE, 2560,
        a[0]->shape[1] + b[0]->shape[1], 1, 1), Tensor::create(Precision::FP32,
        MemoryType::NONE, 2560, a[0]->shape[1] + b[0]->shape[1], 1, 1)};
    if (!alloc_output(temp.input, temp.input.nbytes(), nullptr) ||
        !alloc_output(temp.output, temp.output.nbytes(), nullptr)) {
        impl_->failed = true; return true;
    }
    auto copy = [&](void* dst, const void* src, size_t bytes) {
        if (mollm_cuda::report_cuda(cudaMemcpyAsync(dst, src, bytes,
            cudaMemcpyDeviceToDevice, nullptr), "MoE paired-window copy")) return true;
        impl_->failed = true; return false;
    };
    auto* input = device_pointer<float>(temp.input);
    if (!copy(input, device_pointer_const<float>(*a[0]), a[0]->nbytes()) ||
        !copy(reinterpret_cast<uint8_t*>(input) + a[0]->nbytes(),
            device_pointer_const<float>(*b[0]), b[0]->nbytes())) return true;
    std::vector<const Tensor*> inputs = a; inputs[0] = &temp.input;
    struct MergeScope {
        bool& active; bool previous;
        ~MergeScope() { active = previous; }
    } scope{moe_window_merge_active_, moe_window_merge_active_};
    moe_window_merge_active_ = true;
    if (!try_nvfp4_moe(node, inputs, &temp.output)) return false;
    if (impl_->failed) return true;
    const auto* output = device_pointer_const<float>(temp.output);
    if (!copy(device_pointer<float>(*output_a), output, output_a->nbytes()) ||
        !copy(device_pointer<float>(*output_b), reinterpret_cast<const uint8_t*>(output) +
            output_a->nbytes(), output_b->nbytes())) return true;
    ++impl_->native_ops[static_cast<uint32_t>(OpType::MOE)];
    return true;
}

bool CudaBackend::try_nvfp4_moe(
    const GraphNode& node, const std::vector<const Tensor*>& inputs, Tensor* output) {
    static const bool disabled = std::getenv("MOLLM_CUDA_DISABLE_NVFP4_MOE") != nullptr;
    if (disabled || !impl_->coherent_host_weights || inputs.size() != 8 || !output ||
        !std::all_of(inputs.begin(), inputs.end(), [](const Tensor* t) { return t != nullptr; }))
        return false;
    const auto& p = node.params;
    const int hidden = graph_params::get_i32(p, 0, 0);
    const int experts = graph_params::get_i32(p, 1, 0);
    const int top_k = graph_params::get_i32(p, 2, 0);
    const int inter = graph_params::get_i32(p, 3, 0);
    const int shared_inter = graph_params::get_i32(p, 4, inter);
    const int bias_input = graph_params::get_i32(p, 11, 8);
    if (hidden <= 0 || inter <= 0 || shared_inter <= 0 || experts <= 0 ||
        top_k <= 0 || top_k > experts || hidden % 16 || inter % 16 ||
        graph_params::get_i32(p, 7, 1) != 1 ||
        graph_params::get_i32(p, 10, 1) != 1 ||
        (bias_input >= 0 && bias_input < static_cast<int>(inputs.size())) ||
        graph_params::get_i32(p, 12, -1) >= 0 ||
        graph_params::get_i32(p, 13, -1) >= 0 ||
        graph_params::get_f32(p, 1, 0.0f) != 0.0f ||
        !fp32_contiguous(*inputs[0]) || !fp32_contiguous(*output) ||
        !same_shape(*inputs[0], *output) || inputs[0]->shape[0] != hidden ||
        inputs[0]->shape[2] != 1 || inputs[0]->shape[3] != 1 ||
        inputs[0]->shape[1] <= 0 || inputs[2]->prec != Precision::NVFP4 ||
        inputs[3]->prec != Precision::NVFP4)
        return false;
    const ExpertSource* gate_source = inputs[2]->moe_ssd_source;
    const ExpertSource* down_source = inputs[3]->moe_ssd_source;
    if (!gate_source || !down_source || !gate_source->provider ||
        gate_source->provider != down_source->provider) return false;
    // Prefer calibrated native NVFP4 prefill MMA when the device and package
    // support it; unsupported configurations fall back to the SIMT path.
    bool native_mma = inputs[0]->shape[1] > 1;
    constexpr bool simt_pipeline=true;
    static const bool batch_pipeline = [] {
        const char* value=std::getenv("MOLLM_CUDA_NVFP4_BATCH_PIPELINE");
        return value && std::strcmp(value,"1")==0;
    }();
    const bool gate_codes_kmajor = gate_source->resident_nvfp4_gate_codes_kmajor;
    const bool gate_scales_kmajor=gate_source->resident_nvfp4_gate_scales_kmajor;
    if((gate_scales_kmajor&&!gate_codes_kmajor)||down_source->resident_nvfp4_gate_scales_kmajor)
        return false;
    if (gate_codes_kmajor && (!gate_source->resident_device_views ||
        gate_source->resident_nvfp4_mma_layout || down_source->resident_nvfp4_mma_layout ||
        down_source->resident_nvfp4_gate_codes_kmajor || hidden != 2560 || inter != 640))
        return false;
    static const int native_gate_parts = [] {
        const char* value = std::getenv("MOLLM_CUDA_NVFP4_MMA_GATE_PARTS");
        if (value && std::strcmp(value, "2") == 0) return 2;
        if (value && std::strcmp(value, "4") == 0) return 4;
        if (value && std::strcmp(value, "8") == 0) return 8;
        return 1;
    }();
    if (native_mma &&
        (!impl_->supports_native_nvfp4_mma ||
         gate_source->resident_nvfp4_mma_layout != down_source->resident_nvfp4_mma_layout ||
         hidden > 2560 || inter > 2560 || hidden % 64 || inter % 64 ||
         gate_source->nvfp4_input_scales_per_expert != 2 ||
         down_source->nvfp4_input_scales_per_expert != 1 ||
         gate_source->nvfp4_input_scales.size() != static_cast<size_t>(experts) * 2 ||
         down_source->nvfp4_input_scales.size() != static_cast<size_t>(experts))) {
        native_mma = false;
    }
    const bool canonical_mma = native_mma && !gate_source->resident_nvfp4_mma_layout;
    for (int index : {1, 4, 5, 6, 7}) {
        const auto* prepared = impl_->find_weight(*inputs[index]);
        const int n = index == 1 ? experts : index == 6 ? hidden : index == 7 ? 1 : shared_inter;
        const int k = index == 6 ? shared_inter : hidden;
        if (!prepared || prepared->n != n || prepared->k != k || inputs[index]->device.offset)
            return false;
    }
    const float* x = device_pointer_const<float>(*inputs[0]);
    float* destination = device_pointer<float>(*output);
    if (!x || !destination) return false;
    const int rows = static_cast<int>(inputs[0]->shape[1]);
    const size_t routes = static_cast<size_t>(rows) * top_k;
    if (routes > 65535) return false; // CUDA grid.y limit; retain the generic bridge.
    static const int reuse_prefill_input=[] {
        const char* value=std::getenv("MOLLM_CUDA_MOE_PREFILL_INPUT_HALF_REUSE");
        return value?int(mollm_cuda::logical_prefill_option(value,false)):-1;
    }();
    bool half_reuse=(reuse_prefill_input<0?logical_prefill_batch_active_:reuse_prefill_input!=0)&&rows>=16&&rows<=2048&&hidden<=2560&&
        hidden%32==0&&!impl_->capture_audit;
    for(int index:{1,4,5,7}) {
        const auto layout=impl_->find_weight(*inputs[index])->layout;
        half_reuse=half_reuse&&(layout==Impl::WeightLayout::Q4Bg32Biased||
            layout==Impl::WeightLayout::Q4Bg32Compact||
            layout==Impl::WeightLayout::Q4Bg32CompactInteger);
    }
    const size_t count = static_cast<size_t>(rows) *
        (static_cast<size_t>(experts) + 3 * static_cast<size_t>(shared_inter) + hidden + 1) +
        routes * (3 * static_cast<size_t>(inter) + hidden + 1);
    auto fail = [&]() {
        // Every lease and host descriptor remains alive until GPU reads end,
        // including failures after partially enqueuing the shared projection.
        cudaDeviceSynchronize();
        impl_->failed = true;
        return true;
    };
    const size_t half_elements=half_reuse?size_t(rows)*hidden:0;
    const size_t half_offset=half_reuse?(count+63)&~size_t(63):count;
    if (!impl_->reserve(impl_->moe_scratch, impl_->moe_scratch_bytes,
            half_offset*sizeof(float)+half_elements*sizeof(__half)))
        return fail();
    float* logits = static_cast<float*>(impl_->moe_scratch);
    float* gate_up = logits + static_cast<size_t>(rows) * experts;
    float* activated = gate_up + routes * 2 * inter;
    float* down = activated + routes * inter;
    float* shared_g = down + routes * hidden;
    float* shared_u = shared_g + static_cast<size_t>(rows) * shared_inter;
    float* shared_a = shared_u + static_cast<size_t>(rows) * shared_inter;
    float* shared_d = shared_a + static_cast<size_t>(rows) * shared_inter;
    float* shared_scale = shared_d + static_cast<size_t>(rows) * hidden;
    float* device_route_weights = shared_scale + rows;
    // This shadow belongs to this MoE invocation, not to a graph tensor or
    // recycled input address. Default-stream consumers finish before scratch
    // reuse; the decode-only shared stream never enters this rows>=16 path.
    __half* shared_input_half=half_reuse?
        reinterpret_cast<__half*>(static_cast<float*>(impl_->moe_scratch)+half_offset):nullptr;
    bool shared_input_ready=false;
    const auto projection = [](const Tensor& t, const float* a, float* c) {
        return mollm_cuda::Nvfp4Projection{
            static_cast<const uint8_t*>(t.data), t.nvfp4_scales,
            t.nvfp4_row_scales, a, c, t.nvfp4_q8_pair_data != nullptr, t.nvfp4_mma_layout,
            t.nvfp4_gate_scales_kmajor};
    };
    const auto valid = [](const Tensor& t, int n, int k) {
        return t.prec == Precision::NVFP4 && t.data && t.nvfp4_scales &&
            t.nvfp4_row_scales && t.shape[0] == n && t.shape[1] == k &&
            t.shape[2] == 1 && t.shape[3] == 1 &&
            t.group_size == 16 && t.groups_per_row == static_cast<unsigned>(k / 16) &&
            (!t.nvfp4_q8_pair_data || t.nvfp4_q8_pair_data == t.data);
    };
    cudaStream_t execution_stream = nullptr;
    bool down_output_bf16 = false;
    const auto matmul = [&](const float* a, int index, float* out, int n, int k) {
        const bool reuse=shared_input_half&&a==x&&k==hidden&&!execution_stream;
        if(reuse&&!shared_input_ready) {
            mollm_cuda::launch_fp32_to_fp16(x,shared_input_half,half_elements);
            if(!mollm_cuda::report_cuda(cudaGetLastError(),"MoE shared FP16 input"))return false;
            shared_input_ready=true;
            ++impl_->moe_prefill_input_half_conversions;
        }
        if(reuse)++impl_->moe_prefill_input_half_consumers;
        return impl_->run_matmul_device(a, k, *inputs[index], out, n,
            rows, n, k, Activation::NONE, 0, -1, execution_stream,
            reuse?shared_input_half:nullptr,reuse);
    };
    const auto shared_branch = [&]() {
        if (!matmul(x, 4, shared_g, shared_inter, hidden) ||
            !matmul(x, 5, shared_u, shared_inter, hidden)) return false;
        mollm_cuda::launch_moe_activation(shared_g, shared_u, nullptr,
            shared_a, rows, shared_inter, shared_inter, execution_stream);
        if (!matmul(shared_a, 6, shared_d, hidden, shared_inter) ||
            !matmul(x, 7, shared_scale, 1, hidden)) return false;
        return !impl_->failed && mollm_cuda::report_cuda(cudaGetLastError(), "shared expert branch");
    };
    const auto combine = [&](bool synchronous) {
        if(impl_->moe_hc_chain.armed&&!synchronous&&down_output_bf16&&rows>=16&&rows<=2048&&
           hidden==2560&&top_k==10&&(execution_stream==nullptr||execution_stream==cudaStreamPerThread)){
            impl_->moe_hc_chain={true,true,reinterpret_cast<const uint16_t*>(down),shared_d,shared_scale,destination,rows,execution_stream};
            return true;
        }
        if (down_output_bf16) {
            if (!mollm_cuda::report_cuda(mollm_cuda::launch_moe_combine_bf16(
                    reinterpret_cast<const uint16_t*>(down), shared_d, shared_scale,
                    destination, rows, hidden, top_k, execution_stream),
                    "NVFP4 BF16 down combine")) return fail();
        } else mollm_cuda::launch_moe_combine(down, shared_d, shared_scale,
                    destination, rows, hidden, top_k, execution_stream);
        if (!mollm_cuda::report_cuda(cudaGetLastError(), "NVFP4 MoE launch") ||
            (synchronous && !mollm_cuda::report_cuda(cudaDeviceSynchronize(),
                "NVFP4 MoE lease completion"))) return fail();
        return true;
    };
    const auto finish = [&](bool synchronous) {
        if (!shared_branch()) return fail();
        return combine(synchronous);
    };
    static const bool host_routing = std::getenv("MOLLM_CUDA_DISABLE_RESIDENT_ROUTES") != nullptr;
    const int score_func = graph_params::get_i32(p, 5, 0);
    if (!host_routing && experts <= 1024 && top_k <= 32 &&
        score_func >= 0 && score_func <= 2 && graph_params::get_i32(p, 8, 1) <= 1) {
        auto found = impl_->resident_experts.find(gate_source);
        if (found == impl_->resident_experts.end()) {
            Impl::ResidentExperts resident;
            if (gate_source->provider->borrow_resident_layer(gate_source, down_source, resident.leases) &&
                resident.leases.size() == static_cast<size_t>(experts) &&
                std::all_of(resident.leases.begin(), resident.leases.end(), [&](const ExpertLease& lease) {
                    return valid(lease.gate_up, 2 * inter, hidden) && valid(lease.down, hidden, inter)&&
                        lease.gate_up.nvfp4_gate_scales_kmajor==gate_scales_kmajor&&
                        !lease.down.nvfp4_gate_scales_kmajor;
                })) {
                std::vector<mollm_cuda::Nvfp4Projection> tables(2 * experts);
                resident.gate_async_aligned = gate_source->resident_device_views &&
                    std::all_of(resident.leases.begin(),resident.leases.end(),[](const ExpertLease& lease) {
                        return !(reinterpret_cast<uintptr_t>(lease.gate_up.data)&15) &&
                            !(reinterpret_cast<uintptr_t>(lease.gate_up.nvfp4_scales)&3);
                    });
                for (int i = 0; i < experts; ++i) {
                    tables[i] = projection(resident.leases[i].gate_up, nullptr, nullptr);
                    tables[experts + i] = projection(resident.leases[i].down, nullptr, nullptr);
                    if (native_mma) {
                        tables[i].input_global = gate_source->nvfp4_input_scales[i * 2];
                        tables[i].input_global_second = gate_source->nvfp4_input_scales[i * 2 + 1];
                        tables[experts + i].input_global = down_source->nvfp4_input_scales[i];
                    }
                }
                resident.gate_pair_layout=std::all_of(tables.begin(),tables.begin()+experts,
                    [](const mollm_cuda::Nvfp4Projection& job){return job.pair_layout;});
                // Check the actual descriptors uploaded for this resident layer,
                // not a checkpoint-wide assumption. Different expert/up scales
                // must retain per-route quantization.
                resident.gate_input_global=tables.front().input_global;
                resident.uniform_gate_input_global=std::isfinite(resident.gate_input_global)&&
                    resident.gate_input_global>0.f&&
                    std::all_of(tables.begin(),tables.begin()+experts,[&](const auto& job) {
                        return job.input_global==resident.gate_input_global&&
                            job.input_global_second==resident.gate_input_global;
                    });
                void* device_tables = nullptr;
                if (!mollm_cuda::malloc_device(&device_tables,
                        tables.size() * sizeof(tables[0]), "resident MoE descriptors")) return fail();
                if (!mollm_cuda::copy_memory(device_tables, tables.data(),
                        tables.size() * sizeof(tables[0]), cudaMemcpyHostToDevice,
                        "resident MoE descriptors")) {
                    cudaFree(device_tables); return fail();
                }
                resident.tables = static_cast<mollm_cuda::Nvfp4Projection*>(device_tables);
                resident.down_source = down_source;
                resident.hidden = hidden; resident.inter = inter;
                found = impl_->resident_experts.emplace(gate_source, std::move(resident)).first;
            }
        }
        if (found != impl_->resident_experts.end() && found->second.down_source == down_source &&
            found->second.hidden == hidden && found->second.inter == inter &&
            found->second.leases.size() == static_cast<size_t>(experts)) {
            static const bool no_grouping = std::getenv("MOLLM_CUDA_DISABLE_NVFP4_GROUPING") != nullptr;
            // Qualified grouped batched prefill; per-shape guards below.
            constexpr bool batch_prefill=true;
            // Canonical and native fragment loads may have different crossover
            // points. Keep established defaults and permit bounded tuning.
            // Qualified grouped-MMA policy: no scalar singleton fallback.
            constexpr int batch_gate_min=1;
            constexpr int batch_down_min=1;
            static const bool prepare_singles = [] {
                const char* value = std::getenv("MOLLM_CUDA_NVFP4_MMA_PREPARE_SINGLES");
                return value && std::strcmp(value, "1") == 0;
            }();
            // Admit grouped batching from 32 rows so short prompts benefit.
            constexpr bool batch_small_prefill=true;
            static const int sort_override = [] {
                const char* option = std::getenv("MOLLM_CUDA_NVFP4_PREFILL_SORT");
                if (!option) option = std::getenv("MOLLM_CUDA_NVFP4_MMA_PREFILL_SORT");
                if (!option) return -1;
                return std::strcmp(option, "1") == 0 ? 1 : 0;
            }();
            const bool sort_prefill = sort_override >= 0 ? sort_override != 0 :
                impl_->supports_extended_nvfp4_prefill_sort;
            const bool batch_candidate = native_mma && batch_prefill &&
                rows >= (batch_small_prefill ? 32 : 128) && native_gate_parts == 1;
            const bool sort_candidate = sort_prefill && rows >= 128;
            // Qualified large-route grouping (removes the 4096-route cliff).
            constexpr bool tiled_large_group=true;
            constexpr bool radix_prefill_group=false;
            const bool extended_group = (tiled_large_group || radix_prefill_group) &&
                (batch_candidate || sort_candidate) && routes > 4096 && routes <= 65535;
            const bool group_jobs = !no_grouping && rows > 1 &&
                (extended_group || routes <= ((batch_candidate || sort_candidate) ? 4096 : 2048));
            if (sort_candidate && group_jobs) ++impl_->moe_sorted_prefill_calls;
            const bool use_batched = batch_candidate && group_jobs;
            const bool use_radix_group = radix_prefill_group && group_jobs &&
                (batch_candidate || sort_candidate);
            if (use_radix_group) {
                size_t scratch_bytes = 0;
                if (!mollm_cuda::report_cuda(mollm_cuda::nvfp4_radix_group_scratch_size(
                        routes, &scratch_bytes), "NVFP4 radix scratch query") ||
                    !impl_->reserve(impl_->moe_radix_group_scratch,
                        impl_->moe_radix_group_scratch_bytes, scratch_bytes)) return fail();
            }
            constexpr int compact_scope=0;
            const bool compact_gate_dispatch = (compact_scope & 1) != 0;
            const bool compact_down_dispatch = (compact_scope & 2) != 0;
            const bool compact_dispatch = compact_scope != 0;
            constexpr int mma_batch_routes=8;
            // Gate/up use the qualified width-32 grouped MMA.
            constexpr int gate_batch_routes=32;
            constexpr bool specialized_prepared=false;
            // Qualified N32 async gate staging.
            constexpr bool async_gate_requested=true;
            constexpr bool async_gate_four_warps=true;
            const bool async_gate = async_gate_requested && async_gate_four_warps &&
                rows>=16 && hidden==2560 && inter==640 && canonical_mma &&
                gate_batch_routes==32 && !compact_gate_dispatch && !specialized_prepared &&
                found->second.gate_async_aligned;
            // Qualified fused prepare: packed-only mode 2.
            constexpr int activation_prepare_fusion=2;
            // Qualified BF16 gate/up epilogue boundary.
            constexpr bool bf16_gate_requested=true;
            const bool gate_output_bf16=bf16_gate_requested && rows<=4096 &&
                async_gate && use_batched && batch_gate_min==1 && !prepare_singles &&
                activation_prepare_fusion;
            // Qualified early input-fragment loads before the weight wait.
            constexpr bool early_inputs_requested=true;
            const bool gate_early_inputs=early_inputs_requested && gate_output_bf16 &&
                found->second.gate_pair_layout;
            // Qualified vectorized gate input quantization.
            constexpr bool vector_quantize_requested=true;
            const bool gate_vector_quantize=vector_quantize_requested && gate_output_bf16;
            // Qualified warp-ballot plan builder.
            constexpr bool parallel_plan_requested=true;
            const bool parallel_plan=parallel_plan_requested && rows>=16 && rows<=4096 &&
                use_batched && canonical_mma && !compact_gate_dispatch && !compact_down_dispatch &&
                !specialized_prepared && !prepare_singles;
            // Qualified BF16 down epilogue boundary.
            constexpr bool bf16_down_requested=true;
            // Pair the tightly packed producer and consumer; every unsupported
            // configuration retains FLOAT output. Reuse existing scratch, no
            // additional model cache or weight layout/conversion changes.
            down_output_bf16 = bf16_down_requested && rows>=16 && rows<=4096 &&
                hidden==2560 && inter==640 && native_mma && use_batched &&
                canonical_mma && mma_batch_routes==8 && batch_down_min==1 &&
                !prepare_singles && !compact_down_dispatch && !specialized_prepared &&
                async_gate_four_warps;
            // Qualified fused MMA/activation/quantize dataflow.
            constexpr bool mma_activation_requested=true;
            const mollm_cuda::Nvfp4PrefillBatchPolicy batch_policy{
                logical_prefill_batch_active_,moe_window_merge_active_};
            static const int input_scale_kmajor_override=[] {
                const char* value=std::getenv("MOLLM_CUDA_NVFP4_PREFILL_INPUT_SCALE_KMAJOR");
                return value?int(mollm_cuda::logical_prefill_option(value,false)):-1;
            }();
            static const int compact_logits_override=[] {
                const char* value=std::getenv("MOLLM_CUDA_NVFP4_PREFILL_COMPACT_LOGITS");
                return value?int(mollm_cuda::logical_prefill_option(value,false)):-1;
            }();
            static const int token_quantization_override=[] {
                const char* value=std::getenv("MOLLM_CUDA_NVFP4_PREFILL_TOKEN_QUANTIZE");
                return value?int(mollm_cuda::logical_prefill_option(value,false)):-1;
            }();
            static const int token_pair_words_override=[] {
                const char* value=std::getenv("MOLLM_CUDA_NVFP4_PREFILL_TOKEN_PAIR_WORDS");
                return value?int(mollm_cuda::logical_prefill_option(value,false)):-1;
            }();
            const auto requested=[&](int option) {
                return option<0?logical_prefill_batch_active_:option!=0;
            };
            bool mma_activation_fused=mma_activation_requested&&batch_policy.fused_rows(rows)&&
                gate_output_bf16&&down_output_bf16&&gate_vector_quantize&&parallel_plan&&
                gate_early_inputs&&group_jobs&&activation_prepare_fusion==2;
            // K-major gate codes have a fused consumer only with paired token
            // inputs. Qualify all non-cache prerequisites BEFORE allocating a
            // scale view: a rejected path must not leave an unused resident
            // copy behind (4.69 GiB for the full Qwen3.8 model).
            if(gate_codes_kmajor&&!(requested(input_scale_kmajor_override)&&
                batch_policy.kmajor_rows(rows)&&requested(compact_logits_override)&&
                rows<=2048&&hidden==2560&&inter==640&&
                requested(token_quantization_override)&&found->second.uniform_gate_input_global&&
                !(reinterpret_cast<uintptr_t>(x)&15)&&requested(token_pair_words_override)&&
                found->second.gate_pair_layout))mma_activation_fused=false;
            // Explicit total budget; native descriptors remain immutable for
            // decode and every unfused fallback. Never allocate during capture.
            static const size_t scale_cache_limit=[] {
                const char* value=std::getenv("MOLLM_CUDA_PREFILL_GATE_SCALE_CACHE_MB");
                if(!value||!*value||*value<'0'||*value>'9')return size_t{0};
                char* end=nullptr;
                const auto mb=std::strtoull(value,&end,10);
                return end&&!*end&&mb<=8192?size_t(mb)*1024*1024:size_t{0};
            }();
            auto& resident=found->second;
            const size_t scale_view_bytes=size_t(experts)*204800;
            const size_t scale_table_bytes=size_t(experts)*2*sizeof(mollm_cuda::Nvfp4Projection);
            const size_t scale_cache_bytes=scale_view_bytes+scale_table_bytes;
            if(mma_activation_fused&&rows>1&&hidden==2560&&inter==640&&
                resident.gate_async_aligned&&!gate_scales_kmajor&&!resident.prefill_scale_tables&&
                impl_->prefill_gate_scale_cache_bytes<=scale_cache_limit&&
                scale_cache_bytes<=scale_cache_limit-impl_->prefill_gate_scale_cache_bytes) {
                cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
                if(!mollm_cuda::report_cuda(cudaStreamIsCapturing(execution_stream,&capture),
                        "gate scale cache capture check"))return fail();
                if(capture==cudaStreamCaptureStatusNone) {
                    void* view=nullptr;void* tables=nullptr;
                    if(!mollm_cuda::malloc_device(&view,scale_view_bytes,"prefill gate scale view"))return fail();
                    if(!mollm_cuda::malloc_device(&tables,scale_table_bytes,"prefill gate scale descriptors")) {
                        cudaFree(view);return fail();
                    }
                    if(!mollm_cuda::report_cuda(mollm_cuda::launch_nvfp4_gate_scale_view(
                            resident.tables,static_cast<mollm_cuda::Nvfp4Projection*>(tables),
                            static_cast<uint8_t*>(view),experts,execution_stream),"prefill gate scale transpose")||
                        !mollm_cuda::report_cuda(cudaStreamSynchronize(execution_stream),"prefill gate scale readiness")) {
                        cudaFree(tables);cudaFree(view);return fail();
                    }
                    resident.prefill_scale_view=static_cast<uint8_t*>(view);
                    resident.prefill_scale_tables=static_cast<mollm_cuda::Nvfp4Projection*>(tables);
                    impl_->prefill_gate_scale_cache_bytes+=scale_cache_bytes;
                }
            }
            bool use_scale_view=mma_activation_fused&&rows>1&&
                (gate_scales_kmajor||resident.prefill_scale_tables);
            // Logical prompts also need the optimized layout below1025.
            // Ordinary short windows/decode retain their established policy;
            // paired-window admission still requires rows>1024.
            const bool input_scale_kmajor_requested=input_scale_kmajor_override<0?logical_prefill_batch_active_:input_scale_kmajor_override!=0;
            bool input_scale_kmajor=input_scale_kmajor_requested&&mma_activation_fused&&
                batch_policy.kmajor_rows(rows);
            const bool compact_logits_requested=compact_logits_override<0?logical_prefill_batch_active_:compact_logits_override!=0;
            bool compact_logits=compact_logits_requested&&use_scale_view&&
                input_scale_kmajor&&rows<=2048&&hidden==2560&&inter==640;
            const bool token_quantization_requested=token_quantization_override<0?logical_prefill_batch_active_:token_quantization_override!=0;
            // try_nvfp4_moe requires a contiguous FP32 input. Resident routing
            // derives each input from x+row*hidden; grouping copies that pointer.
            // All consumers remain on execution_stream within this frame.
            bool token_quantization=token_quantization_requested&&compact_logits&&
                resident.uniform_gate_input_global&&!(reinterpret_cast<uintptr_t>(x)&15);
            const bool token_pair_words_requested=token_pair_words_override<0?logical_prefill_batch_active_:token_pair_words_override!=0;
            // Row-layout standalone timings do not establish a gain. Limit
            // runtime selection to uniformly pair-layout resident descriptors.
            const bool token_pair_words=token_pair_words_requested&&token_quantization&&
                resident.gate_pair_layout;
            // New code views have a fused specialization only for paired
            // token inputs. Other lengths/layouts keep canonical scales and
            // the corresponding grouped BF16/FLOAT fallback, never old codes.
            if (gate_codes_kmajor && !token_pair_words) {
                mma_activation_fused=false;
                use_scale_view=false;
                input_scale_kmajor=false;
                compact_logits=false;
                token_quantization=false;
            }
            static const int token_codes_kmajor_requested=[] {
                const char* value=std::getenv("MOLLM_CUDA_NVFP4_PREFILL_TOKEN_CODES_KMAJOR");
                return value?int(mollm_cuda::logical_prefill_option(value,false)):-1;
            }();
            const bool token_codes_kmajor=(token_codes_kmajor_requested<0?logical_prefill_batch_active_:token_codes_kmajor_requested!=0)&&gate_codes_kmajor&&
                mma_activation_fused&&token_pair_words;
            const auto* gate_scale_tables=use_scale_view&&!gate_scales_kmajor?
                resident.prefill_scale_tables:resident.tables;
            if (use_batched && mma_batch_routes == 16) ++impl_->moe_n16_prefill_calls;
            if (use_batched && gate_batch_routes == 32) ++impl_->moe_gate_n32_prefill_calls;
            if (canonical_mma) ++impl_->moe_canonical_mma_prefill_calls;
            if (use_batched) ++impl_->moe_batched_prefill_calls;
            const size_t batch_bytes = (routes * sizeof(mollm_cuda::Nvfp4MmaBatch) + 127) & ~size_t(127);
            const size_t packed_bytes = routes * static_cast<size_t>(std::max(hidden, inter));
            const size_t compact_offset = batch_bytes + packed_bytes + packed_bytes / 8;
            const size_t compact_bytes = compact_dispatch ? (routes + 1) * sizeof(mollm_cuda::Nvfp4MmaBatch) : 0;
            const size_t fused_down_offset=compact_offset+compact_bytes;
            const size_t fused_down_bytes=mma_activation_fused?routes*size_t(inter)*9/16:0;
            const size_t token_indices_offset=(fused_down_offset+fused_down_bytes+3)&~size_t(3);
            const size_t group_scratch_bytes=token_quantization?
                token_indices_offset+routes*sizeof(int):fused_down_offset+fused_down_bytes;
            // A hint, not a correctness bound: the kernel handles overflow if
            // equal-weight sorting yields more runs than the expert count.
            const int compact_grid_y = compact_down_dispatch
                ? static_cast<int>(std::min(routes, (routes + mma_batch_routes - 1) / mma_batch_routes +
                    (static_cast<size_t>(experts) * (mma_batch_routes - 1) + mma_batch_routes - 1) / mma_batch_routes)) : 0;
            const int gate_compact_grid_y = compact_gate_dispatch
                ? static_cast<int>(std::min(routes, (routes + gate_batch_routes - 1) / gate_batch_routes +
                    (static_cast<size_t>(experts) * (gate_batch_routes - 1) + gate_batch_routes - 1) / gate_batch_routes)) : 0;
            if (use_batched && !impl_->reserve(impl_->moe_mma_group_scratch, impl_->moe_mma_group_scratch_bytes,
                    group_scratch_bytes)) return fail();
            if (!impl_->reserve(impl_->moe_jobs, impl_->moe_jobs_bytes,
                    (group_jobs ? 4 : 2) * routes * sizeof(mollm_cuda::Nvfp4Projection))) return fail();
            auto* gate_jobs = static_cast<mollm_cuda::Nvfp4Projection*>(impl_->moe_jobs);
            auto* down_jobs = gate_jobs + routes;
            if (native_mma && native_gate_parts > 1 && !impl_->reserve(impl_->moe_mma_partial, impl_->moe_mma_partial_bytes,
                    static_cast<size_t>(routes) * 2 * inter * native_gate_parts * sizeof(float))) return fail();
            const auto enqueue = [&]() {
                if (!matmul(x, 1, logits, experts, hidden)) return fail();
                // Only canonical single-token Q4 projections are scratch-free
                // on a non-default stream. Batched/FP16/other layouts stay serial.
                bool parallel_shared = impl_->moe_shared_stream && rows == 1;
                for (int index : {4, 5, 6, 7})
                    parallel_shared = parallel_shared &&
                        impl_->find_weight(*inputs[index])->layout == Impl::WeightLayout::Q4Bg32Biased;
                if (parallel_shared) {
                    const cudaStream_t parent = execution_stream;
                    if (!mollm_cuda::report_cuda(cudaEventRecord(impl_->moe_shared_ready, parent), "shared expert fork") ||
                        !mollm_cuda::report_cuda(cudaStreamWaitEvent(impl_->moe_shared_stream, impl_->moe_shared_ready, 0), "shared expert input dependency")) return fail();
                    execution_stream = impl_->moe_shared_stream;
                    const bool shared_ok = shared_branch();
                    const cudaError_t recorded = cudaEventRecord(impl_->moe_shared_done, execution_stream);
                    execution_stream = parent;
                    if (!shared_ok || !mollm_cuda::report_cuda(recorded, "shared expert completion")) return fail();
                    ++impl_->moe_shared_overlap_enqueues;
                }
                mollm_cuda::launch_moe_resident_routes(logits, rows, experts, top_k,
                    score_func, graph_params::get_i32(p, 6, 1) != 0,
                    graph_params::get_f32(p, 0, 1.0f), gate_scale_tables, x,
                    gate_up, activated, down, hidden, inter, gate_jobs, down_jobs,
                    device_route_weights, execution_stream, -1, down_output_bf16,gate_output_bf16);
                static const bool trace_first_routes = [] {
                    const char* value = std::getenv("MOLLM_CUDA_NVFP4_TRACE_FIRST_ROUTES");
                    return value && std::strcmp(value, "1") == 0;
                }();
                static const bool trace_batch_occupancy = [] {
                    const char* value = std::getenv("MOLLM_CUDA_NVFP4_TRACE_BATCH_OCCUPANCY");
                    return value && std::strcmp(value, "1") == 0;
                }();
                if ((trace_first_routes || trace_batch_occupancy) && rows > 1) {
                    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
                    if (!mollm_cuda::report_cuda(cudaStreamIsCapturing(execution_stream, &capture), "first-route capture check")) return fail();
                    if (capture == cudaStreamCaptureStatusNone) {
                        std::vector<mollm_cuda::Nvfp4Projection> selected(trace_batch_occupancy ? routes : top_k), tables(experts);
                        std::vector<float> multipliers(top_k);
                        if (!mollm_cuda::report_cuda(cudaMemcpyAsync(selected.data(), gate_jobs,
                                selected.size() * sizeof(selected[0]), cudaMemcpyDeviceToHost, execution_stream), "first-route jobs") ||
                            !mollm_cuda::report_cuda(cudaMemcpyAsync(tables.data(),
                                gate_scale_tables,
                                tables.size() * sizeof(tables[0]), cudaMemcpyDeviceToHost, execution_stream), "first-route tables") ||
                            !mollm_cuda::report_cuda(cudaMemcpyAsync(multipliers.data(), device_route_weights,
                                multipliers.size() * sizeof(float), cudaMemcpyDeviceToHost, execution_stream), "first-route weights") ||
                            !mollm_cuda::report_cuda(cudaStreamSynchronize(execution_stream), "first-route completion")) return fail();
                        std::fprintf(stderr, "nvfp4_first_routes node=%u rows=%d ids=", node.id, rows);
                        for (int route = 0; route < top_k; ++route) {
                            int expert = -1;
                            for (int e = 0; e < experts; ++e) {
                                const auto& a = selected[route];
                                const auto& b = tables[e];
                                if (a.weight == b.weight && a.block_scales == b.block_scales &&
                                    a.row_scales == b.row_scales && a.pair_layout == b.pair_layout &&
                                    a.mma_layout == b.mma_layout) { expert = e; break; }
                            }
                            std::fprintf(stderr, "%s%d", route ? "," : "", expert);
                        }
                        std::fprintf(stderr, " weights=");
                        for (int route = 0; route < top_k; ++route)
                            std::fprintf(stderr, "%s%.9g", route ? "," : "", multipliers[route]);
                        std::fprintf(stderr, "\n");
                        if (trace_batch_occupancy) {
                            std::vector<int> counts(experts, 0);
                            int unmatched = 0;
                            for (const auto& a : selected) {
                                int expert = -1;
                                for (int e = 0; e < experts; ++e) {
                                    const auto& b = tables[e];
                                    if (a.weight == b.weight && a.block_scales == b.block_scales &&
                                        a.row_scales == b.row_scales && a.pair_layout == b.pair_layout &&
                                        a.mma_layout == b.mma_layout) { expert = e; break; }
                                }
                                if (expert < 0) ++unmatched;
                                else ++counts[expert];
                            }
                            int active = 0, used = 0, singles = 0;
                            for (int n : counts) {
                                used += n > 0;
                                active += (n + gate_batch_routes - 1) / gate_batch_routes;
                                singles += n % gate_batch_routes == 1;
                            }
                            // Valid only for the stable grouped path; unmatched
                            // identities invalidate the estimate, and are reported.
                            std::fprintf(stderr,
                                "nvfp4_batch_occupancy node=%u rows=%d grouped=%d routes=%zu experts_used=%d batches=%d singleton_batches=%d unmatched=%d batch_routes=%d\n",
                                node.id, rows, int(group_jobs), routes, used, active, singles, unmatched, gate_batch_routes);
                            // Diagnostic only: preserve zero-count expert IDs
                            // so a probe can reconstruct the complete run sizes.
                            std::fprintf(stderr,"nvfp4_expert_counts node=%u rows=%d unmatched=%d counts=",
                                node.id,rows,unmatched);
                            for (size_t e=0;e<counts.size();++e)
                                std::fprintf(stderr,"%s%d",e?",":"",counts[e]);
                            std::fprintf(stderr,"\n");
                        }
                    }
                }
                if (group_jobs) {
                    if (use_radix_group) {
                        if (!mollm_cuda::report_cuda(mollm_cuda::launch_group_nvfp4_jobs_radix(
                                gate_jobs, down_jobs, down_jobs + routes, down_jobs + 2 * routes,
                                routes, impl_->moe_radix_group_scratch,
                                impl_->moe_radix_group_scratch_bytes, execution_stream), "NVFP4 radix grouping")) return fail();
                        ++impl_->moe_radix_group_calls;
                    } else mollm_cuda::launch_group_nvfp4_jobs(gate_jobs, down_jobs,
                        down_jobs + routes, down_jobs + 2 * routes, routes, execution_stream, nullptr, extended_group);
                    gate_jobs = down_jobs + routes;
                    down_jobs += 2 * routes;
                    // Diagnostic only: deliberately synchronizes; never use
                    // these runs as throughput evidence or during capture.
                    static const bool trace_pairs = std::getenv("MOLLM_CUDA_NVFP4_TRACE_PAIRS") != nullptr;
                    if (trace_pairs) {
                        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
                        if (!mollm_cuda::report_cuda(cudaStreamIsCapturing(execution_stream, &capture), "NVFP4 route trace capture check")) return fail();
                        if (capture == cudaStreamCaptureStatusNone) {
                            std::vector<mollm_cuda::Nvfp4Projection> trace(routes);
                            if (!mollm_cuda::report_cuda(cudaMemcpyAsync(trace.data(), gate_jobs,
                                    routes * sizeof(trace[0]), cudaMemcpyDeviceToHost, execution_stream), "NVFP4 route trace copy") ||
                                !mollm_cuda::report_cuda(cudaStreamSynchronize(execution_stream), "NVFP4 route trace completion")) return fail();
                            const auto equal = [](const auto& a, const auto& b) {
                                return a.weight == b.weight && a.block_scales == b.block_scales && a.pair_layout == b.pair_layout;
                            };
                            size_t matched = 0;
                            for (size_t i = 0; i + 1 < routes; i += 2)
                                if (equal(trace[i], trace[i + 1])) matched += 2;
                            size_t histogram[5] = {};
                            size_t optimal = 0;
                            for (size_t i = 0; i < routes;) {
                                size_t end = i + 1;
                                while (end < routes && equal(trace[i], trace[end])) ++end;
                                const size_t size = end - i;
                                ++histogram[std::min<size_t>(size, 5) - 1];
                                optimal += size / 2 * 2;
                                i = end;
                            }
                            std::fprintf(stderr, "nvfp4_route_pairs rows=%d routes=%zu matched=%zu optimal=%zu groups_1=%zu groups_2=%zu groups_3=%zu groups_4=%zu groups_5plus=%zu\n",
                                rows, routes, matched, optimal, histogram[0], histogram[1], histogram[2], histogram[3], histogram[4]);
                        }
                    }
                }
                if(mma_activation_fused){
                    auto* scratch=static_cast<uint8_t*>(impl_->moe_mma_group_scratch);
                    if(!mollm_cuda::report_cuda(mollm_cuda::launch_nvfp4_mma_activation_diagnostic(
                        gate_jobs,down_jobs,routes,reinterpret_cast<mollm_cuda::Nvfp4MmaBatch*>(scratch),
                        scratch+batch_bytes,scratch+batch_bytes+packed_bytes,activated,
                        scratch+fused_down_offset,scratch+fused_down_offset+routes*size_t(inter)/2,
                        false,execution_stream,false,32,false,false,nullptr,0,use_scale_view,input_scale_kmajor,compact_logits,
                        token_quantization?x:nullptr,token_quantization?rows:0,resident.gate_input_global,
                        token_quantization?reinterpret_cast<int*>(scratch+token_indices_offset):nullptr,
                        token_pair_words,gate_codes_kmajor,token_codes_kmajor),"NVFP4 MMA activation fusion"))return fail();
                    if(use_scale_view)++impl_->prefill_gate_scale_view_calls;
                    if(input_scale_kmajor)++impl_->prefill_input_scale_kmajor_calls;
                    if(compact_logits)++impl_->prefill_compact_gate_logit_calls;
                    if(token_quantization)++impl_->prefill_token_quantization_calls;
                    if(token_pair_words)++impl_->prefill_token_pair_word_calls;
                    if(token_codes_kmajor)++impl_->prefill_token_code_kmajor_calls;
                    ++impl_->moe_mma_activation_fused_prefill_calls;
                } else if (native_mma) {
                    if (use_batched) {
                        auto* scratch = static_cast<uint8_t*>(impl_->moe_mma_group_scratch);
                        if (!mollm_cuda::report_cuda(mollm_cuda::launch_nvfp4_mma_grouped(
                            gate_jobs, routes, reinterpret_cast<mollm_cuda::Nvfp4MmaBatch*>(scratch),
                            2 * inter, hidden, inter, execution_stream, scratch + batch_bytes,
                            scratch + batch_bytes + packed_bytes, batch_gate_min, canonical_mma, prepare_singles,
                            compact_gate_dispatch ? reinterpret_cast<mollm_cuda::Nvfp4MmaBatch*>(scratch + compact_offset) : nullptr,
                            gate_compact_grid_y, gate_batch_routes, specialized_prepared,false,async_gate,
                            gate_output_bf16,gate_early_inputs,parallel_plan,gate_vector_quantize,gate_codes_kmajor), "NVFP4 batched gate/up MMA")) return fail();
                        if(gate_output_bf16)++impl_->moe_bf16_gate_output_prefill_calls;
                        if(parallel_plan)++impl_->moe_parallel_batch_plan_prefill_calls;
                        if(gate_early_inputs)++impl_->moe_gate_early_inputs_prefill_calls;
                        if(gate_vector_quantize)++impl_->moe_gate_vector_quantize_prefill_calls;
                        if (async_gate) ++impl_->moe_gate_async_weights_calls;
                        if (compact_gate_dispatch) ++impl_->moe_gate_compact_prefill_calls;
                    } else if (!mollm_cuda::report_cuda(mollm_cuda::launch_nvfp4_mma_projections(
                            gate_jobs, routes, 2 * inter, hidden, inter, execution_stream,
                            static_cast<float*>(impl_->moe_mma_partial), native_gate_parts, canonical_mma,
                            nullptr,nullptr,gate_codes_kmajor), "NVFP4 gate/up MMA")) return fail();
                } else if (gate_codes_kmajor) {
                    if (!mollm_cuda::report_cuda(mollm_cuda::launch_nvfp4_gate_code_projections(
                            gate_jobs,routes,2*inter,hidden,
                            (simt_pipeline&&rows==1)||(batch_pipeline&&rows>=2&&rows<=8),
                            execution_stream), "NVFP4 gate code SIMT")) return fail();
                } else if (((simt_pipeline && rows == 1) ||
                    (batch_pipeline && rows >= 2 && rows <= 8)) && !gate_source->resident_nvfp4_mma_layout) {
                    if (!mollm_cuda::report_cuda(mollm_cuda::launch_nvfp4_pipelined_projections(
                            gate_jobs, routes, 2 * inter, hidden, true, execution_stream), "NVFP4 pipelined gate/up")) return fail();
                } else mollm_cuda::launch_nvfp4_projections(gate_jobs, routes, 2 * inter, hidden, execution_stream,
                    gate_source->resident_nvfp4_mma_layout);
                const bool fused_prepare=activation_prepare_fusion && native_mma && use_batched && rows>=16;
                const bool packed_only=fused_prepare && activation_prepare_fusion==2 &&
                    (batch_down_min==1 || prepare_singles);
                if (fused_prepare&&!mma_activation_fused) {
                    auto* scratch=static_cast<uint8_t*>(impl_->moe_mma_group_scratch);
                    if (!mollm_cuda::report_cuda(mollm_cuda::launch_nvfp4_activation_prepare(
                        down_jobs,routes,inter,gate_up,activated,scratch+batch_bytes,
                        scratch+batch_bytes+packed_bytes,packed_only?3:2,execution_stream,gate_output_bf16),"NVFP4 fused activation preparation")) return fail();
                    ++impl_->moe_activation_prepare_fused_calls;
                    if (packed_only) ++impl_->moe_activation_prepare_packed_only_calls;
                } else if(!mma_activation_fused) mollm_cuda::launch_moe_activation(gate_up, gate_up + inter,
                    native_mma ? nullptr : device_route_weights, activated, routes, inter, 2 * inter, execution_stream);
                if (native_mma) {
                    if (use_batched) {
                        auto* scratch = static_cast<uint8_t*>(impl_->moe_mma_group_scratch);
                        if (!mollm_cuda::report_cuda(mollm_cuda::launch_nvfp4_mma_grouped(
                            down_jobs, routes, reinterpret_cast<mollm_cuda::Nvfp4MmaBatch*>(scratch),
                            hidden, inter, 0, execution_stream,
                            scratch+(mma_activation_fused?fused_down_offset:batch_bytes),
                            scratch+(mma_activation_fused?fused_down_offset+routes*size_t(inter)/2:batch_bytes+packed_bytes),
                            batch_down_min, canonical_mma, prepare_singles,
                            compact_down_dispatch ? reinterpret_cast<mollm_cuda::Nvfp4MmaBatch*>(scratch + compact_offset) : nullptr,
                            compact_grid_y, mma_batch_routes, specialized_prepared,fused_prepare,
                            false,down_output_bf16,false,parallel_plan), "NVFP4 batched down MMA")) return fail();
                        if (down_output_bf16) ++impl_->moe_bf16_down_output_prefill_calls;
                        if(parallel_plan)++impl_->moe_parallel_batch_plan_prefill_calls;
                        if (compact_down_dispatch) ++impl_->moe_down_compact_prefill_calls;
                    } else if (!mollm_cuda::report_cuda(mollm_cuda::launch_nvfp4_mma_projections(
                            down_jobs, routes, hidden, inter, 0, execution_stream,
                            nullptr, 1, canonical_mma), "NVFP4 down MMA")) return fail();
                } else if (((simt_pipeline && rows == 1) ||
                    (batch_pipeline && rows >= 2 && rows <= 8)) && !down_source->resident_nvfp4_mma_layout) {
                    static const bool prefetch_down_row_scale = [] {
                        const char* value = std::getenv("MOLLM_CUDA_NVFP4_DOWN_ROW_SCALE_PREFETCH");
                        return value && std::strcmp(value, "1") == 0;
                    }();
                    const int load_hint = prefetch_down_row_scale && rows == 1 &&
                        hidden == 2560 && inter == 640 ? 7 : 0;
                    if (!mollm_cuda::report_cuda(mollm_cuda::launch_nvfp4_pipelined_projections(
                            down_jobs, routes, hidden, inter, true, execution_stream,
                            false, load_hint), "NVFP4 pipelined down")) return fail();
                } else mollm_cuda::launch_nvfp4_projections(down_jobs, routes, hidden, inter, execution_stream,
                    down_source->resident_nvfp4_mma_layout);
                if (parallel_shared) {
                    if (!mollm_cuda::report_cuda(cudaStreamWaitEvent(execution_stream,
                            impl_->moe_shared_done, 0), "shared expert join")) return fail();
                    return combine(false);
                }
                return finish(false);
            };
            // A segment capture records the constituent kernels directly;
            // launching an already instantiated graph into capture is rejected
            // by CUDA. Resident leases still keep every descriptor alive.
            if (impl_->segment_mode == Impl::SegmentMode::CAPTURE) return enqueue();
            static const bool no_graphs = std::getenv("MOLLM_CUDA_DISABLE_MOE_GRAPHS") != nullptr;
            std::array<const void*, 5> graph_weights{};
            bool capture_safe = !no_graphs && rows == 1;
            int weight_index = 0;
            for (int index : {1, 4, 5, 6, 7}) {
                const auto* weight = impl_->find_weight(*inputs[index]);
                capture_safe = capture_safe && weight->layout == Impl::WeightLayout::Q4Bg32Biased;
                graph_weights[weight_index++] = weight->data;
            }
            if (!capture_safe) return enqueue();
            if (resident.graph && resident.graph_input == x && resident.graph_output == destination &&
                resident.graph_scratch == impl_->moe_scratch && resident.graph_jobs == impl_->moe_jobs &&
                resident.graph_weights == graph_weights && resident.graph_i32 == p.i32 && resident.graph_f32 == p.f32) {
                ++impl_->moe_graph_hits;
                if (!mollm_cuda::report_cuda(cudaGraphLaunch(resident.graph, nullptr), "resident MoE graph replay")) return fail();
                return true;
            }
            if (resident.graph) { cudaGraphExecDestroy(resident.graph); resident.graph = nullptr; }
            if (!impl_->moe_capture_stream && !mollm_cuda::report_cuda(
                    cudaStreamCreateWithFlags(&impl_->moe_capture_stream, cudaStreamNonBlocking),
                    "resident MoE capture stream")) return fail();
            if (!mollm_cuda::report_cuda(cudaStreamBeginCapture(impl_->moe_capture_stream,
                    cudaStreamCaptureModeThreadLocal), "resident MoE graph capture")) return fail();
            execution_stream = impl_->moe_capture_stream;
            const bool enqueued = enqueue();
            execution_stream = nullptr;
            cudaGraph_t graph = nullptr;
            const cudaError_t captured = cudaStreamEndCapture(impl_->moe_capture_stream, &graph);
            if (!enqueued || impl_->failed || captured != cudaSuccess || !graph) {
                if (graph) cudaGraphDestroy(graph);
                mollm_cuda::report_cuda(captured, "resident MoE capture completion");
                return fail();
            }
            const cudaError_t instantiated = cudaGraphInstantiateWithFlags(&resident.graph, graph, 0);
            cudaGraphDestroy(graph);
            if (!mollm_cuda::report_cuda(instantiated, "resident MoE graph instantiate")) return fail();
            resident.graph_input = x; resident.graph_output = destination;
            resident.graph_scratch = impl_->moe_scratch; resident.graph_jobs = impl_->moe_jobs;
            resident.graph_weights = graph_weights; resident.graph_i32 = p.i32; resident.graph_f32 = p.f32;
            ++impl_->moe_graph_builds;
            if (!mollm_cuda::report_cuda(cudaGraphLaunch(resident.graph, nullptr), "resident MoE graph launch")) return fail();
            return true;
        }
    }
    if (!matmul(x, 1, logits, experts, hidden)) return fail();
    std::vector<float> host_logits(static_cast<size_t>(rows) * experts);
    if (!mollm_cuda::copy_memory(host_logits.data(), logits,
            host_logits.size() * sizeof(float), cudaMemcpyDeviceToHost, "MoE routing logits"))
        return fail();
    mollm::detail::MoeRoutingParams routing;
    routing.num_experts = experts;
    routing.top_k = top_k;
    routing.score_func = graph_params::get_i32(p, 5, 0);
    routing.normalize_topk = graph_params::get_i32(p, 6, 1) != 0;
    routing.num_groups = graph_params::get_i32(p, 8, 1);
    routing.topk_groups = graph_params::get_i32(p, 9, 1);
    routing.scaling_factor = graph_params::get_f32(p, 0, 1.0f);
    std::vector<int> indices;
    std::vector<float> weights;
    if (!mollm::detail::select_moe_routes(host_logits.data(), rows, nullptr,
                                         routing, indices, weights)) return fail();
    // Accumulate experts in the same ascending-ID order as the CPU provider.
    for (int row = 0; row < rows; ++row) {
        for (int i = 1; i < top_k; ++i) {
            for (int j = i; j > 0 && indices[row * top_k + j] < indices[row * top_k + j - 1]; --j) {
                std::swap(indices[row * top_k + j], indices[row * top_k + j - 1]);
                std::swap(weights[row * top_k + j], weights[row * top_k + j - 1]);
            }
        }
    }
    std::vector<ExpertLease> leases(routes);
    std::vector<mollm_cuda::Nvfp4Projection> gate_jobs(routes), down_jobs(routes);
    for (size_t i = 0; i < routes; ++i) {
        // Never initiate I/O from the CUDA path. A cache miss or unsupported
        // provider is handled by the existing CPU bridge without partial output.
        if (!gate_source->provider->borrow(gate_source, down_source, indices[i], leases[i], false))
            return false;
        if (!valid(leases[i].gate_up, 2 * inter, hidden) ||
            !valid(leases[i].down, hidden, inter)) return false;
        gate_jobs[i] = projection(leases[i].gate_up, x + (i / top_k) * hidden,
                                  gate_up + i * 2 * inter);
        down_jobs[i] = projection(leases[i].down, activated + i * inter, down + i * hidden);
    }
    mollm_cuda::launch_nvfp4_projections(gate_jobs.data(), routes, 2 * inter, hidden);
    mollm_cuda::launch_moe_activation(gate_up, gate_up + inter, weights.data(),
                                      activated, routes, inter, 2 * inter);
    mollm_cuda::launch_nvfp4_projections(down_jobs.data(), routes, hidden, inter);
    return finish(true);
}
