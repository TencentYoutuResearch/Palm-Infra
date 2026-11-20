#include <cuda_runtime.h>

#include "backends/cuda/backend_internal.h"
#include "kernels/cuda/q8_head_screen.h"
#include "kernels/cuda/draft_compact_head.h"

#include <algorithm>

using mollm::cuda_backend_detail::device_pointer_const;

namespace {
// The screened and compact heads are qualified for this vocabulary/hidden
// shape on SM121 only; every other shape keeps the exact Q8 GEMV.
constexpr int qualified_head_rows = 248320;
constexpr int qualified_head_inner = 2560;
constexpr int draft_energy_screen_groups = 16;
}  // namespace

bool CudaBackend::supports_lm_head(const Tensor& weight) const {
    return impl_->find_weight(weight) != nullptr;
}

bool CudaBackend::supports_lm_head_argmax(const Tensor& weight) const {
    return supports_lm_head(weight);
}

bool CudaBackend::supports_lm_head_small_batch(const Tensor& weight) const {
    const auto* prepared=impl_->find_weight(weight);
    return impl_->qualified_verification && prepared &&
        prepared->layout==Impl::WeightLayout::Q8RowMajor && prepared->scales &&
        prepared->groups_per_row==1 && prepared->group_size>=weight.shape[1] &&
        weight.shape[0]>0 && weight.shape[0]<=INT32_MAX-7 &&
        weight.shape[1]>0 && weight.shape[1]<=INT32_MAX && weight.shape[1]%4==0;
}

bool CudaBackend::lm_head_small_batch_argmax_device_and_end_graph(
    const Tensor& activation, const Tensor& weight, int* output,
    int tokens, int columns, int inner, int activation_kind) {
    finish_decode_segment();
    if(impl_->deferred_rms_overlaps(device_pointer_const<float>(activation),activation.view_span_bytes(),true)&&
       !impl_->materialize_deferred_rms())return false;
    if(!impl_->restore_hc_compact(device_pointer_const<float>(activation),activation.view_span_bytes()))return false;
    const auto* prepared=impl_->find_weight(weight);
    const float* source=device_pointer_const<float>(activation);
    const int groups=columns>0 ? std::min(256,columns/256+(columns%256!=0)) : 0;
    if(!output || !source || !supports_lm_head_small_batch(weight) ||
        tokens<2 || tokens>8 || columns<=0 || inner<=0 ||
        columns!=weight.shape[0] || inner!=weight.shape[1] ||
        activation_kind!=0 || !activation.is_contiguous() || activation.shape[0]!=inner ||
        activation.shape[1]<tokens || groups<=0 ||
        !impl_->reserve(impl_->output,impl_->output_bytes,size_t(tokens)*columns*sizeof(float)) ||
        !impl_->reserve(impl_->argmax_partial,impl_->argmax_partial_bytes,size_t(groups)*sizeof(mollm_cuda::ArgMaxPair)) ||
        !impl_->reserve(impl_->argmax_result,impl_->argmax_result_bytes,sizeof(mollm_cuda::ArgMaxPair))) {
        impl_->failed=true;return false;
    }
    // Scalar and batch screening share one auxiliary weight copy. Admit this
    // measured target-head shape only with compact-Q4 drafting: otherwise a
    // screened scalar draft could evict the target copy on every MTP round.
    const bool screen_candidate=!impl_->draft_head_request&&tokens>=3&&
        tokens<=verification_workspace_max_rows()&&columns==qualified_head_rows&&
        inner==qualified_head_inner&&uintptr_t(source)%16==0;
    const bool wide_screen=impl_->prefer_screened_heads;
    const bool screen=screen_candidate&&(tokens==3||wide_screen);
    if(screen) {
        // Reserve the maximum admitted width on the first screened batch;
        // later width transitions reuse the buffer without reallocating it.
        const int capacity=wide_screen?std::max(5,tokens):3;
        if(!impl_->reserve(impl_->q8_head_packed,impl_->q8_head_packed_bytes,
                size_t(columns)*1920,true)||
           !impl_->reserve(impl_->q8_head_scratch,impl_->q8_head_scratch_bytes,
                size_t(columns)*8*capacity+1040*capacity,true)) {impl_->failed=true;return false;}
        if(impl_->q8_head_source!=prepared->data||impl_->q8_head_rows!=columns) {
            if(!mollm_cuda::report_cuda(mollm_cuda::pack_q8_head_screen(
                static_cast<const int8_t*>(prepared->data),
                static_cast<uint8_t*>(impl_->q8_head_packed),columns,cudaStreamPerThread),
                "pack Q8 batch head screen")) {impl_->failed=true;return false;}
            impl_->q8_head_source=prepared->data;impl_->q8_head_rows=columns;
        }
        auto* scratch=static_cast<char*>(impl_->q8_head_scratch);
        const size_t values=size_t(columns)*tokens;
        mollm_cuda::Q8HeadScreenBatchScratch workspace{
            reinterpret_cast<float*>(scratch),
            reinterpret_cast<float*>(scratch+values*8+tokens*12),
            reinterpret_cast<double*>(scratch+values*8),
            reinterpret_cast<unsigned*>(scratch+values*8+tokens*8),
            reinterpret_cast<int*>(scratch+values*4),
            values,size_t(tokens)*257,size_t(tokens),size_t(tokens),values};
        const auto* codes=static_cast<const int8_t*>(prepared->data);
        const auto* packed=static_cast<const uint8_t*>(impl_->q8_head_packed);
        auto* destination=static_cast<float*>(impl_->output);
        cudaError_t status;
        if(tokens==3) {
            mollm_cuda::Q8HeadScreenBatch3Scratch old_workspace{workspace.approximate,
                workspace.lower,workspace.norm,workspace.count,workspace.indices};
            status=mollm_cuda::launch_q8_head_screen_batch3(source,codes,packed,
                prepared->scales,destination,columns,old_workspace,cudaStreamPerThread);
        } else {
            status=mollm_cuda::launch_q8_head_screen_batch(source,codes,packed,
                prepared->scales,destination,tokens,columns,workspace,cudaStreamPerThread);
        }
        if(!mollm_cuda::report_cuda(status,
            "Q8 batch head screen")) {impl_->failed=true;return false;}
    } else if(!mollm_cuda::report_cuda(mollm_cuda::launch_q8_small_batch_exact(source,
        static_cast<const int8_t*>(prepared->data),prepared->scales,
        static_cast<float*>(impl_->output),tokens,columns,inner,inner,columns),"q8_verify_batch")) {
        impl_->failed=true;return false;
    }
    for(int token=0;token<tokens;++token) {
        mollm_cuda::launch_argmax(static_cast<const float*>(impl_->output)+size_t(token)*columns,
            columns,static_cast<mollm_cuda::ArgMaxPair*>(impl_->argmax_partial),groups,
            static_cast<mollm_cuda::ArgMaxPair*>(impl_->argmax_result));
        mollm_cuda::ArgMaxPair result{};
        if(!mollm_cuda::report_cuda(cudaGetLastError(),"q8_verify_argmax") ||
            !mollm_cuda::copy_memory(&result,impl_->argmax_result,sizeof(result),
                cudaMemcpyDeviceToHost,"q8_verify_top1")) {impl_->failed=true;return false;}
        output[token]=result.index;
    }
    ++impl_->native_ops[static_cast<uint32_t>(OpType::MATMUL)];
    return true;
}

void CudaBackend::lm_head_gemv(const float* activation_host,
                               const Tensor& weight, float* output_host,
                               int n, int k, int activation) {
    if (!impl_->run_matmul(
            activation_host, k, weight, output_host, n, 1, n, k,
            static_cast<Activation>(activation), 0, -1))
        impl_->failed = true;
}

void CudaBackend::lm_head_gemv_device_and_end_graph(
    const Tensor& activation, size_t activation_element_offset,
    const Tensor& weight, float* output_host, int n, int k,
    int activation_kind) {
    finish_decode_segment();
    if(impl_->deferred_rms_overlaps(device_pointer_const<float>(activation),activation.view_span_bytes(),true)&&
       !impl_->materialize_deferred_rms())return;
    if(!impl_->restore_hc_compact(device_pointer_const<float>(activation),activation.view_span_bytes()))return;
    const float* activation_base = device_pointer_const<float>(activation);
    const float* source = activation_base
        ? activation_base + activation_element_offset : nullptr;
    const size_t output_size = static_cast<size_t>(n) * sizeof(float);
    if (!source ||
        !impl_->reserve(impl_->output, impl_->output_bytes, output_size) ||
        !impl_->run_matmul_device(
            source, k, weight, static_cast<float*>(impl_->output), n,
            1, n, k, static_cast<Activation>(activation_kind), 0, -1) ||
        !mollm_cuda::copy_memory(
            output_host, impl_->output, output_size, cudaMemcpyDeviceToHost,
            "cudaMemcpy lm_head output")) {
        impl_->failed = true;
        return;
    }
    ++impl_->native_ops[static_cast<uint32_t>(OpType::MATMUL)];
}

int CudaBackend::lm_head_draft_argmax_device_and_end_graph(
    const Tensor& input, size_t offset, const Tensor& weight,
    int n, int k, int activation, Tensor* hidden_copy, float* probability) {
    struct DraftScope {
        bool& flag;
        bool previous;
        float*& destination;
        float* previous_destination;
        ~DraftScope() { flag = previous; destination = previous_destination; }
    } scope{impl_->draft_head_request, impl_->draft_head_request,
            impl_->draft_probability_output, impl_->draft_probability_output};
    impl_->draft_head_request = true;
    impl_->draft_probability_output = probability;
    if (probability) *probability = -1.0f;
    return lm_head_argmax_device_and_end_graph(
        input, offset, weight, n, k, activation, hidden_copy);
}

int CudaBackend::lm_head_argmax_device_and_end_graph(
    const Tensor& activation, size_t activation_element_offset,
    const Tensor& weight, int n, int k, int activation_kind,
    Tensor* hidden_copy) {
    finish_decode_segment();
    if(impl_->deferred_rms_overlaps(device_pointer_const<float>(activation),activation.view_span_bytes(),true)&&
       !impl_->materialize_deferred_rms())return -1;
    if(!impl_->restore_hc_compact(device_pointer_const<float>(activation),activation.view_span_bytes()))return -1;
    if(hidden_copy&&!impl_->restore_hc_compact(device_pointer_const<float>(*hidden_copy),hidden_copy->view_span_bytes()))return -1;
    const float* activation_base = device_pointer_const<float>(activation);
    const float* source = activation_base
        ? activation_base + activation_element_offset : nullptr;
    const size_t output_size = static_cast<size_t>(n) * sizeof(float);
    const int groups = std::min(256, (n + 255) / 256);
    const size_t partial_size =
        static_cast<size_t>(groups) * sizeof(mollm_cuda::ArgMaxPair);
    if (!source || n <= 0 || groups <= 0 ||
        !impl_->reserve(impl_->output, impl_->output_bytes, output_size) ||
        !impl_->reserve(
            impl_->argmax_partial, impl_->argmax_partial_bytes,
            partial_size) ||
        !impl_->reserve(
            impl_->argmax_result, impl_->argmax_result_bytes,
            impl_->draft_probability_output ? sizeof(mollm_cuda::DraftHeadResult) :
                sizeof(mollm_cuda::ArgMaxPair))) {
        impl_->failed = true;
        return -1;
    }
    const auto* prepared=impl_->find_weight(weight);
    const bool screen=prepared&&
        prepared->layout==Impl::WeightLayout::Q8RowMajor&&prepared->scales&&
        prepared->groups_per_row==1&&prepared->group_size>=k&&
        n>=65536&&k==2560&&activation_kind==0&&uintptr_t(source)%16==0;
    const bool draft_q4=impl_->draft_head_request&&prepared&&
        prepared->layout==Impl::WeightLayout::Q8RowMajor&&prepared->scales&&
        prepared->groups_per_row==1&&prepared->group_size>=k&&prepared->n==n&&prepared->k==k&&
        k>0&&k%32==0&&n<=INT32_MAX-7&&uintptr_t(source)%16==0&&activation_kind==0;
    bool computed=false;
    if(draft_q4) {
        // Confidence requests need Q4 logits/probabilities; plain greedy drafts
        // on the qualified shape use the smaller Q2 head.
        const bool qualified_shape=n==qualified_head_rows&&k==qualified_head_inner;
        const int bits=!impl_->draft_probability_output&&qualified_shape ? 2 : 4;
        const bool fitted_q4=bits==4&&impl_->draft_probability_output&&qualified_shape&&
            impl_->prefer_screened_heads;
        const auto kind = bits == 2 ? Impl::DraftCompactHeadKind::Q2 :
            fitted_q4 ? Impl::DraftCompactHeadKind::FittedQ4 : Impl::DraftCompactHeadKind::Q4;
        const size_t block_bytes=bits==2 ? sizeof(mollm_cuda::Q2DraftG32) : sizeof(mollm_cuda::Q4CompactG32);
        const size_t bytes=size_t((n+7)/8)*(k/32)*block_bytes;
        if(!impl_->reserve(impl_->draft_q4_head,impl_->draft_q4_head_bytes,bytes,true)) {impl_->failed=true;return -1;}
        if(impl_->draft_q4_head_source!=prepared->data||impl_->draft_q4_rows!=n||
           impl_->draft_q4_inner!=k||impl_->draft_compact_head_kind!=kind) {
            cudaError_t status;
            if (bits == 2) status = mollm_cuda::pack_q8_draft_head_q2(
                static_cast<const int8_t*>(prepared->data), prepared->scales,
                static_cast<mollm_cuda::Q2DraftG32*>(impl_->draft_q4_head), n, k, cudaStreamPerThread);
            else if (fitted_q4) status = mollm_cuda::pack_q8_draft_head_compact_fitted(
                static_cast<const int8_t*>(prepared->data), prepared->scales,
                static_cast<mollm_cuda::Q4CompactG32*>(impl_->draft_q4_head), n, k, cudaStreamPerThread);
            else status = mollm_cuda::pack_q8_draft_head_compact(
                static_cast<const int8_t*>(prepared->data), prepared->scales,
                static_cast<mollm_cuda::Q4CompactG32*>(impl_->draft_q4_head), n, k, cudaStreamPerThread);
            if(!mollm_cuda::report_cuda(status,"pack compact draft head")) {impl_->failed=true;return -1;}
            impl_->draft_q4_head_source=prepared->data;impl_->draft_q4_rows=n;impl_->draft_q4_inner=k;
            impl_->draft_compact_head_kind=kind;
        }
        if(bits==2) computed=mollm_cuda::report_cuda(mollm_cuda::launch_q2_draft_head_gemv(
            source,static_cast<const mollm_cuda::Q2DraftG32*>(impl_->draft_q4_head),
            static_cast<float*>(impl_->output),n,k,cudaStreamPerThread),"Q2 draft head");
        else {
            if (fitted_q4) {
                // Fitted Q4 implies a probability request on the qualified
                // shape. Scratch persists across eager draft calls.
                constexpr size_t array_bytes = size_t(qualified_head_rows) * sizeof(float);
                constexpr size_t bytes = array_bytes * 4 + 32 * sizeof(int);
                if (!impl_->reserve(impl_->draft_screen_scratch,
                        impl_->draft_screen_scratch_bytes, bytes, true)) {impl_->failed=true;return -1;}
                auto* base = static_cast<char*>(impl_->draft_screen_scratch);
                mollm_cuda::DraftHeadScreenScratch scratch{
                    reinterpret_cast<float*>(base), reinterpret_cast<float*>(base + array_bytes),
                    reinterpret_cast<int*>(base + array_bytes * 2),
                    reinterpret_cast<int*>(base + array_bytes * 3),
                    reinterpret_cast<int*>(base + array_bytes * 4),
                    impl_->draft_screen_sort, impl_->draft_screen_sort_bytes};
                if (!impl_->draft_screen_initialized) {
                    size_t required = 0;
                    if (!mollm_cuda::report_cuda(
                            mollm_cuda::draft_head_screen_temp_bytes(scratch, required),
                            "draft energy screen scratch query") ||
                        !impl_->reserve(impl_->draft_screen_sort,
                            impl_->draft_screen_sort_bytes, required, true)) {impl_->failed=true;return -1;}
                    scratch.sort_temp = impl_->draft_screen_sort;
                    scratch.sort_bytes = impl_->draft_screen_sort_bytes;
                    if (!mollm_cuda::report_cuda(
                            mollm_cuda::initialize_draft_head_screen(scratch, cudaStreamPerThread),
                            "draft energy screen initialize")) {impl_->failed=true;return -1;}
                    impl_->draft_screen_initialized = true;
                }
                computed = mollm_cuda::report_cuda(mollm_cuda::launch_draft_head_screen(
                    source, static_cast<const mollm_cuda::Q4CompactG32*>(impl_->draft_q4_head),
                    static_cast<float*>(impl_->output), n, k, scratch, cudaStreamPerThread,
                    draft_energy_screen_groups),
                    "draft energy screen");
                if (computed) ++impl_->draft_screen_calls;
            } else {
                mollm_cuda::launch_q4_g32_compact_gemv(source,static_cast<const mollm_cuda::Q4CompactG32*>(impl_->draft_q4_head),
                    static_cast<float*>(impl_->output),n,k,cudaStreamPerThread);
                computed=mollm_cuda::report_cuda(cudaGetLastError(),"Q4 draft head");
            }
        }
    } else if(screen) {
        // The qualified shape uses the smaller Q4 screen; others keep Q6.
        const bool scalar_alternative=impl_->prefer_screened_heads&&
            n==qualified_head_rows&&prepared->n==n&&prepared->k==k;
        const int bits=scalar_alternative ? 4 : 6;
        // Allocate only the scalar format in use. MTP can additionally create
        // its independent Q6 batch view without changing or evicting this one.
        void*& packed = scalar_alternative ? impl_->q8_head_scalar_packed : impl_->q8_head_packed;
        size_t& packed_bytes = scalar_alternative ? impl_->q8_head_scalar_packed_bytes : impl_->q8_head_packed_bytes;
        const void*& packed_source = scalar_alternative ? impl_->q8_head_scalar_source : impl_->q8_head_source;
        int& packed_rows = scalar_alternative ? impl_->q8_head_scalar_rows : impl_->q8_head_rows;
        if(!impl_->reserve(packed,packed_bytes,size_t(n)*(320*bits),true)||
           !impl_->reserve(impl_->q8_head_scratch,impl_->q8_head_scratch_bytes,size_t(n)*8+1040,true)) {
            impl_->failed=true;return -1;
        }
        if(packed_source!=prepared->data||packed_rows!=n||
           (scalar_alternative&&impl_->q8_head_scalar_bits!=bits)) {
            const auto pack=bits==4?mollm_cuda::pack_q8_head_screen_q4:mollm_cuda::pack_q8_head_screen;
            if(!mollm_cuda::report_cuda(pack(
                static_cast<const int8_t*>(prepared->data),static_cast<uint8_t*>(packed),n,cudaStreamPerThread),"pack Q8 head screen")) {
                impl_->failed=true;return -1;
            }
            packed_source=prepared->data;packed_rows=n;
            if(scalar_alternative){impl_->q8_head_scalar_bits=bits;++impl_->q8_head_scalar_packs;}
        }
        auto* scratch=static_cast<char*>(impl_->q8_head_scratch);
        mollm_cuda::Q8HeadScreenScratch workspace{
            reinterpret_cast<float*>(scratch),reinterpret_cast<float*>(scratch+size_t(n)*8+12),
            reinterpret_cast<double*>(scratch+size_t(n)*8),reinterpret_cast<unsigned*>(scratch+size_t(n)*8+8),
            reinterpret_cast<int*>(scratch+size_t(n)*4)};
        const auto launch=bits==4?mollm_cuda::launch_q8_head_screen_q4:mollm_cuda::launch_q8_head_screen;
        computed=mollm_cuda::report_cuda(launch(source,
            static_cast<const int8_t*>(prepared->data),static_cast<const uint8_t*>(packed),
            prepared->scales,static_cast<float*>(impl_->output),n,workspace,cudaStreamPerThread),"Q8 head screen");
        if (computed && scalar_alternative) ++impl_->q8_head_scalar_calls;
    }else{
        computed=impl_->run_matmul_device(source,k,weight,static_cast<float*>(impl_->output),n,
            1,n,k,static_cast<Activation>(activation_kind),0,-1);
    }
    if(!computed){impl_->failed=true;return -1;}
    if (hidden_copy && hidden_copy->device.buffer &&
        hidden_copy->nbytes() >= static_cast<size_t>(k) * sizeof(float) &&
        !mollm_cuda::copy_memory(
            hidden_copy->device.buffer, source,
            static_cast<size_t>(k) * sizeof(float),
            cudaMemcpyDeviceToDevice, "cudaMemcpy lm_head hidden")) {
        impl_->failed = true;
        return -1;
    }
    mollm_cuda::launch_argmax(
        static_cast<const float*>(impl_->output), n,
        static_cast<mollm_cuda::ArgMaxPair*>(impl_->argmax_partial), groups,
        static_cast<mollm_cuda::ArgMaxPair*>(impl_->argmax_result));
    const bool probability_requested = draft_q4 && impl_->draft_probability_output;
    if (probability_requested && !mollm_cuda::report_cuda(
            mollm_cuda::launch_draft_head_probability(
                static_cast<const float*>(impl_->output), n,
                static_cast<mollm_cuda::DraftHeadResult*>(impl_->argmax_result),
                static_cast<float*>(impl_->argmax_partial), groups, cudaStreamPerThread),
            "draft head probability")) {
        impl_->failed = true;
        return -1;
    }
    mollm_cuda::DraftHeadResult combined{};
    auto& result = combined.top1;
    if (!mollm_cuda::report_cuda(cudaGetLastError(), "argmax_cuda") ||
        !mollm_cuda::copy_memory(
            &combined, impl_->argmax_result,
            probability_requested ? sizeof(combined) : sizeof(result),
            cudaMemcpyDeviceToHost, "cudaMemcpy lm_head argmax") ||
        result.index < 0 || result.index >= n) {
        impl_->failed = true;
        return -1;
    }
    if (probability_requested)
        *impl_->draft_probability_output = combined.probability;
    ++impl_->native_ops[static_cast<uint32_t>(OpType::MATMUL)];
    return result.index;
}
