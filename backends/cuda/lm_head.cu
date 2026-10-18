#include <cuda_runtime.h>

#include "backends/cuda/backend_internal.h"

#include <algorithm>

using mollm::cuda_backend_detail::device_pointer_const;

bool CudaBackend::supports_lm_head(const Tensor& weight) const {
    return impl_->find_weight(weight) != nullptr;
}

bool CudaBackend::supports_lm_head_argmax(const Tensor& weight) const {
    return supports_lm_head(weight);
}

bool CudaBackend::supports_lm_head_small_batch(const Tensor& weight) const {
    static const bool enabled=[] {
        const char* value=std::getenv("MOLLM_CUDA_MTP_VERIFY_Q8_BATCH");
        return value && std::strcmp(value,"1")==0;
    }();
    const auto* prepared=impl_->find_weight(weight);
    return enabled && prepared &&
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
    if(!mollm_cuda::report_cuda(mollm_cuda::launch_q8_small_batch_exact(source,
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
            sizeof(mollm_cuda::ArgMaxPair)) ||
        !impl_->run_matmul_device(
            source, k, weight, static_cast<float*>(impl_->output), n,
            1, n, k, static_cast<Activation>(activation_kind), 0, -1)) {
        impl_->failed = true;
        return -1;
    }
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
    mollm_cuda::ArgMaxPair result{};
    if (!mollm_cuda::report_cuda(cudaGetLastError(), "argmax_cuda") ||
        !mollm_cuda::copy_memory(
            &result, impl_->argmax_result, sizeof(result),
            cudaMemcpyDeviceToHost, "cudaMemcpy lm_head argmax") ||
        result.index < 0 || result.index >= n) {
        impl_->failed = true;
        return -1;
    }
    ++impl_->native_ops[static_cast<uint32_t>(OpType::MATMUL)];
    return result.index;
}
