#include <cuda_runtime.h>

#include "backends/cuda/backend_internal.h"
#include "backends/cuda/prefill_batch_policy.h"
#include "core/cache_layout.h"
#include "graph/graph.h"
#include "kernels/cuda/attention.h"
#include "kernels/cuda/qsa.h"
#include "kernels/cuda/elementwise.h"
#include "kernels/cuda/matmul.h"
#include "kernels/cuda/recurrent.h"
#include "kernels/cpu/ple.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using mollm::cuda_backend_detail::device_pointer;
using mollm::cuda_backend_detail::device_pointer_const;
using mollm::cuda_backend_detail::fp32_contiguous;
using mollm::cuda_backend_detail::same_shape;

bool CudaBackend::supports_sigmoid_gr_reduce_fusion() const {
    // Qualified prefill fusion; capture-audit and reference backends keep the
    // separate nodes.
    return impl_->qualified_fusions && !impl_->capture_audit;
}

int CudaBackend::hc_up_reduce_max_rows() const {
    return mollm::hc::prefill_max_rows;
}

bool CudaBackend::supports_hc_up_reduce_fusion() const {
    const char* reference=std::getenv("MOLLM_CUDA_DENSE_PREFILL_FP32_REFERENCE");
    return impl_->qualified_fusions&&impl_->hc_deferred_enabled&&
        supports_sigmoid_gr_reduce_fusion()&&
        !(reference&&!std::strcmp(reference,"1"));
}

bool CudaBackend::dispatch_hc_up_reduce(const GraphNode& up,const GraphNode& reduce,
    const std::vector<const Tensor*>& inputs,Tensor* output,ThreadPool*) {
    if(!supports_hc_up_reduce_fusion()||impl_->failed||inputs.size()!=3||
        !inputs[0]||!inputs[1]||!inputs[2]||!output||up.op_type!=OpType::MATMUL||
        reduce.op_type!=OpType::GR_REDUCE||graph_params::get_i32(up.params,0,0)!=0||
        graph_params::get_i32(up.params,1,0)!=0||graph_params::get_i32(up.params,2,-1)!=-1||
        graph_params::get_i32(reduce.params,0,0)!=2560||
        graph_params::get_i32(reduce.params,1,4)!=4)return false;
    const auto& norm=*inputs[0];const auto& low=*inputs[1];const auto& weight=*inputs[2];
    const int64_t rows=low.shape[1];
    if(!fp32_contiguous(norm)||!fp32_contiguous(low)||!fp32_contiguous(*output)||
        rows<16||rows>mollm::hc::prefill_max_rows||low.shape[0]!=320||low.shape[2]!=1||low.shape[3]!=1||
        norm.shape[0]!=10240||norm.shape[1]!=rows||norm.shape[2]!=1||norm.shape[3]!=1||
        output->shape[0]!=2560||output->shape[1]!=rows||output->shape[2]!=1||output->shape[3]!=1||
        weight.shape[0]!=10240||weight.shape[1]!=320||weight.shape[2]!=1||weight.shape[3]!=1)return false;
    const auto* normalized=device_pointer_const<float>(norm);
    const auto* activation=device_pointer_const<float>(low);auto* destination=device_pointer<float>(*output);
    for(const auto* input:inputs)
        if(!impl_->restore_hc_compact(device_pointer_const<float>(*input),input->view_span_bytes()))return true;
    if(!impl_->restore_hc_compact(destination,output->view_span_bytes()))return true;
    const auto& deferred=impl_->hc_deferred;
    if(!normalized||!activation||!destination||!deferred.owner||deferred.reduced||
        deferred.normalized!=normalized||deferred.owner->deferred_reduce!=&reduce||
        deferred.epoch!=impl_->hc_execution_epoch||deferred.rows!=rows||
        !deferred.carry||!deferred.gamma||!impl_->hc_deferred_inverses)return false;
    const auto* prepared=impl_->find_weight(weight);
    if(!prepared||prepared->n!=10240||prepared->k!=320)return false;
    const bool canonical=prepared->layout==Impl::WeightLayout::Q4Bg32Biased;
    const bool compact=prepared->layout==Impl::WeightLayout::Q4Bg32Compact;
    const bool dense=prepared->layout==Impl::WeightLayout::Dense&&prepared->type==CUDA_R_16F;
    if(!canonical&&!compact&&!dense)return false;
    const uintptr_t out=reinterpret_cast<uintptr_t>(destination);
    const size_t out_bytes=size_t(rows)*2560*sizeof(float);
    for(const Tensor* input:inputs)
        if(hc_byte_ranges_overlap(out,out_bytes,reinterpret_cast<uintptr_t>(
            device_pointer_const<float>(*input)),input->view_span_bytes()))return false;
    if(hc_byte_ranges_overlap(out,out_bytes,reinterpret_cast<uintptr_t>(deferred.carry),size_t(rows)*10240*4)||
        hc_byte_ranges_overlap(out,out_bytes,reinterpret_cast<uintptr_t>(deferred.gamma),10240*4)||
        hc_byte_ranges_overlap(out,out_bytes,reinterpret_cast<uintptr_t>(impl_->hc_deferred_inverses),size_t(rows)*4*4))return false;
    cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
    if(!mollm_cuda::report_cuda(cudaStreamIsCapturing(cudaStreamPerThread,&capture),"HC up fusion capture check")){
        impl_->failed=true;return true;
    }
    if(capture!=cudaStreamCaptureStatusNone)return false;
    const size_t elements=size_t(rows)*320;
    if(!impl_->reserve(impl_->activation_fp16,impl_->activation_fp16_bytes,elements*sizeof(__half))){
        impl_->failed=true;return true;
    }
    auto* half=static_cast<__half*>(impl_->activation_fp16);
    mollm_cuda::launch_fp32_to_fp16(activation,half,elements);
    const void* linear=prepared->prefill_fp16?prepared->prefill_fp16:prepared->data;
    if(!dense&&!prepared->prefill_fp16){
        if(!impl_->reserve(impl_->quantized_weight_scratch,impl_->quantized_weight_scratch_bytes,size_t(10240)*320*2)){
            impl_->failed=true;return true;
        }
        if(canonical)mollm_cuda::launch_prepare_q4_g32_dense_gemm(
            static_cast<const Q4B8G32Block*>(prepared->data),reinterpret_cast<__half2*>(impl_->quantized_weight_scratch),
            nullptr,nullptr,0,10240,10);
        else mollm_cuda::launch_prepare_q4_compact_g32_dense_gemm(
            static_cast<const mollm_cuda::Q4CompactG32*>(prepared->data),reinterpret_cast<__half2*>(impl_->quantized_weight_scratch),
            nullptr,nullptr,0,10240,10);
        linear=impl_->quantized_weight_scratch;
    }
    if(!mollm_cuda::report_cuda(cudaGetLastError(),"HC up fusion FP16 preparation")){
        impl_->failed=true;return true;
    }
    if((reinterpret_cast<uintptr_t>(half)&31)||(reinterpret_cast<uintptr_t>(linear)&31)||
        (out&3)||(reinterpret_cast<uintptr_t>(deferred.carry)&3)||
        (reinterpret_cast<uintptr_t>(deferred.gamma)&3)||
        (reinterpret_cast<uintptr_t>(impl_->hc_deferred_inverses)&3))return false;
    if(hc_byte_ranges_overlap(out,out_bytes,reinterpret_cast<uintptr_t>(half),elements*2)||
        hc_byte_ranges_overlap(out,out_bytes,reinterpret_cast<uintptr_t>(linear),size_t(10240)*320*2))return false;
    constexpr int up_input_pitch=320;
    constexpr bool up_wide_columns=true;
    // Wider tiles lose on tiny tasks; ownership/shape/capture gates above
    // remain authoritative, and decode never enters this fusion.
    const int up_columns=up_wide_columns&&rows>=256?32:16;
    if(!mollm_cuda::launch_hc_up_reduce_fused(reinterpret_cast<const uint16_t*>(half),
        static_cast<const uint16_t*>(linear),deferred.carry,deferred.gamma,impl_->hc_deferred_inverses,
        destination,static_cast<int>(rows),nullptr,up_input_pitch,up_columns,
        impl_->hc_compact_exact(deferred.carry,static_cast<int>(rows)))){impl_->failed=true;return true;}
    if(impl_->hc_compact_exact(deferred.carry,static_cast<int>(rows)))++impl_->hc_compact_consumers;
    if(!mollm_cuda::report_cuda(cudaGetLastError(),"HC up reduce fused")){impl_->failed=true;return true;}
    impl_->hc_deferred.reduced=true;++impl_->hc_up_reduce_fused_calls;++impl_->hc_deferred_reduces;
    if(up_input_pitch!=320)++impl_->hc_up_reduce_padded_calls;
    if(up_columns==32)++impl_->hc_up_reduce_wide_calls;
    ++impl_->native_ops[static_cast<uint32_t>(reduce.op_type)];return true;
}

bool CudaBackend::dispatch_sigmoid_gr_reduce(const GraphNode& node,
    const std::vector<const Tensor*>& inputs, Tensor* output, ThreadPool*) {
    if (!supports_sigmoid_gr_reduce_fusion() || node.op_type != OpType::GR_REDUCE ||
        inputs.size() != 2 || !inputs[0] || !inputs[1] || !output) return false;
    const auto& x = *inputs[0];
    const auto& logits = *inputs[1];
    const int hidden = graph_params::get_i32(node.params, 0, 0);
    const int streams = graph_params::get_i32(node.params, 1, 4);
    if (hidden <= 0 || streams <= 0 || !fp32_contiguous(x) ||
        !fp32_contiguous(logits) || !fp32_contiguous(*output) ||
        !same_shape(x, logits) || x.shape[0] != int64_t(hidden) * streams ||
        x.shape[1] <= 1 || x.shape[1] > INT_MAX || x.shape[2] != 1 || x.shape[3] != 1 ||
        output->shape[0] != hidden || output->shape[1] != x.shape[1] ||
        output->shape[2] != 1 || output->shape[3] != 1) return false;
    const auto* source = device_pointer_const<float>(x);
    const auto* raw = device_pointer_const<float>(logits);
    auto* destination = device_pointer<float>(*output);
    if (!source || !raw || !destination) return false;
    if(!impl_->restore_hc_compact(source,x.view_span_bytes())||
       !impl_->restore_hc_compact(raw,logits.view_span_bytes())||
       !impl_->restore_hc_compact(destination,output->view_span_bytes()))return true;
    const bool deferred=impl_->hc_deferred.owner&&impl_->hc_deferred.normalized==source&&
        impl_->hc_deferred.owner->deferred_reduce==&node&&
        impl_->hc_deferred.epoch==impl_->hc_execution_epoch&&
        hidden==2560&&streams==4&&x.shape[1]==impl_->hc_deferred.rows;
    if(deferred) {
        if(!mollm_cuda::launch_gr_reduce_sigmoid_deferred_rms(impl_->hc_deferred.carry,
            raw,impl_->hc_deferred.gamma,impl_->hc_deferred_inverses,destination,
            impl_->hc_deferred.rows,nullptr,impl_->hc_compact_exact(impl_->hc_deferred.carry,impl_->hc_deferred.rows))) {impl_->failed=true;return true;}
        if(impl_->hc_compact_exact(impl_->hc_deferred.carry,impl_->hc_deferred.rows))++impl_->hc_compact_consumers;
        impl_->hc_deferred.reduced=true;
        ++impl_->hc_deferred_reduces;
    } else {
        if((impl_->deferred_rms_overlaps(source,x.view_span_bytes(),true)||
            impl_->deferred_rms_overlaps(raw,logits.view_span_bytes(),true))&&
           !impl_->materialize_deferred_rms())return true;
        mollm_cuda::launch_gr_reduce_sigmoid(source, raw, destination,
            hidden, streams, static_cast<int>(x.shape[1]));
    }
    if (!mollm_cuda::report_cuda(cudaGetLastError(), "sigmoid_gr_reduce_cuda"))
        impl_->failed = true;
    ++impl_->sigmoid_gr_reduce_fused_calls;
    ++impl_->native_ops[static_cast<uint32_t>(node.op_type)];
    return true;
}

bool CudaBackend::supports_gr_inject_rms_fusion() const {
    // Qualified prefill fusion; capture-audit and reference backends keep the
    // unfused nodes.
    return impl_->qualified_fusions && !impl_->capture_audit;
}

size_t CudaBackend::gr_inject_rms_fused_calls() const {
    return impl_->gr_inject_rms_fused_calls;
}
bool CudaBackend::supports_moe_gr_inject_rms_fusion() const {
    return impl_->qualified_fusions&&supports_gr_inject_rms_fusion()&&
        !impl_->segment_enabled&&!impl_->capture_audit&&
        impl_->hc_deferred_enabled&&impl_->hc_compact_enabled;
}
size_t CudaBackend::moe_gr_inject_rms_fused_calls() const {return impl_->moe_gr_inject_rms_calls;}

bool CudaBackend::dispatch_moe_gr_inject_rms(const GraphNode& moe,const GraphNode& inject,
    const GraphNode& rms,const std::vector<const Tensor*>& moe_inputs,
    const std::vector<const Tensor*>& hc_inputs,Tensor* branch,Tensor* carry,Tensor* output,ThreadPool* pool) {
    if(!supports_moe_gr_inject_rms_fusion()||impl_->failed||impl_->moe_hc_chain.armed||
       moe.op_type!=OpType::MOE||inject.op_type!=OpType::GR_INJECT||rms.op_type!=OpType::GROUP_RMS_NORM||
       moe_inputs.size()!=8||hc_inputs.size()!=4||!branch||!carry||!output||hc_inputs[0]!=branch||
       !fp32_contiguous(*branch)||branch->shape[0]!=2560||branch->shape[1]<16||branch->shape[1]>2048||
       !device_pointer<float>(*branch)||!moe_inputs[2]||!moe_inputs[3]||
       moe_inputs[2]->prec!=Precision::NVFP4||moe_inputs[3]->prec!=Precision::NVFP4)return false;
    cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
    if(!mollm_cuda::report_cuda(cudaStreamIsCapturing(cudaStreamPerThread,&capture),"MoE HC atomic capture check")){impl_->failed=true;return true;}
    if(capture!=cudaStreamCaptureStatusNone)return false;
    // No pending state may escape this one synchronous host dispatch scope.
    struct Scope {Impl* impl;~Scope(){if(!impl->failed)impl->materialize_moe_hc_branch();impl->moe_hc_chain={};}} scope{impl_.get()};
    impl_->moe_hc_chain.armed=true;
    dispatch(moe,moe_inputs,branch,pool);
    if(impl_->failed)return true;
    if(!dispatch_gr_inject_rms(inject,rms,hc_inputs,carry,output,pool)&&!impl_->failed){
        if(!impl_->materialize_moe_hc_branch())return true;
        dispatch(inject,{hc_inputs[0],hc_inputs[1],hc_inputs[2]},carry,pool);
        if(!impl_->failed)dispatch(rms,{carry,hc_inputs[3]},output,pool);
    }
    return true;
}
size_t CudaBackend::deferred_rms_producer_calls() const {return impl_->hc_deferred_producers;}
size_t CudaBackend::deferred_rms_materializations() const {return impl_->hc_deferred_materializations;}
size_t CudaBackend::compact_carry_producer_calls() const {return impl_->hc_compact_producers;}
size_t CudaBackend::compact_carry_consumer_calls() const {return impl_->hc_compact_consumers;}
size_t CudaBackend::compact_carry_residual_reads() const {return impl_->hc_compact_residual_reads;}
size_t CudaBackend::compact_carry_restorations() const {return impl_->hc_compact_restores;}
size_t CudaBackend::compact_carry_allocation_registrations() const {return impl_->hc_allocation_generation;}
size_t CudaBackend::attention_head_group_calls() const {return impl_->attention_prefix_head_group_enqueues;}
size_t CudaBackend::attention_native_pv_calls() const {return impl_->attention_pv_gemm_enqueues;}
size_t CudaBackend::attention_head_lane_calls() const {return impl_->attention_head_lane_pair_enqueues;}
size_t CudaBackend::attention_score_workspace_bytes() const {return impl_->attention_scores_bytes;}
size_t CudaBackend::attention_query_group_calls() const {return impl_->attention_prefix_query_group_enqueues;}

bool CudaBackend::dispatch_gr_inject_rms(const GraphNode& inject, const GraphNode& rms,
    const std::vector<const Tensor*>& inputs, Tensor* carry, Tensor* output, ThreadPool*) {
    if (!supports_gr_inject_rms_fusion() || impl_->failed ||
        inject.op_type != OpType::GR_INJECT || rms.op_type != OpType::GROUP_RMS_NORM ||
        inputs.size() != 4 || !carry || !output ||
        !impl_->hc_prefill_reuse_enabled || !impl_->hc_prefill_rms_half_enabled ||
        !impl_->hc_execution_epoch) return false;
    const auto dense = [](const Tensor* t) {
        return t && fp32_contiguous(*t) && t->shape[0] > 0 && t->shape[1] > 0 &&
            t->shape[2] == 1 && t->shape[3] == 1 && device_pointer_const<float>(*t);
    };
    if (!dense(carry) || !dense(output) ||
        !std::all_of(inputs.begin(), inputs.end(), dense)) return false;
    const int hidden = graph_params::get_i32(inject.params, 0, 0);
    const int streams = graph_params::get_i32(inject.params, 1, 4);
    const auto& branch = *inputs[0];
    const auto& residual = *inputs[1];
    const auto& gates = *inputs[2];
    const auto& weight = *inputs[3];
    if (hidden != 2560 || streams != 4 ||
        graph_params::get_i32(rms.params, 0, 0) != hidden ||
        branch.shape[0] != hidden || branch.shape[1] < 16 || branch.shape[1] > 4096 ||
        residual.shape[0] != 10240 || residual.shape[1] != branch.shape[1] ||
        gates.shape[0] != streams || gates.shape[1] != branch.shape[1] ||
        weight.shape[0] != 10240 || weight.shape[1] != 1 ||
        !same_shape(residual, *carry) || !same_shape(residual, *output)) return false;
    auto entry = impl_->hc_prefill_rms_nodes.find(&rms);
    if (entry == impl_->hc_prefill_rms_nodes.end() || entry->second->input != output)
        return false;
    const int rows = static_cast<int>(branch.shape[1]);
    if (!impl_->reserve_hc_prefill_half(size_t(rows) * 10240 * sizeof(__half)))
        return impl_->failed; // handled error; never dispatch a fallback after failure
    auto* destination = device_pointer<float>(*output);
    if(!impl_->materialize_deferred_rms())return true;
    const bool deferred=impl_->hc_deferred_enabled&&!impl_->segment_enabled&&
        impl_->hc_deferred_inverses&&entry->second->deferred_reduce&&
        supports_sigmoid_gr_reduce_fusion();
    const auto* residual_pointer=device_pointer_const<float>(residual);
    auto* carry_pointer=device_pointer<float>(*carry);
    const auto allocation=impl_->hc_allocations.find(carry->device.buffer);
    cudaStreamCaptureStatus compact_capture=cudaStreamCaptureStatusNone;
    if(impl_->hc_compact_enabled&&!mollm_cuda::report_cuda(cudaStreamIsCapturing(cudaStreamPerThread,&compact_capture),"compact HC producer capture check")){
        impl_->failed=true;return true;
    }
    const bool compact=impl_->hc_compact_enabled&&deferred&&rows>=16&&rows<=mollm::hc::prefill_max_rows&&
        compact_capture==cudaStreamCaptureStatusNone&&carry->device.offset==0&&carry_pointer==carry->device.buffer&&
        allocation!=impl_->hc_allocations.end()&&allocation->second.bytes>=size_t(rows)*10240*4&&
        impl_->reserve_hc_compact_scratch();
    if(impl_->failed)return true;
    bool compact_residual=compact&&impl_->hc_compact_exact(residual_pointer,rows)&&residual.device.offset==0;
    for(size_t i=0;i<inputs.size();++i)
        if(!(i==1&&compact_residual)&&!impl_->restore_hc_compact(device_pointer_const<float>(*inputs[i]),inputs[i]->view_span_bytes()))return true;
    if(!impl_->restore_hc_compact(carry_pointer,carry->view_span_bytes())||
       !impl_->restore_hc_compact(destination,output->view_span_bytes()))return true;
    // Another input/output view may have forced this same owner to FP32.
    compact_residual=compact_residual&&impl_->hc_compact_exact(residual_pointer,rows);
    auto& moe_chain=impl_->moe_hc_chain;
    const bool moe_fused=moe_chain.ready&&deferred&&compact&&compact_residual&&
        moe_chain.branch==device_pointer_const<float>(branch)&&moe_chain.rows==rows;
    if(moe_chain.ready&&!moe_fused&&!impl_->materialize_moe_hc_branch())return true;
    if(deferred) {
        if(moe_fused){
            if(!mollm_cuda::launch_moe_combine_gr_inject_rms_deferred(moe_chain.routed,moe_chain.shared,moe_chain.scale,
                residual_pointer,device_pointer_const<float>(gates),device_pointer_const<float>(weight),
                device_pointer<float>(*const_cast<Tensor*>(inputs[0])),carry_pointer,static_cast<uint16_t*>(impl_->hc_prefill_half),
                impl_->hc_deferred_inverses,rows,graph_params::get_f32(rms.params,0,1e-6f),moe_chain.stream,false)){
                impl_->failed=true;return true;}
            moe_chain.ready=false;++impl_->moe_gr_inject_rms_calls;
        } else if(!mollm_cuda::launch_gr_inject_rms_deferred(device_pointer_const<float>(branch),
            device_pointer_const<float>(residual),device_pointer_const<float>(gates),
            device_pointer_const<float>(weight),device_pointer<float>(*carry),
            static_cast<uint16_t*>(impl_->hc_prefill_half),impl_->hc_deferred_inverses,
            rows,graph_params::get_f32(rms.params,0,1e-6f),nullptr,compact,compact_residual)) {impl_->failed=true;return true;}
        if(compact){
            impl_->hc_compact_carries[carry->device.buffer]={rows,allocation->second.generation};
            ++impl_->hc_compact_producers;if(compact_residual)++impl_->hc_compact_residual_reads;
        }
        impl_->hc_deferred={entry->second,destination,device_pointer_const<float>(*carry),
            device_pointer_const<float>(weight),rows,impl_->hc_execution_epoch,false};
        ++impl_->hc_deferred_producers;
    } else if (!mollm_cuda::launch_gr_inject_group_rms_half(
            device_pointer_const<float>(branch), device_pointer_const<float>(residual),
            device_pointer_const<float>(gates), device_pointer_const<float>(weight),
            device_pointer<float>(*carry), destination,
            static_cast<uint16_t*>(impl_->hc_prefill_half), hidden, streams, rows,
            graph_params::get_f32(rms.params, 0, 1e-6f))) return false;
    if (!mollm_cuda::report_cuda(cudaGetLastError(), "gr_inject_group_rms_half_cuda")) {
        impl_->failed = true;
        return true;
    }
    impl_->hc_prefill_owner = entry->second;
    impl_->hc_prefill_epoch = impl_->hc_execution_epoch;
    impl_->hc_prefill_input = destination;
    impl_->hc_prefill_rows = rows;
    impl_->hc_prefill_from_rms = true;
    ++impl_->hc_prefill_rms_half_calls;
    ++impl_->gr_inject_rms_fused_calls;
    ++impl_->native_ops[static_cast<uint32_t>(inject.op_type)];
    ++impl_->native_ops[static_cast<uint32_t>(rms.op_type)];
    return true;
}

void CudaBackend::dispatch(const GraphNode& node,
                           const std::vector<const Tensor*>& inputs,
                           Tensor* output, ThreadPool* thread_pool) {
    // Views and unknown/native/fallback consumers must see the full logical
    // FP32 owner, not packed bits. Compact-aware fusions bypass this entry.
    if(output&&!impl_->hc_compact_carries.empty()&&
       impl_->deferred_rms_overlaps(device_pointer_const<uint8_t>(*output),output->view_span_bytes())&&
       !impl_->materialize_deferred_rms())return;
    if(!impl_->hc_compact_carries.empty()) {
        for(const auto* input:inputs)
            if(input&&!impl_->restore_hc_compact(device_pointer_const<uint8_t>(*input),input->view_span_bytes()))return;
        if(output&&!impl_->restore_hc_compact(device_pointer_const<uint8_t>(*output),output->view_span_bytes()))return;
    }
    // Only the two validated HC half projections may read a deferred tensor
    // without materializing it. Every ordinary/fallback input sees real FP32.
    if(impl_->hc_deferred.owner) {
        const auto& d=impl_->hc_deferred;const auto* pair=d.owner;
        bool reads=false;
        bool operand_mutation=false;
        for(const auto* input:inputs)
            if(input) {
                const auto* p=device_pointer_const<float>(*input);
                if(impl_->deferred_rms_overlaps(p,input->view_span_bytes(),true))reads=true;
                if(hc_byte_ranges_overlap(reinterpret_cast<uintptr_t>(p),input->view_span_bytes(),
                    reinterpret_cast<uintptr_t>(d.carry),size_t(d.rows)*10240*sizeof(float))||
                   hc_byte_ranges_overlap(reinterpret_cast<uintptr_t>(p),input->view_span_bytes(),
                    reinterpret_cast<uintptr_t>(d.gamma),10240*sizeof(float)))operand_mutation=true;
            }
        if(operand_mutation&&!impl_->materialize_deferred_rms())return;
        if(!impl_->hc_deferred.owner)reads=false;
        if(reads) {
            const bool first=pair->first==&node,second=pair->second==&node;
            const auto* w=inputs.size()==2&&inputs[1]?impl_->find_weight(*inputs[1]):nullptr;
            const int n=first?320:4;
            const bool shadow=(first||second)&&node.op_type==OpType::MATMUL&&
                inputs.size()==2&&inputs[0]==pair->input&&output&&
                impl_->hc_prefill_reuse_enabled&&impl_->hc_prefill_owner==pair&&
                impl_->hc_prefill_epoch==d.epoch&&d.epoch==impl_->hc_execution_epoch&&
                impl_->hc_prefill_input==d.normalized&&impl_->hc_prefill_rows==d.rows&&
                (first?impl_->hc_prefill_from_rms:(!impl_->hc_prefill_from_rms&&d.reduced))&&
                size_t(d.rows)*10240*sizeof(__half)<=impl_->hc_prefill_half_bytes&&
                fp32_contiguous(*inputs[0])&&fp32_contiguous(*output)&&
                inputs[0]->shape[0]==10240&&inputs[0]->shape[1]==d.rows&&
                output->shape[0]==n&&output->shape[1]==d.rows&&
                inputs[0]->shape[2]==1&&inputs[0]->shape[3]==1&&
                output->shape[2]==1&&output->shape[3]==1&&device_pointer<float>(*output)&&
                w&&w->n==n&&w->k==10240&&inputs[1]->device.offset==0&&
                (first||inputs[1]==pair->second_weight)&&
                (w->layout==Impl::WeightLayout::Q4Bg32Biased||
                 w->layout==Impl::WeightLayout::Q4Bg32Compact||
                 w->layout==Impl::WeightLayout::Q4Bg32CompactInteger)&&
                graph_params::get_i32(node.params,0,0)==0&&
                graph_params::get_i32(node.params,1,0)==0&&
                graph_params::get_i32(node.params,2,-1)==-1;
            if(!shadow&&!impl_->materialize_deferred_rms())return;
        }
    }
    if (impl_->capture_audit) {
        Impl::CaptureSignature signature;
        auto append = [&](const Tensor* tensor) {
            if (!tensor) { signature.storage.push_back(0); return; }
            signature.storage.push_back(reinterpret_cast<uintptr_t>(tensor->device.buffer
                ? tensor->device.buffer : tensor->data));
            signature.storage.push_back(tensor->device.offset);
            signature.storage.push_back(static_cast<uint64_t>(tensor->prec));
            for (int d = 0; d < 4; ++d) {
                signature.storage.push_back(tensor->shape[d]);
                signature.storage.push_back(tensor->stride[d]);
            }
        };
        append(output);
        for (const auto* input : inputs) append(input);
        signature.i32 = node.params.i32;
        signature.f32 = node.params.f32;
        auto& counts = impl_->capture_audit_counts[static_cast<uint32_t>(node.op_type)];
        auto found = impl_->capture_signatures.find(&node);
        if (found == impl_->capture_signatures.end()) ++counts.first;
        else {
            const bool same_storage = signature.storage == found->second.storage;
            const bool same_params = signature.i32 == found->second.i32 && signature.f32 == found->second.f32;
            if (same_storage && same_params) ++counts.stable;
            if (!same_storage) ++counts.storage_changed;
            if (!same_params) ++counts.params_changed;
        }
        impl_->capture_signatures[&node] = std::move(signature);
    }
    auto record_native = [&]() {
        ++impl_->native_ops[static_cast<uint32_t>(node.op_type)];
    };
    if (impl_->segment_enabled && node.op_type == OpType::TILE &&
        !inputs.empty() && inputs[0] && inputs[0]->shape[1] == 1) {
        if (impl_->segment_decode_node != &node) {
            impl_->segment_decode_node = &node;
            impl_->segment_decode_steps = 0;
        }
        impl_->segment_candidate = ++impl_->segment_decode_steps > 1 && !impl_->resident_experts.empty();
    }
    if (node.op_type == OpType::SDPA || node.op_type == OpType::SDPA_MLA ||
        node.op_type == OpType::QSA_INDEXER || node.op_type == OpType::QSA_SDPA ||
        node.op_type == OpType::PLE_LOOKUP) finish_decode_segment();
    if (node.op_type == OpType::INPUT ||
        node.op_type == OpType::CONSTANT) {
        record_native();
        return;
    }

    if (node.op_type == OpType::RESHAPE && inputs.size() == 1 && inputs[0] && output) {
        size_t strides[4];
        if (reshape_view_strides(*inputs[0], *output, strides)) {
            const int64_t shape[4] = {output->shape[0], output->shape[1],
                                     output->shape[2], output->shape[3]};
            *output = *inputs[0];
            std::copy(shape, shape + 4, output->shape);
            std::copy(strides, strides + 4, output->stride);
            ++impl_->strided_reshape_calls;
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::RESHAPE && !inputs.empty() && inputs[0] &&
        output && inputs[0]->is_contiguous()) {
        const int64_t shape[4] = {output->shape[0], output->shape[1],
                                  output->shape[2], output->shape[3]};
        *output = *inputs[0];
        for (int dimension = 0; dimension < 4; ++dimension)
            output->shape[dimension] = shape[dimension];
        output->compute_strides();
        record_native();
        return;
    }

    if (node.op_type == OpType::PERMUTE && !inputs.empty() && inputs[0] &&
        output) {
        const Tensor& source = *inputs[0];
        const int axis[4] = {
            graph_params::get_i32(node.params, 0, 0),
            graph_params::get_i32(node.params, 1, 1),
            graph_params::get_i32(node.params, 2, 2),
            graph_params::get_i32(node.params, 3, 3),
        };
        Tensor view = source;
        for (int dimension = 0; dimension < 4; ++dimension) {
            view.shape[axis[dimension]] = source.shape[dimension];
            view.stride[axis[dimension]] = source.stride[dimension];
        }
        *output = view;
        record_native();
        return;
    }

    if (node.op_type == OpType::SLICE && !inputs.empty() && inputs[0] &&
        output) {
        const Tensor& source = *inputs[0];
        const int dimension = graph_params::get_i32(node.params, 0, 0);
        const int offset = graph_params::get_i32(node.params, 1, 0);
        const int size = graph_params::get_i32(
            node.params, 2, static_cast<int>(source.shape[dimension]));
        *output = source;
        const size_t byte_offset =
            static_cast<size_t>(offset) * source.stride[dimension];
        if (source.data)
            output->data = static_cast<uint8_t*>(source.data) + byte_offset;
        output->device.offset = source.device.offset + byte_offset;
        output->shape[dimension] = size;
        record_native();
        return;
    }

    Impl::HcProjectionPair* hc_pair = nullptr;
    const Impl::DeviceWeight* hc_first_weight = nullptr;
    const Impl::DeviceWeight* hc_second_weight = nullptr;
    const auto hc_action = [&]() -> uint64_t {
        if (!impl_->hc_projection_enabled || !impl_->hc_execution_epoch ||
            node.op_type != OpType::MATMUL) return 0;
        auto found = impl_->hc_projection_nodes.find(&node);
        if (found == impl_->hc_projection_nodes.end()) return 0;
        hc_pair = found->second;
        const bool first = hc_pair->first == &node;
        if (first) hc_pair->computed_epoch = 0;
        if (node.op_type != OpType::MATMUL || inputs.size() != 2 ||
            inputs[0] != hc_pair->input || !inputs[1] || !output ||
            !fp32_contiguous(*inputs[0]) || !fp32_contiguous(*output) ||
            inputs[0]->shape[0] != 10240 || inputs[0]->shape[1] != 1 ||
            inputs[0]->shape[2] != 1 || inputs[0]->shape[3] != 1 ||
            output->shape[0] != (first ? 320 : 4) || output->shape[1] != 1 ||
            output->shape[2] != 1 || output->shape[3] != 1 ||
            graph_params::get_i32(node.params, 0, 0) != 0 ||
            graph_params::get_i32(node.params, 1, 0) != 0 ||
            graph_params::get_i32(node.params, 2, -1) != -1 ||
            inputs[1]->device.offset || hc_pair->second_weight->device.offset) return 0;
        const auto* activation = device_pointer_const<float>(*inputs[0]);
        if (!activation || reinterpret_cast<uintptr_t>(activation) % alignof(float2) ||
            !device_pointer<float>(*output)) return 0;
        hc_second_weight = impl_->find_weight(*hc_pair->second_weight);
        if (!hc_second_weight || hc_second_weight->row_g32 ||
            hc_second_weight->layout != Impl::WeightLayout::Q4Bg32Biased ||
            hc_second_weight->n != 4 || hc_second_weight->k != 10240) return 0;
        if (!first) {
            return inputs[1] == hc_pair->second_weight &&
                hc_pair->computed_epoch == impl_->hc_execution_epoch &&
                hc_pair->computed_input == activation &&
                hc_pair->computed_weight == hc_second_weight ? 2 : 0;
        }
        hc_first_weight = impl_->find_weight(*inputs[1]);
        if (!hc_first_weight || hc_first_weight->row_g32 ||
            hc_first_weight->layout != Impl::WeightLayout::Q4Bg32Biased ||
            hc_first_weight->n != 320 || hc_first_weight->k != 10240) return 0;
        hc_pair->computed_input = activation;
        hc_pair->computed_weight = hc_second_weight;
        hc_pair->computed_epoch = impl_->hc_execution_epoch;
        return 1;
    };
    uint64_t hc_tag = hc_action();
    const uint64_t invalidations_before = impl_->segment_invalidations;
    if (node.op_type != OpType::SDPA && node.op_type != OpType::SDPA_MLA &&
        node.op_type != OpType::QSA_INDEXER && node.op_type != OpType::QSA_SDPA &&
        node.op_type != OpType::PLE_LOOKUP &&
        try_decode_segment(node, inputs, output, thread_pool, hc_tag,
            hc_tag ? hc_second_weight->data : nullptr, hc_tag ? hc_pair->cache : nullptr)) {
        record_native();
        return;
    }
    // Segment invalidation can redispatch earlier records; re-evaluate the
    // epoch after that recovery instead of trusting pre-recovery host state.
    if (impl_->segment_invalidations != invalidations_before) hc_tag = hc_action();
    if (hc_tag) {
        const cudaError_t result = hc_tag == 1
            ? mollm_cuda::launch_q4_g32_unequal_projections(
                hc_pair->computed_input,
                static_cast<const Q4B8G32Block*>(hc_first_weight->data),
                static_cast<const Q4B8G32Block*>(hc_second_weight->data),
                device_pointer<float>(*output), hc_pair->cache, 320, 4, 10240, cudaStreamPerThread)
            : cudaMemcpyAsync(device_pointer<float>(*output), hc_pair->cache,
                4 * sizeof(float), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        if (!mollm_cuda::report_cuda(result, "HC paired projection")) impl_->failed = true;
        else if (hc_tag == 1) ++impl_->hc_projection_enqueues;
        else ++impl_->hc_projection_copies;
        record_native();
        return;
    }

    if (node.op_type == OpType::TILE && !inputs.empty() && inputs[0] && output &&
        inputs[0]->prec == Precision::FP32 && fp32_contiguous(*output)) {
        const Tensor& source = *inputs[0];
        bool valid = true;
        for (int d = 0; d < 4; ++d) {
            const int repeat = graph_params::get_i32(node.params, d, 1);
            valid = valid && repeat > 0 && source.shape[d] > 0 &&
                source.shape[d] <= INT64_MAX / std::max(repeat, 1) &&
                output->shape[d] == source.shape[d] * repeat &&
                source.stride[d] % sizeof(float) == 0;
        }
        const float* input = device_pointer_const<float>(source);
        float* destination = device_pointer<float>(*output);
        if (valid && input && destination && input != destination) {
            mollm_cuda::launch_tile(input, destination, output->nelements(),
                output->shape[0], output->shape[1], output->shape[2],
                source.shape[0], source.shape[1], source.shape[2], source.shape[3],
                source.stride[0] / sizeof(float), source.stride[1] / sizeof(float),
                source.stride[2] / sizeof(float), source.stride[3] / sizeof(float));
            if (!mollm_cuda::report_cuda(cudaGetLastError(), "tile_cuda")) impl_->failed = true;
            else record_native();
            return;
        }
    }

    if ((node.op_type == OpType::CONTIGUOUS ||
         node.op_type == OpType::RESHAPE) &&
        !inputs.empty() && inputs[0] && output &&
        inputs[0]->prec == Precision::FP32 &&
        output->prec == Precision::FP32 && output->is_contiguous()) {
        const Tensor& source = *inputs[0];
        const float* input = device_pointer_const<float>(source);
        float* destination = device_pointer<float>(*output);
        const size_t count = static_cast<size_t>(output->nelements());
        if (input && destination && source.nelements() == output->nelements()) {
            mollm_cuda::launch_contiguous(
                input, destination, count, source.shape[0], source.shape[1],
                source.shape[2], source.stride[0] / sizeof(float),
                source.stride[1] / sizeof(float),
                source.stride[2] / sizeof(float),
                source.stride[3] / sizeof(float));
            if (!mollm_cuda::report_cuda(
                    cudaGetLastError(), "contiguous_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::ROTARY_EMBED && inputs.size() >= 3 &&
        inputs[0] && inputs[1] && inputs[2] && output &&
        inputs[0]->prec == Precision::FP32 &&
        inputs[1]->prec == Precision::FP32 &&
        inputs[2]->prec == Precision::FP32 &&
        output->prec == Precision::FP32) {
        const Tensor& source = *inputs[0];
        const Tensor& cosine = *inputs[1];
        const Tensor& sine = *inputs[2];
        const float* input = device_pointer_const<float>(source);
        const float* cos_data = device_pointer_const<float>(cosine);
        const float* sin_data = device_pointer_const<float>(sine);
        float* destination = device_pointer<float>(*output);
        const int feature_dim = static_cast<int>(source.shape[0]);
        const int sequence_length = static_cast<int>(source.shape[1]);
        const int channels = static_cast<int>(source.shape[2] * source.shape[3]);
        const int rope_dim = graph_params::get_i32(node.params, 0, 64);
        const bool interleave =
            graph_params::get_i32(node.params, 1, 1) != 0;
        if (input && cos_data && sin_data && destination &&
            rope_dim > 0 && rope_dim <= feature_dim && rope_dim % 2 == 0 &&
            cosine.shape[0] >= rope_dim / 2 &&
            sine.shape[0] >= rope_dim / 2) {
            mollm_cuda::launch_rope(
                input, cos_data, sin_data, destination, feature_dim,
                sequence_length, channels, static_cast<int>(source.shape[2]),
                rope_dim, interleave,
                source.stride[0] / sizeof(float),
                source.stride[1] / sizeof(float),
                source.stride[2] / sizeof(float),
                source.stride[3] / sizeof(float),
                cosine.stride[0] / sizeof(float),
                cosine.stride[1] / sizeof(float),
                sine.stride[0] / sizeof(float),
                sine.stride[1] / sizeof(float),
                output->stride[0] / sizeof(float),
                output->stride[1] / sizeof(float),
                output->stride[2] / sizeof(float),
                output->stride[3] / sizeof(float));
            if (!mollm_cuda::report_cuda(cudaGetLastError(), "rope_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if(node.op_type==OpType::QSA_INDEXER) {
        const auto fail=[&](const char* reason){std::fprintf(stderr,"CudaBackend: QSA indexer rejected: %s\n",reason);impl_->failed=true;};
        if(inputs.size()!=8||!output||output->prec!=Precision::INT32||node.params.i32.size()!=5||
           std::any_of(inputs.begin(),inputs.end(),[](const Tensor* t){return !t||!t->data;})) {
            fail("invalid operands/parameters");return;
        }
        mollm_cuda::QsaIndexerParams p{};
        p.heads=node.params.i32[0];p.dimension=node.params.i32[1];p.rope_dim=node.params.i32[2];
        p.ratio=node.params.i32[3];const int budget=node.params.i32[4];
        p.eps=graph_params::get_f32(node.params,0,1e-6f);
        if(p.heads<1||p.heads>16||p.dimension<2||p.dimension>256||p.rope_dim<2||
           p.rope_dim>p.dimension||p.rope_dim%2||p.ratio<1||p.ratio>16||budget<1||
           budget>4096-p.ratio+1||budget%p.ratio||inputs[0]->shape[1]<1||inputs[0]->shape[1]>65535||
           inputs[0]->prec!=Precision::FP32||inputs[0]->stride[0]!=sizeof(float)||inputs[0]->stride[1]%4||
           inputs[0]->shape[0]!=(p.heads+1)*p.dimension||inputs[0]->shape[2]!=1||inputs[0]->shape[3]!=1||
           output->shape[0]!=budget+p.ratio-1||output->shape[1]!=inputs[0]->shape[1]||
           output->shape[2]!=1||output->shape[3]!=1||output->stride[0]!=sizeof(int32_t)||output->stride[1]%4||
           output->stride[1]<size_t(output->shape[0])*sizeof(int32_t)||!std::isfinite(p.eps)||p.eps<=0) {
            fail("unsupported geometry/precision");return;
        }
        for(int i:{1,2,3,4,5})if(!fp32_contiguous(*inputs[i])){fail("noncontiguous norm/frequency/state");return;}
        if(inputs[1]->nelements()!=p.dimension||inputs[2]->nelements()!=p.dimension||
           inputs[3]->nelements()!=p.rope_dim/2||inputs[4]->shape[0]!=p.dimension||
           inputs[5]->shape[0]!=p.dimension||!inputs[6]->device.buffer||
           inputs[7]->prec!=Precision::FP32) {fail("invalid norm/frequency/cache/mask");return;}
        const auto* meta=cache_meta(static_cast<const uint8_t*>(inputs[6]->data)+inputs[6]->device.offset);
        if(meta->max_seq_len<1||meta->max_seq_len>mollm_qsa::max_context||meta->current_seq_len>meta->max_seq_len||
           size_t(inputs[0]->shape[1])>meta->max_seq_len-meta->current_seq_len) {
            fail("cache length/capacity");return;
        }
        p.queries=inputs[0]->shape[1];p.past=meta->current_seq_len;p.capacity=meta->max_seq_len;
        p.projection_stride=inputs[0]->stride[1]/sizeof(float);
        const int keys=p.past+p.queries;
        p.score_stride=std::max(1,keys/p.ratio);
        if(inputs[4]->shape[1]<p.capacity||inputs[5]->shape[1]<p.capacity/p.ratio||
           inputs[7]->shape[0]<keys||inputs[7]->shape[1]<p.queries||
           inputs[7]->shape[2]!=1||inputs[7]->shape[3]!=1||
           inputs[7]->stride[0]%sizeof(float)||inputs[7]->stride[1]%sizeof(float)) {
            fail("insufficient state or mask span");return;
        }
        cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
        if(!mollm_cuda::report_cuda(cudaStreamIsCapturing(cudaStreamPerThread,&capture),"QSA capture check")||
           capture!=cudaStreamCaptureStatusNone) {fail("mask validation requires noncaptured dispatch");return;}
        if(!impl_->reserve(impl_->qsa_query,impl_->qsa_query_bytes,size_t(p.queries)*p.heads*p.dimension*4,true)||
           !impl_->reserve(impl_->qsa_scores,impl_->qsa_scores_bytes,size_t(p.queries)*p.score_stride*4,true)||
           !impl_->reserve(impl_->qsa_mask_invalid,impl_->qsa_mask_invalid_bytes,sizeof(uint32_t),true)) {
            fail("workspace allocation");return;
        }
        const float* mask=device_pointer_const<float>(*inputs[7]);
        auto* invalid=static_cast<uint32_t*>(impl_->qsa_mask_invalid);
        uint32_t host_invalid=1;
        // Correctness-first gate. Never silently apply contiguous grouping to
        // a masked gap. This tiny flag readback is not a kernel-only timing.
        if(!mollm_cuda::report_cuda(mollm_cuda::launch_qsa_validate_causal_mask(mask,invalid,
             p.queries,keys,p.past,inputs[7]->stride[0]/4,inputs[7]->stride[1]/4,cudaStreamPerThread),"QSA mask validation")||
           !mollm_cuda::report_cuda(cudaMemcpyAsync(&host_invalid,invalid,4,cudaMemcpyDeviceToHost,cudaStreamPerThread),"QSA mask flag")||
           !mollm_cuda::report_cuda(cudaStreamSynchronize(cudaStreamPerThread),"QSA mask synchronize")||host_invalid) {
            fail("requires contiguous causal visibility; masked gaps are not implemented");return;
        }
        auto* query=static_cast<float*>(impl_->qsa_query);auto* scores=static_cast<float*>(impl_->qsa_scores);
        const bool streaming=keys/p.ratio>mollm_qsa::sort_tile_groups;
        if(streaming) {
            const size_t bytes=size_t(p.queries)*2*(budget/p.ratio)*4;
            if(!impl_->reserve(impl_->qsa_topk_scores,impl_->qsa_topk_scores_bytes,bytes,true)||
               !impl_->reserve(impl_->qsa_topk_ids,impl_->qsa_topk_ids_bytes,bytes,true)) {fail("top-k workspace allocation");return;}
        }
        if(!mollm_cuda::report_cuda(mollm_cuda::launch_qsa_indexer_update(
            device_pointer_const<float>(*inputs[0]),device_pointer_const<float>(*inputs[1]),
            device_pointer_const<float>(*inputs[2]),device_pointer_const<float>(*inputs[3]),
            device_pointer<float>(*const_cast<Tensor*>(inputs[4])),device_pointer<float>(*const_cast<Tensor*>(inputs[5])),
            query,scores,p,cudaStreamPerThread),"QSA indexer update")||
           !mollm_cuda::report_cuda(mollm_cuda::launch_qsa_causal_block_topk(scores,device_pointer<int32_t>(*output),
            p.queries,keys,p.past,budget,p.ratio,p.score_stride,output->stride[1]/4,cudaStreamPerThread,
            streaming?static_cast<float*>(impl_->qsa_topk_scores):nullptr,
            streaming?static_cast<int32_t*>(impl_->qsa_topk_ids):nullptr,false),"QSA top-k")) {
            fail("native launch");return;
        }
        ++impl_->qsa_indexer_enqueues;
        if(streaming)++impl_->qsa_streaming_topk_enqueues;record_native();return;
    }

    if ((node.op_type == OpType::SDPA || node.op_type == OpType::QSA_SDPA ||
         node.op_type == OpType::SDPA_MLA) &&
        inputs.size() >= 3 && inputs[0] && inputs[1] && inputs[2] &&
        output && inputs[0]->prec == Precision::FP32 &&
        inputs[1]->prec == Precision::FP32 &&
        inputs[2]->prec == Precision::FP32 &&
        output->prec == Precision::FP32) {
        const Tensor& query = *inputs[0];
        const Tensor& current_key = *inputs[1];
        const Tensor& current_value = *inputs[2];
        const Tensor* mask = inputs.size() > 3 && inputs[3] &&
                inputs[3]->data && inputs[3]->nelements() > 0
            ? inputs[3] : nullptr;
        const Tensor* key_cache = inputs.size() > 4 && inputs[4] &&
                inputs[4]->data
            ? inputs[4] : nullptr;
        const Tensor* value_cache = inputs.size() > 5 && inputs[5] &&
                inputs[5]->data
            ? inputs[5] : nullptr;
        const int cache_mode = graph_params::get_i32(node.params, 0, 2);
        const bool causal = graph_params::get_i32(node.params, 1, 1) != 0;
        const int num_heads = graph_params::get_i32(
            node.params, 2, static_cast<int>(query.shape[2]));
        const int num_kv_heads = graph_params::get_i32(
            node.params, 3, static_cast<int>(current_key.shape[2]));
        const int key_dim = graph_params::get_i32(
            node.params, 4, static_cast<int>(query.shape[0]));
        const int value_dim = graph_params::get_i32(
            node.params, 5, static_cast<int>(current_value.shape[0]));
        float scale = graph_params::get_f32(node.params, 0, 0.0f);
        if (scale == 0.0f)
            scale = 1.0f / std::sqrt(static_cast<float>(key_dim));
        const int query_length = static_cast<int>(query.shape[1]);
        const int current_length = static_cast<int>(current_key.shape[1]);
        int past_length = 0;
        int key_capacity = current_length;
        bool cached = false;
        const float* current_key_data =
            device_pointer_const<float>(current_key);
        const float* current_value_data =
            device_pointer_const<float>(current_value);
        const void* key_data = current_key_data;
        const void* value_data = current_value_data;
        void* key_cache_data = nullptr;
        void* value_cache_data = nullptr;
        bool fp16_cache = false;
        if (cache_mode == 2 && key_cache && value_cache &&
            key_cache->prec != value_cache->prec) {
            std::fprintf(
                stderr,
                "CudaBackend: SDPA requires matching K/V cache precision\n");
            impl_->failed = true;
            return;
        }
        if (cache_mode == 2 && key_cache && value_cache &&
            key_cache->prec == value_cache->prec &&
            (key_cache->prec == Precision::FP16 ||
             key_cache->prec == Precision::FP32) &&
            key_cache->device.buffer && value_cache->device.buffer) {
            const auto* key_metadata = cache_meta(
                static_cast<const uint8_t*>(key_cache->data) +
                key_cache->device.offset);
            const auto* value_metadata = cache_meta(
                static_cast<const uint8_t*>(value_cache->data) +
                value_cache->device.offset);
            past_length = static_cast<int>(key_metadata->current_seq_len);
            key_capacity = static_cast<int>(key_metadata->max_seq_len);
            key_cache_data =
                static_cast<uint8_t*>(key_cache->device.buffer) +
                key_cache->device.offset + CacheMetadata::SIZE;
            value_cache_data =
                static_cast<uint8_t*>(value_cache->device.buffer) +
                value_cache->device.offset + CacheMetadata::SIZE;
            fp16_cache = key_cache->prec == Precision::FP16;
            cached =
                value_metadata->current_seq_len ==
                    key_metadata->current_seq_len &&
                value_metadata->max_seq_len == key_metadata->max_seq_len &&
                key_metadata->num_kv_heads ==
                    static_cast<uint64_t>(num_kv_heads) &&
                value_metadata->num_kv_heads ==
                    static_cast<uint64_t>(num_kv_heads) &&
                key_metadata->head_dim == static_cast<uint64_t>(key_dim) &&
                value_metadata->v_head_dim ==
                    static_cast<uint64_t>(value_dim);
        }
        const int key_length = past_length + current_length;
        const float* query_data = device_pointer_const<float>(query);
        float* output_data = device_pointer<float>(*output);
        const float* mask_data = mask
            ? device_pointer_const<float>(*mask) : nullptr;
        const bool valid = query_data && current_key_data &&
            current_value_data &&
            output_data && num_heads > 0 && num_kv_heads > 0 &&
            num_heads % num_kv_heads == 0 && query_length > 0 &&
            current_length > 0 && key_dim > 0 && value_dim > 0 &&
            query.shape[0] >= key_dim && current_key.shape[0] >= key_dim &&
            current_value.shape[0] >= value_dim &&
            output->shape[0] >= value_dim &&
            (cache_mode == 0 || cached) &&
            (!cached || (key_cache_data && value_cache_data &&
                         key_length <= key_capacity)) &&
            (!mask || mask_data);
        if (valid) {
            bool cache_appended = false;
            const auto append_cache = [&]() {
                if (!cached || cache_appended)
                    return true;
                mollm_cuda::launch_append_kv(
                    current_key_data, current_value_data, key_cache_data,
                    value_cache_data, fp16_cache, num_kv_heads,
                    current_length, past_length, key_capacity, key_dim,
                    value_dim,
                    current_key.stride[1] / sizeof(float),
                    current_key.stride[2] / sizeof(float),
                    current_value.stride[1] / sizeof(float),
                    current_value.stride[2] / sizeof(float));
                if (!mollm_cuda::report_cuda(
                        cudaGetLastError(), "append_kv_cuda"))
                    return false;
                cache_appended = true;
                return true;
            };
            if (cached) {
                key_data = key_cache_data;
                value_data = value_cache_data;
                // Decode may fold the one-token append into its score stage.
                // Prefill and generic decode layouts keep the standalone
                // append path below.
                if (query_length != 1 && node.op_type != OpType::QSA_SDPA && !append_cache()) {
                    impl_->failed = true;
                    return;
                }
            }
            if(node.op_type==OpType::QSA_SDPA) {
                const Tensor* indices=inputs.size()==7?inputs[6]:nullptr;
                if(!indices||indices->prec!=Precision::INT32||indices->shape[0]<1||indices->shape[0]>4096||
                   indices->shape[1]!=query_length||indices->shape[2]!=1||indices->shape[3]!=1||
                   indices->stride[0]!=4||indices->stride[1]%4||!cached||!causal||current_length!=query_length||
                   query.shape[2]!=num_heads||query.shape[3]!=1||current_key.shape[2]!=num_kv_heads||current_key.shape[3]!=1||
                   current_value.shape[1]!=query_length||current_value.shape[2]!=num_kv_heads||current_value.shape[3]!=1||
                   output->shape[2]!=num_heads||output->shape[3]!=1||query.stride[1]%4||query.stride[2]%4||
                   output->stride[1]%4||output->stride[2]%4||current_key.stride[1]%4||current_key.stride[2]%4||
                   current_value.stride[1]%4||current_value.stride[2]%4||
                   query.stride[0]!=4||current_key.stride[0]!=4||current_value.stride[0]!=4||output->stride[0]!=4||
                   !mask||mask->prec!=Precision::FP32||mask->shape[0]<key_length||mask->shape[1]<query_length||
                   mask->stride[0]%4||mask->stride[1]%4) {
                    std::fprintf(stderr,"CudaBackend: QSA attention invalid operands/layout\n");impl_->failed=true;return;
                }
                mollm_cuda::QsaSelectedAttentionParams p{};
                p.heads=num_heads;p.kv_heads=num_kv_heads;p.queries=query_length;p.keys=key_length;p.past=past_length;
                p.key_dim=key_dim;p.value_dim=value_dim;p.slots=indices->shape[0];p.scale=scale;
                p.query_position_stride=query.stride[1]/4;p.query_head_stride=query.stride[2]/4;
                p.output_position_stride=output->stride[1]/4;p.output_head_stride=output->stride[2]/4;
                p.key_position_stride=key_dim;p.key_head_stride=size_t(key_capacity)*key_dim;
                p.value_position_stride=value_dim;p.value_head_stride=size_t(key_capacity)*value_dim;
                p.index_position_stride=indices->stride[1]/4;p.mask_column_stride=mask->stride[0]/4;p.mask_row_stride=mask->stride[1]/4;
                static const bool grouped=[] {
                    const char* value=std::getenv("MOLLM_CUDA_QSA_GQA_REUSE");return !value||!std::strcmp(value,"1");
                }();
                static const bool split=[] {
                    const char* value=std::getenv("MOLLM_CUDA_QSA_DECODE_SPLIT");return !value||!std::strcmp(value,"1");
                }();
                const int group=num_heads/num_kv_heads;
                p.reuse_gqa=grouped&&query_length>=32&&key_dim==256&&value_dim==256&&group>=2&&group<=16;
                if(split&&query_length==1&&p.slots>=128) {
                    p.partitions=16;
                }
                if(p.reuse_gqa||p.partitions>1) {
                    const size_t bytes=size_t(query_length)*num_heads*p.slots*sizeof(float);
                    if(!impl_->reserve(impl_->qsa_selected_scores,impl_->qsa_selected_scores_bytes,bytes,true)) {
                        impl_->failed=true;return;
                    }
                    p.selected_scores=static_cast<float*>(impl_->qsa_selected_scores);
                }
                if(!append_cache()){impl_->failed=true;return;}
                if(!mollm_cuda::report_cuda(mollm_cuda::launch_qsa_selected_attention(query_data,key_data,value_data,fp16_cache,
                   device_pointer_const<int32_t>(*indices),output_data,p,cudaStreamPerThread,mask_data),"QSA selected attention")) {
                    impl_->failed=true;return;
                }
                ++impl_->qsa_attention_enqueues;
                if(p.reuse_gqa)++impl_->qsa_gqa_enqueues;
                if(p.partitions>1)++impl_->qsa_split_enqueues;
                record_native();return;
            }
            const size_t score_count = static_cast<size_t>(num_heads) *
                query_length * key_length;
            const size_t score_bytes = score_count * sizeof(float);
            // Bounded logical chunks can require multi-GiB score matrices.
            // Power-of-two spare capacity (2.25 -> 4 GiB at 12K KV) can
            // consume the next chunk's admission margin. Retain high-water
            // reuse, but size new prefill allocations exactly; decode keeps
            // geometric growth to avoid per-token allocator synchronization.
            if (!impl_->reserve(impl_->attention_scores,
                                impl_->attention_scores_bytes,
                                score_bytes,logical_prefill_batch_active_)) {
                impl_->failed = true;
                return;
            }
            auto* scores = static_cast<float*>(impl_->attention_scores);
            // Diagnostic schedule equivalence: preserve the decode score,
            // softmax and value reductions for each short causal query.
            // Appending the whole cache above is safe: each query is limited
            // to its own prefix, so later keys cannot enter its reduction.
            static const bool exact_small_attention = [] {
                const char* value = std::getenv("MOLLM_CUDA_SDPA_SMALL_BATCH_EXACT");
                return value && std::strcmp(value, "1") == 0;
            }();
            if (exact_small_attention && cached && causal &&
                query_length > 1 && query_length <= 8 &&
                current_length == query_length) {
                for (int token = 0; token < query_length; ++token) {
                    const float* token_query = query_data +
                        static_cast<size_t>(token) * query.stride[1] / sizeof(float);
                    const float* token_key = current_key_data +
                        static_cast<size_t>(token) * current_key.stride[1] / sizeof(float);
                    const float* token_value = current_value_data +
                        static_cast<size_t>(token) * current_value.stride[1] / sizeof(float);
                    float* token_output = output_data +
                        static_cast<size_t>(token) * output->stride[1] / sizeof(float);
                    const float* token_mask = mask_data ? mask_data +
                        static_cast<size_t>(token) * mask->stride[1] / sizeof(float) : nullptr;
                    const int token_past = past_length + token;
                    const bool fused = fp16_cache &&
                        mollm_cuda::try_launch_sdpa_decode_fp16_cached(
                            token_query, token_key, token_value, key_cache_data,
                            value_cache_data, scores, token_output, token_mask,
                            num_heads, num_kv_heads, token_past + 1, token_past,
                            key_dim, value_dim, key_capacity, causal, scale,
                            query.stride[0] / sizeof(float), query.stride[2] / sizeof(float),
                            current_key.stride[0] / sizeof(float), current_key.stride[2] / sizeof(float),
                            current_value.stride[0] / sizeof(float), current_value.stride[2] / sizeof(float),
                            mask ? mask->stride[0] / sizeof(float) : 0,
                            output->stride[0] / sizeof(float), output->stride[2] / sizeof(float));
                    if (!fused)
                        mollm_cuda::launch_sdpa_decode(
                            token_query, key_data, value_data, scores, token_output,
                            token_mask, num_heads, num_kv_heads, token_past + 1,
                            token_past, key_dim, value_dim, key_capacity, true,
                            fp16_cache, causal, scale,
                            query.stride[0] / sizeof(float), query.stride[2] / sizeof(float),
                            1, key_dim, static_cast<size_t>(key_capacity) * key_dim,
                            1, value_dim, static_cast<size_t>(key_capacity) * value_dim,
                            mask ? mask->stride[0] / sizeof(float) : 0,
                            output->stride[0] / sizeof(float), output->stride[2] / sizeof(float));
                }
                if (!mollm_cuda::report_cuda(cudaGetLastError(), "sdpa_exact_small_batch_cuda"))
                    impl_->failed = true;
                else
                    record_native();
                return;
            }
            if (query_length == 1) {
                const bool fused_append = cached && fp16_cache &&
                    mollm_cuda::try_launch_sdpa_decode_fp16_cached(
                        query_data, current_key_data, current_value_data,
                        key_cache_data, value_cache_data, scores, output_data,
                        mask_data, num_heads, num_kv_heads, key_length,
                        past_length, key_dim, value_dim, key_capacity, causal,
                        scale, query.stride[0] / sizeof(float),
                        query.stride[2] / sizeof(float),
                        current_key.stride[0] / sizeof(float),
                        current_key.stride[2] / sizeof(float),
                        current_value.stride[0] / sizeof(float),
                        current_value.stride[2] / sizeof(float),
                        mask ? mask->stride[0] / sizeof(float) : 0,
                        output->stride[0] / sizeof(float),
                        output->stride[2] / sizeof(float));
                if (!fused_append) {
                    if (!append_cache()) {
                        impl_->failed = true;
                        return;
                    }
                    mollm_cuda::launch_sdpa_decode(
                        query_data, key_data, value_data, scores, output_data,
                        mask_data, num_heads, num_kv_heads, key_length,
                        past_length, key_dim, value_dim, key_capacity, cached,
                        fp16_cache, causal, scale,
                        query.stride[0] / sizeof(float),
                        query.stride[2] / sizeof(float),
                        cached ? 1 : current_key.stride[0] / sizeof(float),
                        cached ? static_cast<size_t>(key_dim)
                               : current_key.stride[1] / sizeof(float),
                        cached ? static_cast<size_t>(key_capacity) * key_dim
                               : current_key.stride[2] / sizeof(float),
                        cached ? 1 : current_value.stride[0] / sizeof(float),
                        cached ? static_cast<size_t>(value_dim)
                               : current_value.stride[1] / sizeof(float),
                        cached
                            ? static_cast<size_t>(key_capacity) * value_dim
                            : current_value.stride[2] / sizeof(float),
                        mask ? mask->stride[0] / sizeof(float) : 0,
                        output->stride[0] / sizeof(float),
                        output->stride[2] / sizeof(float));
                }
                if (!mollm_cuda::report_cuda(
                        cudaGetLastError(), "sdpa_decode_cuda")) {
                    impl_->failed = true;
                    return;
                }
                record_native();
                return;
            }
            static const int validated_prefix_requested=[] {
                const char* value=std::getenv("MOLLM_CUDA_SDPA_PREFILL_VALIDATED_PREFIX");
                return value?int(mollm_cuda::logical_prefill_option(value,false)):-1;
            }();
            const bool validated_prefix=validated_prefix_requested<0?
                logical_prefill_batch_active_:validated_prefix_requested!=0;
            if (validated_prefix && cached && fp16_cache && key_dim==256 &&
                value_dim==256 && mask_data && query_length>1 &&
                query.stride[0]==sizeof(float) && output->stride[0]==sizeof(float)) {
                constexpr int prefix_shared_limit=1024;
                static const int prefix_skip_zero_pv_requested=[] {
                    const char* value=std::getenv("MOLLM_CUDA_SDPA_PREFILL_SKIP_ZERO_PV");
                    return value?int(mollm_cuda::logical_prefill_option(value,false)):-1;
                }();
                const bool prefix_skip_zero_pv=prefix_skip_zero_pv_requested<0?
                    logical_prefill_batch_active_:prefix_skip_zero_pv_requested!=0;
                // PV pair supersedes loop unrolling; retain the unroll branch
                // only for callers that explicitly disable pair.
                constexpr bool prefix_unroll_pv=false;
                static const int prefix_pair_pv_requested=[] {
                    const char* value=std::getenv("MOLLM_CUDA_SDPA_PREFILL_PV_PAIR");
                    return value?int(mollm_cuda::logical_prefill_option(value,false)):-1;
                }();
                const bool prefix_pair_pv=prefix_pair_pv_requested<0?
                    logical_prefill_batch_active_:prefix_pair_pv_requested!=0;
                // Qualified dense-prefill policy: four query heads share one
                // KV head. Measured +4.6% prefill over ungrouped on GB10 with
                // identical greedy output.
                constexpr int prefix_query_group=4;
                cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
                if (!mollm_cuda::report_cuda(cudaStreamIsCapturing(cudaStreamPerThread,&capture),
                        "validated prefix capture check")) { impl_->failed=true;return; }
                // Qualified logical SM121 default. Other devices retain the
                // accepted16K limit; this extends single-head PV, not the
                // two-head/mask-elision scope.
                const int pv_max_keys=impl_->supports_sdpa_head_lane_pair?32768:16384;
                const bool mid_pv=logical_prefill_batch_active_&&
                    impl_->supports_sdpa_head_lane_pair&&capture==cudaStreamCaptureStatusNone&&
                    prefix_skip_zero_pv&&prefix_pair_pv&&prefix_shared_limit==1024&&
                    num_heads==24&&num_kv_heads==2&&query_length>=32&&query_length<=2048&&
                    key_length>2048&&key_length<=4096&&past_length>=2048&&key_capacity<=32768&&
                    output->stride[1]/sizeof(float)==256&&
                    output->stride[2]/sizeof(float)==size_t(query_length)*256;
                const bool grouped_queries=!mid_pv&&prefix_query_group&&prefix_skip_zero_pv&&prefix_shared_limit==1024&&
                    query_length>=16&&query_length<=mollm_cuda::sdpa_prefill_group_max_queries&&key_length<=4096;
                // Qualified dense-prefill policy: four query heads share one
                // KV head. Measured +1.5% prefill over ungrouped on GB10 with
                // identical greedy output.
                constexpr bool prefix_head_group=true;
                // Experimental long/global-score-only reuse axis. Keep short
                // windows and unsupported GQA ratios on their existing path.
                const bool grouped_heads=prefix_head_group&&grouped_queries&&prefix_query_group==4&&
                    key_length>1024&&num_kv_heads>0&&num_heads%num_kv_heads==0&&
                    (num_heads/num_kv_heads)%4==0;
                // Opt-in dimension pairing supersedes loop unrolling. Both
                // remain restricted to validated D256 cached prefill with
                // zero-PV guards; decode and default behavior are unchanged.
                // Shared-score operator gains did not reproduce reliably in
                // same-chunk whole-model tests. Retain the global-only scope.
                const bool use_pv_pair=!grouped_heads && prefix_pair_pv && prefix_skip_zero_pv && key_length>prefix_shared_limit;
                const bool use_pv_unroll=!grouped_queries && !use_pv_pair && prefix_unroll_pv && prefix_skip_zero_pv && key_length>prefix_shared_limit;
                if (impl_->attention_prefix_valid || capture==cudaStreamCaptureStatusNone) {
                    if (!impl_->reserve(impl_->attention_prefix_valid,impl_->attention_prefix_valid_bytes,
                            sizeof(uint32_t))) { impl_->failed=true;return; }
                    mollm_cuda::SdpaGemmPrefillParams params{num_heads,num_kv_heads,query_length,
                        key_length,past_length,key_dim,key_capacity,causal,scale,
                        query.stride[1]/sizeof(float),query.stride[2]/sizeof(float),
                        mask->stride[0]/sizeof(float),mask->stride[1]/sizeof(float),
                        output->stride[1]/sizeof(float),output->stride[2]/sizeof(float)};
                    cudaError_t error=cudaSuccess;
                    // Qualified bounded logical-prefill policy. Reuse the
                    // native score allocation and never cache expanded V across
                    // appends. Short/decode/captured/unsupported layouts retain
                    // their existing paths without allocating candidate scratch.
                    const bool pv_gemm=logical_prefill_batch_active_&&
                        capture==cudaStreamCaptureStatusNone&&prefix_skip_zero_pv&&
                        prefix_shared_limit==1024&&use_pv_pair&&!grouped_queries&&!grouped_heads&&
                        num_heads==24&&num_kv_heads==2&&query_length>=32&&query_length<=2048&&
                        (mid_pv||(key_length>4096&&key_length<=pv_max_keys))&&key_capacity<=32768&&
                        params.output_position_stride==256&&
                        params.output_head_stride==size_t(query_length)*256;
                    // Qualified SM121 long two-head layout. Automatic
                    // admission stays16K pending model/checker qualification
                    // of the combined head/mask policy.
                    constexpr int head_lane_max_keys=16384;
                    const bool head_lane_pair=pv_gemm&&!mid_pv&&key_length<=head_lane_max_keys&&impl_->supports_sdpa_head_lane_pair;
                    // Qualified SM121 long two-head and mid-KV single-head
                    // paths share one policy.
                    const bool known_mask=head_lane_pair||mid_pv;
                    if(pv_gemm) {
                        if(!impl_->reserve(impl_->attention_pv_values,impl_->attention_pv_values_bytes,
                                size_t(num_kv_heads)*key_length*256*sizeof(float))||
                           !impl_->reserve(impl_->attention_pv_output,impl_->attention_pv_output_bytes,
                                size_t(num_heads)*query_length*256*sizeof(float))||
                           !impl_->reserve(impl_->attention_pv_pointers,impl_->attention_pv_pointers_bytes,
                                size_t(num_heads)*3*sizeof(float*))) {impl_->failed=true;return;}
                        if(!impl_->attention_pv_handle) {
                            if(cublasCreate(&impl_->attention_pv_handle)!=CUBLAS_STATUS_SUCCESS||
                               cublasSetStream(impl_->attention_pv_handle,cudaStreamPerThread)!=CUBLAS_STATUS_SUCCESS) {
                                impl_->failed=true;return;
                            }
                        }
                    }
                    const bool launched=mollm_cuda::try_launch_sdpa_prefill_validated_prefix(query_data,
                        key_data,value_data,scores,output_data,mask_data,
                        static_cast<uint32_t*>(impl_->attention_prefix_valid),params,cudaStreamPerThread,&error,
                        prefix_shared_limit,prefix_skip_zero_pv,use_pv_unroll,use_pv_pair,grouped_queries,
                        grouped_queries?prefix_query_group:2,false,grouped_heads,pv_gemm,head_lane_pair?2:1,known_mask);
                    if (!mollm_cuda::report_cuda(error,"validated prefix attention")) { impl_->failed=true;return; }
                    if (launched) {
                        if(pv_gemm) {
                            if(head_lane_pair)++impl_->attention_head_lane_pair_enqueues;
                            const auto status=mollm_cuda::launch_sdpa_prefill_fp32_pv(impl_->attention_pv_handle,
                                value_data,scores,static_cast<float*>(impl_->attention_pv_output),
                                static_cast<float*>(impl_->attention_pv_values),
                                static_cast<float**>(impl_->attention_pv_pointers),params,cudaStreamPerThread);
                            if(status!=CUBLAS_STATUS_SUCCESS) {
                                std::fprintf(stderr,"native-probability PV cuBLAS status=%d\n",int(status));
                                impl_->failed=true;return;
                            }
                            mollm_cuda::launch_sdpa_validated_pv_copy(static_cast<float*>(impl_->attention_pv_output),
                                output_data,static_cast<uint32_t*>(impl_->attention_prefix_valid),params,cudaStreamPerThread);
                            if(!mollm_cuda::report_cuda(cudaGetLastError(),"native-probability PV copy")) {
                                impl_->failed=true;return;
                            }
                            ++impl_->attention_pv_gemm_enqueues;
                        }
                        ++impl_->attention_prefix_enqueues;
                        if(grouped_queries)++impl_->attention_prefix_query_group_enqueues;
                        if(grouped_heads)++impl_->attention_prefix_head_group_enqueues;
                        if (key_length>1024 && key_length<=prefix_shared_limit)
                            ++impl_->attention_prefix_long_shared_enqueues;
                        if (prefix_skip_zero_pv) ++impl_->attention_prefix_zero_pv_enqueues;
                        if (use_pv_unroll) ++impl_->attention_prefix_pv_unroll_enqueues;
                        if (use_pv_pair) ++impl_->attention_prefix_pv_pair_enqueues;
                        // Diagnostic synchronization only; never use trace runs
                        // for throughput. Capture skips host readback entirely.
                        static const bool trace=std::getenv("MOLLM_CUDA_SDPA_PREFIX_TRACE_VALIDATION")!=nullptr;
                        if (trace && capture==cudaStreamCaptureStatusNone) {
                            uint32_t valid=0;
                            if (!mollm_cuda::report_cuda(cudaMemcpy(&valid,impl_->attention_prefix_valid,
                                    sizeof(valid),cudaMemcpyDeviceToHost),"prefix validation trace")) {
                                impl_->failed=true;return;
                            }
                            std::fprintf(stderr,"sdpa_prefix_validation node=%u queries=%d keys=%d valid=%d known_mask=%d score_workspace_bytes=%zu pv_gemm=%d\n",
                                node.id,query_length,key_length,int(valid!=0),int(known_mask),impl_->attention_scores_bytes,int(pv_gemm));
                        }
                        record_native();return;
                    }
                }
            }
            if (mollm_cuda::try_launch_sdpa_prefill(
                    query_data, key_data, value_data, scores, output_data,
                    mask_data, num_heads, num_kv_heads, query_length,
                    key_length, past_length, key_dim, value_dim,
                    key_capacity, cached, fp16_cache, causal, scale,
                    query.stride[0] / sizeof(float),
                    query.stride[1] / sizeof(float),
                    query.stride[2] / sizeof(float),
                    mask ? mask->stride[0] / sizeof(float) : 0,
                    mask ? mask->stride[1] / sizeof(float) : 0,
                    output->stride[0] / sizeof(float),
                    output->stride[1] / sizeof(float),
                    output->stride[2] / sizeof(float))) {
                if (!mollm_cuda::report_cuda(
                        cudaGetLastError(), "sdpa_prefill_fp16_cuda")) {
                    impl_->failed = true;
                    return;
                }
                record_native();
                return;
            }
            mollm_cuda::launch_sdpa_scores(
                query_data, key_data, scores, mask_data, num_heads,
                num_kv_heads, query_length, key_length, past_length, key_dim,
                key_capacity, cached, fp16_cache, causal, scale,
                query.stride[0] / sizeof(float),
                query.stride[1] / sizeof(float),
                query.stride[2] / sizeof(float),
                cached ? 1 : current_key.stride[0] / sizeof(float),
                cached ? static_cast<size_t>(key_dim)
                       : current_key.stride[1] / sizeof(float),
                cached ? static_cast<size_t>(key_capacity) * key_dim
                       : current_key.stride[2] / sizeof(float),
                mask ? mask->stride[0] / sizeof(float) : 0,
                mask ? mask->stride[1] / sizeof(float) : 0);
            if (!mollm_cuda::report_cuda(
                    cudaGetLastError(), "sdpa_scores_cuda")) {
                impl_->failed = true;
                return;
            }
            mollm_cuda::launch_sdpa_output(
                scores, value_data, output_data, num_heads, num_kv_heads,
                query_length, key_length, value_dim, key_capacity, cached,
                fp16_cache,
                cached ? 1 : current_value.stride[0] / sizeof(float),
                cached ? static_cast<size_t>(value_dim)
                       : current_value.stride[1] / sizeof(float),
                cached ? static_cast<size_t>(key_capacity) * value_dim
                       : current_value.stride[2] / sizeof(float),
                output->stride[0] / sizeof(float),
                output->stride[1] / sizeof(float),
                output->stride[2] / sizeof(float));
            if (!mollm_cuda::report_cuda(
                    cudaGetLastError(), "sdpa_output_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    const auto* matmul_weight =
        inputs.size() >= 2 && inputs[1] ? impl_->find_weight(*inputs[1])
                                        : nullptr;
    if (node.op_type == OpType::MATMUL && inputs.size() >= 2 && inputs[0] &&
        inputs[1] && output && inputs[0]->prec == Precision::FP32 &&
        output->prec == Precision::FP32 &&
        matmul_weight &&
        matmul_weight->n == inputs[1]->shape[0] &&
        matmul_weight->k == inputs[0]->shape[0] &&
        inputs[1]->device.offset == 0) {
        const Tensor& a = *inputs[0];
        const Tensor& weight = *inputs[1];
        const int m = static_cast<int>(a.shape[1]);
        const int k = static_cast<int>(a.shape[0]);
        const int n = static_cast<int>(weight.shape[0]);
        const int lda = static_cast<int>(a.stride[1] / sizeof(float));
        const int ldc = static_cast<int>(output->stride[1] / sizeof(float));
        const Activation activation_kind = static_cast<Activation>(
            graph_params::get_i32(node.params, 0, 0));
        const float* device_a = device_pointer_const<float>(a);
        float* device_c = device_pointer<float>(*output);
        Impl::HcProjectionPair* prefill_pair = nullptr;
        __half* shared_half = nullptr;
        bool shared_ready = false, shared_first = false;
        if (impl_->hc_prefill_reuse_enabled && impl_->hc_execution_epoch) {
            auto pair_entry=impl_->hc_projection_nodes.find(&node);
            if (pair_entry!=impl_->hc_projection_nodes.end()) {
                prefill_pair=pair_entry->second;
                shared_first=prefill_pair->first==&node;
                const bool eligible=m>=16 && k==10240 && lda==k && ldc==n &&
                    fp32_contiguous(a) && fp32_contiguous(*output) &&
                    a.shape[2]==1 && a.shape[3]==1 && output->shape[1]==m &&
                    output->shape[2]==1 && output->shape[3]==1 &&
                    prefill_pair->input==&a && device_a && device_c &&
                    n==(shared_first?320:4) &&
                    (shared_first || prefill_pair->second_weight==&weight) &&
                    graph_params::get_i32(node.params,0,0)==0 &&
                    graph_params::get_i32(node.params,1,0)==0 &&
                    graph_params::get_i32(node.params,2,-1)==-1 &&
                    (matmul_weight->layout==Impl::WeightLayout::Q4Bg32Biased ||
                     matmul_weight->layout==Impl::WeightLayout::Q4Bg32Compact ||
                     matmul_weight->layout==Impl::WeightLayout::Q4Bg32CompactInteger);
                const size_t bytes=static_cast<size_t>(m)*k*sizeof(__half);
                const bool fresh=impl_->hc_prefill_owner==prefill_pair &&
                    impl_->hc_prefill_epoch==impl_->hc_execution_epoch &&
                    impl_->hc_prefill_input==device_a && impl_->hc_prefill_rows==m &&
                    bytes<=impl_->hc_prefill_half_bytes;
                if (eligible && shared_first) {
                    shared_ready=fresh && impl_->hc_prefill_from_rms;
                    impl_->hc_prefill_owner=nullptr;
                    if (impl_->reserve_hc_prefill_half(bytes)) {
                        shared_half=static_cast<__half*>(impl_->hc_prefill_half);
                    }
                    if (impl_->failed) return;
                } else if (eligible && !shared_first && fresh && !impl_->hc_prefill_from_rms) {
                    shared_half=static_cast<__half*>(impl_->hc_prefill_half);
                    shared_ready=true;
                } else if (shared_first) {
                    impl_->hc_prefill_owner=nullptr;
                }
            }
        }
        const bool ok = device_a && device_c
            ? impl_->run_matmul_device(
                  device_a, lda, weight, device_c, ldc, m, n, k,
                  activation_kind,
                  graph_params::get_i32(node.params, 1, 0),
                  graph_params::get_i32(node.params, 2, -1), nullptr, shared_half, shared_ready)
            : impl_->run_matmul(
                  a.ptr<float>(), lda, weight, output->ptr<float>(), ldc,
                  m, n, k, activation_kind,
                  graph_params::get_i32(node.params, 1, 0),
                  graph_params::get_i32(node.params, 2, -1));
        if (ok) {
            if(impl_->hc_deferred.owner&&impl_->hc_deferred.owner->second==&node)
                impl_->hc_deferred={}; // All three graph consumers completed.
            if (shared_half && shared_first) {
                impl_->hc_prefill_owner=prefill_pair;
                impl_->hc_prefill_epoch=impl_->hc_execution_epoch;
                impl_->hc_prefill_input=device_a;
                impl_->hc_prefill_rows=m;
                impl_->hc_prefill_from_rms=false;
            } else if (shared_ready) {
                ++impl_->hc_prefill_reuses;
                impl_->hc_prefill_owner=nullptr;
            } else if (prefill_pair && impl_->hc_prefill_owner==prefill_pair) {
                impl_->hc_prefill_owner=nullptr;
            }
            record_native();
            return;
        }
        impl_->failed = true;
        return;
    }

    if (node.op_type == OpType::QK_RMS_NORM_ROPE && inputs.size() >= 6 &&
        inputs[0] && inputs[1] && inputs[2] && inputs[3] && inputs[4] &&
        inputs[5] && output && inputs[0]->prec == Precision::FP32 &&
        inputs[1]->prec == Precision::FP32 &&
        fp32_contiguous(*inputs[2]) && fp32_contiguous(*inputs[3]) &&
        inputs[4]->prec == Precision::FP32 &&
        inputs[5]->prec == Precision::FP32 &&
        output->prec == Precision::FP32) {
        const Tensor& query = *inputs[0];
        const Tensor& key = *inputs[1];
        const Tensor& query_weight = *inputs[2];
        const Tensor& key_weight = *inputs[3];
        const Tensor& cosine = *inputs[4];
        const Tensor& sine = *inputs[5];
        const int width = static_cast<int>(output->shape[0]);
        const int sequence_length = static_cast<int>(output->shape[1]);
        const int total_heads = static_cast<int>(output->shape[2]);
        const int query_heads = graph_params::get_i32(
            node.params, 2, total_heads);
        const int key_heads = total_heads - query_heads;
        const int rope_dim = graph_params::get_i32(node.params, 0, width);
        const bool interleave =
            graph_params::get_i32(node.params, 1, 1) != 0;
        const auto element_aligned = [](const Tensor& tensor) {
            for (int dimension = 0; dimension < 4; ++dimension)
                if (tensor.stride[dimension] % sizeof(float) != 0)
                    return false;
            return true;
        };
        const float* query_data = device_pointer_const<float>(query);
        const float* key_data = device_pointer_const<float>(key);
        const float* query_weight_data =
            device_pointer_const<float>(query_weight);
        const float* key_weight_data =
            device_pointer_const<float>(key_weight);
        const float* cosine_data = device_pointer_const<float>(cosine);
        const float* sine_data = device_pointer_const<float>(sine);
        float* destination = device_pointer<float>(*output);
        if (query_data && key_data && query_weight_data && key_weight_data &&
            cosine_data && sine_data && destination && width > 0 &&
            sequence_length > 0 && query_heads > 0 && key_heads > 0 &&
            rope_dim > 0 && rope_dim <= width && rope_dim % 2 == 0 &&
            query.shape[0] == width && key.shape[0] == width &&
            query.shape[1] >= sequence_length * query_heads &&
            key.shape[1] >= sequence_length * key_heads &&
            query.shape[2] == 1 && query.shape[3] == 1 &&
            key.shape[2] == 1 && key.shape[3] == 1 &&
            query_weight.shape[0] >= width && key_weight.shape[0] >= width &&
            cosine.shape[0] >= rope_dim / 2 &&
            sine.shape[0] >= rope_dim / 2 &&
            cosine.shape[1] >= sequence_length &&
            sine.shape[1] >= sequence_length && output->shape[3] == 1 &&
            element_aligned(query) && element_aligned(key) &&
            element_aligned(cosine) && element_aligned(sine) &&
            element_aligned(*output)) {
            mollm_cuda::launch_qk_rms_norm_rope(
                query_data, key_data, query_weight_data, key_weight_data,
                cosine_data, sine_data, destination, width, sequence_length,
                query_heads, total_heads, rope_dim, interleave,
                query.stride[0] / sizeof(float),
                query.stride[1] / sizeof(float),
                key.stride[0] / sizeof(float),
                key.stride[1] / sizeof(float),
                cosine.stride[0] / sizeof(float),
                cosine.stride[1] / sizeof(float),
                sine.stride[0] / sizeof(float),
                sine.stride[1] / sizeof(float),
                output->stride[0] / sizeof(float),
                output->stride[1] / sizeof(float),
                output->stride[2] / sizeof(float),
                graph_params::get_f32(node.params, 0, 1e-6f));
            if (!mollm_cuda::report_cuda(
                    cudaGetLastError(), "qk_rms_norm_rope_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::RMS_NORM_ROPE && inputs.size() >= 4 &&
        inputs[0] && inputs[1] && inputs[2] && inputs[3] && output &&
        fp32_contiguous(*inputs[0]) && fp32_contiguous(*inputs[1]) &&
        inputs[2]->prec == Precision::FP32 &&
        inputs[3]->prec == Precision::FP32 &&
        output->prec == Precision::FP32) {
        const Tensor& source = *inputs[0];
        const Tensor& weight = *inputs[1];
        const Tensor& cosine = *inputs[2];
        const Tensor& sine = *inputs[3];
        const float* source_data = device_pointer_const<float>(source);
        const float* weight_data = device_pointer_const<float>(weight);
        const float* cosine_data = device_pointer_const<float>(cosine);
        const float* sine_data = device_pointer_const<float>(sine);
        float* destination = device_pointer<float>(*output);
        const int width = static_cast<int>(source.shape[0]);
        const int sequence_length = static_cast<int>(output->shape[1]);
        const int channels =
            static_cast<int>(output->shape[2] * output->shape[3]);
        const int rows = width > 0
            ? static_cast<int>(source.nelements() / width) : 0;
        const int rope_dim = graph_params::get_i32(node.params, 0, width);
        const bool interleave =
            graph_params::get_i32(node.params, 1, 1) != 0;
        const size_t count = static_cast<size_t>(source.nelements());
        const size_t scratch_bytes = count * sizeof(float);
        if (source_data && weight_data && cosine_data && sine_data &&
            destination && width > 0 && rows > 0 && rope_dim > 0 &&
            rope_dim <= width && rope_dim % 2 == 0 &&
            rows == sequence_length * channels &&
            cosine.shape[0] >= rope_dim / 2 &&
            sine.shape[0] >= rope_dim / 2 &&
            impl_->reserve(impl_->norm_scratch,
                           impl_->norm_scratch_bytes, scratch_bytes)) {
            auto* normalized = static_cast<float*>(impl_->norm_scratch);
            mollm_cuda::launch_rms_norm(
                source_data, weight_data, normalized, width, rows,
                graph_params::get_f32(node.params, 0, 1e-6f));
            mollm_cuda::launch_rope(
                normalized, cosine_data, sine_data, destination, width,
                sequence_length, channels, static_cast<int>(output->shape[2]),
                rope_dim, interleave, 1, static_cast<size_t>(width),
                static_cast<size_t>(width) * sequence_length,
                static_cast<size_t>(width) * sequence_length *
                    output->shape[2],
                cosine.stride[0] / sizeof(float),
                cosine.stride[1] / sizeof(float),
                sine.stride[0] / sizeof(float),
                sine.stride[1] / sizeof(float),
                output->stride[0] / sizeof(float),
                output->stride[1] / sizeof(float),
                output->stride[2] / sizeof(float),
                output->stride[3] / sizeof(float));
            if (!mollm_cuda::report_cuda(
                    cudaGetLastError(), "rms_norm_rope_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::SHORTCONV && inputs.size() >= 3 &&
        inputs[0] && inputs[1] && inputs[2] && output &&
        inputs[0]->prec == Precision::FP32 &&
        inputs[1]->prec == Precision::FP32 &&
        inputs[2]->prec == Precision::FP32 &&
        output->prec == Precision::FP32) {
        const Tensor& input = *inputs[0];
        const Tensor& weight = *inputs[1];
        const Tensor& state = *inputs[2];
        const int groups = static_cast<int>(input.shape[0]);
        const int sequence_length = static_cast<int>(input.shape[1]);
        const int kernel_size = graph_params::get_i32(node.params, 0, 4);
        const int real_tokens = graph_params::get_i32(
            node.params, 1, sequence_length);
        const float* input_data = device_pointer_const<float>(input);
        const float* weight_data = device_pointer_const<float>(weight);
        float* state_data = device_pointer<float>(state);
        float* destination = device_pointer<float>(*output);
        if (input_data && weight_data && state_data && destination &&
            groups > 0 && sequence_length > 0 && kernel_size > 0 &&
            input.shape[2] == 1 && input.shape[3] == 1 &&
            output->shape[0] == input.shape[0] &&
            output->shape[1] == input.shape[1] &&
            output->shape[2] == 1 && output->shape[3] == 1 &&
            weight.nelements() >=
                static_cast<int64_t>(groups) * kernel_size &&
            state.nelements() >=
                static_cast<int64_t>(groups) * (kernel_size - 1) &&
            output->nelements() >=
                static_cast<int64_t>(groups) * sequence_length &&
            input.stride[0] % sizeof(float) == 0 &&
            input.stride[1] % sizeof(float) == 0 &&
            [&]() {
                float* prefixes=impl_->state_prefix_tokens>0 && real_tokens==sequence_length ? impl_->state_prefix_output(
                    state_data,size_t(groups)*(kernel_size-1)*sizeof(float),sequence_length) : nullptr;
                if(prefixes)return mollm_cuda::launch_shortconv_prefixes(
                    input_data,weight_data,state_data,destination,prefixes,groups,
                    sequence_length,kernel_size,real_tokens,input.stride[0]/sizeof(float),
                    input.stride[1]/sizeof(float));
                // Qualified coalesced store mapping for unit feature stride.
                constexpr bool transpose_output=true;
                const int mapping = transpose_output && input.stride[0] == sizeof(float) &&
                    sequence_length >= 8 && sequence_length <= 65535 * 8 ? 2 : 0;
                const bool launched = mollm_cuda::launch_shortconv(input_data,weight_data,state_data,destination,
                    groups,sequence_length,kernel_size,real_tokens,input.stride[0]/sizeof(float),
                    input.stride[1]/sizeof(float), mapping);
                if (launched && mapping == 2) ++impl_->shortconv_transpose_dispatches;
                return launched;
            }()) {
            if (!mollm_cuda::report_cuda(
                    cudaGetLastError(), "shortconv_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    const bool gdn_core =
        node.op_type == OpType::GATED_DELTANET_PREFILL ||
        node.op_type == OpType::GATED_DELTANET_DECODE;
    const bool gdn_conv_decode =
        node.op_type == OpType::GATED_DELTANET_CONV_DECODE;
    if ((gdn_core || gdn_conv_decode) &&
        inputs.size() >= (gdn_conv_decode ? 10u : 8u) && output &&
        std::all_of(
            inputs.begin(), inputs.begin() + 8,
            [](const Tensor* tensor) {
                return tensor && tensor->prec == Precision::FP32;
            }) &&
        (!gdn_conv_decode ||
         (inputs[8] && inputs[8]->prec == Precision::FP32 &&
          inputs[9] && inputs[9]->prec == Precision::FP32)) &&
        output->prec == Precision::FP32) {
        const int num_heads = graph_params::get_i32(node.params, 0, 16);
        const int key_dimension = graph_params::get_i32(node.params, 1, 128);
        const int value_dimension =
            graph_params::get_i32(node.params, 2, 128);
        const int sequence_length = graph_params::get_i32(node.params, 3, 1);
        const int real_tokens =
            graph_params::get_i32(node.params, 6, sequence_length);
        const int num_value_heads =
            graph_params::get_i32(node.params, 7, num_heads);
        const int qkv_key_width = num_heads * key_dimension;
        const int qkv_width =
            2 * qkv_key_width + num_value_heads * value_dimension;
        const int output_width = num_value_heads * value_dimension;
        const float* qkv = device_pointer_const<float>(*inputs[0]);
        const float* a = device_pointer_const<float>(*inputs[1]);
        const float* b = device_pointer_const<float>(*inputs[2]);
        const float* z = device_pointer_const<float>(*inputs[3]);
        const float* a_log = device_pointer_const<float>(*inputs[4]);
        const float* dt_bias = device_pointer_const<float>(*inputs[5]);
        const float* norm_weight = device_pointer_const<float>(*inputs[6]);
        float* state = device_pointer<float>(*inputs[7]);
        float* destination = device_pointer<float>(*output);
        float scale = graph_params::get_f32(node.params, 2, 0.0f);
        if (scale == 0.0f && key_dimension > 0)
            scale = 1.0f / std::sqrt(static_cast<float>(key_dimension));
        const bool valid = qkv && a && b && z && a_log && dt_bias &&
            norm_weight && state && destination && num_heads > 0 &&
            num_value_heads > 0 && num_value_heads % num_heads == 0 &&
            key_dimension == 128 && value_dimension == 128 &&
            sequence_length > 0 && inputs[0]->nelements() >=
                static_cast<int64_t>(qkv_width) * sequence_length &&
            inputs[1]->shape[0] >= num_value_heads &&
            inputs[2]->shape[0] >= num_value_heads &&
            inputs[3]->shape[0] >= output_width &&
            inputs[4]->nelements() >= num_value_heads &&
            inputs[5]->nelements() >= num_value_heads &&
            inputs[6]->nelements() >= value_dimension &&
            inputs[7]->nelements() >=
                static_cast<int64_t>(num_value_heads) * key_dimension *
                    value_dimension &&
            output->nelements() >=
                static_cast<int64_t>(output_width) * sequence_length &&
            inputs[1]->stride[0] == sizeof(float) &&
            inputs[2]->stride[0] == sizeof(float) &&
            inputs[3]->stride[0] == sizeof(float) &&
            output->is_contiguous();
        bool launched = false;
        if (valid && gdn_conv_decode) {
            const int convolution_kernel =
                graph_params::get_i32(node.params, 5, 4);
            const float* convolution_weight =
                device_pointer_const<float>(*inputs[8]);
            float* convolution_state = device_pointer<float>(*inputs[9]);
            const size_t scratch_bytes =
                static_cast<size_t>(qkv_width) * sizeof(float);
            const bool valid_convolution = sequence_length == 1 &&
                convolution_weight && convolution_state &&
                convolution_kernel > 0 && inputs[8]->nelements() >=
                    static_cast<int64_t>(qkv_width) * convolution_kernel &&
                inputs[9]->nelements() >=
                    static_cast<int64_t>(qkv_width) *
                        (convolution_kernel - 1) &&
                inputs[0]->stride[0] % sizeof(float) == 0 &&
                inputs[0]->stride[1] % sizeof(float) == 0;
            if (valid_convolution &&
                impl_->reserve(
                    impl_->recurrent_scratch,
                    impl_->recurrent_scratch_bytes, scratch_bytes)) {
                auto* convolved =
                    static_cast<float*>(impl_->recurrent_scratch);
                const bool convolution_launched =
                    mollm_cuda::launch_shortconv(
                        qkv, convolution_weight, convolution_state,
                        convolved, qkv_width, 1, convolution_kernel, 1,
                        inputs[0]->stride[0] / sizeof(float),
                        inputs[0]->stride[1] / sizeof(float));
                launched = convolution_launched && mollm_cuda::launch_gdn(
                    convolved, a, b, z, a_log, dt_bias, norm_weight, state,
                    destination, num_heads, num_value_heads, key_dimension,
                    value_dimension, 1, 1,
                    graph_params::get_i32(node.params, 4, 1),
                    graph_params::get_f32(node.params, 0, 1e-6f),
                    graph_params::get_f32(node.params, 1, 1e-6f), scale,
                    inputs[1]->stride[1] / sizeof(float),
                    inputs[2]->stride[1] / sizeof(float),
                    inputs[3]->stride[1] / sizeof(float));
                if (convolution_launched && !launched) {
                    impl_->failed = true;
                    return;
                }
            }
        } else if (valid) {
            const bool prefill =
                node.op_type == OpType::GATED_DELTANET_PREFILL;
            const size_t prefill_scratch_bytes = prefill
                ? mollm_cuda::gdn_prefill_scratch_bytes(
                      num_heads, num_value_heads, key_dimension,
                      value_dimension, sequence_length)
                : 0;
            void* prefill_scratch = nullptr;
            if (prefill_scratch_bytes > 0) {
                if (!impl_->reserve(
                        impl_->recurrent_scratch,
                        impl_->recurrent_scratch_bytes,
                        prefill_scratch_bytes)) {
                    impl_->failed = true;
                    return;
                }
                prefill_scratch = impl_->recurrent_scratch;
            }
            float* prefixes=impl_->state_prefix_tokens>0 && real_tokens==sequence_length && key_dimension==128 && value_dimension==128
                ? impl_->state_prefix_output(state,size_t(num_value_heads)*128*128*sizeof(float),sequence_length) : nullptr;
            if(prefixes)launched=mollm_cuda::launch_gdn_prefixes(
                qkv,a,b,z,a_log,dt_bias,norm_weight,state,destination,prefixes,
                num_heads,num_value_heads,sequence_length,real_tokens,
                graph_params::get_i32(node.params,4,1),graph_params::get_f32(node.params,0,1e-6f),
                graph_params::get_f32(node.params,1,1e-6f),scale,inputs[1]->stride[1]/sizeof(float),
                inputs[2]->stride[1]/sizeof(float),inputs[3]->stride[1]/sizeof(float));
            else launched = mollm_cuda::launch_gdn(
                qkv, a, b, z, a_log, dt_bias, norm_weight, state,
                destination, num_heads, num_value_heads, key_dimension,
                value_dimension, sequence_length, real_tokens,
                graph_params::get_i32(node.params, 4, 1),
                graph_params::get_f32(node.params, 0, 1e-6f),
                graph_params::get_f32(node.params, 1, 1e-6f), scale,
                inputs[1]->stride[1] / sizeof(float),
                inputs[2]->stride[1] / sizeof(float),
                inputs[3]->stride[1] / sizeof(float), prefill_scratch,
                prefill_scratch_bytes);
        }
        if (launched) {
            if (!mollm_cuda::report_cuda(cudaGetLastError(), "gdn_128_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::ADD_RMS_NORM && inputs.size() >= 3 &&
        inputs[0] && inputs[1] && inputs[2] && output &&
        inputs[0]->prec == Precision::FP32 &&
        inputs[1]->prec == Precision::FP32 &&
        fp32_contiguous(*inputs[2]) && output->prec == Precision::FP32 &&
        inputs[0]->stride[0] == sizeof(float) &&
        inputs[1]->stride[0] == sizeof(float) &&
        output->stride[0] == sizeof(float) &&
        inputs[0]->shape[2] == 1 && inputs[0]->shape[3] == 1 &&
        inputs[1]->shape[2] == 1 && inputs[1]->shape[3] == 1 &&
        output->shape[2] == 1 && output->shape[3] == 1 &&
        inputs[0]->nelements() == inputs[1]->nelements() &&
        inputs[0]->nelements() == output->nelements()) {
        float* residual = device_pointer<float>(*inputs[0]);
        const float* update = device_pointer_const<float>(*inputs[1]);
        const float* weight = device_pointer_const<float>(*inputs[2]);
        float* destination = device_pointer<float>(*output);
        const int width = static_cast<int>(inputs[0]->shape[0]);
        const int rows = static_cast<int>(inputs[0]->shape[1]);
        if (residual && update && weight && destination && width > 0 &&
            rows > 0) {
            mollm_cuda::launch_add_rms_norm(
                residual, update, weight, destination, width, rows,
                inputs[0]->stride[1] / sizeof(float),
                inputs[1]->stride[1] / sizeof(float),
                output->stride[1] / sizeof(float),
                graph_params::get_f32(node.params, 0, 1e-6f));
            if (!mollm_cuda::report_cuda(
                    cudaGetLastError(), "add_rms_norm_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    // Qwen3.8 gated residual streams. Keep branch boundaries BF16-rounded,
    // matching the reference kernels even though graph storage is FP32.
    if (node.op_type == OpType::GROUP_RMS_NORM ||
        node.op_type == OpType::GR_REDUCE || node.op_type == OpType::GR_INJECT) {
        const bool inject = node.op_type == OpType::GR_INJECT;
        const size_t arity = inject ? 3 : 2;
        const auto dense_2d = [](const Tensor* tensor) {
            return tensor && fp32_contiguous(*tensor) &&
                tensor->shape[0] > 0 && tensor->shape[1] > 0 &&
                tensor->shape[2] == 1 && tensor->shape[3] == 1 &&
                device_pointer_const<float>(*tensor);
        };
        if (inputs.size() >= arity && dense_2d(output) &&
            std::all_of(inputs.begin(), inputs.begin() + arity, dense_2d)) {
            const Tensor& x = *inputs[0];
            const Tensor& second = *inputs[1];
            const int hidden = graph_params::get_i32(node.params, 0, 0);
            const int streams = graph_params::get_i32(node.params, 1, 4);
            const int rows = static_cast<int>(x.shape[1]);
            const int64_t wide = static_cast<int64_t>(hidden) * streams;
            const float* source = device_pointer_const<float>(x);
            const float* other = device_pointer_const<float>(second);
            float* destination = device_pointer<float>(*output);
            bool launched = false;
            if (node.op_type == OpType::GROUP_RMS_NORM && hidden > 0 &&
                x.shape[0] % hidden == 0 && second.shape[0] == x.shape[0] &&
                second.shape[1] == 1 && same_shape(x, *output)) {
                Impl::HcProjectionPair* rms_pair=nullptr;
                uint16_t* half=nullptr;
                if (impl_->hc_prefill_reuse_enabled && impl_->hc_prefill_rms_half_enabled &&
                    impl_->hc_execution_epoch) {
                    auto entry=impl_->hc_prefill_rms_nodes.find(&node);
                    if (entry!=impl_->hc_prefill_rms_nodes.end()) {
                        impl_->hc_prefill_owner=nullptr;
                        rms_pair=entry->second;
                        if (rows>=16 && x.shape[0]==10240 && rms_pair->input==output &&
                            impl_->reserve_hc_prefill_half(size_t(rows)*10240*sizeof(__half)))
                            half=static_cast<uint16_t*>(impl_->hc_prefill_half);
                        if (impl_->failed) return;
                    }
                }
                const float epsilon=graph_params::get_f32(node.params,0,1e-6f);
                bool half_launched=false;
                if(half) half_launched=mollm_cuda::launch_group_rms_norm_half(
                    source,other,destination,half,hidden,x.shape[0]/hidden,rows,epsilon);
                if (half_launched) {
                    impl_->hc_prefill_owner=rms_pair;
                    impl_->hc_prefill_epoch=impl_->hc_execution_epoch;
                    impl_->hc_prefill_input=destination;
                    impl_->hc_prefill_rows=rows;
                    impl_->hc_prefill_from_rms=true;
                    ++impl_->hc_prefill_rms_half_calls;
                } else {
                    mollm_cuda::launch_group_rms_norm(source,other,destination,hidden,
                        x.shape[0]/hidden,rows,epsilon);
                }
                launched = true;
            } else if (node.op_type == OpType::GR_REDUCE && hidden > 0 &&
                       streams > 0 && x.shape[0] == wide &&
                       same_shape(x, second) && output->shape[0] == hidden &&
                       output->shape[1] == rows) {
                mollm_cuda::launch_gr_reduce(
                    source, other, destination, hidden, streams, rows);
                launched = true;
            } else if (inject && hidden > 0 && streams > 0 &&
                       x.shape[0] == hidden && second.shape[0] == wide &&
                       second.shape[1] == rows && same_shape(second, *output) &&
                       inputs[2]->shape[0] == streams &&
                       inputs[2]->shape[1] == rows) {
                mollm_cuda::launch_gr_inject(
                    source, other, device_pointer_const<float>(*inputs[2]),
                    destination, hidden, streams, rows);
                launched = true;
            }
            if (launched) {
                if (!mollm_cuda::report_cuda(cudaGetLastError(),
                                            "gated_residual_cuda")) {
                    impl_->failed = true;
                    return;
                }
                record_native();
                return;
            }
        }
    }

    if (node.op_type == OpType::RMS_NORM && inputs.size() >= 2 &&
        inputs[0] && inputs[1] && output && fp32_contiguous(*inputs[0]) &&
        fp32_contiguous(*inputs[1]) && fp32_contiguous(*output)) {
        const float* source = device_pointer_const<float>(*inputs[0]);
        const float* weight = device_pointer_const<float>(*inputs[1]);
        float* destination = device_pointer<float>(*output);
        const int width = static_cast<int>(inputs[0]->shape[0]);
        const int rows = width > 0
            ? static_cast<int>(inputs[0]->nelements() / width) : 0;
        if (source && weight && destination && rows > 0) {
            mollm_cuda::launch_rms_norm(
                source, weight, destination, width, rows,
                graph_params::get_f32(node.params, 0, 1e-6f));
            if (!mollm_cuda::report_cuda(cudaGetLastError(), "rms_norm_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if ((node.op_type == OpType::ADD || node.op_type == OpType::MUL) &&
        inputs.size() >= 2 && inputs[0] && inputs[1] && output &&
        fp32_contiguous(*inputs[0]) && fp32_contiguous(*inputs[1]) &&
        fp32_contiguous(*output) &&
        (same_shape(*inputs[0], *output) || inputs[0]->nelements() == 1) &&
        (same_shape(*inputs[1], *output) || inputs[1]->nelements() == 1)) {
        const float* lhs = device_pointer_const<float>(*inputs[0]);
        const float* rhs = device_pointer_const<float>(*inputs[1]);
        float* destination = device_pointer<float>(*output);
        const size_t count = static_cast<size_t>(output->nelements());
        if (lhs && rhs && destination) {
            mollm_cuda::launch_binary(
                lhs, rhs, destination, count,
                node.op_type == OpType::MUL,
                inputs[0]->nelements() == 1,
                inputs[1]->nelements() == 1);
            if (!mollm_cuda::report_cuda(cudaGetLastError(), "binary_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::SIGMOID_MUL && inputs.size() >= 2 &&
        inputs[0] && inputs[1] && output &&
        inputs[0]->prec == Precision::FP32 &&
        inputs[1]->prec == Precision::FP32 && fp32_contiguous(*output) &&
        same_shape(*inputs[0], *inputs[1]) &&
        inputs[0]->nelements() == output->nelements()) {
        const float* value = device_pointer_const<float>(*inputs[0]);
        const float* gate = device_pointer_const<float>(*inputs[1]);
        float* destination = device_pointer<float>(*output);
        const size_t count = static_cast<size_t>(output->nelements());
        if (value && gate && destination) {
            if (inputs[0]->is_contiguous() && inputs[1]->is_contiguous()) {
                mollm_cuda::launch_sigmoid_mul(value, gate, destination, count);
            } else {
                mollm_cuda::launch_sigmoid_mul_strided(
                    value, gate, destination, count, inputs[0]->shape[0],
                    inputs[0]->shape[1], inputs[0]->shape[2],
                    inputs[0]->stride[0] / sizeof(float),
                    inputs[0]->stride[1] / sizeof(float),
                    inputs[0]->stride[2] / sizeof(float),
                    inputs[0]->stride[3] / sizeof(float),
                    inputs[1]->stride[0] / sizeof(float),
                    inputs[1]->stride[1] / sizeof(float),
                    inputs[1]->stride[2] / sizeof(float),
                    inputs[1]->stride[3] / sizeof(float));
            }
            if (!mollm_cuda::report_cuda(
                    cudaGetLastError(), "sigmoid_mul_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    int unary_operation = -1;
    switch (node.op_type) {
    case OpType::SILU: unary_operation = 0; break;
    case OpType::GELU: unary_operation = 1; break;
    case OpType::TANH: unary_operation = 2; break;
    case OpType::SIGMOID:
    case OpType::SIGMOID_EXACT: unary_operation = 3; break;
    case OpType::EXP:
    case OpType::EXP_EXACT: unary_operation = 4; break;
    case OpType::SOFTPLUS: unary_operation = 5; break;
    default: break;
    }
    if (unary_operation >= 0 && !inputs.empty() && inputs[0] && output &&
        inputs[0]->prec == Precision::FP32 && fp32_contiguous(*output) &&
        same_shape(*inputs[0], *output)) {
        const float* source = device_pointer_const<float>(*inputs[0]);
        float* destination = device_pointer<float>(*output);
        const size_t count = static_cast<size_t>(output->nelements());
        if (source && destination) {
            if (inputs[0]->is_contiguous()) {
                mollm_cuda::launch_unary(
                    source, destination, count, unary_operation);
            } else {
                const Tensor& src = *inputs[0];
                mollm_cuda::launch_unary_strided(
                    source, destination, count, unary_operation,
                    src.shape[0], src.shape[1], src.shape[2],
                    src.stride[0] / sizeof(float),
                    src.stride[1] / sizeof(float),
                    src.stride[2] / sizeof(float),
                    src.stride[3] / sizeof(float));
            }
            if (!mollm_cuda::report_cuda(cudaGetLastError(), "unary_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::SWIGLU && !inputs.empty() && inputs[0] &&
        output && fp32_contiguous(*inputs[0]) && fp32_contiguous(*output) &&
        inputs[0]->shape[0] == output->shape[0] * 2) {
        const float* source = device_pointer_const<float>(*inputs[0]);
        float* destination = device_pointer<float>(*output);
        const size_t count = static_cast<size_t>(output->nelements());
        if (source && destination) {
            mollm_cuda::launch_swiglu(
                source, destination, count,
                static_cast<size_t>(output->shape[0]));
            if (!mollm_cuda::report_cuda(cudaGetLastError(), "swiglu_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if ((node.op_type == OpType::PLE_GATE || node.op_type == OpType::PLE_DILATED_CONV) &&
        inputs.size() == 3 && inputs[0] && inputs[1] && inputs[2] && output &&
        fp32_contiguous(*inputs[0]) && fp32_contiguous(*inputs[1]) &&
        fp32_contiguous(*inputs[2]) && fp32_contiguous(*output) &&
        same_shape(*inputs[0], *output) && inputs[0]->shape[2] == 1 && inputs[0]->shape[3] == 1 &&
        inputs[0]->shape[0] > 0 && inputs[0]->shape[1] > 0) {
        const float* a = device_pointer_const<float>(*inputs[0]);
        const float* b = device_pointer_const<float>(*inputs[1]);
        const float* c = device_pointer_const<float>(*inputs[2]);
        float* destination = device_pointer<float>(*output);
        bool launched = false;
        if (a && b && c && destination) {
            if (node.op_type == OpType::PLE_GATE) {
                const int hidden = graph_params::get_i32(node.params, 0, 0);
                const int streams = graph_params::get_i32(node.params, 1, 4);
                if (hidden > 0 && streams > 0 &&
                    static_cast<int64_t>(hidden) * streams == inputs[0]->shape[0] &&
                    same_shape(*inputs[0], *inputs[1]) &&
                    inputs[2]->shape[0] == hidden && inputs[2]->shape[1] == inputs[0]->shape[1] &&
                    inputs[2]->shape[2] == 1 && inputs[2]->shape[3] == 1) {
                    mollm_cuda::launch_ple_gate(a, b, c, destination, hidden, streams,
                        static_cast<int>(inputs[0]->shape[1]));
                    launched = true;
                }
            } else {
                const int channels = static_cast<int>(inputs[0]->shape[0]);
                const int tokens = static_cast<int>(inputs[0]->shape[1]);
                const int kernel_size = graph_params::get_i32(node.params, 0, 4);
                const int dilation = graph_params::get_i32(node.params, 1, 3);
                const int requested = graph_params::get_i32(node.params, 2, tokens);
                const int real = requested >= 0 && requested < tokens ? requested : tokens;
                const int64_t state_len = static_cast<int64_t>(kernel_size - 1) * dilation;
                if (kernel_size > 0 && dilation > 0 && state_len <= INT32_MAX &&
                    inputs[1]->shape[0] == channels && inputs[1]->shape[1] == kernel_size &&
                    inputs[2]->shape[0] == channels && inputs[2]->shape[1] == state_len &&
                    inputs[1]->shape[2] == 1 && inputs[1]->shape[3] == 1 &&
                    inputs[2]->shape[2] == 1 && inputs[2]->shape[3] == 1 &&
                    a != destination && a != c && destination != c) {
                    float* prefixes=impl_->state_prefix_tokens>0 && real==tokens ? impl_->state_prefix_output(
                        const_cast<float*>(c),size_t(channels)*state_len*sizeof(float),tokens) : nullptr;
                    if(prefixes)launched=mollm_cuda::launch_ple_dilated_conv_prefixes(a,b,
                        const_cast<float*>(c),destination,prefixes,channels,tokens,kernel_size,dilation,real);
                    else {
                        mollm_cuda::launch_ple_dilated_conv(a,b,const_cast<float*>(c),
                            destination,channels,tokens,kernel_size,dilation,real);
                        launched=true;
                    }
                }
            }
        }
        if (launched) {
            if (!mollm_cuda::report_cuda(cudaGetLastError(), "PLE CUDA launch")) impl_->failed = true;
            else record_native();
            return;
        }
    }

    if (node.op_type == OpType::MOE &&
        (try_fp16_moe(node, inputs, output) || try_nvfp4_moe(node, inputs, output))) {
        if (!impl_->failed) record_native();
        return;
    }

    if (impl_->segment_mode == Impl::SegmentMode::CAPTURE && impl_->active_segment) {
        // Unsupported op was recorded before dispatch but launched no work.
        impl_->active_segment->records.pop_back();
        --impl_->segment_cursor;
    }
    finish_decode_segment();
    if (node.op_type != OpType::PLE_LOOKUP) impl_->segment_disabled = true;
    impl_->segment_ready = false;
    if (impl_->operator_fallback ==
        OperatorFallbackPolicy::REQUIRE_NATIVE) {
        std::fprintf(
            stderr,
            "CudaBackend: native-only mode rejected %s operator fallback\n",
            op_type_name(node.op_type));
        impl_->failed = true;
        return;
    }

    synchronize_for_host_read();
    if (impl_->failed)
        return;
    std::vector<Tensor> host_tensors;
    std::vector<const Tensor*> host_inputs;
    std::vector<std::vector<uint64_t>> host_storage;
    std::vector<int> staged_storage(inputs.size(), -1);
    host_tensors.reserve(inputs.size());
    host_inputs.reserve(inputs.size());
    host_storage.reserve(inputs.size());
    for (size_t input_index = 0; input_index < inputs.size(); ++input_index) {
        const Tensor* input = inputs[input_index];
        if (!input) {
            host_inputs.push_back(nullptr);
            continue;
        }
        host_tensors.push_back(*input);
        Tensor& host = host_tensors.back();
        // Prepared constants retain their package-native host representation.
        // Device intermediates and persistent state are staged from the first
        // byte of the tensor view, preserving their original shape/strides.
        const bool device_weight = impl_->find_weight(*input) != nullptr;
        if (input->device.buffer && !device_weight) {
            const size_t bytes = input->view_span_bytes();
            host_storage.emplace_back(
                (bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t));
            staged_storage[input_index] =
                static_cast<int>(host_storage.size() - 1);
            void* destination = host_storage.back().data();
            if (!copy_to_host(*input, destination, bytes)) {
                impl_->failed = true;
                return;
            }
            host.data = destination;
        } else if (device_weight && !host.data) {
            std::fprintf(
                stderr, "CudaBackend: %s fallback lacks host weight storage\n",
                op_type_name(node.op_type));
            impl_->failed = true;
            return;
        }
        host.device.buffer = nullptr;
        host.device.offset = 0;
        host_inputs.push_back(&host);
    }
    if (!output || !output->device.buffer) {
        impl_->failed = true;
        return;
    }
    const size_t output_bytes = output->view_span_bytes();
    std::vector<uint64_t> host_output_storage(
        (output_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t));
    Tensor host_output = *output;
    host_output.data = host_output_storage.data();
    host_output.device.buffer = nullptr;
    host_output.device.offset = 0;
    // Save PLE's small INT32 history using the exact existing CPU update
    // routine. Keep the actual lookup dispatch and in-place update unchanged.
    if(impl_->state_prefix_tokens>0 && node.op_type==OpType::PLE_LOOKUP &&
        host_inputs.size()>=6 && host_inputs[0] && host_inputs[1] &&
        host_inputs[4] && host_inputs[5] && host_inputs[0]->prec==Precision::INT32 &&
        host_inputs[1]->prec==Precision::INT32) {
        const int tokens=static_cast<int>(host_inputs[0]->nelements());
        const int ngram=graph_params::get_i32(node.params,0,3);
        const int heads_per_ngram=graph_params::get_i32(node.params,1,8);
        const int eos=graph_params::get_i32(node.params,2,0);
        const int vocab=graph_params::get_i32(node.params,3,0);
        const int layer=graph_params::get_i32(node.params,4,0);
        const int real=graph_params::get_i32(node.params,7,tokens);
        if(ngram>=2 && heads_per_ngram>0 && tokens==impl_->state_prefix_tokens && real==tokens &&
            host_inputs[1]->nelements()==ngram && host_inputs[1]->is_contiguous()) {
            float* storage=impl_->state_prefix_output(device_pointer<int32_t>(*inputs[1]),
                size_t(ngram)*sizeof(int32_t),tokens);
            if(storage) {
                std::vector<int32_t> history(host_inputs[1]->ptr<int32_t>(),host_inputs[1]->ptr<int32_t>()+ngram);
                if(!history.back()){std::fill(history.begin(),history.end()-1,eos);history.back()=1;}
                std::vector<int32_t> snapshots(size_t(tokens)*ngram),indices(size_t(ngram-1)*heads_per_ngram);
                const uint64_t seed=static_cast<uint32_t>(graph_params::get_i32(node.params,5,1234)) |
                    (uint64_t(static_cast<uint32_t>(graph_params::get_i32(node.params,6,0)))<<32);
                for(int token=0;token<tokens;++token) {
                    if(!ple_build_indices(host_inputs[0]->ptr<int32_t>()+token,1,history.data(),ngram-1,
                        host_inputs[4]->ptr<int32_t>(),host_inputs[5]->ptr<int32_t>(),ngram,heads_per_ngram,
                        eos,vocab,layer,seed,indices.data())) {impl_->failed=true;return;}
                    std::copy(history.begin(),history.end(),snapshots.begin()+size_t(token)*ngram);
                }
                if(!mollm_cuda::copy_memory(storage,snapshots.data(),snapshots.size()*sizeof(int32_t),
                    cudaMemcpyHostToDevice,"PLE history prefixes")) {impl_->failed=true;return;}
            }
        }
    }
    // Staging allocations can reuse addresses/storage IDs across fallback
    // operators within one device graph. Their contents have a new lifetime.
    impl_->cpu.begin_execution();
    impl_->cpu.clear_dispatch_error();
    impl_->cpu.dispatch(node, host_inputs, &host_output, thread_pool);
    if (impl_->cpu.dispatch_failed()) {
        impl_->failed = true;
        return;
    }
    // Reference kernels may mutate cache or recurrent-state inputs. Copy every
    // staged input back; this is intentionally centralized so new fallback
    // operators cannot silently lose in-place updates.
    for (size_t input_index = 0; input_index < inputs.size(); ++input_index) {
        const int storage_index = staged_storage[input_index];
        if (storage_index < 0)
            continue;
        Tensor& input = *const_cast<Tensor*>(inputs[input_index]);
        const size_t bytes = input.view_span_bytes();
        if (!copy_from_host(
                host_storage[static_cast<size_t>(storage_index)].data(),
                input, bytes)) {
            impl_->failed = true;
            return;
        }
    }
    if (host_output.data != host_output_storage.data() ||
        !copy_from_host(host_output.data, *output, output_bytes)) {
        impl_->failed = true;
        return;
    }
    ++impl_->fallback_ops[static_cast<uint32_t>(node.op_type)];
    if (node.op_type == OpType::PLE_LOOKUP && impl_->segment_candidate && !impl_->segment_disabled)
        impl_->segment_ready = true;
}
