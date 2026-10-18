#include "kernels/cuda/attention.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(cudaError_t error) {
    if (error!=cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(error));std::exit(1); }
}
template<class T> struct Buffer {
    T* data=nullptr;
    explicit Buffer(size_t count) { check(cudaMalloc(&data,count*sizeof(T))); }
    ~Buffer() { cudaFree(data); }
};
static bool long_shared = false;
static bool long_shared_zero_pv = false;
static bool skip_zero_pv = false;
static bool long_pv = false;
static bool pv_unroll = false;
static bool pair_pv = false;
static bool pair_pv_tuned = false;
static bool pair_queries = false;
static bool small_only = false;
static bool full_batch = false;
static bool full_head_sanitize = false;
static bool long_kv_heads = false;
static bool native_pv_gemm = false;
static int head_lane_probe = 0;
static bool known_mask_probe = false;
static bool query_slices = false;
static bool query_slices_focused = false;
static bool query_slices_sanitize = false;
static bool actual_32k_pv = false;
static bool actual_32k_sanitize = false;
static bool head_lane_actual32k = false;
static bool head_lane_actual32k_sanitize = false;
static int query_slice_rows = 256;
static bool known_mask_mid = false;
static bool known_mask_sanitize = false;
static bool head_lane_long = false;
static bool head_lane_sanitize = false;
static bool native_pv_capacity32k = false;
static bool native_pv_capacity32k_sanitize = false;
static bool native_pv_sanitize = false;
static bool native_pv_sanitize_finite = false;
static bool native_pv_sanitize_long = false;
static bool native_pv_long_context = false;
static bool native_pv_short_context = false;
static bool native_pv_short_finite = false;
static bool native_pv_mid_context = false;
static bool native_pv_mid_sanitize = false;
static cublasHandle_t pv_handle=nullptr;
static int query_group = 2;
static bool grouped_long_shared = false;
static bool grouped_heads = false;
static bool run(int queries,int keys,bool interleaved,bool benchmark,float scale=.0625f) {
    const bool dense_bench=benchmark&&(pair_queries||native_pv_gemm);
    const int heads=query_slices_sanitize||head_lane_actual32k_sanitize?4:benchmark?24:(head_lane_probe||grouped_heads||native_pv_short_context?8:4),kh=2,dim=256,capacity=native_pv_capacity32k||native_pv_mid_context?32768:dense_bench?(long_kv_heads||native_pv_gemm?16384:4096):keys+3,past=keys-queries;
    const int qwidth=dense_bench?256:260,owidth=dense_bench?256:264;
    const size_t qs=interleaved?size_t(heads)*qwidth:qwidth;
    const size_t qh=interleaved?qwidth:size_t(queries)*qs;
    const size_t os=interleaved?size_t(heads)*owidth:owidth;
    const size_t oh=interleaved?owidth:size_t(queries)*os;
    const size_t qsize=size_t(heads-1)*qh+size_t(queries-1)*qs+dim+8;
    const size_t outsize=size_t(heads-1)*oh+size_t(queries-1)*os+dim+8;
    const size_t kvsize=size_t(kh)*capacity*dim,msize=size_t(queries)*(keys+2);
    std::vector<float> q(qsize),mask(msize),reference(outsize),actual(outsize);
    std::vector<__half> k(kvsize),v(kvsize);
    Buffer<float> dq(qsize),dm(msize),scores(size_t(heads)*queries*keys),out(outsize);
    Buffer<float> sliced_scores(query_slices?size_t(heads)*std::min(query_slice_rows,queries)*keys:1);
    Buffer<__half> dk(kvsize),dv(kvsize);Buffer<uint32_t> valid(1);
    Buffer<float> pv_values(native_pv_gemm?kvsize:1),pv_output(native_pv_gemm?outsize:1);
    Buffer<float*> pv_pointers(native_pv_gemm?size_t(heads)*3:1);
    mollm_cuda::SdpaGemmPrefillParams p{heads,kh,queries,keys,past,dim,capacity,
        true,scale,qs,qh,1,size_t(keys+2),os,oh};
    auto launch=[&](bool candidate,int offset=0,int length=0) {
        auto part=p;
        if(length) {part.queries=length;part.past+=offset;}
        const auto* part_q=dq.data+size_t(offset)*qs;
        auto* part_out=out.data+size_t(offset)*os;
        const auto* part_mask=dm.data+size_t(offset)*(keys+2);
        bool ok;
        if(query_slices&&candidate&&part.queries>=32) {
            cudaError_t error=cudaSuccess;
            ok=mollm_cuda::try_launch_sdpa_prefill_query_slices(pv_handle,part_q,dk.data,dv.data,
                sliced_scores.data,part_out,part_mask,valid.data,pv_values.data,pv_output.data+size_t(offset)*os,
                pv_pointers.data,part,query_slice_rows,head_lane_probe,cudaStreamPerThread,&error);
            check(error);
        } else if(native_pv_gemm&&(candidate||head_lane_probe)&&part.queries>=32) {
            ok=mollm_cuda::try_launch_sdpa_prefill_validated_prefix(part_q,dk.data,dv.data,scores.data,part_out,
                part_mask,valid.data,part,cudaStreamPerThread,nullptr,1024,true,false,true,false,2,false,false,true,
                ((known_mask_probe&&!head_lane_actual32k)||(candidate&&head_lane_probe))?head_lane_probe:1,known_mask_probe&&(candidate||query_slices));
            if(ok) {
                const auto status=mollm_cuda::launch_sdpa_prefill_fp32_pv(pv_handle,dv.data,scores.data,
                    pv_output.data+size_t(offset)*os,pv_values.data,pv_pointers.data,part,cudaStreamPerThread);
                if(status!=CUBLAS_STATUS_SUCCESS){std::fprintf(stderr,"PV cuBLAS status=%d\n",int(status));std::exit(2);}
                mollm_cuda::launch_sdpa_validated_pv_copy(pv_output.data+size_t(offset)*os,part_out,valid.data,part,cudaStreamPerThread);
            }
        } else if(native_pv_short_context&&part.queries>=16) {
            // Compare with the RETAINED short/global-score head-group4 path,
            // not the slower single-query reference. Small candidate parts
            // keep that path; parts below16 use its ordinary fallback below.
            ok=mollm_cuda::try_launch_sdpa_prefill_validated_prefix(part_q,dk.data,dv.data,scores.data,part_out,
                part_mask,valid.data,part,cudaStreamPerThread,nullptr,1024,true,false,false,true,4,false,true);
        } else if (candidate || long_shared || skip_zero_pv) ok=mollm_cuda::try_launch_sdpa_prefill_validated_prefix(
            part_q,dk.data,dv.data,scores.data,part_out,part_mask,valid.data,part,cudaStreamPerThread,
            nullptr,grouped_long_shared&&candidate?8192/query_group:(long_pv || (candidate && long_shared) ? 4096 : 1024),
            native_pv_gemm || pair_queries || pair_pv || pv_unroll || long_shared_zero_pv || (candidate && skip_zero_pv),
            (pv_unroll && candidate) || (pair_pv_tuned && !candidate && keys > 1024),
            (native_pv_gemm&&keys>1024) || (pair_pv && candidate) || (pair_queries && !candidate && !grouped_long_shared && (!grouped_heads||long_kv_heads) && keys>1024),
            pair_queries&&(candidate||grouped_long_shared||(grouped_heads&&!long_kv_heads)),query_group,grouped_long_shared&&candidate,grouped_heads&&candidate);
        else ok=mollm_cuda::try_launch_sdpa_prefill(dq.data,dk.data,dv.data,scores.data,
            out.data,dm.data,heads,kh,queries,keys,past,dim,dim,capacity,true,true,
            true,p.scale,1,qs,qh,1,keys+2,1,os,oh);
        if (!ok) std::exit(2);check(cudaGetLastError());
    };
    auto inputs=[&](int variant,int fault) {
        for (size_t i=0;i<qsize;++i) q[i]=std::sin(float(i)*.19f+variant);
        for (size_t i=0;i<kvsize;++i) {
            k[i]=__float2half_rn(std::cos(float(i)*.13f+variant));
            v[i]=__float2half_rn(std::sin(float(i)*.11f+variant));
        }
        for (int row=0;row<queries;++row) for (int col=0;col<keys;++col)
            mask[size_t(row)*(keys+2)+col]=col<=past+row?0.f:-1e38f;
        if(fault==11)mask[0]=-0.f; // Numerically equal, not the validated +0 bit pattern.
        if (fault==1) mask[0]=.25f;
        if (fault==2) q[0]=1e21f;
        if (fault==3) k[0]=__float2half_rn(NAN);
        if (fault==4) q[0]=NAN;
        if (fault==5) k[0]=__float2half_rn(INFINITY);
        if (fault==6) v[0]=__float2half_rn(NAN); // Must keep all PV operations.
        if (fault==8) v[0]=__float2half_rn(INFINITY);
        if (fault==10) {
            for (int head=0;head<kh;++head) for (int row=keys;row<capacity;++row)
                for (int dim=0;dim<256;++dim) {
                    const size_t index=(size_t(head)*capacity+row)*256+dim;
                    k[index]=__float2half_rn(NAN);
                    v[index]=__float2half_rn(INFINITY);
                }
        }
        if (fault==7 || fault==9) {
            // A dominant first score makes tiny later PV terms underflow to
            // signed zero. Positive/negative future V must still affect its sign.
            std::fill(k.begin(),k.end(),__float2half_rn(0.f));
            std::fill(v.begin(),v.end(),__float2half_rn(0.f));
            for (int head=0;head<heads;++head) for (int row=0;row<queries;++row)
                for (int dim=0;dim<256;++dim)
                    q[size_t(head)*qh+size_t(row)*qs+dim]=dim==0?1.f:0.f;
            for (int head=0;head<kh;++head) {
                k[size_t(head)*capacity*256]=__float2half_rn(fault==7?1616.f:1280.f);
                for (int row=1;row<keys;++row) for (int dim=0;dim<256;++dim)
                    v[(size_t(head)*capacity+row)*256+dim]=__float2half_rn(
                        row>=2 && dim%2==0?0x1p-24f:-0x1p-24f);
            }
        }
        check(cudaMemcpy(dq.data,q.data(),qsize*4,cudaMemcpyHostToDevice));
        check(cudaMemcpy(dk.data,k.data(),kvsize*2,cudaMemcpyHostToDevice));
        check(cudaMemcpy(dv.data,v.data(),kvsize*2,cudaMemcpyHostToDevice));
        check(cudaMemcpy(dm.data,mask.data(),msize*4,cudaMemcpyHostToDevice));
    };
    inputs(0,0);
    if(query_slices) {
        const uint32_t sentinel=0x12345678;
        check(cudaMemcpy(valid.data,&sentinel,4,cudaMemcpyHostToDevice));
        const auto rejected=[&](mollm_cuda::SdpaGemmPrefillParams invalid,int tile,int lanes,
                                float* workspace,float* temporary) {
            cudaError_t error=cudaErrorUnknown;
            if(mollm_cuda::try_launch_sdpa_prefill_query_slices(pv_handle,dq.data,dk.data,dv.data,
                workspace,out.data,dm.data,valid.data,pv_values.data,temporary,pv_pointers.data,
                invalid,tile,lanes,cudaStreamPerThread,&error)||error!=cudaSuccess)return false;
            uint32_t after=0;check(cudaMemcpy(&after,valid.data,4,cudaMemcpyDeviceToHost));
            return after==sentinel;
        };
        for(int tile:{0,1,255,2049})if(!rejected(p,tile,2,sliced_scores.data,pv_output.data))return false;
        for(int lanes:{0,3})if(!rejected(p,256,lanes,sliced_scores.data,pv_output.data))return false;
        if(!rejected(p,256,2,dq.data,pv_output.data)||!rejected(p,256,2,sliced_scores.data,out.data))return false;
        auto invalid=p;invalid.query_position_stride=SIZE_MAX;
        if(!rejected(invalid,256,2,sliced_scores.data,pv_output.data))return false;
        invalid=p;invalid.mask_row_stride=SIZE_MAX;
        if(!rejected(invalid,256,2,sliced_scores.data,pv_output.data))return false;
        invalid=p;invalid.capacity=32769;
        if(!rejected(invalid,256,2,sliced_scores.data,pv_output.data))return false;
        std::puts("query_slice_guards rejected=11 flag_untouched=1");
    }
    if(head_lane_probe) {
        for(int rejected:{0,3,4,5}) {
            cudaError_t error=cudaErrorUnknown;
            if(mollm_cuda::try_launch_sdpa_prefill_validated_prefix(dq.data,dk.data,dv.data,scores.data,out.data,
                dm.data,valid.data,p,cudaStreamPerThread,&error,1024,true,false,true,false,2,false,false,true,rejected)||error!=cudaSuccess)return false;
        }
        auto incompatible=p;incompatible.heads=6;
        if(mollm_cuda::try_launch_sdpa_prefill_validated_prefix(dq.data,dk.data,dv.data,scores.data,out.data,
            dm.data,valid.data,incompatible,cudaStreamPerThread,nullptr,1024,true,false,true,false,2,false,false,true,2))return false;
    }
    if(native_pv_gemm){launch(true);check(cudaStreamSynchronize(cudaStreamPerThread));}
    cudaGraph_t graph;cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
    launch(true);
    check(cudaStreamEndCapture(cudaStreamPerThread,&graph));
    check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));check(cudaGraphDestroy(graph));
    cudaGraphExec_t partitioned=nullptr;
    if(full_batch) {
        const int cut=long_kv_heads||native_pv_gemm?std::min(1024,queries/2):(queries==1025?1008:1024);
        if(native_pv_gemm){launch(true,0,cut);launch(true,cut,queries-cut);check(cudaStreamSynchronize(cudaStreamPerThread));}
        check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
        launch(true,0,cut);launch(true,cut,queries-cut);
        check(cudaStreamEndCapture(cudaStreamPerThread,&graph));
        check(cudaGraphInstantiate(&partitioned,graph,nullptr,nullptr,0));check(cudaGraphDestroy(graph));
    }
    int variant=0;
    const std::vector<int> faults=known_mask_sanitize?std::vector<int>{0,1,0,6,11}:
        known_mask_probe?std::vector<int>{0,0,1,2,3,4,5,6,7,8,9,10,11}:
        head_lane_sanitize||native_pv_sanitize_long||native_pv_capacity32k_sanitize||native_pv_mid_sanitize?std::vector<int>{0,1,0}:
        (full_head_sanitize||native_pv_sanitize_finite)?std::vector<int>{0,0,0}:
        std::vector<int>{0,0,1,2,3,4,5,6,7,8,9,10};
    for (int fault : faults) {
        inputs(++variant,fault);
        check(cudaMemset(out.data,0x7f,outsize*4));launch(false);
        check(cudaMemcpy(reference.data(),out.data,outsize*4,cudaMemcpyDeviceToHost));
        check(cudaMemset(out.data,0x7f,outsize*4));
        check(cudaGraphLaunch(executable,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
        check(cudaMemcpy(actual.data(),out.data,outsize*4,cudaMemcpyDeviceToHost));
        if(full_batch) {
            check(cudaMemset(out.data,0x7f,outsize*4));
            check(cudaGraphLaunch(partitioned,cudaStreamPerThread));
            check(cudaStreamSynchronize(cudaStreamPerThread));
            std::vector<float> split(outsize);
            check(cudaMemcpy(split.data(),out.data,outsize*4,cudaMemcpyDeviceToHost));
            if(std::memcmp(split.data(),actual.data(),outsize*4)) {
                std::fprintf(stderr,"whole/split mismatch queries=%d keys=%d interleaved=%d fault=%d\n",queries,keys,interleaved,fault);
                return false;
            }
        }
        // The per-launch flag belongs to the whole batch, not the last slice.
        if(full_batch){check(cudaGraphLaunch(executable,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));}
        uint32_t selected;check(cudaMemcpy(&selected,valid.data,4,cudaMemcpyDeviceToHost));
        const bool expected=std::isfinite(scale) && std::fabs(scale)<=1.f &&
            (fault==0 || fault==7 || fault==9 || fault==10 || (!skip_zero_pv && !long_shared_zero_pv && (fault==6 || fault==8)));
        if (past==0 && queries>=2 && keys>=3 && scale==.0625f && (fault==7 || fault==9)) {
            uint32_t even,odd;
            std::memcpy(&even,&reference[os],4);
            std::memcpy(&odd,&reference[os+1],4);
            if (fault==7 ? (even!=0u || odd!=0x80000000u)
                         : ((odd&0x7f800000u)!=0u || (odd&0x007fffffu)==0u)) {
                std::fprintf(stderr,"zero/subnormal trap did not trigger fault=%d even=%08x odd=%08x\n",fault,even,odd);
                return false;
            }
        }
        if (bool(selected)!=expected || std::memcmp(reference.data(),actual.data(),outsize*4)) {
            std::fprintf(stderr,"mismatch Q=%d K=%d interleaved=%d fault=%d selected=%u expected=%d\n",
                queries,keys,interleaved,fault,selected,expected);
            size_t changed=0,first=outsize;uint32_t first_ref=0,first_actual=0;
            for(size_t i=0;i<outsize;++i) {
                uint32_t a,b;std::memcpy(&a,&reference[i],4);std::memcpy(&b,&actual[i],4);
                if(a!=b){if(first==outsize){first=i;first_ref=a;first_actual=b;}++changed;}
            }
            std::fprintf(stderr,"difference_count=%zu first_index=%zu reference_bits=%08x candidate_bits=%08x\n",
                changed,first,first_ref,first_actual);return false;
        }
        for (size_t i=outsize-8;i<outsize;++i) {
            uint32_t bits;std::memcpy(&bits,&actual[i],4);if (bits!=0x7f7f7f7f) return false;
        }
    }
    check(cudaGraphExecDestroy(executable));
    if(partitioned)check(cudaGraphExecDestroy(partitioned));
    if(full_batch)std::printf("whole_vs_partitioned queries=%d keys=%d interleaved=%d faults=%zu byte_exact=1 grouped=%d\n",queries,keys,interleaved,faults.size(),int(pair_queries));
    std::printf("Q=%d K=%d interleaved=%d scale=%g cases=%zu exact=1 flags=1 guards=1 capacity=%d q_position=%zu q_head=%zu out_position=%zu out_head=%zu pv_gemm=%d\n",queries,keys,interleaved,p.scale,faults.size(),capacity,qs,qh,os,oh,int(native_pv_gemm&&queries>=32));
    if (benchmark && !std::getenv("MOLLM_SDPA_VALIDATED_PREFIX_CHECK_ONLY")) {
        inputs(11,0);
        // Long-context operators already take hundreds of milliseconds each.
        // Keep seven alternating pairs, but bound repeated work per graph.
        cudaGraphExec_t graphs[2];const int calls=actual_32k_pv||head_lane_probe||native_pv_long_context||native_pv_short_context?2:12;
        std::printf("timing_graph_calls=%d warm_graph_replays=2 timed_graph_replays=3 pairs=7\n",calls);
        for (int mode=0;mode<2;++mode) {
            check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
            for (int i=0;i<calls;++i) launch(mode!=0);
            check(cudaStreamEndCapture(cudaStreamPerThread,&graph));
            check(cudaGraphInstantiate(&graphs[mode],graph,nullptr,nullptr,0));check(cudaGraphDestroy(graph));
        }
        for (int order=0;order<7;++order) {
            float us[2];
            for (int step=0;step<2;++step) {
                const int mode=step^(order&1);
                for (int i=0;i<2;++i) check(cudaGraphLaunch(graphs[mode],cudaStreamPerThread));
                check(cudaStreamSynchronize(cudaStreamPerThread));
                cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
                check(cudaEventRecord(begin,cudaStreamPerThread));
                for (int i=0;i<3;++i) check(cudaGraphLaunch(graphs[mode],cudaStreamPerThread));
                check(cudaEventRecord(end,cudaStreamPerThread));check(cudaEventSynchronize(end));
                float ms;check(cudaEventElapsedTime(&ms,begin,end));us[mode]=ms*1000/(3*calls);
                check(cudaEventDestroy(begin));check(cudaEventDestroy(end));
            }
            std::printf("timing interleaved=%d order=%d baseline_us=%.3f validated_us=%.3f queries=%d keys=%d query_group=%d\n",interleaved,order,us[0],us[1],queries,keys,query_group);
        }
        for (auto executable : graphs) check(cudaGraphExecDestroy(executable));
    }
    return true;
}
int main(int argc,char** argv) {
    // Guarded long-running checks must retain completed-case evidence even if
    // a later timing/checker phase times out.
    std::setvbuf(stdout,nullptr,_IOLBF,0);
    head_lane_actual32k_sanitize=argc==2&&!std::strcmp(argv[1],"--head-lane-known-mask-actual32k-sanitize");
    head_lane_actual32k=head_lane_actual32k_sanitize||(argc==2&&!std::strcmp(argv[1],"--head-lane-known-mask-actual32k"));
    head_lane_long=argc==2&&(!std::strcmp(argv[1],"--head-lane-reuse2-long")||!std::strcmp(argv[1],"--head-lane-known-mask-long"));
    query_slices_sanitize=argc==2&&!std::strcmp(argv[1],"--query-slices1024-sanitize");
    known_mask_sanitize=head_lane_actual32k_sanitize||query_slices_sanitize||(argc==2&&(!std::strcmp(argv[1],"--head-lane-known-mask-sanitize")||!std::strcmp(argv[1],"--mid-known-mask-sanitize")));
    head_lane_sanitize=known_mask_sanitize||(argc==2&&(!std::strcmp(argv[1],"--head-lane-reuse2-sanitize")||!std::strcmp(argv[1],"--head-lane-known-mask-zero-scale")));
    query_slice_rows=query_slices_sanitize||(argc==2&&!std::strcmp(argv[1],"--query-slices1024"))?1024:256;
    query_slices_focused=query_slice_rows==1024||(argc==2&&!std::strcmp(argv[1],"--query-slices-focused"));
    query_slices=query_slices_focused||(argc==2&&!std::strcmp(argv[1],"--query-slices"));
    known_mask_mid=argc==2&&(!std::strcmp(argv[1],"--mid-known-mask")||!std::strcmp(argv[1],"--mid-known-mask-sanitize"));
    known_mask_probe=head_lane_actual32k||query_slices||known_mask_mid||(argc==2&&(!std::strcmp(argv[1],"--head-lane-known-mask")||
        !std::strcmp(argv[1],"--head-lane-known-mask-long")||!std::strcmp(argv[1],"--head-lane-known-mask-sanitize")||!std::strcmp(argv[1],"--head-lane-known-mask-zero-scale")));
    head_lane_probe=known_mask_mid?1:known_mask_probe||head_lane_long||head_lane_sanitize||(argc==2&&!std::strcmp(argv[1],"--head-lane-reuse2"))?2:0;
    native_pv_short_finite=argc==2&&!std::strcmp(argv[1],"--native-pv-gemm-short-finite");
    native_pv_mid_sanitize=argc==2&&!std::strcmp(argv[1],"--native-pv-gemm-mid-sanitize");
    native_pv_mid_context=known_mask_mid||native_pv_mid_sanitize||(argc==2&&!std::strcmp(argv[1],"--native-pv-gemm-mid-context"));
    native_pv_short_context=native_pv_mid_context||native_pv_short_finite||(argc==2&&!std::strcmp(argv[1],"--native-pv-gemm-short-context"));
    native_pv_long_context=argc==2&&!std::strcmp(argv[1],"--native-pv-gemm-long-context");
    native_pv_sanitize_long=argc==2&&!std::strcmp(argv[1],"--native-pv-gemm-long-sanitize");
    native_pv_sanitize_finite=native_pv_short_finite||native_pv_sanitize_long||(argc==2&&!std::strcmp(argv[1],"--native-pv-gemm-sanitize-finite"));
    native_pv_sanitize=native_pv_sanitize_finite||(argc==2&&!std::strcmp(argv[1],"--native-pv-gemm-sanitize"));
    actual_32k_sanitize=argc==2&&!std::strcmp(argv[1],"--native-pv-actual32k-sanitize");
    native_pv_capacity32k_sanitize=actual_32k_sanitize||(argc==2&&!std::strcmp(argv[1],"--native-pv-capacity32k-sanitize"));
    actual_32k_pv=actual_32k_sanitize||(argc==2&&!std::strcmp(argv[1],"--native-pv-actual32k"));
    native_pv_capacity32k=head_lane_actual32k||actual_32k_pv||head_lane_long||head_lane_sanitize||native_pv_capacity32k_sanitize||(argc==2&&!std::strcmp(argv[1],"--native-pv-capacity32k"));
    native_pv_gemm=head_lane_probe||native_pv_capacity32k||native_pv_short_context||native_pv_long_context||native_pv_sanitize||(argc==2&&!std::strcmp(argv[1],"--native-pv-gemm"));
    long_kv_heads=argc==2&&!std::strcmp(argv[1],"--long-kv-heads4");
    full_head_sanitize=argc==2&&!std::strcmp(argv[1],"--full-batch-heads4-sanitize");
    full_batch=native_pv_gemm||long_kv_heads||full_head_sanitize||(argc==2&&(!std::strcmp(argv[1],"--full-batch")||
        !std::strcmp(argv[1],"--full-batch-group2")||!std::strcmp(argv[1],"--full-batch-group4")||
        !std::strcmp(argv[1],"--full-batch-heads4")));
    grouped_heads=long_kv_heads||full_head_sanitize||(argc==2&&(!std::strcmp(argv[1],"--group-heads4")||!std::strcmp(argv[1],"--group-heads4-small")||
        !std::strcmp(argv[1],"--full-batch-heads4")));
    grouped_long_shared=argc==2&&(!std::strcmp(argv[1],"--pair-queries4-long-shared")||
        !std::strcmp(argv[1],"--pair-queries4-long-shared-small"));
    small_only=argc==2&&(!std::strcmp(argv[1],"--pair-queries-small")||!std::strcmp(argv[1],"--pair-queries4-small"));
    small_only=small_only||(grouped_long_shared&&!std::strcmp(argv[1],"--pair-queries4-long-shared-small"));
    small_only=small_only||(grouped_heads&&!std::strcmp(argv[1],"--group-heads4-small"));
    query_group=grouped_heads||grouped_long_shared||(argc==2&&(!std::strcmp(argv[1],"--pair-queries4")||!std::strcmp(argv[1],"--pair-queries4-small")||!std::strcmp(argv[1],"--full-batch-group4")))?4:2;
    pair_queries=small_only||query_group==4||(argc==2&&!std::strcmp(argv[1],"--pair-queries"));
    pair_queries=pair_queries||(full_batch&&std::strcmp(argv[1],"--full-batch"));
    if(native_pv_gemm)pair_queries=false;
    long_shared_zero_pv = argc == 2 && !std::strcmp(argv[1],"--long-shared-zero-pv");
    long_shared = long_shared_zero_pv || (argc == 2 && !std::strcmp(argv[1],"--long-shared"));
    long_pv = argc == 2 && !std::strcmp(argv[1],"--skip-zero-pv-long");
    pv_unroll = argc == 2 && !std::strcmp(argv[1],"--pv-unroll");
    pair_pv_tuned = argc == 2 && !std::strcmp(argv[1],"--pair-pv-tuned");
    pair_pv = pair_pv_tuned || (argc == 2 && !std::strcmp(argv[1],"--pair-pv"));
    skip_zero_pv = full_batch || pair_queries || pair_pv || pv_unroll || long_pv || (argc == 2 && !std::strcmp(argv[1],"--skip-zero-pv"));
    if (argc > 2 || (argc == 2 && !long_shared && !skip_zero_pv)) return 64;
    if(long_shared_zero_pv)
        std::puts("comparison=shared-score-limit baseline=1024 candidate=4096 validated-prefix=both zero-pv=both");
    if(pv_unroll)
        std::puts("comparison=PV-unroll baseline=original candidate=unroll4 shared-limit=1024 validated-prefix=both zero-pv=both");
    if(pair_pv)
        std::puts("comparison=PV-dimension-pair baseline=scalar candidate=half2 shared-limit=1024 validated-prefix=both zero-pv=both");
    if(pair_pv_tuned)
        std::puts("baseline_override=global-score-unroll4 shared-score-scalar matches-retained-PV-selection=1");
    if(long_kv_heads)std::puts("comparison=long-KV-PV-reuse baseline=current-single-query-PV-pair candidate=4-heads-one-KV shared-limit1024 zero-pv=both runtime-selection-unchanged=1");
    else if(grouped_heads)std::puts("comparison=PV-reuse-axis baseline=4-adjacent-queries candidate=4-heads-one-KV shared-limit1024 zero-pv=both");
    else if(grouped_long_shared)std::puts("comparison=group4-score-storage baseline=group4-limit1024 candidate=group4-limit2048 zero-pv=both payload_max32KiB");
    else if(pair_queries)std::printf("comparison=validated-query-PV-reuse baseline=current-prefix-scalar-shared-half2-global candidate=query-group-%d-half2 shared-limit=1024 zero-pv=both\n",query_group);
    int devices=0;if (cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    if(native_pv_gemm) {
        std::puts(query_slices?
            "comparison=query-slices baseline=retained-native-QK-softmax-GEMM-PV-and-mask-elision candidate=compact-query-slices includes=full-parent-validation-V-expansion-pointer-binding-output-copy":head_lane_actual32k?
            "comparison=actual-long-KV-combined-head-and-mask-policy baseline=head1-mask-load-GEMM-PV candidate=head2-mask-elision-GEMM-PV includes=validation-V-expansion-pointer-binding-output-copy":known_mask_probe?
            "comparison=known-positive-zero-mask baseline=retained-native-QK-softmax-GEMM-PV candidate=mask-load-elision includes=validation-V-expansion-pointer-binding-output-copy invalid-prefix=original-PV-fallback":native_pv_short_context?
            "comparison=short-native-QK-softmax-plus-pedantic-GEMM-PV baseline=current-head-group4 includes=V-expansion-pointer-binding-validation-output-copy queries-below32=original-path runtime-range-unchanged=1":
            "comparison=native-QK-softmax-plus-pedantic-GEMM-PV baseline=current-single-query-paired-PV includes=V-expansion-pointer-binding-validation-output-copy invalid-prefix=original-PV-fallback queries-below32=original-path");
        if(cublasCreate(&pv_handle)!=CUBLAS_STATUS_SUCCESS||cublasSetStream(pv_handle,cudaStreamPerThread)!=CUBLAS_STATUS_SUCCESS)return 2;
        bool ok=true;
        if(head_lane_probe) {
            if(known_mask_probe&&!query_slices&&!head_lane_actual32k)std::printf("scope=known-mask-load-elision-only BOTH-arms-head-lanes%d preserves-positive-zero-addition runtime-selection-unchanged=1\n",head_lane_probe);
            if(!known_mask_probe)std::printf("comparison=warp-head-lanes baseline=current-native-probabilities-GEMM-PV candidate=%d-heads-one-warp includes=validation-V-expand-PV-copy runtime-selection-unchanged=1\n",head_lane_probe);
            if(head_lane_actual32k) {
                std::puts("scope=actualKeys16385_24576_32768_capacity32768 baseline=head1-mask-load-GEMM-PV candidate=head2-positive-zero-mask-elision-GEMM-PV combined-policy-qualification runtime-unchanged");
                if(head_lane_actual32k_sanitize) {
                    std::puts("sanitizer_scope=actualKeys32768_capacity32768_H4_Q32Q33_padded_interleaved_changed5_including_invalid_mask_direct_capture_partition_guards_no_timing");
                    for(bool interleaved:{false,true})for(int rows:{32,33})
                        ok &= run(rows,32768,interleaved,false);
                } else {
                    for(bool interleaved:{false,true})for(int keys:{16385,24576,32768})
                        for(int rows:{32,129})ok &= run(rows,keys,interleaved,false);
                    if(ok)ok=run(2048,24576,false,true)&&run(997,32768,false,true);
                }
            } else if(query_slices) {
                std::printf("scope=query-slices%d BOTH-arms-retained-head-lanes2-and-mask-elision full-batch-validation-once V-expansion-once no-runtime-change\n",query_slice_rows);
                if(query_slices_sanitize) {
                    // A real split, including the adjusted 128-row tail, with
                    // full-parent invalidation and padded/interleaved guards.
                    // Reduced checker scope supplements, not replaces, the
                    // six-geometry/thirteen-fault numeric qualification.
                    std::puts("sanitizer_scope=H4_KVH2_Q1025_K4097_capacity32768_tile1024_padded_interleaved_faults5_direct_capture_partition_guards_no_timing");
                    for(bool interleaved:{false,true})ok &= run(1025,4097,interleaved,false);
                } else if(query_slices_focused) {
                    for(bool interleaved:{false,true})for(int q:query_slice_rows==1024?std::vector<int>{1025,2047}:std::vector<int>{257,513})ok &= run(q,8192,interleaved,false);
                } else for(bool interleaved:{false,true})for(int keys:{4097,8192})
                    for(int q:{32,33,257,513})ok &= run(q,keys,interleaved,false);
                if(ok&&!query_slices_sanitize)ok=run(2048,6144,false,true)&&run(957,7101,false,true);
            } else if(known_mask_mid) {
                std::printf("scope=mid-KV-single-head-capacity32768 faults%d no-runtime-selection-change\n",known_mask_sanitize?5:13);
                if(known_mask_sanitize) {
                    for(bool interleaved:{false,true})ok &= run(32,2080,interleaved,false);
                    for(float scale:{0.f,-0.f,2.f})ok &= run(32,2080,false,false,scale);
                } else {
                    for(bool interleaved:{false,true})for(int keys:{2080,3072,4096})
                        for(int q:{32,33,65,129})ok &= run(q,keys,interleaved,false);
                    if(ok)ok=run(2048,4096,false,true)&&run(1024,3072,false,true);
                    if(ok)for(float scale:{0.f,-0.f,-.0625f,1.f,-1.f,2.f})ok &= run(32,2080,false,false,scale);
                }
            } else if(head_lane_sanitize) {
                std::printf("scope=sanitize_H8_padded_interleaved_Q32Q33_K4097_capacity32768_faults%d_including_invalid_mask\n",known_mask_sanitize?5:known_mask_probe?13:3);
                for(bool interleaved:{false,true})ok &= run(32,4097,interleaved,false)&&run(33,4097,interleaved,false);
                if(known_mask_probe)for(float scale:known_mask_sanitize?std::vector<float>{0.f,-0.f,2.f}:std::vector<float>{0.f,-0.f,-.0625f,1.f,-1.f,2.f})
                    ok &= run(32,4097,false,false,scale);
            } else if(head_lane_long) {
                std::printf("scope=long_actualKeys14336_16384_capacity32768_faults%d\n",known_mask_probe?13:12);
                for(bool interleaved:{false,true})ok &= run(32,16384,interleaved,false)&&run(129,16384,interleaved,false);
                if(ok)ok=run(2048,14336,false,true)&&run(997,16384,false,true);
            } else {
                for(bool interleaved:{false,true})for(int keys:{4097,8192})
                    for(int rows:{32,33,65,129})ok &= run(rows,keys,interleaved,false);
                if(ok)ok=run(2048,6144,false,true)&&run(957,7101,false,true);
            }
            cublasDestroy(pv_handle);return ok?0:1;
        }
        if(native_pv_capacity32k) {
            if(actual_32k_sanitize) {
                std::puts("sanitizer_scope=actualKeys32768_capacity32768_H4_Q32Q33_padded_interleaved_changed3_including_invalid_mask_direct_capture_partition_guards_no_timing");
                for(bool interleaved:{false,true})for(int rows:{32,33})ok &= run(rows,32768,interleaved,false);
            } else if(actual_32k_pv) {
                std::puts("scope=actualKeys16385_24576_32768_capacity32768_native-PV-versus-pedantic-GEMM-PV_runtime-selection-unchanged");
                for(bool interleaved:{false,true})for(int keys:{16385,24576,32768})
                    for(int rows:{32,129})ok &= run(rows,keys,interleaved,false);
                if(ok)ok=run(2048,24576,false,true)&&run(997,32768,false,true);
            } else if(native_pv_capacity32k_sanitize) {
                std::puts("scope=capacity32768_actualKeysStillAtMost16384_strideQualification_not32kPromptSupport");
                for(bool interleaved:{false,true})ok &= run(32,4097,interleaved,false)&&run(33,4097,interleaved,false);
            } else {
                std::puts("scope=capacity32768_actualKeysStillAtMost16384_strideQualification_not32kPromptSupport");
                for(bool interleaved:{false,true})ok &= run(32,4097,interleaved,false)&&run(129,16384,interleaved,false);
                if(ok)ok=run(2048,6144,false,true)&&run(997,16384,false,true);
            }
            cublasDestroy(pv_handle);return ok?0:1;
        }
        if(native_pv_short_context) {
            if(native_pv_mid_context) {
                if(native_pv_mid_sanitize) {
                    std::puts("scope=midKV_sanitize_H8_Q32Q33_K2081K4096_capacity32768_changed3_including_invalid_mask");
                    for(bool interleaved:{false,true})for(int keys:{2081,4096})
                        ok &= run(32,keys,interleaved,false)&&run(33,keys,interleaved,false);
                    cublasDestroy(pv_handle);return ok?0:1;
                }
                std::puts("scope=midKV_past_at_least2048_capacity32768_all12_fault_inputs_current_head_group4_baseline_runtime_unchanged");
                for(bool interleaved:{false,true}) {
                    ok &= run(32,2080,interleaved,false);
                    for(int q:{17,31,32,33,65,129})ok &= run(q,3072,interleaved,false);
                }
                if(ok)ok=run(1024,3072,false,true);
                if(ok)ok=run(1021,3069,false,true);
                for(bool interleaved:{false,true})for(int q:{17,31,32,33,65,129})
                    ok &= run(q,4096,interleaved,false);
                if(ok)ok=run(2048,4096,false,true);
                if(ok)ok=run(957,4096,false,true);
                cublasDestroy(pv_handle);return ok?0:1;
            }
            if(native_pv_short_finite)std::puts("scope=FINITE_INPUT_TIMING_ONLY_not_all_fault_qualification_signed_zero_fault7_failed_separately");
            for(bool interleaved:{false,true})for(int keys:{1743,4096})
                for(int q:{17,31,32,33,65,129})ok &= run(q,keys,interleaved,false);
            if(ok)ok=run(1743,1743,false,true);
            if(ok)ok=run(2048,2048,false,true);
            if(ok)ok=run(2048,4096,false,true);
            cublasDestroy(pv_handle);return ok?0:1;
        }
        if(native_pv_long_context) {
            std::puts("qualification_scope=K16384_padded_interleaved_and_H24_Q2048K14336_Q997K16384_capacity16384_faults12_direct_capture_partition_guards");
            for(bool interleaved:{false,true})ok &= run(32,16384,interleaved,false)&&run(129,16384,interleaved,false);
            if(ok)ok=run(2048,14336,false,true);
            if(ok)ok=run(997,16384,false,true);
            cublasDestroy(pv_handle);return ok?0:1;
        }
        if(native_pv_sanitize) {
            std::puts(native_pv_sanitize_long?
                "sanitizer_scope=Q32Q33_K16384_padded_interleaved_changed_inputs3_including_invalid_mask_direct_capture_partition_guards_no_timing":native_pv_sanitize_finite?
                "sanitizer_scope=Q32Q33_K4097_padded_interleaved_finite_changed_inputs3_direct_capture_partition_guards_no_timing":
                "sanitizer_scope=Q32K4097_Q129K8192_padded_interleaved_faults12_direct_capture_partition_guards_no_timing");
            for(bool interleaved:{false,true}) {
                ok &= run(32,native_pv_sanitize_long?16384:4097,interleaved,false)&&
                    run(native_pv_sanitize_finite?33:129,native_pv_sanitize_long?16384:native_pv_sanitize_finite?4097:8192,interleaved,false);
            }
            cublasDestroy(pv_handle);return ok?0:1;
        }
        for(bool interleaved:{false,true}) {
            for(int q:{17,31,32,33,63,64,65,127,128,129})ok &= run(q,4097,interleaved,false);
            ok &= run(128,8192,interleaved,false);
        }
        if(ok)ok=run(2048,6144,false,true);
        cublasDestroy(pv_handle);
        return ok?0:1;
    }
    if(long_kv_heads) {
        for(bool interleaved:{false,true}) {
            if(!run(17,4097,interleaved,false)||!run(128,8192,interleaved,false))return 1;
        }
        return run(2048,6144,false,true)?0:1;
    }
    if(full_head_sanitize) {
        std::puts("sanitizer_scope=Q1743_K1743_H8_KVH2_finite_changed_inputs3_direct_capture_partition_guards_no_timing");
        return run(1743,1743,false,false)?0:1;
    }
    if(full_batch) {
        for(bool interleaved:{false,true}) {
            for(int queries:{1025,1743,2048})
                if(!run(queries,queries,interleaved,false))return 1;
            if(!run(1743,2048,interleaved,false))return 1;
        }
        if(std::getenv("MOLLM_SDPA_FULL_BATCH_BENCH")&&
           !run(1743,1743,false,true))return 1;
        return 0;
    }
    Buffer<float> stub(8);Buffer<uint32_t> flag(1);
    mollm_cuda::SdpaGemmPrefillParams good{4,2,3,3,0,256,3,true,.0625f,256,768,1,3,256,768};
    int rejected=0;
    auto reject=[&](const mollm_cuda::SdpaGemmPrefillParams& p) {
        if (mollm_cuda::try_launch_sdpa_prefill_validated_prefix(stub.data,stub.data,stub.data,
            stub.data,stub.data,stub.data,flag.data,p)) std::exit(3);
        ++rejected;
    };
    auto bad=good;bad.heads=0;reject(bad);bad=good;bad.heads=65536;reject(bad);
    bad=good;bad.kv_heads=0;reject(bad);bad=good;bad.heads=3;reject(bad);
    bad=good;bad.queries=1;reject(bad);bad=good;bad.queries=65536;reject(bad);
    bad=good;bad.keys=0;reject(bad);bad=good;bad.keys=65536;reject(bad);
    bad=good;bad.dimension=128;reject(bad);bad=good;bad.capacity=2;reject(bad);
    bad=good;bad.past=-1;reject(bad);bad=good;bad.past=INT32_MAX;reject(bad);
    bad=good;bad.query_position_stride=255;reject(bad);bad=good;bad.query_head_stride=769;reject(bad);
    bad=good;bad.mask_column_stride=0;reject(bad);bad=good;bad.output_head_stride=0;reject(bad);
    bad=good;bad.query_position_stride=SIZE_MAX-1;reject(bad);
    bad=good;bad.output_head_stride=SIZE_MAX;reject(bad);bad=good;bad.mask_row_stride=SIZE_MAX;reject(bad);
    if (mollm_cuda::try_launch_sdpa_prefill_validated_prefix(stub.data,stub.data,stub.data,
            stub.data,stub.data,nullptr,flag.data,good) ||
        mollm_cuda::try_launch_sdpa_prefill_validated_prefix(stub.data,stub.data,stub.data,
            stub.data,stub.data,stub.data,nullptr,good)) return 3;
    for (int limit : {0,8193}) {
        cudaError_t error = cudaSuccess;
        if (mollm_cuda::try_launch_sdpa_prefill_validated_prefix(stub.data,stub.data,stub.data,
            stub.data,stub.data,stub.data,flag.data,good,cudaStreamPerThread,&error,limit) ||
            error != cudaSuccess) return 3;
    }
    check(cudaGetLastError());std::printf("invalid_shapes=%d missing_mask_flag=2 invalid_shared_limits=2 rejected=1\n",rejected);
    if (pv_unroll) {
        if (mollm_cuda::try_launch_sdpa_prefill_validated_prefix(stub.data,stub.data,stub.data,
                stub.data,stub.data,stub.data,flag.data,good,cudaStreamPerThread,nullptr,1024,false,true)) return 3;
        std::puts("pv_unroll_requires_zero_pv=1 rejected=1");
    }
    if (pair_pv) {
        if (mollm_cuda::try_launch_sdpa_prefill_validated_prefix(stub.data,stub.data,stub.data,
                stub.data,stub.data,stub.data,flag.data,good,cudaStreamPerThread,nullptr,1024,true,true,true)) return 3;
        std::puts("pv_pair_and_unroll_combination_rejected=1");
    }
    if(pair_queries) {
        if(mollm_cuda::try_launch_sdpa_prefill_validated_prefix(stub.data,stub.data,stub.data,stub.data,stub.data,stub.data,flag.data,
            good,cudaStreamPerThread,nullptr,1024,false,false,false,true)||
           mollm_cuda::try_launch_sdpa_prefill_validated_prefix(stub.data,stub.data,stub.data,stub.data,stub.data,stub.data,flag.data,
            good,cudaStreamPerThread,nullptr,2048,true,false,false,true)||
           mollm_cuda::try_launch_sdpa_prefill_validated_prefix(stub.data,stub.data,stub.data,stub.data,stub.data,stub.data,flag.data,
            good,cudaStreamPerThread,nullptr,1024,true,true,false,true))return 3;
        std::puts("query-pair-unsupported-policies-rejected=3");
        for(int group:{0,1,3,5})if(mollm_cuda::try_launch_sdpa_prefill_validated_prefix(stub.data,stub.data,stub.data,
            stub.data,stub.data,stub.data,flag.data,good,cudaStreamPerThread,nullptr,1024,true,false,false,true,group))return 3;
        std::puts("query-pair-invalid-groups-rejected=4");
        if(mollm_cuda::try_launch_sdpa_prefill_validated_prefix(stub.data,stub.data,stub.data,
            stub.data,stub.data,stub.data,flag.data,good,cudaStreamPerThread,nullptr,1024,true,false,false,true,query_group))return 3;
        std::puts("query-pair-writable-alias-rejected=1");
    }
    if(grouped_long_shared) {
        // Disjoint, fully sized operands: rejection must be due to policy,
        // not the tiny aliasing stubs used by the historical shape checks.
        Buffer<float> q(3072),out(3072),scores(36),mask(9);
        Buffer<__half> k(1536),v(1536);
        cudaGraph_t empty;check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
        auto reject_policy=[&](int limit,int group,bool paired,bool zero,bool unroll){
            cudaError_t error=cudaSuccess;
            if(mollm_cuda::try_launch_sdpa_prefill_validated_prefix(q.data,k.data,v.data,scores.data,out.data,
                mask.data,flag.data,good,cudaStreamPerThread,&error,limit,zero,unroll,false,paired,group,true)||
                error!=cudaSuccess)std::exit(3);
        };
        reject_policy(2049,4,true,true,false);reject_policy(4096,4,true,true,false);
        reject_policy(4097,2,true,true,false);reject_policy(2048,0,true,true,false);
        reject_policy(2048,3,true,true,false);reject_policy(2048,4,false,true,false);
        reject_policy(2048,4,true,false,false);reject_policy(2048,4,true,true,true);
        reject_policy(0,4,true,true,false);
        check(cudaStreamEndCapture(cudaStreamPerThread,&empty));size_t nodes=0;
        check(cudaGraphGetNodes(empty,nullptr,&nodes));if(nodes)return 3;check(cudaGraphDestroy(empty));
        std::puts("grouped-long-shared-policy-rejections=9 empty_capture=1 payload_bound32KiB=1");
    }
    if(grouped_heads) {
        Buffer<float> q(6144),out(6144),scores(72),mask(9);
        Buffer<__half> k(1536),v(1536);
        auto head_params=good;head_params.heads=8;
        cudaGraph_t empty;check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
        for(int policy=0;policy<7;++policy) {
            auto p=head_params;if(policy==0)p.heads=6;if(policy==1)p.kv_heads=0;
            if(mollm_cuda::try_launch_sdpa_prefill_validated_prefix(q.data,k.data,v.data,scores.data,out.data,
                mask.data,flag.data,p,cudaStreamPerThread,nullptr,1024,true,policy==6,policy==2,
                policy!=3,policy==4?2:4,policy==5,true))return 3;
        }
        check(cudaStreamEndCapture(cudaStreamPerThread,&empty));size_t nodes=0;
        check(cudaGraphGetNodes(empty,nullptr,&nodes));if(nodes)return 3;check(cudaGraphDestroy(empty));
        std::puts("grouped-head-policy-rejections=7 empty_capture=1");
    }
    for (int keys : {3,7,129,857,1024,1025}) for (bool interleaved : {false,true})
        if(!small_only||keys<=129||keys==1025)
        if (!run(std::min(17,keys),keys,interleaved,false)) return 1;
    if (long_shared || long_pv || pv_unroll || pair_pv || grouped_long_shared || grouped_heads)
        for (int keys : {1743,4096,4097}) for (bool interleaved : {false,true})
            if (!run(17,keys,interleaved,false)) return 1;
    if(grouped_long_shared)for(int keys:{2048,2049})for(bool interleaved:{false,true})
        if(!run(17,keys,interleaved,false))return 1;
    if(grouped_long_shared){
        query_group=2;
        for(int keys:{4096,4097})for(bool interleaved:{false,true})
            if(!run(17,keys,interleaved,false))return 1;
        query_group=4;
    }
    for (float scale : {2.f,INFINITY,NAN,-.0625f,0.f})
        if (!run(3,7,false,false,scale)) return 1;
    if(pair_queries) {
        if(!run(2,3,false,false)||!run(16,129,true,false))return 1;
        if(!small_only)for(bool interleaved:{false,true}) {
            if(!run(1024,1024,interleaved,true)||!run(719,1743,interleaved,true))return 1;
        }
        return 0;
    }
    for (bool interleaved : {false,true}) if (!run(857,857,interleaved,true)) return 1;
    if (long_shared || long_pv || pv_unroll || pair_pv)
        for (bool interleaved : {false,true}) if (!run(1743,1743,interleaved,true)) return 1;
    if (pv_unroll || pair_pv)
        for (bool interleaved : {false,true}) if (!run(719,1743,interleaved,true)) return 1;
}
