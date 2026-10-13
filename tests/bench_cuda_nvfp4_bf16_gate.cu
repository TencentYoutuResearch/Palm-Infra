// Bounded gate producer + activation/quantization consumer, not model TPS.
#include "kernels/cuda/moe.h"
#include "runtime/cuda_host_memory_budget.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

static bool compact_logits=false;
static bool token_quantization=false;
static bool scattered_token_inputs=false;
static bool paired_token_words=false;
static bool weight_codes_kmajor=false;
static bool token_codes_kmajor=false;
static bool hoist_token_sources=false;
static bool weight_locality=false;
static int locality_mask=0; // bit1: FP4 codes; bit2: K-major block scales.
static int token_row(int job,int tokens,int variant=0) {
    int stride=1;
    if(scattered_token_inputs){stride=37;while(std::gcd(stride,tokens)!=1||(tokens>2&&stride%tokens==1))++stride;}
    return (int64_t(job)*stride+variant*7)%tokens;
}
static void check(cudaError_t s){if(s!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(s));std::exit(1);}}
template<class T>struct Buffer{T* p=nullptr;explicit Buffer(size_t n){check(cudaMalloc(&p,n*sizeof(T)));}~Buffer(){cudaFree(p);}};
template<class T>static void copy(T* p,const std::vector<T>& v){check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
template<class T>static std::vector<T> read(T* p,size_t n){std::vector<T> v(n);check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
static uint16_t code(float x){uint32_t b;std::memcpy(&b,&x,4);if((b&0x7f800000u)!=0x7f800000u)b+=0x7fffu+((b>>16)&1u);return b>>16;}
static float from_bits(uint32_t b){float x;std::memcpy(&x,&b,4);return x;}
template<class T>static void guards(const std::vector<T>& v,size_t n){const auto* p=reinterpret_cast<const uint8_t*>(v.data()+n);for(size_t i=0;i<8*sizeof(T);++i)if(p[i]!=0x5a)std::exit(2);}
template<class T>static void equal(const std::vector<T>& a,const std::vector<T>& b){if(a.size()!=b.size()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T)))std::exit(3);}
static void scale_view_copy_check(){
    constexpr int experts=3;constexpr size_t bytes=experts*204800;
    Buffer<uint8_t> source(bytes+8),target(bytes+8);
    Buffer<mollm_cuda::Nvfp4Projection> tables(2*experts),cloned(2*experts+1);
    std::vector<uint8_t> raw(bytes+8,0x5a),expected(bytes+8,0x5a);
    std::vector<mollm_cuda::Nvfp4Projection> descriptors(2*experts);
    for(int e=0;e<2*experts;++e){
        descriptors[e].block_scales=source.p+size_t(e%experts)*204800;
        descriptors[e].weight=source.p+e;
        descriptors[e].pair_layout=e%2;
        descriptors[e].input_global=e+.25f;
        descriptors[e].input_global_second=e+.75f;
        descriptors[e].output_scale=e+1.f;
    }
    copy(tables.p,descriptors);
    auto expected_tables=descriptors;
    for(int e=0;e<experts;++e)expected_tables[e].block_scales=target.p+size_t(e)*204800;
    for(int replay=0;replay<2;++replay){
        for(size_t i=0;i<bytes;++i)raw[i]=uint8_t(i*37+(i/160)*11+replay*73);
        for(int e=0;e<experts;++e)for(int row=0;row<1280;++row)for(int g=0;g<40;++g)
            std::memcpy(expected.data()+size_t(e)*204800+(size_t(g)*1280+row)*4,
                raw.data()+size_t(e)*204800+size_t(row)*160+g*4,4);
        copy(source.p,raw);check(cudaMemset(target.p,0x5a,bytes+8));
        check(cudaMemset(cloned.p,0x5a,(2*experts+1)*sizeof(*cloned.p)));
        check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
        check(mollm_cuda::launch_nvfp4_gate_scale_view(tables.p,cloned.p,target.p,experts,cudaStreamPerThread));
        cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
        check(cudaStreamEndCapture(cudaStreamPerThread,&graph));
        check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
        for(int i=0;i<3;++i)check(cudaGraphLaunch(executable,cudaStreamPerThread));
        check(cudaStreamSynchronize(cudaStreamPerThread));
        equal(read(target.p,bytes+8),expected);equal(read(source.p,bytes+8),raw);
        equal(read(tables.p,2*experts),descriptors);equal(read(cloned.p,2*experts),expected_tables);
        const auto tail=read(reinterpret_cast<uint8_t*>(cloned.p+2*experts),sizeof(*cloned.p));
        for(auto byte:tail)if(byte!=0x5a)std::exit(2);
        check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));
    }
    check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
    for(int bad=0;bad<9;++bad){
        const auto status=mollm_cuda::launch_nvfp4_gate_scale_view(
            bad==0?nullptr:tables.p,bad==1?nullptr:bad==5?tables.p:cloned.p,
            bad==2?nullptr:bad==6?reinterpret_cast<uint8_t*>(cloned.p):bad==7?target.p+1:target.p,
            bad==3?0:bad==4?1025:bad==8?-1:experts,cudaStreamPerThread);
        if(status!=cudaErrorInvalidValue)std::exit(4);
    }
    cudaGraph_t rejected=nullptr;check(cudaStreamEndCapture(cudaStreamPerThread,&rejected));
    size_t nodes=0;check(cudaGraphGetNodes(rejected,nullptr,&nodes));if(nodes)std::exit(4);
    check(cudaGraphDestroy(rejected));
    std::printf("weight_sf_gpu_copy byte_exact=true descriptors_preserved=true guards=true capture=true rejects=9\n");
}
static double time_graph(cudaGraphExec_t graph){
    check(cudaGraphLaunch(graph,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
    cudaEvent_t a,b;check(cudaEventCreate(&a));check(cudaEventCreate(&b));check(cudaEventRecord(a,cudaStreamPerThread));
    for(int i=0;i<32;++i)check(cudaGraphLaunch(graph,cudaStreamPerThread));
    check(cudaEventRecord(b,cudaStreamPerThread));check(cudaEventSynchronize(b));float ms;check(cudaEventElapsedTime(&ms,a,b));
    check(cudaEventDestroy(a));check(cudaEventDestroy(b));return ms*1000/32;
}

static void consumer_edges(){
    constexpr int count=17,inner=640;constexpr size_t n=size_t(count)*inner;
    Buffer<float> gu(n*2),a(n+8),b(n+8);Buffer<uint16_t> codes(n*2);
    Buffer<uint8_t> pa(n/2+8),pb(n/2+8),sa(n/16+8),sb(n/16+8);
    Buffer<mollm_cuda::Nvfp4Projection> jobs(count);
    const uint32_t bits[]={0,0x80000000,1,0x80000001,0x3f808000,0x3f818000,0xbf808000,0xbf818000,
        0x007fffff,0x7f800000,0xff800000,0x7fc01234,0x7f800001,0xffffffff,0x3f800000};
    std::vector<float> x(n*2);std::vector<uint16_t> q(n*2);
    for(size_t i=0;i<x.size();++i){x[i]=from_bits(bits[i%15]);q[i]=code(x[i]);}
    copy(gu.p,x);copy(codes.p,q);
    int checked=0;
    for(int mode:{2,3}){
        std::vector<mollm_cuda::Nvfp4Projection> d(count);
        for(int j=0;j<count;++j){d[j].input=a.p+size_t((j*7)%count)*inner;d[j].input_global=.75f+j*.01f;}
        copy(jobs.p,d);check(cudaMemset(a.p,0x5a,(n+8)*4));check(cudaMemset(b.p,0x5a,(n+8)*4));
        for(auto p:{pa.p,pb.p})check(cudaMemset(p,0x5a,n/2+8));for(auto p:{sa.p,sb.p})check(cudaMemset(p,0x5a,n/16+8));
        check(mollm_cuda::launch_nvfp4_activation_prepare(jobs.p,count,inner,gu.p,a.p,pa.p,sa.p,mode,cudaStreamPerThread));
        for(int j=0;j<count;++j)d[j].input=b.p+size_t((j*7)%count)*inner;copy(jobs.p,d);
        check(mollm_cuda::launch_nvfp4_activation_prepare(jobs.p,count,inner,reinterpret_cast<const float*>(codes.p),b.p,pb.p,sb.p,mode,cudaStreamPerThread,true));
        const auto av=read(a.p,n+8),bv=read(b.p,n+8);equal(av,bv);guards(av,n);guards(bv,n);
        const auto pav=read(pa.p,n/2+8),pbv=read(pb.p,n/2+8),sav=read(sa.p,n/16+8),sbv=read(sb.p,n/16+8);
        equal(pav,pbv);equal(sav,sbv);guards(pav,n/2);guards(pbv,n/2);guards(sav,n/16);guards(sbv,n/16);
        if(mode==3){const auto* bytes=reinterpret_cast<const uint8_t*>(bv.data());for(size_t i=0;i<bv.size()*4;++i)if(bytes[i]!=0x5a)std::exit(4);}
        ++checked;
    }
    check(cudaMemset(b.p,0x5a,(n+8)*4));check(cudaMemset(pb.p,0x5a,n/2+8));check(cudaMemset(sb.p,0x5a,n/16+8));int rejected=0;
    auto reject=[&](cudaError_t e){if(e!=cudaErrorInvalidValue)std::exit(4);++rejected;};
    for(int mode:{-1,0,1,4})reject(mollm_cuda::launch_nvfp4_activation_prepare(jobs.p,count,inner,reinterpret_cast<const float*>(codes.p),b.p,pb.p,sb.p,mode,cudaStreamPerThread,true));
    reject(mollm_cuda::launch_nvfp4_activation_prepare(jobs.p,count,1280,reinterpret_cast<const float*>(codes.p),b.p,pb.p,sb.p,2,cudaStreamPerThread,true));
    reject(mollm_cuda::launch_nvfp4_activation_prepare(jobs.p,count,inner,reinterpret_cast<const float*>(reinterpret_cast<uint8_t*>(codes.p)+1),b.p,pb.p,sb.p,2,cudaStreamPerThread,true));
    reject(mollm_cuda::launch_nvfp4_activation_prepare(nullptr,count,inner,reinterpret_cast<const float*>(codes.p),b.p,pb.p,sb.p,2,cudaStreamPerThread,true));
    reject(mollm_cuda::launch_nvfp4_activation_prepare(jobs.p,count,inner,nullptr,b.p,pb.p,sb.p,2,cudaStreamPerThread,true));
    reject(mollm_cuda::launch_nvfp4_activation_prepare(jobs.p,count,inner,reinterpret_cast<const float*>(codes.p),nullptr,pb.p,sb.p,2,cudaStreamPerThread,true));
    reject(mollm_cuda::launch_nvfp4_activation_prepare(jobs.p,count,inner,reinterpret_cast<const float*>(codes.p),b.p,nullptr,sb.p,2,cudaStreamPerThread,true));
    reject(mollm_cuda::launch_nvfp4_activation_prepare(jobs.p,count,inner,reinterpret_cast<const float*>(codes.p),b.p,pb.p,nullptr,2,cudaStreamPerThread,true));
    const auto after=read(reinterpret_cast<uint8_t*>(b.p),(n+8)*4);for(auto v:after)if(v!=0x5a)std::exit(4);
    for(auto v:read(pb.p,n/2+8))if(v!=0x5a)std::exit(4);for(auto v:read(sb.p,n/16+8))if(v!=0x5a)std::exit(4);
    std::printf("bf16_gate_consumer edge_cases=%d rejected=%d full_activation_packed_scales_bytes=true packed_only_untouched=true\n",checked,rejected);
}

static void run(const std::vector<int>& runs,bool pair,bool timing,bool early,bool vector,bool fused,bool shared_inputs,bool stage_inputs,int input_pitch,bool pad_logits,bool skew_weights,bool preplanned,bool weight_sf,bool input_sf){
    constexpr int h=1280,k=2560,inner=640,sets=2;
    const int experts=runs.size(),count=std::accumulate(runs.begin(),runs.end(),0);
    const int token_count=token_quantization?(count+9)/10:0;
    Buffer<int> token_indices(count+8);
    check(cudaMemset(token_indices.p,0x5a,(count+8)*sizeof(int)));
    const size_t values=size_t(count)*h,act_values=size_t(count)*inner;
    constexpr size_t wbytes=size_t(h)*k/2,sbytes=size_t(h)*k/16,stride=wbytes+sbytes+h*4;
    Buffer<uint8_t> weights(size_t(sets)*experts*stride),packed(size_t(count)*k+8),scales(size_t(count)*k/8+8);
    Buffer<float> input(size_t(count)*k),fp(values+8),a(act_values+8),b(act_values+8);
    Buffer<uint16_t> bf(values+8);
    Buffer<uint8_t> pa(act_values/2+8),pb(act_values/2+8),sa(act_values/16+8),sb(act_values/16+8);
    Buffer<mollm_cuda::Nvfp4Projection> jobs(size_t(sets)*count*2),down(count*2);
    Buffer<uint8_t> sf_view(weight_sf?size_t(sets)*experts*sbytes+8:8);
    Buffer<mollm_cuda::Nvfp4Projection> sf_jobs(weight_sf?size_t(sets)*count:1);
    Buffer<uint8_t> code_view(weight_codes_kmajor?size_t(sets)*experts*wbytes+8:8);
    Buffer<mollm_cuda::Nvfp4Projection> code_jobs(weight_codes_kmajor?size_t(sets)*count:1);
    std::vector<uint8_t> code_host;
    Buffer<mollm_cuda::Nvfp4Projection> locality_jobs(weight_locality?size_t(sets)*count:1);
    Buffer<mollm_cuda::Nvfp4MmaBatch> plan(count+1);
    std::vector<mollm_cuda::Nvfp4MmaBatch> compact_host;
    int route_start=0;
    for(int n:runs) {
        for(int offset=0;offset<n;offset+=32)compact_host.push_back({route_start+offset,std::min(32,n-offset)});
        route_start+=n;
    }
    if(route_start!=count||compact_host.empty())std::exit(4);
    const int compact_count=compact_host.size();
    compact_host.push_back({0x5a5a5a5a,0x5a5a5a5a});
    Buffer<mollm_cuda::Nvfp4MmaBatch> compact(compact_host.size());
    copy(compact.p,compact_host);
    std::vector<uint8_t> payload(size_t(sets)*experts*stride);
    for(int expert=0;expert<sets*experts;++expert){auto* p=payload.data()+size_t(expert)*stride;
        for(int row=0;row<h;++row){const float global=(1+(row+expert)%7)/64.f;std::memcpy(p+wbytes+sbytes+row*4,&global,4);
            for(int group=0;group<k/16;++group){const uint8_t codes[]={0,1,7,8,48,56,64,126};p[wbytes+size_t(row)*160+group]=codes[(row+group+expert)%8];
                for(int byte=0;byte<8;++byte){const size_t offset=pair?size_t(row/4)*5120+(row%4)*8+group*32+byte:size_t(row)*1280+group*8+byte;p[offset]=(row*29+group*17+byte*13+expert*7)&255;}
            }
        }
    }
    // Counterfactual ONLY: identical bytes at distinct expert addresses.
    // Keep each expert's row-scale pointer unique even in the aliased variant
    // so same_nvfp4_expert and every batch boundary remain unchanged.
    if(weight_locality)for(int set=0;set<sets;++set)for(int expert=1;expert<experts;++expert)
        std::memcpy(payload.data()+size_t(set*experts+expert)*stride,
            payload.data()+size_t(set*experts)*stride,stride);
    std::vector<uint8_t> sf_host;
    if(weight_sf) {
        sf_host.resize(size_t(sets)*experts*sbytes+8,0x5a);
        for(int expert=0;expert<sets*experts;++expert)for(int row=0;row<h;++row)
            for(int q=0;q<40;++q)std::memcpy(sf_host.data()+size_t(expert)*sbytes+(size_t(q)*h+row)*4,
                payload.data()+size_t(expert)*stride+wbytes+size_t(row)*160+q*4,4);
        copy(sf_view.p,sf_host);
    }
    if(weight_codes_kmajor) {
        code_host.resize(size_t(sets)*experts*wbytes+8,0x5a);
        const int blocks=pair?h/4:h, bytes=pair?128:32;
        for(int expert=0;expert<sets*experts;++expert)for(int q=0;q<40;++q)for(int row=0;row<blocks;++row)
            std::memcpy(code_host.data()+size_t(expert)*wbytes+(size_t(q)*blocks+row)*bytes,
                payload.data()+size_t(expert)*stride+(size_t(row)*40+q)*bytes,bytes);
    }
    copy(weights.p,payload);payload.clear();payload.shrink_to_fit();
    if(weight_codes_kmajor) {
        check(cudaMemset(code_view.p,0x5a,code_host.size()));
        for(int expert=0;expert<sets*experts;++expert) {
            mollm_cuda::Nvfp4Projection source{};
            source.weight=weights.p+size_t(expert)*stride;
            source.pair_layout=pair;
            check(mollm_cuda::launch_repack_nvfp4_gate_codes(source,
                code_view.p+size_t(expert)*wbytes,h,k));
        }
        equal(read(code_view.p,code_host.size()),code_host);
    }
    std::vector<mollm_cuda::Nvfp4Projection> descriptors(size_t(sets)*count*2),consumers(count*2);
    int j=0;for(int expert=0;expert<experts;++expert)for(int token=0;token<runs[expert];++token,++j){
        const size_t route=count==114?(size_t(j)*7)%count:size_t(j);
        for(int set=0;set<sets;++set){const auto* p=weights.p+size_t(set*experts+expert)*stride;
            const size_t input_row=token_quantization?token_row(j,token_count):shared_inputs?size_t(j)%(count/10):size_t(j);
            auto d=mollm_cuda::Nvfp4Projection{p,p+wbytes,reinterpret_cast<const float*>(p+wbytes+sbytes),input.p+input_row*k,fp.p+route*h,pair,false,false,.75f+expert*.001f,1.25f+expert*.001f,(j+1)/float(count)};
            if(token_quantization)d.input_global=d.input_global_second=.75f;
            descriptors[set*count+j]=d;d.output=reinterpret_cast<float*>(bf.p+route*h);descriptors[(sets+set)*count+j]=d;
        }
        consumers[j].input=a.p+route*inner;consumers[j].input_global=.75f+expert*.001f;
        consumers[count+j]=consumers[j];consumers[count+j].input=b.p+route*inner;
    }
    copy(jobs.p,descriptors);copy(down.p,consumers);
    if(weight_sf) {
        std::vector<mollm_cuda::Nvfp4Projection> sf_descriptors(size_t(sets)*count);
        int route=0;
        for(int expert=0;expert<experts;++expert)for(int token=0;token<runs[expert];++token,++route)
            for(int set=0;set<sets;++set) {
                auto d=descriptors[size_t(sets+set)*count+route];
                d.block_scales=sf_view.p+size_t(set*experts+expert)*sbytes;
                sf_descriptors[size_t(set)*count+route]=d;
            }
        copy(sf_jobs.p,sf_descriptors);
        if(weight_codes_kmajor) {
            auto code_descriptors=sf_descriptors;
            int route=0;
            for(int expert=0;expert<experts;++expert)for(int token=0;token<runs[expert];++token,++route)
                for(int set=0;set<sets;++set)
                    code_descriptors[size_t(set)*count+route].weight=code_view.p+size_t(set*experts+expert)*wbytes;
            copy(code_jobs.p,code_descriptors);
        }
        if(weight_locality){
            for(int set=0;set<sets;++set)for(int route=0;route<count;++route){
                auto& d=sf_descriptors[size_t(set)*count+route];
                if(locality_mask&1)d.weight=weights.p+size_t(set*experts)*stride;
                if(locality_mask&2)d.block_scales=sf_view.p+size_t(set*experts)*sbytes;
                // row_scales deliberately NOT aliased: retain expert identity.
            }
            copy(locality_jobs.p,sf_descriptors);
        }
    }
    auto producer=[&](bool compressed,int set,bool ahead){return mollm_cuda::launch_nvfp4_mma_grouped(jobs.p+(set+(compressed?sets:0))*count,count,plan.p,h,k,inner,cudaStreamPerThread,packed.p,scales.p,1,true,false,nullptr,0,32,false,false,true,compressed,vector?compressed&&pair:ahead,vector,vector&&compressed&&ahead);};
    check(cudaMemset(bf.p,0x5a,(values+8)*2));check(cudaMemset(plan.p,0x5a,(count+1)*sizeof(*plan.p)));
    int rejects=0;for(int id=0;id<12;++id){
        const auto status=mollm_cuda::launch_nvfp4_mma_grouped(jobs.p+sets*count,count,plan.p,
            id==0?2560:h,id==1?640:k,id==2?0:inner,cudaStreamPerThread,packed.p,scales.p,
            id==3?2:1,id!=4,id==5,id==6?plan.p:nullptr,0,id==7?8:32,id==8,id==9,id!=10,true);
        if(status!=cudaErrorInvalidValue)std::exit(4);++rejects;
        // id11 explicitly exercises a non-four-warp environment on next call.
        if(id==10)setenv("MOLLM_CUDA_NVFP4_MMA_BATCH_WARPS","8",1);
    }
    setenv("MOLLM_CUDA_NVFP4_MMA_BATCH_WARPS","4",1);
    if(early){
        for(int id=0;id<12;++id){
            const auto status=mollm_cuda::launch_nvfp4_mma_grouped(jobs.p+sets*count,count,plan.p,
                id==0?2560:h,id==1?640:k,id==2?0:inner,cudaStreamPerThread,packed.p,scales.p,
                id==3?2:1,id!=4,id==5,id==6?plan.p:nullptr,0,id==7?8:32,id==8,id==9,id!=10,id!=11,true);
            if(status!=cudaErrorInvalidValue)std::exit(4);++rejects;
        }
    }
    if(vector){
        for(int id=0;id<18;++id){
            if(id==17)setenv("MOLLM_CUDA_NVFP4_MMA_BATCH_WARPS","8",1);
            const auto status=mollm_cuda::launch_nvfp4_mma_grouped(
                id==12?nullptr:jobs.p+sets*count,id==16?0:count,id==13?nullptr:plan.p,
                id==0?2560:h,id==1?640:k,id==2?0:inner,cudaStreamPerThread,
                id==14?nullptr:packed.p,id==15?nullptr:scales.p,id==3?2:1,
                id!=4,id==5,id==6?plan.p:nullptr,0,id==7?8:32,
                id==8,id==9,id!=10,id!=11,false,false,true);
            if(status!=cudaErrorInvalidValue)std::exit(4);++rejects;
        }
        setenv("MOLLM_CUDA_NVFP4_MMA_BATCH_WARPS","4",1);
    }
    if(fused)for(int id=0;id<11;++id){
        const auto status=mollm_cuda::launch_nvfp4_mma_activation_diagnostic(
            id==0?nullptr:jobs.p+sets*count,id==1?nullptr:down.p+count,
            id==8?0:id==9?-1:id==10?65536:count,id==2?nullptr:plan.p,
            id==3?nullptr:packed.p,id==4?nullptr:scales.p,id==5?nullptr:b.p,
            id==6?nullptr:pb.p,id==7?nullptr:sb.p,true,cudaStreamPerThread,stage_inputs,input_pitch,pad_logits,skew_weights,nullptr,0,weight_sf,input_sf);
        if(status!=cudaErrorInvalidValue)std::exit(4);++rejects;
    }
    if(fused){
        cudaGraph_t empty;
        check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
        if(weight_codes_kmajor) {
            const auto status=mollm_cuda::launch_nvfp4_mma_activation_diagnostic(
                sf_jobs.p,down.p+count,count,plan.p,packed.p,scales.p,b.p,pb.p,sb.p,
                false,cudaStreamPerThread,false,32,false,false,nullptr,0,true,true,true,
                input.p,token_count,.75f,token_indices.p,false,true);
            if(status!=cudaErrorInvalidValue)std::exit(4);++rejects;
        }
        if(token_codes_kmajor)for(int invalid=0;invalid<2;++invalid) {
            const auto status=mollm_cuda::launch_nvfp4_mma_activation_diagnostic(
                sf_jobs.p,down.p+count,count,plan.p,packed.p,scales.p,b.p,pb.p,sb.p,
                false,cudaStreamPerThread,false,32,false,false,nullptr,0,true,true,true,
                input.p,token_count,.75f,token_indices.p,invalid==0,invalid!=0,true);
            if(status!=cudaErrorInvalidValue)std::exit(4);++rejects;
        }
        if(hoist_token_sources) {
            const auto status=mollm_cuda::launch_nvfp4_mma_activation_diagnostic(
                sf_jobs.p,down.p+count,count,plan.p,packed.p,scales.p,b.p,pb.p,sb.p,
                false,cudaStreamPerThread,false,32,false,false,nullptr,0,true,true,true,
                input.p,token_count,.75f,token_indices.p,true,true,false,true);
            if(status!=cudaErrorInvalidValue)std::exit(4);++rejects;
        }
        if(token_quantization)for(int invalid=0;invalid<10;++invalid) {
            const auto status=mollm_cuda::launch_nvfp4_mma_activation_diagnostic(
                sf_jobs.p,down.p+count,count,plan.p,packed.p,scales.p,b.p,pb.p,sb.p,
                false,cudaStreamPerThread,false,32,false,false,nullptr,0,true,true,invalid!=0,
                invalid==1?nullptr:invalid==2?input.p+1:invalid==3?b.p:input.p,
                invalid==4?0:invalid==5?2049:token_count,invalid==6?0.f:invalid==7?from_bits(0x7fc00000):.75f,
                invalid==8?nullptr:invalid==9?reinterpret_cast<int*>(packed.p):token_indices.p);
            if(status!=cudaErrorInvalidValue)std::exit(4);++rejects;
        }
        if(paired_token_words)for(int invalid=0;invalid<2;++invalid) {
            const auto status=mollm_cuda::launch_nvfp4_mma_activation_diagnostic(
                sf_jobs.p,down.p+count,count,plan.p,invalid?packed.p+4:packed.p,scales.p,b.p,pb.p,sb.p,
                false,cudaStreamPerThread,false,32,false,false,nullptr,0,true,true,true,
                invalid?input.p:nullptr,invalid?token_count:0,.75f,invalid?token_indices.p:nullptr,true);
            if(status!=cudaErrorInvalidValue)std::exit(4);++rejects;
        }
        if(compact_logits)for(int invalid=0;invalid<6;++invalid) {
            const auto status=mollm_cuda::launch_nvfp4_mma_activation_diagnostic(
                sf_jobs.p,down.p+count,count,plan.p,packed.p,scales.p,b.p,pb.p,sb.p,
                false,cudaStreamPerThread,invalid==0,32,invalid==1,invalid==2,
                invalid==3?compact.p:nullptr,invalid==3?compact_count:0,
                invalid!=4,invalid!=5,true);
            if(status!=cudaErrorInvalidValue)std::exit(4);++rejects;
        }
        if(weight_sf&&input_sf) {
            const auto status=mollm_cuda::launch_nvfp4_mma_activation_diagnostic(
                sf_jobs.p,down.p+count,count,plan.p,packed.p,scales.p,
                b.p,pb.p,sb.p,false,cudaStreamPerThread,true,48,
                false,false,nullptr,0,true,true);
            if(status!=cudaErrorInvalidValue)std::exit(4);++rejects;
        }
        if(input_sf)for(int combination=0;combination<4;++combination) {
            const auto status=mollm_cuda::launch_nvfp4_mma_activation_diagnostic(
                jobs.p+sets*count,down.p+count,count,plan.p,packed.p,scales.p,
                b.p,pb.p,sb.p,false,cudaStreamPerThread,combination==0,32,
                combination==1,combination==2,combination==3?compact.p:nullptr,
                combination==3?compact_count:0,false,true);
            if(status!=cudaErrorInvalidValue)std::exit(4);++rejects;
        }
        if(weight_sf)for(int combination=0;combination<4;++combination) {
            const auto status=mollm_cuda::launch_nvfp4_mma_activation_diagnostic(
                sf_jobs.p,count?down.p+count:nullptr,count,plan.p,packed.p,scales.p,
                b.p,pb.p,sb.p,false,cudaStreamPerThread,combination==0,32,
                combination==1,combination==2,combination==3?compact.p:nullptr,
                combination==3?compact_count:0,true);
            if(status!=cudaErrorInvalidValue)std::exit(4);++rejects;
        }
        if(preplanned)for(int id=0;id<7;++id) {
            const auto* table=id==0?nullptr:id==4?plan.p:id==5?
                reinterpret_cast<const mollm_cuda::Nvfp4MmaBatch*>(packed.p+1):id==6?
                reinterpret_cast<const mollm_cuda::Nvfp4MmaBatch*>(UINTPTR_MAX-3):compact.p;
            const int n=id==0?1:id==1?0:id==2?-1:id==3?count+1:compact_count;
            const auto status=mollm_cuda::launch_nvfp4_mma_activation_diagnostic(
                jobs.p+sets*count,down.p+count,count,plan.p,packed.p,scales.p,
                b.p,pb.p,sb.p,false,cudaStreamPerThread,false,32,false,false,table,n);
            if(status!=cudaErrorInvalidValue)std::exit(4);++rejects;
        }
        if(skew_weights)for(int combination:{0,1}) {
            const auto status=mollm_cuda::launch_nvfp4_mma_activation_diagnostic(
                jobs.p+sets*count,down.p+count,count,plan.p,packed.p,scales.p,
                b.p,pb.p,sb.p,false,cudaStreamPerThread,combination==0,32,combination==1,true);
            if(status!=cudaErrorInvalidValue)std::exit(4);++rejects;
        }
        if(pad_logits) {
            const auto status=mollm_cuda::launch_nvfp4_mma_activation_diagnostic(
                jobs.p+sets*count,down.p+count,count,plan.p,packed.p,scales.p,
                b.p,pb.p,sb.p,false,cudaStreamPerThread,true,32,true);
            if(status!=cudaErrorInvalidValue)std::exit(4);++rejects;
        }
        if(stage_inputs)for(int invalid_pitch:{0,16,64}) {
            const auto status=mollm_cuda::launch_nvfp4_mma_activation_diagnostic(
                jobs.p+sets*count,down.p+count,count,plan.p,packed.p,scales.p,
                b.p,pb.p,sb.p,false,cudaStreamPerThread,true,invalid_pitch);
            if(status!=cudaErrorInvalidValue)std::exit(4);++rejects;
        }
        for(int id=0;id<8;++id){
            const auto status=mollm_cuda::launch_nvfp4_mma_activation_diagnostic(
                jobs.p+sets*count,down.p+count,count,plan.p,
                id==4?packed.p+1:packed.p,id==5?scales.p+1:scales.p,
                id==3?reinterpret_cast<float*>(plan.p):b.p,
                id==0?packed.p:id==1?packed.p+8:id==6?reinterpret_cast<uint8_t*>(UINTPTR_MAX-15):pb.p,
                id==2?scales.p:id==7?reinterpret_cast<uint8_t*>(jobs.p+sets*count):sb.p,
                false,cudaStreamPerThread,stage_inputs,input_pitch,pad_logits,skew_weights,nullptr,0,weight_sf,input_sf);
            if(status!=cudaErrorInvalidValue)std::exit(4);++rejects;
        }
        check(cudaStreamEndCapture(cudaStreamPerThread,&empty));size_t nodes=0;
        check(cudaGraphGetNodes(empty,nullptr,&nodes));if(nodes)std::exit(4);
        check(cudaGraphDestroy(empty));
    }
    for(auto v:read(bf.p,values+8))if(v!=0x5a5a)std::exit(4);
    for(auto v:read(reinterpret_cast<uint8_t*>(plan.p),(count+1)*sizeof(*plan.p)))if(v!=0x5a)std::exit(4);
    auto chain=[&](bool compressed,int set,int mode,bool ahead=false,bool allow_fused=true,bool staged=true,bool alias_weights=false){
        const bool staged_sf=(stage_inputs&&weight_sf&&input_sf)||compact_logits;
        if(compressed&&fused&&allow_fused){
            check(mollm_cuda::launch_nvfp4_mma_activation_diagnostic(
                weight_codes_kmajor&&(staged||token_codes_kmajor)?code_jobs.p+size_t(set)*count:alias_weights?locality_jobs.p+size_t(set)*count:weight_sf&&(staged||input_sf||staged_sf)?sf_jobs.p+size_t(set)*count:jobs.p+(sets+set)*count,down.p+count,count,plan.p,packed.p,scales.p,
                b.p,pb.p,sb.p,mode==2,cudaStreamPerThread,stage_inputs&&staged,input_pitch,pad_logits&&staged,skew_weights&&staged,
                preplanned&&staged?compact.p:nullptr,preplanned&&staged?compact_count:0,
                weight_sf&&(staged||input_sf||staged_sf),input_sf&&(staged||staged_sf),compact_logits&&(staged||token_quantization),
                token_quantization&&(staged||paired_token_words)?input.p:nullptr,
                token_quantization&&(staged||paired_token_words)?token_count:0,.75f,
                token_quantization&&(staged||paired_token_words)?token_indices.p:nullptr,
                paired_token_words&&(staged||weight_codes_kmajor),weight_codes_kmajor&&(staged||token_codes_kmajor),token_codes_kmajor&&(staged||hoist_token_sources),hoist_token_sources&&staged));return;
        }
        check(producer(compressed,set,ahead));
        check(mollm_cuda::launch_nvfp4_activation_prepare(down.p+(compressed?count:0),count,inner,
            compressed?reinterpret_cast<const float*>(bf.p):fp.p,compressed?b.p:a.p,
            compressed?pb.p:pa.p,compressed?sb.p:sa.p,mode,cudaStreamPerThread,compressed));
    };
    int checked=0;
    for(int set=0;set<sets;++set)for(int mode:{2,3}){
        cudaGraph_t graph;cudaGraphExec_t replay;check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));chain(true,set,mode,early||vector);
        check(cudaStreamEndCapture(cudaStreamPerThread,&graph));check(cudaGraphInstantiateWithFlags(&replay,graph,0));check(cudaGraphDestroy(graph));
        for(int variant=0;variant<4;++variant){
            std::vector<float> x(size_t(count)*k);for(size_t i=0;i<x.size();++i)x[i]=variant==2?0.f:variant==3?-0.f:(int((i*13+variant*11+set)%101)-50)/37.f;
            copy(input.p,x);
            if(token_quantization) {
                for(int set_index=0;set_index<sets*2;++set_index)for(int job_index=0;job_index<count;++job_index)
                    descriptors[size_t(set_index)*count+job_index].input=input.p+size_t(token_row(job_index,token_count,variant))*k;
                copy(jobs.p,descriptors);
                auto current=read(sf_jobs.p,size_t(sets)*count);
                for(int set_index=0;set_index<sets;++set_index)for(int job_index=0;job_index<count;++job_index)
                    current[size_t(set_index)*count+job_index].input=input.p+size_t(token_row(job_index,token_count,variant))*k;
                copy(sf_jobs.p,current);
                if(weight_codes_kmajor) {
                    auto codes=read(code_jobs.p,size_t(sets)*count);
                    for(int set_index=0;set_index<sets;++set_index)for(int job_index=0;job_index<count;++job_index)
                        codes[size_t(set_index)*count+job_index].input=input.p+size_t(token_row(job_index,token_count,variant))*k;
                    copy(code_jobs.p,codes);
                }
                if(weight_locality){
                    auto aliases=read(locality_jobs.p,size_t(sets)*count);
                    for(int set_index=0;set_index<sets;++set_index)for(int job_index=0;job_index<count;++job_index)
                        aliases[size_t(set_index)*count+job_index].input=input.p+size_t(token_row(job_index,token_count,variant))*k;
                    copy(locality_jobs.p,aliases);
                }
            }
            check(cudaMemset(fp.p,0x5a,(values+8)*4));check(cudaMemset(bf.p,0x5a,(values+8)*2));
            for(auto p:{a.p,b.p})check(cudaMemset(p,0x5a,(act_values+8)*4));
            for(auto p:{pa.p,pb.p})check(cudaMemset(p,0x5a,act_values/2+8));for(auto p:{sa.p,sb.p})check(cudaMemset(p,0x5a,act_values/16+8));
            check(cudaMemset(packed.p,0x5a,size_t(count)*k+8));
            check(cudaMemset(scales.p,0x5a,size_t(count)*k/8+8));
            chain(false,set,mode);check(cudaStreamSynchronize(cudaStreamPerThread));
            std::vector<uint8_t> input_p,input_s;
            if(input_sf){input_p=read(packed.p,size_t(count)*k+8);input_s=read(scales.p,size_t(count)*k/8+8);}
            if(token_quantization) {
                check(cudaMemset(packed.p,0x5a,size_t(count)*k+8));
                check(cudaMemset(scales.p,0x5a,size_t(count)*k/8+8));
                check(cudaMemset(token_indices.p,0x5a,(count+8)*sizeof(int)));
            }
            check(cudaGraphLaunch(replay,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
            if(token_quantization) {
                std::vector<uint8_t> expected_p(size_t(count)*k+8,0x5a),expected_s(size_t(count)*k/8+8,0x5a);
                std::vector<int> representative(token_count);
                for(int job_index=0;job_index<token_count;++job_index)representative[token_row(job_index,token_count,variant)]=job_index;
                for(int token=0;token<token_count;++token) {
                    const int original=representative[token];
                    if(!token_codes_kmajor)std::memcpy(expected_p.data()+size_t(token)*1280,input_p.data()+size_t(original)*1280,1280);
                    if(paired_token_words&&!token_codes_kmajor)for(int block=0;block<40;++block)for(int word=0;word<8;++word)
                        std::memcpy(expected_p.data()+size_t(token)*1280+block*32+((word%4)*2+word/4)*4,
                            input_p.data()+size_t(original)*1280+block*32+word*4,4);
                    if(token_codes_kmajor)for(int block=0;block<40;++block) {
                        uint8_t tile[32];
                        for(int word=0;word<8;++word)std::memcpy(tile+((word%4)*2+word/4)*4,
                            input_p.data()+size_t(original)*1280+block*32+word*4,4);
                        std::memcpy(expected_p.data()+(size_t(block)*token_count+token)*32,tile,32);
                    }
                    for(int block=0;block<40;++block)std::memcpy(expected_s.data()+(size_t(block)*token_count+token)*4,
                        input_s.data()+size_t(original)*160+block*4,4);
                }
                equal(read(packed.p,expected_p.size()),expected_p);equal(read(scales.p,expected_s.size()),expected_s);
                std::vector<int> expected_indices(count+8,0x5a5a5a5a);
                for(int job_index=0;job_index<count;++job_index)expected_indices[job_index]=token_row(job_index,token_count,variant);
                equal(read(token_indices.p,count+8),expected_indices);
            } else if(input_sf){
                equal(read(packed.p,size_t(count)*k+8),input_p);
                auto expected=input_s;
                for(int q=0;q<40;++q)for(int source=0;source<count*2;++source)
                    std::memcpy(expected.data()+(size_t(q)*count*2+source)*4,
                        input_s.data()+size_t(source)*160+q*4,4);
                equal(read(scales.p,size_t(count)*k/8+8),expected);
            }
            const auto f=read(fp.p,values+8);const auto q=read(bf.p,values+8);guards(f,values);guards(q,values);
            if(fused){for(auto v:q)if(v!=0x5a5a)std::exit(5);}
            else for(size_t i=0;i<values;++i)if(code(f[i])!=q[i]){std::fprintf(stderr,"gate code mismatch pair=%d set=%d mode=%d variant=%d index=%zu\n",pair,set,mode,variant,i);std::exit(5);}
            const auto av=read(a.p,act_values+8),bv=read(b.p,act_values+8);equal(av,bv);guards(av,act_values);guards(bv,act_values);
            if(mode==3){const auto* bytes=reinterpret_cast<const uint8_t*>(bv.data());for(size_t i=0;i<bv.size()*4;++i)if(bytes[i]!=0x5a)std::exit(4);}
            const auto pav=read(pa.p,act_values/2+8),pbv=read(pb.p,act_values/2+8),sav=read(sa.p,act_values/16+8),sbv=read(sb.p,act_values/16+8);
            equal(pav,pbv);equal(sav,sbv);guards(pav,act_values/2);guards(pbv,act_values/2);guards(sav,act_values/16);guards(sbv,act_values/16);++checked;
            const auto compact_after=read(compact.p,compact_host.size());
            if(std::memcmp(compact_after.data(),compact_host.data(),compact_host.size()*sizeof(*compact.p)))std::exit(4);
            if(preplanned) {
                const auto sparse_after=read(plan.p,count);
                size_t ordinal=0;
                for(const auto& batch:sparse_after)if(batch.first>=0) {
                    if(ordinal>=size_t(compact_count)||std::memcmp(&batch,&compact_host[ordinal],sizeof(batch)))std::exit(4);
                    ++ordinal;
                }
                if(ordinal!=size_t(compact_count))std::exit(4);
            }
        }
        check(cudaGraphExecDestroy(replay));
    }
    std::printf("bf16_gate pair=%d jobs=%d captured_replays=%d rejects=%d fused=%d shared_inputs=%d stage_inputs=%d input_pitch=%d pad_logits=%d skew_weights=%d gate_intermediate_checked=%s full_activation_packed_scales_bytes=true guards=true\n",pair,count,checked,rejects,fused,shared_inputs,stage_inputs,input_pitch,pad_logits,skew_weights,fused?"untouched":"rounded_codes");std::fflush(stdout);
    if(weight_sf) {
        const auto after=read(sf_view.p,sf_host.size());equal(after,sf_host);
        std::printf("weight_sf_oracle pair=%d jobs=%d sidecar_bytes=%zu all_view_bytes_unchanged=true construction_excluded=true\n",pair,count,sf_host.size()-8);
    }
    if(weight_locality){
        std::vector<float> x(size_t(count)*k);
        cudaGraphExec_t graphs[2];
        for(int alias=0;alias<2;++alias){cudaGraph_t graph;
            check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
            for(int set=0;set<sets;++set)chain(true,set,3,true,true,true,alias);
            check(cudaStreamEndCapture(cudaStreamPerThread,&graph));
            check(cudaGraphInstantiateWithFlags(&graphs[alias],graph,0));check(cudaGraphDestroy(graph));
        }
        for(int changed=0;changed<3;++changed){
            for(size_t i=0;i<x.size();++i)x[i]=(int((i+changed*7)%101)-50)/37.f;
            copy(input.p,x);
            for(int set=0;set<sets;++set)for(int mode:{2,3}){
                check(cudaMemset(b.p,0x5a,(act_values+8)*4));
                check(cudaMemset(pb.p,0x5a,act_values/2+8));check(cudaMemset(sb.p,0x5a,act_values/16+8));
                chain(true,set,mode,true,true,true,false);check(cudaStreamSynchronize(cudaStreamPerThread));
                const auto output=read(b.p,act_values+8);
                const auto p=read(pb.p,act_values/2+8),s=read(sb.p,act_values/16+8);
                const auto batches=read(reinterpret_cast<uint8_t*>(plan.p),(count+1)*sizeof(*plan.p));
                chain(true,set,mode,true,true,true,true);check(cudaStreamSynchronize(cudaStreamPerThread));
                equal(output,read(b.p,act_values+8));equal(p,read(pb.p,act_values/2+8));equal(s,read(sb.p,act_values/16+8));
                equal(batches,read(reinterpret_cast<uint8_t*>(plan.p),(count+1)*sizeof(*plan.p)));
                guards(output,act_values);guards(p,act_values/2);guards(s,act_values/16);
            }
            check(cudaGraphLaunch(graphs[0],cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
            const auto p=read(pb.p,act_values/2+8),s=read(sb.p,act_values/16+8);
            const auto batches=read(reinterpret_cast<uint8_t*>(plan.p),(count+1)*sizeof(*plan.p));
            check(cudaGraphLaunch(graphs[1],cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
            equal(p,read(pb.p,act_values/2+8));equal(s,read(sb.p,act_values/16+8));
            equal(batches,read(reinterpret_cast<uint8_t*>(plan.p),(count+1)*sizeof(*plan.p)));
        }
        std::printf("bf16_gate_weight_locality_check pair=%d jobs=%d experts=%d locality_mask=%d changed_inputs=3 sets=%d direct_modes=2,3 full_activation_packed_scales_plan_and_guards_byte_exact=true captured_packed_scales_plan_byte_exact=true row_scale_identity_preserved=true scope=counterfactual_identical_weights_not_model\n",pair,count,experts,locality_mask,sets);std::fflush(stdout);
        if(timing)for(int repeat=0;repeat<7;++repeat){double us[2];
            for(int order=0;order<2;++order){const int mode=order^(repeat&1);us[mode]=time_graph(graphs[mode])/sets;}
            std::printf("bf16_gate_weight_locality_timing pair=%d jobs=%d experts=%d locality_mask=%d repeat=%d distinct_us=%.6f reused_us=%.6f scope=same_kernel_routes_batches_arithmetic_counterfactual_not_model\n",pair,count,experts,locality_mask,repeat,us[0],us[1]);std::fflush(stdout);
        }
        for(auto graph:graphs)check(cudaGraphExecDestroy(graph));
        equal(read(sf_view.p,sf_host.size()),sf_host);
        return;
    }
    if(timing){
        if(preplanned)std::printf("preplanned_oracle routes=%d batches=%d expert_order_preserved=true construction_excluded=true\n",count,compact_count);
        std::vector<float> x(size_t(count)*k);for(size_t i=0;i<x.size();++i)x[i]=(int(i%101)-50)/37.f;copy(input.p,x);
        cudaGraphExec_t graphs[2];for(int mode=0;mode<2;++mode){cudaGraph_t graph;
            check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));for(int set=0;set<sets;++set)chain((early||vector||fused)?true:mode,set,3,(early||vector)&&(fused||mode),!fused||mode||stage_inputs||pad_logits||skew_weights||preplanned||weight_sf||input_sf,!(stage_inputs||pad_logits||skew_weights||preplanned||weight_sf||input_sf)||mode);
            check(cudaStreamEndCapture(cudaStreamPerThread,&graph));check(cudaGraphInstantiateWithFlags(&graphs[mode],graph,0));check(cudaGraphDestroy(graph));
        }
        for(int repeat=0;repeat<7;++repeat){double us[2];for(int order=0;order<2;++order){const int mode=order^(repeat&1);us[mode]=time_graph(graphs[mode])/sets;}
            if(hoist_token_sources)std::printf("bf16_gate_token_hoist_timing pair=%d jobs=%d tokens=%d repeat=%d baseline_us=%.3f hoist_us=%.3f scope=both_current_kmajor_views_only_token_map_hoisting_changes_producer_included_not_model\n",pair,count,token_count,repeat,us[0],us[1]);
            else if(token_codes_kmajor)std::printf("bf16_gate_token_code_layout_timing pair=%d jobs=%d tokens=%d repeat=%d baseline_us=%.3f kmajor_us=%.3f scope=both_kmajor_weights_only_token_code_layout_changes_producer_included_not_model\n",pair,count,token_count,repeat,us[0],us[1]);
            else if(weight_codes_kmajor)std::printf("bf16_gate_code_layout_timing pair=%d jobs=%d tokens=%d repeat=%d baseline_us=%.3f kmajor_us=%.3f scope=both_paired_token_inputs_only_code_layout_changes_reorder_excluded_not_model\n",pair,count,token_count,repeat,us[0],us[1]);
            else if(paired_token_words)std::printf("bf16_gate_token_words_timing pair=%d jobs=%d tokens=%d repeat=%d baseline_us=%.3f paired_us=%.3f scattered=%d scope=both_shared_token_compact_logits_includes_map_and_quantization_not_model\n",pair,count,token_count,repeat,us[0],us[1],scattered_token_inputs);
            else if(token_quantization)std::printf("bf16_gate_token_quant_timing pair=%d jobs=%d tokens=%d repeat=%d baseline_us=%.3f token_us=%.3f scattered=%d scope=both_compact_logits_includes_map_and_quantization_not_model\n",pair,count,token_count,repeat,us[0],us[1],scattered_token_inputs);
            else if(compact_logits)std::printf("bf16_gate_compact_logits_timing pair=%d jobs=%d repeat=%d baseline_us=%.3f compact_us=%.3f scope=same_kmajor_sf_quantization_and_plan_only_logit_storage_changes_not_model\n",pair,count,repeat,us[0],us[1]);
            else if(stage_inputs&&weight_sf&&input_sf)std::printf("bf16_gate_staged_sf_timing pair=%d jobs=%d repeat=%d baseline_us=%.3f staged_us=%.3f scope=both_sides_kmajor_sf_only_input_staging_changes_not_model\n",pair,count,repeat,us[0],us[1]);
            else if(input_sf)std::printf("bf16_gate_input_sf_timing pair=%d jobs=%d repeat=%d weight_sf=%d baseline_us=%.3f transposed_us=%.3f scope=fused_chain_includes_reordered_quantization_no_extra_storage_not_model\n",pair,count,repeat,weight_sf,us[0],us[1]);
            else if(weight_sf)std::printf("bf16_gate_weight_sf_timing pair=%d jobs=%d repeat=%d baseline_us=%.3f transposed_us=%.3f scope=current_fused_vs_k64_major_weight_sf_view_excludes_preparation_not_model\n",pair,count,repeat,us[0],us[1]);
            else if(preplanned)std::printf("bf16_gate_oracle_timing pair=%d jobs=%d repeat=%d baseline_us=%.3f compact_us=%.3f scope=current_fused_vs_order_preserved_oracle_compact_chain_excludes_compaction_not_model\n",pair,count,repeat,us[0],us[1]);
            else if(skew_weights)std::printf("bf16_gate_weights_timing pair=%d jobs=%d repeat=%d baseline_us=%.3f skewed_us=%.3f scope=current_fused_vs_shared_weight_skew_fused_chain_synthetic_not_model\n",pair,count,repeat,us[0],us[1]);
            else if(pad_logits)std::printf("bf16_gate_logits_timing pair=%d jobs=%d repeat=%d baseline_us=%.3f padded_us=%.3f scope=current_fused_vs_logits_pitch33_fused_chain_synthetic_not_model\n",pair,count,repeat,us[0],us[1]);
            else if(stage_inputs)std::printf("bf16_gate_staged_timing pair=%d jobs=%d repeat=%d baseline_us=%.3f staged_us=%.3f scope=current_fused_vs_shared_input_fused_chain_synthetic_not_model\n",pair,count,repeat,us[0],us[1]);
            else if(fused)std::printf("bf16_gate_fused_timing pair=%d jobs=%d repeat=%d baseline_us=%.3f fused_us=%.3f scope=parallel_plan_vector_input_quantize_gate_activation_synthetic_not_model\n",pair,count,repeat,us[0],us[1]);
            else if(vector)std::printf("bf16_gate_vector_timing pair=%d jobs=%d repeat=%d baseline_us=%.3f vector_us=%.3f scope=parallel_plan_quantize_bf16_gate_plus_packed_activation_synthetic_not_model\n",pair,count,repeat,us[0],us[1]);
            else if(early)std::printf("bf16_gate_early_timing pair=%d jobs=%d repeat=%d baseline_us=%.3f early_us=%.3f scope=bf16_gate_quantize_mma_plus_packed_activation_synthetic_not_model\n",pair,count,repeat,us[0],us[1]);
            else std::printf("bf16_gate_timing pair=%d jobs=%d repeat=%d fp32_us=%.3f bf16_us=%.3f scope=gate_quantize_async_mma_plus_packed_activation_synthetic_not_model\n",pair,count,repeat,us[0],us[1]);std::fflush(stdout);
        }
        for(auto graph:graphs)check(cudaGraphExecDestroy(graph));
        if(fused){
            chain(true,1,3,true,false);check(cudaStreamSynchronize(cudaStreamPerThread));
            const auto expected_p=read(pb.p,act_values/2+8),expected_s=read(sb.p,act_values/16+8);
            chain(true,1,3,true,true);check(cudaStreamSynchronize(cudaStreamPerThread));
            equal(expected_p,read(pb.p,act_values/2+8));equal(expected_s,read(sb.p,act_values/16+8));
            std::printf("bf16_gate_fused_final pair=%d jobs=%d packed_scales_full_bytes=true\n",pair,count);
            if(weight_sf)equal(read(sf_view.p,sf_host.size()),sf_host);
        }
    }
    if(weight_codes_kmajor) {
        equal(read(code_view.p,code_host.size()),code_host);
        std::printf("code_kmajor_view pair=%d bytes=%zu immutable_guarded=1 reorder_excluded=1\n",pair,code_host.size());
    }
}

int main(int argc,char** argv){
    const bool only=argc==2&&!std::strcmp(argv[1],"--check-only");std::vector<int> runs{1,7,8,16,17,32,33};
    if(argc>1&&!only){
        if(argc!=5||std::strcmp(argv[1],"--rows")||std::strcmp(argv[3],"--counts"))return 64;
        const char* row_arg=argv[2];for(const char* t=row_arg;*t;++t)if(*t<'0'||*t>'9')return 64;
        if(!*row_arg||std::strlen(row_arg)>4||std::atoi(row_arg)<1||std::atoi(row_arg)>4096)return 64;
        runs.clear();const char* p=argv[4];while(*p){const char* begin=p;while(*p>='0'&&*p<='9')++p;if(p==begin||p-begin>5||(*p&&*p!=','))return 64;
            const long n=std::strtol(begin,nullptr,10);if(n<0||n>65535)return 64;runs.push_back(n);if(!*p)break;++p;if(!*p)return 64;}
        if(runs.size()!=512||std::accumulate(runs.begin(),runs.end(),0)!=std::atoi(argv[2])*10)return 64;
    }
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0));if(prop.major!=12||prop.minor>1)return 77;
    setenv("MOLLM_CUDA_NVFP4_MMA_BATCH_WARPS","4",1);setenv("MOLLM_CUDA_HOST_HEADROOM_MB","102400",1);
    if(!mollm_memory::cuda_host_allocation_allowed(size_t(5)<<30,"bounded BF16 gate diagnostic"))return 1;
    const char* option=std::getenv("MOLLM_TEST_NVFP4_EARLY_INPUTS");
    const bool early=option&&!std::strcmp(option,"1");
    const char* v_option=std::getenv("MOLLM_TEST_NVFP4_GATE_VECTOR_QUANTIZE");
    const bool vector=v_option&&!std::strcmp(v_option,"1");
    const char* f_option=std::getenv("MOLLM_TEST_NVFP4_MMA_ACTIVATION_FUSED");
    const bool fused=f_option&&!std::strcmp(f_option,"1");
    const char* staged_option=std::getenv("MOLLM_TEST_NVFP4_GATE_STAGE_INPUTS");
    const bool stage_inputs=staged_option&&!std::strcmp(staged_option,"1");
    const char* logits_option=std::getenv("MOLLM_TEST_NVFP4_GATE_PAD_LOGITS");
    const bool pad_logits=logits_option&&!std::strcmp(logits_option,"1");
    const char* weights_option=std::getenv("MOLLM_TEST_NVFP4_GATE_SKEW_WEIGHTS");
    const bool skew_weights=weights_option&&!std::strcmp(weights_option,"1");
    const char* plan_option=std::getenv("MOLLM_TEST_NVFP4_GATE_PREPLANNED");
    const bool preplanned=plan_option&&!std::strcmp(plan_option,"1");
    const char* sf_option=std::getenv("MOLLM_TEST_NVFP4_GATE_WEIGHT_SF_KMAJOR");
    const bool weight_sf=sf_option&&!std::strcmp(sf_option,"1");
    const char* input_sf_option=std::getenv("MOLLM_TEST_NVFP4_GATE_INPUT_SF_KMAJOR");
    const bool input_sf=input_sf_option&&!std::strcmp(input_sf_option,"1");
    const char* compact_option=std::getenv("MOLLM_TEST_NVFP4_GATE_COMPACT_LOGITS");
    compact_logits=compact_option&&!std::strcmp(compact_option,"1");
    const char* token_option=std::getenv("MOLLM_TEST_NVFP4_GATE_TOKEN_QUANTIZE");
    token_quantization=token_option&&!std::strcmp(token_option,"1");
    const char* scattered_option=std::getenv("MOLLM_TEST_NVFP4_GATE_TOKEN_SCATTER");
    scattered_token_inputs=scattered_option&&!std::strcmp(scattered_option,"1");
    const char* paired_option=std::getenv("MOLLM_TEST_NVFP4_GATE_TOKEN_PAIR_WORDS");
    paired_token_words=paired_option&&!std::strcmp(paired_option,"1");
    const char* code_option=std::getenv("MOLLM_TEST_NVFP4_GATE_CODE_KMAJOR");
    weight_codes_kmajor=code_option&&!std::strcmp(code_option,"1");
    const char* token_code_option=std::getenv("MOLLM_TEST_NVFP4_GATE_TOKEN_CODE_KMAJOR");
    token_codes_kmajor=token_code_option&&!std::strcmp(token_code_option,"1");
    const char* hoist_option=std::getenv("MOLLM_TEST_NVFP4_GATE_TOKEN_HOIST");
    hoist_token_sources=hoist_option&&!std::strcmp(hoist_option,"1");
    if(hoist_token_sources&&!token_codes_kmajor)return 64;
    const char* locality_option=std::getenv("MOLLM_TEST_NVFP4_GATE_WEIGHT_LOCALITY");
    if(locality_option&&std::strcmp(locality_option,"0")&&std::strcmp(locality_option,"1")&&
        std::strcmp(locality_option,"codes")&&std::strcmp(locality_option,"scales"))return 64;
    locality_mask=locality_option?(!std::strcmp(locality_option,"1")?3:
        !std::strcmp(locality_option,"codes")?1:!std::strcmp(locality_option,"scales")?2:0):0;
    weight_locality=locality_mask!=0;
    const char* pitch_option=std::getenv("MOLLM_TEST_NVFP4_GATE_STAGE_INPUT_PITCH");
    const int input_pitch=pitch_option&&!std::strcmp(pitch_option,"48")?48:32;
    if(pitch_option&&std::strcmp(pitch_option,"32")&&std::strcmp(pitch_option,"48"))return 64;
    const char* shared_option=std::getenv("MOLLM_TEST_NVFP4_GATE_SHARED_INPUTS");
    const bool shared_inputs=shared_option&&!std::strcmp(shared_option,"1");
    if(shared_inputs&&(runs.size()!=512||std::accumulate(runs.begin(),runs.end(),0)%10))return 64;
    if(fused&&(!vector||early))return 64;
    if(stage_inputs&&!fused)return 64;
    if(pad_logits&&(!fused||stage_inputs))return 64;
    if(skew_weights&&(!fused||stage_inputs||pad_logits))return 64;
    if(preplanned&&(!fused||stage_inputs||pad_logits||skew_weights))return 64;
    if(weight_sf&&(!fused||pad_logits||skew_weights||preplanned))return 64;
    if(input_sf&&(!fused||pad_logits||skew_weights||preplanned))return 64;
    if(stage_inputs&&(weight_sf||input_sf)&&!(weight_sf&&input_sf&&input_pitch==32))return 64;
    if(compact_logits&&(!fused||!weight_sf||!input_sf||stage_inputs||pad_logits||skew_weights||preplanned))return 64;
    if(token_quantization&&(!compact_logits||(std::accumulate(runs.begin(),runs.end(),0)+9)/10>2048))return 64;
    if(scattered_token_inputs&&!token_quantization)return 64;
    if(paired_token_words&&!token_quantization)return 64;
    if(weight_codes_kmajor&&(!paired_token_words||!fused||!weight_sf||!input_sf||!compact_logits||weight_locality||stage_inputs||pad_logits||skew_weights||preplanned))return 64;
    if(token_codes_kmajor&&!weight_codes_kmajor)return 64;
    if(weight_locality&&(!paired_token_words||!fused||!weight_sf||!input_sf||!compact_logits||stage_inputs||pad_logits||skew_weights||preplanned))return 64;
    if(vector&&early)return 64;
    if(weight_sf)scale_view_copy_check();
    consumer_edges();for(bool pair:{true,false})run(runs,pair,!only,early,vector,fused,shared_inputs,stage_inputs,input_pitch,pad_logits,skew_weights,preplanned,weight_sf,input_sf);
}
