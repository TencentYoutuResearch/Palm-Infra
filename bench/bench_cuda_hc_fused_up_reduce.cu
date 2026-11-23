// Standalone structural diagnostic: HC up projection + deferred RMS reduction.
// Numerical validation remains mandatory for every optional tile/layout.
#include "kernels/cuda/elementwise.h"
#include "runtime/cuda_host_memory_budget.h"
#include "core/quant_layouts.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <mma.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <fstream>
#include <vector>
static bool compact_probe=false;

static void check(cudaError_t s){if(s!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(s));std::exit(1);}}
static void blas(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS){std::fprintf(stderr,"cuBLAS %d\n",int(s));std::exit(1);}}
template<class T>struct Buffer{T* p;explicit Buffer(size_t n){check(cudaMalloc(&p,n*sizeof(T)));}~Buffer(){cudaFree(p);}};
template<class T>static void copy(T* p,const std::vector<T>& x){check(cudaMemcpy(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));}
template<class T>static std::vector<T> read(T* p,size_t n){std::vector<T> x(n);check(cudaMemcpy(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return x;}

static size_t run(int rows,const std::vector<__half>* real_weight,bool sanitize_only,bool check_only,int input_pitch,int tile_columns){
    constexpr int hidden=2560,wide=10240,k=320;
    const size_t n=size_t(rows)*hidden,wn=size_t(rows)*wide;
    Buffer<__half> input(size_t(rows)*k),weight(size_t(wide)*k);
    Buffer<float> carry(wn),gamma(wide),inv(rows*4),logits(wn),baseline(n+8),fused(n+8);
    Buffer<float> compact_storage(wn+8);
    Buffer<float> native_output(n+8);
    const float* fused_carry=compact_probe?compact_storage.p:carry.p;
    cublasHandle_t handle=nullptr;
    if(!sanitize_only){blas(cublasCreate(&handle));blas(cublasSetStream(handle,cudaStreamPerThread));
        blas(cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH));}
    std::vector<__half> w(size_t(wide)*k);
    if(real_weight)w=*real_weight;
    else for(size_t i=0;i<w.size();++i)w[i]=__float2half(std::sin(float(i%7919)*.037f)*(.003f+(i%13)*.003f));
    copy(weight.p,w);
    std::vector<float> g(wide);for(int i=0;i<wide;++i)g[i]=.5f+(i%17)/16.f;copy(gamma.p,g);
    auto reference=[&]{const float alpha=1.f,beta=0.f;
        if(sanitize_only)check(cudaMemsetAsync(logits.p,0,wn*4,cudaStreamPerThread));
        else blas(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,wide,rows,k,&alpha,
            weight.p,CUDA_R_16F,k,input.p,CUDA_R_16F,k,&beta,logits.p,CUDA_R_32F,wide,
            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        if(!mollm_cuda::launch_gr_reduce_sigmoid_deferred_rms(carry.p,logits.p,gamma.p,inv.p,
            baseline.p,rows,cudaStreamPerThread))std::exit(2);
    };
    auto candidate=[&]{if(!mollm_cuda::launch_hc_up_reduce_fused(
        reinterpret_cast<const uint16_t*>(input.p),reinterpret_cast<const uint16_t*>(weight.p),
        fused_carry,gamma.p,inv.p,fused.p,rows,cudaStreamPerThread,input_pitch,tile_columns,compact_probe))std::exit(2);};
    check(cudaMemset(fused.p,0x5a,(n+8)*4));
    cudaGraph_t invalid_graph;
    check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
    for(int id=0;id<25;++id){
        const auto* a=reinterpret_cast<const uint16_t*>(input.p);
        const auto* w=reinterpret_cast<const uint16_t*>(weight.p);
        const float* x=carry.p;const float* g=gamma.p;const float* v=inv.p;float* y=fused.p;
        int r=rows;
        if(id==0)a=nullptr;if(id==1)w=nullptr;if(id==2)x=nullptr;
        if(id==3)g=nullptr;if(id==4)v=nullptr;if(id==5)y=nullptr;
        if(id==6)r=0;if(id==7)r=-1;if(id==8)r=15;if(id==9)r=2049;
        if(id==10)++a;if(id==11)++w;
        if(id==12)x=reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(x)+1);
        if(id==13)g=reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(g)+1);
        if(id==14)v=reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(v)+1);
        if(id==15)y=reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(y)+1);
        if(id==16)y=reinterpret_cast<float*>(input.p);
        if(id==17)y=reinterpret_cast<float*>(weight.p);
        if(id==18)y=carry.p;if(id==19)y=gamma.p;if(id==20)y=inv.p;
        if(id==21)y=reinterpret_cast<float*>(UINTPTR_MAX-3);
        if(id==22)a=reinterpret_cast<const uint16_t*>(UINTPTR_MAX-31);
        if(id==23)w=reinterpret_cast<const uint16_t*>(UINTPTR_MAX-31);
        if(id==24)y=carry.p+1;
        if(mollm_cuda::launch_hc_up_reduce_fused(a,w,x,g,v,y,r,cudaStreamPerThread,input_pitch,tile_columns))std::exit(4);
    }
    for(int bad:{0,16,319,321,327,329,337})
        if(mollm_cuda::launch_hc_up_reduce_fused(reinterpret_cast<const uint16_t*>(input.p),
                reinterpret_cast<const uint16_t*>(weight.p),carry.p,gamma.p,inv.p,fused.p,
                rows,cudaStreamPerThread,bad,tile_columns))std::exit(4);
    for(int bad:{0,8,17,48,64})
        if(mollm_cuda::launch_hc_up_reduce_fused(reinterpret_cast<const uint16_t*>(input.p),
                reinterpret_cast<const uint16_t*>(weight.p),carry.p,gamma.p,inv.p,fused.p,
                rows,cudaStreamPerThread,input_pitch,bad))std::exit(4);
    check(cudaStreamEndCapture(cudaStreamPerThread,&invalid_graph));
    size_t rejected_nodes=0;check(cudaGraphGetNodes(invalid_graph,nullptr,&rejected_nodes));
    check(cudaGraphDestroy(invalid_graph));if(rejected_nodes)std::exit(4);
    const auto rejected_output=read(fused.p,n+8);
    for(float value:rejected_output){uint32_t bits;std::memcpy(&bits,&value,4);if(bits!=0x5a5a5a5a)std::exit(4);}
    std::printf("hc_up_reduce_api_rejections rows=%d cases=37 captured_nodes=0 whole_output_untouched=true pitch=%d columns=%d\n",rows,input_pitch,tile_columns);
    cudaGraph_t graph;cudaGraphExec_t replay;
    check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));candidate();
    check(cudaStreamEndCapture(cudaStreamPerThread,&graph));check(cudaGraphInstantiateWithFlags(&replay,graph,0));check(cudaGraphDestroy(graph));
    cudaGraphExec_t native_replay;
    check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
    if(!mollm_cuda::launch_hc_up_reduce_fused(reinterpret_cast<const uint16_t*>(input.p),
            reinterpret_cast<const uint16_t*>(weight.p),fused_carry,gamma.p,inv.p,native_output.p,
            rows,cudaStreamPerThread,input_pitch,tile_columns,compact_probe))std::exit(2);
    check(cudaStreamEndCapture(cudaStreamPerThread,&graph));
    check(cudaGraphInstantiateWithFlags(&native_replay,graph,0));check(cudaGraphDestroy(graph));
    size_t all_mismatches=0;
    for(int changed=0;changed<4;++changed){
        if(!real_weight)for(size_t i=0;i<w.size();++i)w[i]=__float2half(std::sin(float(i%7919)*.037f+(changed%2)*.13f)*(.003f+(i%13)*.003f));
        copy(weight.p,w);
        std::vector<__half> a(size_t(rows)*k);for(size_t i=0;i<a.size();++i)
            a[i]=__float2half(sanitize_only||changed==2?0.f:changed==3?-0.f:(int((i*13+changed*11)%97)-48)/31.f);
        std::vector<float> x(wn),v(rows*4);for(size_t i=0;i<wn;++i)x[i]=(int((i*17+changed*7)%103)-51)/7.f;
        std::vector<uint16_t> compact_bytes(wn*2+16,0x5a5a);
        if(compact_probe)for(size_t i=0;i<wn;++i) {
            uint32_t bits;std::memcpy(&bits,&x[i],4);bits+=0x7fffu+((bits>>16)&1u);bits&=0xffff0000u;
            std::memcpy(&x[i],&bits,4);compact_bytes[i]=uint16_t(bits>>16);
        }
        for(size_t i=0;i<v.size();++i)v[i]=.25f+(i%13)/32.f;
        copy(input.p,a);copy(carry.p,x);copy(inv.p,v);
        if(compact_probe)copy(reinterpret_cast<uint16_t*>(compact_storage.p),compact_bytes);
        check(cudaMemset(baseline.p,0x5a,(n+8)*4));check(cudaMemset(fused.p,0x5a,(n+8)*4));
        check(cudaMemset(native_output.p,0x5a,(n+8)*4));
        reference();check(cudaGraphLaunch(native_replay,cudaStreamPerThread));
        check(cudaGraphLaunch(replay,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
        if(compact_probe&&read(reinterpret_cast<uint16_t*>(compact_storage.p),compact_bytes.size())!=compact_bytes)std::exit(5);
        const auto expected=read(baseline.p,n+8),actual=read(fused.p,n+8);
        const auto native=read(native_output.p,n+8);
        if(std::memcmp(native.data(),actual.data(),(n+8)*4))std::exit(6);
        if(rows>1024) {
            // Logical-batch equivalence: row-local HC may tile internally without
            // imposing the same partition on the rest of the graph.
            check(cudaMemset(native_output.p,0x5a,(n+8)*4));
            for(int offset=0;offset<rows;offset+=1024) {
                const int part=std::min(1024,rows-offset);
                // The one-row tail at 1025 is below the fused API's 16-row
                // minimum; other geometries cover partition equivalence.
                if(part<16)break;
                const float* part_carry=compact_probe?
                    reinterpret_cast<const float*>(reinterpret_cast<const uint16_t*>(fused_carry)+size_t(offset)*wide):
                    fused_carry+size_t(offset)*wide;
                if(!mollm_cuda::launch_hc_up_reduce_fused(
                    reinterpret_cast<const uint16_t*>(input.p)+size_t(offset)*k,
                    reinterpret_cast<const uint16_t*>(weight.p),part_carry,gamma.p,inv.p+offset*4,
                    native_output.p+size_t(offset)*hidden,part,cudaStreamPerThread,
                    input_pitch,tile_columns,compact_probe))std::exit(7);
            }
            if(rows!=1025) {
                check(cudaStreamSynchronize(cudaStreamPerThread));
                const auto split=read(native_output.p,n+8);
                if(std::memcmp(split.data(),actual.data(),(n+8)*4))std::exit(8);
                std::printf("hc_up_reduce rows=%d changed=%d whole_vs_split_byte_exact=true compact=%d\n",
                    rows,changed,int(compact_probe));
            }
        }
        std::printf("hc_up_reduce_native_capture rows=%d changed=%d whole_output_and_guards_byte_exact=true compact=%d\n",
            rows,changed,int(compact_probe));
        size_t mismatches=0;float max_abs=0;
        for(size_t i=0;i<n;++i){if(std::memcmp(&expected[i],&actual[i],4))++mismatches;
            max_abs=std::max(max_abs,std::abs(expected[i]-actual[i]));}
        for(size_t i=n;i<n+8;++i){uint32_t bits;std::memcpy(&bits,&actual[i],4);if(bits!=0x5a5a5a5a)std::exit(3);
            std::memcpy(&bits,&expected[i],4);if(bits!=0x5a5a5a5a)std::exit(3);}
        std::printf("hc_up_reduce rows=%d changed=%d mismatches=%zu/%zu max_abs=%g guards=true\n",rows,changed,mismatches,n,max_abs);std::fflush(stdout);
        all_mismatches+=mismatches;
    }
    if(sanitize_only||check_only){check(cudaGraphExecDestroy(native_replay));check(cudaGraphExecDestroy(replay));if(handle)blas(cublasDestroy(handle));return all_mismatches;}
    std::vector<__half> timed_input(size_t(rows)*k);
    for(size_t i=0;i<timed_input.size();++i)timed_input[i]=__float2half((int(i*13%97)-48)/31.f);
    copy(input.p,timed_input);
    // Report research timing even if quality fails; no inference dispatch exists.
    for(int repeat=0;repeat<7;++repeat){double times[2];
        for(int order=0;order<2;++order){const int mode=order^(repeat&1);
            auto launch=[&]{if(mode)candidate();else if(input_pitch!=320||tile_columns!=16){
                if(!mollm_cuda::launch_hc_up_reduce_fused(reinterpret_cast<const uint16_t*>(input.p),
                    reinterpret_cast<const uint16_t*>(weight.p),fused_carry,gamma.p,inv.p,baseline.p,
                    rows,cudaStreamPerThread,tile_columns==32?input_pitch:320,16,compact_probe))std::exit(2);
            }else reference();};launch();check(cudaStreamSynchronize(cudaStreamPerThread));
            cudaEvent_t a,b;check(cudaEventCreate(&a));check(cudaEventCreate(&b));check(cudaEventRecord(a,cudaStreamPerThread));
            for(int i=0;i<32;++i)launch();check(cudaEventRecord(b,cudaStreamPerThread));check(cudaEventSynchronize(b));
            float ms;check(cudaEventElapsedTime(&ms,a,b));times[mode]=ms*1000./32;check(cudaEventDestroy(a));check(cudaEventDestroy(b));
        }
        if(tile_columns==32)std::printf("hc_up_reduce_columns_timing rows=%d repeat=%d pitch=%d native_us=%.3f wider_us=%.3f all_mismatches=%zu scope=same_pitch_columns16_vs_columns32_not_model\n",rows,repeat,input_pitch,times[0],times[1],all_mismatches);
        else if(input_pitch!=320)std::printf("hc_up_reduce_pitch_timing rows=%d repeat=%d pitch=%d native_us=%.3f padded_us=%.3f all_mismatches=%zu scope=current_fused_vs_padded_shared_input_not_model\n",rows,repeat,input_pitch,times[0],times[1],all_mismatches);
        else std::printf("hc_up_reduce_timing rows=%d repeat=%d original_us=%.3f candidate_us=%.3f all_mismatches=%zu not_model=true\n",rows,repeat,times[0],times[1],all_mismatches);std::fflush(stdout);
    }
    check(cudaGraphExecDestroy(native_replay));check(cudaGraphExecDestroy(replay));blas(cublasDestroy(handle));
    return all_mismatches;
}
int main(int argc,char** argv){
    const char* pitch_option=std::getenv("MOLLM_TEST_CUDA_HC_UP_INPUT_PITCH");
    if(pitch_option&&std::strcmp(pitch_option,"320")&&std::strcmp(pitch_option,"328")&&
        std::strcmp(pitch_option,"336"))return 64;
    const int input_pitch=pitch_option?std::atoi(pitch_option):320;
    const char* columns_option=std::getenv("MOLLM_TEST_CUDA_HC_UP_TILE_COLUMNS");
    if(columns_option&&std::strcmp(columns_option,"16")&&std::strcmp(columns_option,"32"))return 64;
    const int tile_columns=columns_option?std::atoi(columns_option):16;
    const char* compact_option=std::getenv("MOLLM_TEST_CUDA_HC_UP_COMPACT_CARRY");
    compact_probe=compact_option&&!std::strcmp(compact_option,"1");
    const bool sanitize_only=argc==2&&!std::strcmp(argv[1],"--sanitize-only");
    const bool check_only=argc==2&&!std::strcmp(argv[1],"--check-only");
    std::vector<__half> real_weight;
    if(argc!=1&&!sanitize_only&&!check_only){
        if(argc!=4||std::strcmp(argv[1],"--weight-record"))return 64;
        const char* option=argv[3];if(!*option||std::strlen(option)>20)return 64;
        for(const char* p=option;*p;++p)if(*p<'0'||*p>'9')return 64;
        errno=0;char* end=nullptr;const auto offset=std::strtoull(option,&end,10);
        if(errno||*end)return 64;
        std::ifstream file(argv[2],std::ios::binary|std::ios::ate);
        const auto length=file.tellg();if(!file||length<0||offset>uint64_t(length)||uint64_t(length)-offset<88)return 64;
        struct Header{uint32_t magic,flags,version,precision;uint64_t shape[4],data_offset,data_size,scale_offset,scale_size;uint32_t group,groups;};
        static_assert(sizeof(Header)==88,"weight header ABI");Header header{};
        file.seekg(offset);file.read(reinterpret_cast<char*>(&header),88);
        if(!file||header.magic!=0x50414d58||header.flags!=4||header.version!=2||header.precision!=3||
            header.shape[0]!=10240||header.shape[1]!=320||header.shape[2]!=1||header.shape[3]!=1||
            header.data_offset<88||header.data_size!=2048000||header.scale_offset||header.scale_size||
            header.group!=32||header.groups!=102400||header.data_offset>uint64_t(length)-offset||
            header.data_size>uint64_t(length)-offset-header.data_offset)return 64;
        std::vector<Q4B8G32Block> packed(102400);file.seekg(offset+header.data_offset);
        file.read(reinterpret_cast<char*>(packed.data()),header.data_size);if(!file)return 64;
        real_weight.resize(size_t(10240)*320);
        for(int row=0;row<10240;++row)for(int j=0;j<320;++j){
            const auto& block=packed[(row/8)*10+j/32];const int lane=row%8;
            const int q=(block.q[lane][(j%32)/2]>>((j&1)*4))&15;
            real_weight[size_t(row)*320+j]=__float2half(float(q<8?q:q-16)*block.scales[lane]);
        }
        std::printf("real_package_weight_record=%llu canonical_signed_Q4_to_FP16=true synthetic_other_inputs=true\n",offset);
    }
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
    setenv("MOLLM_CUDA_HOST_HEADROOM_MB","102400",1);
    if(!mollm_memory::cuda_host_allocation_allowed(size_t(1)<<30,"HC fused up/reduce probe"))return 1;
    size_t mismatches=0;
    for(int rows:{17,719,1024,1025,1743,2048})mismatches+=run(rows,real_weight.empty()?nullptr:&real_weight,sanitize_only,check_only,input_pitch,tile_columns);
    return mismatches?2:0;
}
