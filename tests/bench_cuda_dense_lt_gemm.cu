// Bounded FP16/F32-output dense GEMM algorithm diagnostic, not model throughput.
#include "runtime/cuda_host_memory_budget.h"
#include "backends/cuda/dense_lt_gemm.h"
#include "kernels/cuda/matmul.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

static void check(cudaError_t s){if(s!=cudaSuccess){std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(s));std::exit(1);}}
static void blas(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS){std::fprintf(stderr,"cuBLAS: %d\n",int(s));std::exit(2);}}
template<class T>struct Buffer{T* p=nullptr;explicit Buffer(size_t n){check(cudaMalloc(&p,n*sizeof(T)));}~Buffer(){cudaFree(p);}};
template<class T>static void upload(T* p,const std::vector<T>& v){check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
static std::vector<float> read(float* p,size_t n){std::vector<float> v(n);check(cudaMemcpy(v.data(),p,n*4,cudaMemcpyDeviceToHost));return v;}
template<class Launch>static cudaGraphExec_t capture(Launch&& launch){
    cudaGraph_t g;cudaGraphExec_t e;check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
    launch();check(cudaStreamEndCapture(cudaStreamPerThread,&g));check(cudaGraphInstantiateWithFlags(&e,g,0));check(cudaGraphDestroy(g));return e;
}
static double time_graph(cudaGraphExec_t e,int replays=16){
    check(cudaGraphLaunch(e,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
    cudaEvent_t a,b;check(cudaEventCreate(&a));check(cudaEventCreate(&b));check(cudaEventRecord(a,cudaStreamPerThread));
    for(int i=0;i<replays;++i)check(cudaGraphLaunch(e,cudaStreamPerThread));
    check(cudaEventRecord(b,cudaStreamPerThread));check(cudaEventSynchronize(b));float ms;check(cudaEventElapsedTime(&ms,a,b));
    check(cudaEventDestroy(a));check(cudaEventDestroy(b));return ms*1000/replays;
}
static void fill(std::vector<__half>& v,unsigned seed){
    for(auto& x:v){seed=1664525u*seed+1013904223u;x=__float2half(float(int(seed>>17)-16384)/16381.f);}
}
static void run(cublasHandle_t classic,cublasLtHandle_t lt,int m,int n,int k,bool timing,int repeats=3,bool tile_sweep=false,bool finalists=false,bool zero_workspace=false,bool classic_reference=false,int tile_only=0){
    const size_t budget=zero_workspace?0:size_t(64)<<20;
    constexpr int sets=4;
    const size_t weight_values=size_t(n)*k;
    Buffer<__half> weight(sets*weight_values),input(size_t(m)*k);
    const size_t values=size_t(m)*n;
    const size_t output_stride=(values+8+63)&~size_t(63); // Each set retains 256-byte alignment.
    const size_t output_values=sets*output_stride;
    Buffer<float> baseline(output_values),candidate(output_values);Buffer<unsigned char> workspace(std::max<size_t>(budget,1));
    std::vector<__half> w(sets*weight_values),x(size_t(m)*k);fill(w,519);upload(weight.p,w);w.clear();w.shrink_to_fit();fill(x,12980);upload(input.p,x);
    const float alpha=1.f,beta=0.f;
    cublasLtMatmulDesc_t op;cublasLtMatrixLayout_t a,b,c;cublasLtMatmulPreference_t pref;
    blas(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
    cublasOperation_t transpose=CUBLAS_OP_T;blas(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&transpose,sizeof(transpose)));
    blas(cublasLtMatrixLayoutCreate(&a,CUDA_R_16F,k,n,k));blas(cublasLtMatrixLayoutCreate(&b,CUDA_R_16F,k,m,k));
    blas(cublasLtMatrixLayoutCreate(&c,CUDA_R_32F,n,m,n));blas(cublasLtMatmulPreferenceCreate(&pref));
    blas(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&budget,sizeof(budget)));
    std::vector<cublasLtMatmulHeuristicResult_t> options(4096);int found=0;
    blas(cublasLtMatmulAlgoGetHeuristic(lt,op,a,b,c,c,pref,16,options.data(),&found));
    cublasLtMatmulAlgo_t control{};
    if(tile_sweep){
        bool present=false;
        for(int i=0;i<found;++i){int id=-1,tile=-1,split=-1;size_t written=0;
            blas(cublasLtMatmulAlgoConfigGetAttribute(&options[i].algo,CUBLASLT_ALGO_CONFIG_ID,&id,sizeof(id),&written));
            blas(cublasLtMatmulAlgoConfigGetAttribute(&options[i].algo,CUBLASLT_ALGO_CONFIG_TILE_ID,&tile,sizeof(tile),&written));
            blas(cublasLtMatmulAlgoConfigGetAttribute(&options[i].algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split),&written));
            if(id==67&&tile==412&&split==1&&options[i].state==CUBLAS_STATUS_SUCCESS&&!options[i].workspaceSize){control=options[i].algo;present=true;break;}}
        if(!present)std::exit(20);
        size_t bytes=0;blas(cublasLtMatmulAlgoCapGetAttribute(&control,CUBLASLT_ALGO_CAP_TILE_IDS,nullptr,0,&bytes));
        std::printf("dense_tile_capability bytes=%zu\n",bytes);std::fflush(stdout);
        if(!bytes||bytes%sizeof(int)||bytes>options.size()*sizeof(int))std::exit(21);
        std::vector<int> tiles(bytes/sizeof(int));blas(cublasLtMatmulAlgoCapGetAttribute(&control,CUBLASLT_ALGO_CAP_TILE_IDS,tiles.data(),bytes,&bytes));
        found=0;int rejected=0;
        for(int tile:tiles){auto algo=control;
            if(tile_only&&tile!=tile_only)continue;
            if(finalists&&tile!=23&&tile!=409&&tile!=414&&tile!=467)continue;
            if(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_TILE_ID,&tile,sizeof(tile))!=CUBLAS_STATUS_SUCCESS){++rejected;continue;}
            cublasLtMatmulHeuristicResult_t result{};
            if(cublasLtMatmulAlgoCheck(lt,op,a,b,c,c,&algo,&result)!=CUBLAS_STATUS_SUCCESS||result.state!=CUBLAS_STATUS_SUCCESS||result.workspaceSize){++rejected;continue;}
            result.algo=algo;options[found++]=result;
        }
        std::printf("dense_tile_sweep algorithm=67 control_tile=412 split=1 advertised_tiles=%zu accepted=%d rejected=%d workspace=0 other_config_unchanged=true\n",tiles.size(),found,rejected);
        std::printf("dense_tile_reference=%s\n",classic_reference?"classic_GemmEx":"current_Lt412");
    }
    auto original=[&]{for(int set=0;set<sets;++set){
        if(tile_sweep&&!classic_reference)blas(cublasLtMatmul(lt,op,&alpha,weight.p+set*weight_values,a,input.p,b,&beta,
            baseline.p+set*output_stride,c,baseline.p+set*output_stride,c,&control,nullptr,0,cudaStreamPerThread));
        else blas(cublasGemmEx(classic,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&alpha,
            weight.p+set*weight_values,CUDA_R_16F,k,input.p,CUDA_R_16F,k,&beta,baseline.p+set*output_stride,CUDA_R_32F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));}};
    original();check(cudaStreamSynchronize(cudaStreamPerThread));auto reference_graph=capture(original);
    std::printf("dense_lt_shape m=%d n=%d k=%d candidates=%d workspace_budget=%zu weight_sets=%d scope=synthetic_fp16_to_fp32_not_model\n",m,n,k,found,budget,sets);std::fflush(stdout);
    for(int index=0;index<found;++index){const auto& option=options[index];if(option.state!=CUBLAS_STATUS_SUCCESS||option.workspaceSize>budget)continue;
        auto tuned=[&]{for(int set=0;set<sets;++set)blas(cublasLtMatmul(lt,op,&alpha,weight.p+set*weight_values,a,input.p,b,&beta,candidate.p+set*output_stride,c,candidate.p+set*output_stride,c,&option.algo,workspace.p,budget,cudaStreamPerThread));};
        tuned();check(cudaStreamSynchronize(cudaStreamPerThread));auto graph=capture(tuned);
        size_t mismatch=0;double max_abs=0,max_rel=0;
        for(int replay=0;replay<2;++replay){fill(x,12980+replay*919);upload(input.p,x);
            check(cudaMemset(baseline.p,0x5a,output_values*4));check(cudaMemset(candidate.p,0x5a,output_values*4));
            check(cudaGraphLaunch(reference_graph,cudaStreamPerThread));check(cudaGraphLaunch(graph,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
            auto ref=read(baseline.p,output_values),actual=read(candidate.p,output_values);
            for(int set=0;set<sets;++set){const size_t offset=set*output_stride;
            for(size_t j=0;j<values;++j){const size_t i=offset+j;const double error=std::abs(double(ref[i])-actual[i]);
                if(!std::isfinite(ref[i])||!std::isfinite(actual[i])||error>0.005+0.002*std::abs(double(ref[i])))std::exit(3);
                mismatch+=std::memcmp(&ref[i],&actual[i],4)!=0;max_abs=std::max(max_abs,error);max_rel=std::max(max_rel,error/std::max(1e-6,std::abs(double(ref[i]))));}
            for(auto* v:{&ref,&actual}){const auto* bytes=reinterpret_cast<const unsigned char*>(v->data()+offset+values);for(size_t i=0;i<(output_stride-values)*4;++i)if(bytes[i]!=0x5a)std::exit(4);}}
        }
        int algorithm=-1,tile=-1,split=-1;size_t written=0;
        blas(cublasLtMatmulAlgoConfigGetAttribute(&option.algo,CUBLASLT_ALGO_CONFIG_ID,&algorithm,sizeof(algorithm),&written));
        blas(cublasLtMatmulAlgoConfigGetAttribute(&option.algo,CUBLASLT_ALGO_CONFIG_TILE_ID,&tile,sizeof(tile),&written));
        blas(cublasLtMatmulAlgoConfigGetAttribute(&option.algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split),&written));
        std::printf("dense_lt_check m=%d n=%d k=%d index=%d algorithm=%d tile=%d split=%d workspace=%zu bit_mismatches=%zu max_abs=%.9g max_relative_near_zero=%.9g changed_replays=2 guards=true\n",
            m,n,k,index,algorithm,tile,split,option.workspaceSize,mismatch,max_abs,max_rel);std::fflush(stdout);
        if(timing)for(int repeat=0;repeat<repeats;++repeat){double us[2];for(int order=0;order<2;++order){const int mode=order^(repeat&1);us[mode]=time_graph(mode?graph:reference_graph)/sets;}
            std::printf("dense_lt_timing m=%d n=%d k=%d index=%d repeat=%d baseline_us=%.3f candidate_us=%.3f\n",m,n,k,index,repeat,us[0],us[1]);std::fflush(stdout);}
        check(cudaGraphExecDestroy(graph));
    }
    check(cudaGraphExecDestroy(reference_graph));blas(cublasLtMatmulPreferenceDestroy(pref));blas(cublasLtMatrixLayoutDestroy(a));
    blas(cublasLtMatrixLayoutDestroy(b));blas(cublasLtMatrixLayoutDestroy(c));blas(cublasLtMatmulDescDestroy(op));
}
static void policy_check(cublasHandle_t classic,int m,int n,int padding=0){
    constexpr int k=2560;const int lda=k+padding;const size_t values=size_t(m)*n,outputs=values+64;
    Buffer<__half> weight(size_t(n)*k),input(size_t(m)*lda);
    Buffer<float> baseline(outputs),candidate(outputs);
    std::vector<__half> w(size_t(n)*k),x(size_t(m)*lda);fill(w,147);upload(weight.p,w);w.clear();w.shrink_to_fit();
    mollm_cuda::PrefillLtGemm policy,fresh;const float alpha=1.f,beta=0.f;
    auto original=[&](float* output){blas(cublasGemmEx(classic,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&alpha,
        weight.p,CUDA_R_16F,k,input.p,CUDA_R_16F,lda,&beta,output,CUDA_R_32F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));};
    bool selected=false;auto tuned=[&]{if(!policy.run(classic,weight.p,input.p,candidate.p,m,n,k,lda,selected))std::exit(5);if(!selected)original(candidate.p);};
    fill(x,471);upload(input.p,x);tuned();check(cudaStreamSynchronize(cudaStreamPerThread));const bool expected_selection=selected;
    const char* full_flag=std::getenv("MOLLM_TEST_CUDA_DENSE_WIDE_FULL_LT");
    if(full_flag&&!std::strcmp(full_flag,"1")&&m==1743&&n==16480&&!selected)std::exit(8);
    const char* tile_flag=std::getenv("MOLLM_TEST_CUDA_DENSE_WIDE_TILE");
    const bool expect_tile23=full_flag&&!std::strcmp(full_flag,"1")&&tile_flag&&!std::strcmp(tile_flag,"23")&&m>1024&&n==16480;
    if(expect_tile23&&!policy.wide_tile23_calls())std::exit(22);
    if(!expect_tile23&&policy.wide_tile23_calls())std::exit(23);
    if(m>1024&&n==13312&&selected)std::exit(8);
    auto graph=capture(tuned);
    for(int replay=0;replay<2;++replay){fill(x,471+19*replay);upload(input.p,x);
        check(cudaMemset(baseline.p,0x5a,outputs*4));check(cudaMemset(candidate.p,0x5a,outputs*4));
        original(baseline.p);check(cudaGraphLaunch(graph,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
        auto ref=read(baseline.p,outputs),actual=read(candidate.p,outputs);
        if(selected!=expected_selection||std::memcmp(ref.data(),actual.data(),outputs*4))std::exit(6);
        const auto* bytes=reinterpret_cast<const unsigned char*>(actual.data()+values);for(int i=0;i<256;++i)if(bytes[i]!=0x5a)std::exit(4);
    }
    check(cudaGraphExecDestroy(graph));
    bool handled=true;check(cudaMemset(candidate.p,0x5a,outputs*4));
    if(!fresh.run(classic,weight.p,input.p,candidate.p,32,n,k,k,handled)||handled)std::exit(7);
    if(!fresh.run(classic,weight.p,input.p,candidate.p,2049,n,k,lda,handled)||handled)std::exit(7);
    if(!fresh.run(classic,weight.p,input.p,candidate.p,m,n,k,k-1,handled)||handled)std::exit(7);
    if(!fresh.run(classic,weight.p,input.p+1,candidate.p,m,n,k,lda,handled)||handled)std::exit(7);
    if(!fresh.run(classic,weight.p,input.p,candidate.p+1,m,n,k,lda,handled)||handled)std::exit(7);
    check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
    if(!fresh.run(classic,weight.p,input.p,candidate.p,m,n,k,lda,handled)||handled)std::exit(7);
    cudaGraph_t empty;check(cudaStreamEndCapture(cudaStreamPerThread,&empty));check(cudaGraphDestroy(empty));
    auto untouched=read(candidate.p,outputs);const auto* bytes=reinterpret_cast<const unsigned char*>(untouched.data());for(size_t i=0;i<outputs*4;++i)if(bytes[i]!=0x5a)std::exit(7);
    std::printf("dense_lt_policy_check m=%d n=%d k=%d lda=%d selected=%d captured_changed_inputs=2 byte_exact=true guards=true decode_bad_stride_capture_miss_untouched=true\n",m,n,k,lda,expected_selection);std::fflush(stdout);
}
// Structural control: same inputs and weights, different token-window partition.
// Only the dense production Lt policy/classic GEMM is represented. No claim
// about HC fusion, attention, recurrent state, route grouping or model quality.
static void window_merge_check(cublasHandle_t classic,int n,int k,int nodes,bool timing,bool balanced=false){
    constexpr int first=1024,second=719,rows=first+second,sets=4;
    const size_t weight_values=size_t(n)*k,values=size_t(rows)*n;
    const size_t stride=(values+8+63)&~size_t(63),count=sets*stride;
    Buffer<__half> weight(sets*weight_values),input(size_t(rows)*k);
    Buffer<float> split(count),merged(count);
    std::vector<__half> w(sets*weight_values),x(size_t(rows)*k);
    fill(w,193);upload(weight.p,w);w.clear();w.shrink_to_fit();
    fill(x,139);upload(input.p,x);
    mollm_cuda::PrefillLtGemm policy;
    const float alpha=1.f,beta=0.f;
    bool selected_first=false,selected_second=false,selected_merged=false,selected_tail=false;
    auto gemm=[&](int m,int offset,int set,float* output,bool& selected){
        const auto* a=weight.p+set*weight_values;
        const auto* b=input.p+size_t(offset)*k;
        auto* c=output+set*stride+size_t(offset)*n;
        selected=false;
        if(!policy.run(classic,a,b,c,m,n,k,k,selected))std::exit(5);
        if(!selected)blas(cublasGemmEx(classic,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&alpha,
            a,CUDA_R_16F,k,b,CUDA_R_16F,k,&beta,c,CUDA_R_32F,n,
            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };
    auto partitioned=[&]{for(int set=0;set<sets;++set){
        gemm(first,0,set,split.p,selected_first);
        gemm(second,first,set,split.p,selected_second);
    }};
    auto combined=[&]{for(int set=0;set<sets;++set){
        gemm(balanced?896:rows,0,set,merged.p,selected_merged);
        if(balanced)gemm(rows-896,896,set,merged.p,selected_tail);
    }};
    // Populate any pointer-independent Lt plans before capture.
    partitioned();combined();check(cudaStreamSynchronize(cudaStreamPerThread));
    const bool first_plan=selected_first,second_plan=selected_second,merged_plan=selected_merged,tail_plan=selected_tail;
    auto split_graph=capture(partitioned),merged_graph=capture(combined);
    size_t mismatches=0;double max_abs=0;
    for(int changed=0;changed<3;++changed){
        fill(x,139+changed*17);upload(input.p,x);
        check(cudaMemset(split.p,0x5a,count*4));check(cudaMemset(merged.p,0x5a,count*4));
        if(changed&1){check(cudaGraphLaunch(merged_graph,cudaStreamPerThread));check(cudaGraphLaunch(split_graph,cudaStreamPerThread));}
        else{check(cudaGraphLaunch(split_graph,cudaStreamPerThread));check(cudaGraphLaunch(merged_graph,cudaStreamPerThread));}
        check(cudaStreamSynchronize(cudaStreamPerThread));
        auto a=read(split.p,count),b=read(merged.p,count);
        for(int set=0;set<sets;++set){
            const size_t offset=set*stride;
            for(size_t i=0;i<values;++i){
                const double error=std::abs(double(a[offset+i])-b[offset+i]);
                if(!std::isfinite(a[offset+i])||!std::isfinite(b[offset+i])||
                    error>0.005+0.002*std::abs(double(a[offset+i])))std::exit(3);
                mismatches+=std::memcmp(&a[offset+i],&b[offset+i],4)!=0;
                max_abs=std::max(max_abs,error);
            }
            for(const auto* v:{&a,&b}){
                const auto* bytes=reinterpret_cast<const unsigned char*>(v->data()+offset+values);
                for(size_t i=0;i<(stride-values)*4;++i)if(bytes[i]!=0x5a)std::exit(4);
            }
        }
    }
    if(selected_first!=first_plan||selected_second!=second_plan||selected_merged!=merged_plan||selected_tail!=tail_plan)std::exit(6);
    const char* wide_tile=std::getenv("MOLLM_TEST_CUDA_DENSE_WIDE_ROW_TILE");
    if(!balanced&&n==16480&&k==2560&&wide_tile&&!std::strcmp(wide_tile,"1024")&&
       (!merged_plan||mismatches))std::exit(13);
    std::printf("dense_window_check n=%d k=%d declared_nodes=%d rows=%d+%d candidate_rows=%d+%d weight_sets=%d lt_split=%d,%d lt_merged=%d,%d changed_replays=3 bit_mismatches=%zu max_abs=%.9g finite=true guards=true scope=synthetic_dense_only_not_model_quality\n",
        n,k,nodes,first,second,balanced?896:rows,balanced?rows-896:0,sets,int(first_plan),int(second_plan),int(merged_plan),int(tail_plan),mismatches,max_abs);std::fflush(stdout);
    if(timing)for(int repeat=0;repeat<7;++repeat){
        double us[2];
        for(int order=0;order<2;++order){const int mode=order^(repeat&1);
            us[mode]=time_graph(mode?merged_graph:split_graph,64)/sets;
        }
        std::printf("dense_window_timing n=%d k=%d declared_nodes=%d candidate_rows=%d+%d repeat=%d split_us=%.6f merged_us=%.6f scope=dense_only_not_model\n",
            n,k,nodes,balanced?896:rows,balanced?rows-896:0,repeat,us[0],us[1]);std::fflush(stdout);
    }
    check(cudaGraphExecDestroy(split_graph));check(cudaGraphExecDestroy(merged_graph));
}
static void hc_gate_tile_check(cublasHandle_t classic,int rows,int padding,int n){
    constexpr int k=10240,guard=64;
    const int lda=k+padding; const size_t values=size_t(rows)*n;
    Buffer<__half> weight(size_t(n)*k),input(size_t(rows)*lda);
    Buffer<float> reference(values+guard),actual(values+guard);
    std::vector<__half> w(size_t(n)*k),x(size_t(rows)*lda);
    fill(w,197);upload(weight.p,w);
    auto split=[&]{
        if(rows<=1024||rows>2048){
            if(!mollm_cuda::run_dense_matmul(classic,weight.p,CUDA_R_16F,
                input.p,CUDA_R_16F,reference.p,rows,n,k,lda))std::exit(8);
            return;
        }
        for(int offset=0;offset<rows;offset+=1024)
            if(!mollm_cuda::run_dense_matmul(classic,weight.p,CUDA_R_16F,
                input.p+size_t(offset)*lda,CUDA_R_16F,reference.p+size_t(offset)*n,
                std::min(1024,rows-offset),n,k,lda))std::exit(8);
    };
    auto tiled=[&]{if(!mollm_cuda::run_dense_matmul(classic,weight.p,CUDA_R_16F,
        input.p,CUDA_R_16F,actual.p,rows,n,k,lda,false,1024))std::exit(8);};
    fill(x,381);upload(input.p,x);split();tiled();
    if(mollm_cuda::run_dense_matmul(classic,weight.p,CUDA_R_16F,
        input.p,CUDA_R_16F,actual.p,rows,n,k,lda,false,7))std::exit(12);
    check(cudaStreamSynchronize(cudaStreamPerThread));
    auto control=capture(split),candidate=capture(tiled);
    for(int changed=0;changed<3;++changed){
        fill(x,381+changed*93);upload(input.p,x);
        check(cudaMemset(reference.p,0x5a,(values+guard)*4));
        check(cudaMemset(actual.p,0x5a,(values+guard)*4));
        check(cudaGraphLaunch(control,cudaStreamPerThread));
        check(cudaGraphLaunch(candidate,cudaStreamPerThread));
        auto a=read(reference.p,values+guard),b=read(actual.p,values+guard);
        if(std::memcmp(a.data(),b.data(),values*4))std::exit(9);
        for(size_t i=0;i<values;++i)if(!std::isfinite(b[i]))std::exit(10);
        for(const auto* v:{&a,&b}){
            const auto* bytes=reinterpret_cast<const unsigned char*>(v->data()+values);
            for(int j=0;j<guard*4;++j)if(bytes[j]!=0x5a)std::exit(11);
        }
    }
    check(cudaGraphExecDestroy(control));check(cudaGraphExecDestroy(candidate));
    std::printf("hc_projection_tile n=%d rows=%d lda=%d changed_replays=3 byte_exact=1 guards=1 finite=1\n",n,rows,lda);
}
int main(int argc,char** argv){
    const bool only=argc==2&&!std::strcmp(argv[1],"--check-only");
    const bool remaining=argc==2&&!std::strcmp(argv[1],"--remaining-sweep");
    const bool attention_tiles=argc==2&&!std::strcmp(argv[1],"--attention-logical-tiles");
    const bool attention_finalists=argc==2&&!std::strcmp(argv[1],"--attention-logical-finalists");
    const bool shared_logical=argc==2&&!std::strcmp(argv[1],"--shared-logical-sweep");
    const bool shared_tiles=argc==2&&!std::strcmp(argv[1],"--shared-logical-tiles");
    const bool shared_finalist=argc==2&&!std::strcmp(argv[1],"--shared-logical-finalist");
    const bool logical_sweep=argc==2&&!std::strcmp(argv[1],"--logical-batch-sweep");
    const bool tile_sweep=argc==2&&!std::strcmp(argv[1],"--wide-tile-sweep");
    const bool tile_finalists=argc==2&&!std::strcmp(argv[1],"--wide-tile-finalists");
    const bool logical_policy=argc==2&&!std::strcmp(argv[1],"--logical-policy-only");
    const bool windows=argc==2&&!std::strcmp(argv[1],"--window-merge");
    const bool windows_check=argc==2&&!std::strcmp(argv[1],"--window-merge-check");
    const bool balanced=argc==2&&!std::strcmp(argv[1],"--window-balance");
    const bool balanced_check=argc==2&&!std::strcmp(argv[1],"--window-balance-check");
    const bool gate_tile=argc==2&&!std::strcmp(argv[1],"--hc-gate-tile-check");
    const bool policy_only=argc==2&&!std::strcmp(argv[1],"--policy-only");if(argc>1&&!only&&!policy_only&&!remaining&&!attention_tiles&&!attention_finalists&&!shared_logical&&!shared_tiles&&!shared_finalist&&!logical_sweep&&!tile_sweep&&!tile_finalists&&!logical_policy&&!windows&&!windows_check&&!balanced&&!balanced_check&&!gate_tile)return 64;
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0));if(prop.major!=12||prop.minor>1)return 77;
    setenv("MOLLM_CUDA_HOST_HEADROOM_MB","102400",1);if(!mollm_memory::cuda_host_allocation_allowed(size_t(3)<<30,"bounded dense Lt diagnostic"))return 1;
    std::printf("dense_lt_device name=%s l2_bytes=%d weight_sets=%d clocks_uncontrolled=true\n",prop.name,prop.l2CacheSize,policy_only?1:4);
    cublasHandle_t classic;cublasLtHandle_t lt;blas(cublasCreate(&classic));blas(cublasSetStream(classic,cudaStreamPerThread));blas(cublasLtCreate(&lt));
    if(gate_tile)for(int n:{4,320})for(int rows:{1,1024,1025,1743,2048,2049})for(int padding:{0,8})hc_gate_tile_check(classic,rows,padding,n);
    else if(windows||windows_check||balanced||balanced_check){
        // Declared non-MoE MATMUL shapes in the current calibrated package.
        struct Shape{int n,k,nodes;};
        for(auto s:{Shape{4,10240,96},Shape{320,10240,97},Shape{2560,2560,1},
                    Shape{2560,6144,48},Shape{10240,320,97},Shape{10240,2560,1},
                    Shape{13312,2560,12},Shape{16480,2560,36}})
            window_merge_check(classic,s.n,s.k,s.nodes,windows||balanced,balanced||balanced_check);
    }
    else if(policy_only)for(int m:{512,719,857,896,1024})for(int n:{13312,16480})policy_check(classic,m,n);
    else if(logical_policy)for(int m:{1025,1743,2048})for(int n:{16480,13312})for(int padding:{0,8})policy_check(classic,m,n,padding);
    // Whole logical prefill currently falls back to classic GEMM above1024.
    // Probe advertised algorithms independently; do not change model policy.
    else if(attention_tiles)run(classic,lt,1743,13312,2560,true,3,true,false,true,true);
    else if(attention_finalists)for(int tile:{29,409})run(classic,lt,1743,13312,2560,true,7,true,false,true,true,tile);
    else if(shared_logical)run(classic,lt,1743,2560,6144,true,7,false,false,true);
    else if(shared_tiles)run(classic,lt,1743,2560,6144,true,3,true,false,true,true);
    else if(shared_finalist)run(classic,lt,1743,2560,6144,true,7,true,false,true,true,467);
    else if(tile_finalists)run(classic,lt,1743,16480,2560,true,7,true,true);
    else if(tile_sweep)run(classic,lt,1743,16480,2560,true,3,true);
    else if(logical_sweep)for(int n:{16480,13312})run(classic,lt,1743,n,2560,true,7);
    else if(remaining) {
        const std::pair<int,int> shapes[]={{320,10240},{2560,6144},{6144,2560}};
        for(int m:{1024,719})for(auto shape:shapes)run(classic,lt,m,shape.first,shape.second,true,7);
    } else {
        const std::pair<int,int> shapes[]={{320,2560},{640,2560},{2560,6144},{6144,2560},{10240,320},{13312,2560},{16480,2560}};
        for(int m:{1024,719})for(auto shape:shapes)run(classic,lt,m,shape.first,shape.second,!only);
    }
    blas(cublasDestroy(classic));blas(cublasLtDestroy(lt));
}
