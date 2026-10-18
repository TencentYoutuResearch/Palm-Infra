#pragma once

#include "backends/cuda/device_runtime.h"
#include "kernels/cuda/matmul.h"
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace mollm_cuda {
// Experimental SM121 prefill policy. Cache only advertised, zero-workspace
// algorithms; no weight conversion or tuning launches during model execution.
class PrefillLtGemm {
    struct Plan {
        int m=0,n=0,k=0,lda=0,alignment=0;
        cublasLtMatmulDesc_t op=nullptr;
        cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
        cublasLtMatmulAlgo_t algo{};
        bool selected=false;
        bool wide_tile23=false;
        ~Plan(){if(op)cublasLtMatmulDescDestroy(op);if(a)cublasLtMatrixLayoutDestroy(a);
            if(b)cublasLtMatrixLayoutDestroy(b);if(c)cublasLtMatrixLayoutDestroy(c);}
    };
    cublasLtHandle_t handle_=nullptr;
    std::vector<std::unique_ptr<Plan>> plans_;
    size_t calls_=0,fallbacks_=0,wide_tile23_calls_=0;
    static int alignment(const void* p){int a=256;while(a>1&&(reinterpret_cast<uintptr_t>(p)&(a-1)))a/=2;return a;}
public:
    PrefillLtGemm(){if(cublasLtCreate(&handle_)!=CUBLAS_STATUS_SUCCESS)handle_=nullptr;}
    ~PrefillLtGemm(){
        if(calls_||fallbacks_)std::fprintf(stderr,"CudaBackend: prefill Lt GEMM calls=%zu fallbacks=%zu plans=%zu workspace_bytes=0\n",calls_,fallbacks_,plans_.size());
        if(wide_tile23_calls_)std::fprintf(stderr,"CudaBackend: wide Lt tile23 calls=%zu workspace_bytes=0\n",wide_tile23_calls_);
        plans_.clear();if(handle_)cublasLtDestroy(handle_);
    }
    PrefillLtGemm(const PrefillLtGemm&)=delete;
    PrefillLtGemm& operator=(const PrefillLtGemm&)=delete;
    size_t wide_tile23_calls()const{return wide_tile23_calls_;}
    bool run(cublasHandle_t classic,const void* weight,const void* input,float* output,
             int m,int n,int k,int lda,bool& handled){
        handled=false;
        static const bool wide_row_tile=[] {
            const char* value=std::getenv("MOLLM_TEST_CUDA_DENSE_WIDE_ROW_TILE");
            return value&&!std::strcmp(value,"1024");
        }();
        static const bool wide_full_lt=[] {
            const char* value=std::getenv("MOLLM_TEST_CUDA_DENSE_WIDE_FULL_LT");
            return value&&!std::strcmp(value,"1");
        }();
        static const bool wide_tile23=[] {
            const char* value=std::getenv("MOLLM_TEST_CUDA_DENSE_WIDE_TILE");
            return value&&!std::strcmp(value,"23");
        }();
        // Keep logical prompt execution whole while recovering existing Lt
        // choices for this particular projection. No new tuning/workspace.
        if(wide_row_tile&&handle_&&weight&&input&&output&&m>1024&&m<=2048&&
           n==16480&&k==2560&&lda>=k) {
            for(int offset=0;offset<m;offset+=1024) {
                const int rows=std::min(1024,m-offset);
                const auto* activation=static_cast<const __half*>(input)+size_t(offset)*lda;
                auto* destination=output+size_t(offset)*n;
                bool selected=false;
                if(!run(classic,weight,activation,destination,rows,n,k,lda,selected))return false;
                if(!selected&&!run_dense_matmul(classic,weight,CUDA_R_16F,
                        activation,CUDA_R_16F,destination,rows,n,k,lda))return false;
            }
            handled=true;return true;
        }
        const bool full_wide=wide_full_lt&&m>1024&&m<=2048&&n==16480;
        if(!handle_||!weight||!input||!output||m<512||(m>1024&&!full_wide)||k!=2560||(n!=13312&&n!=16480)||lda<k)return true;
        const int align=std::min({alignment(weight),alignment(input),alignment(output)});
        if(align<16){++fallbacks_;return true;}
        cudaStream_t stream=nullptr;
        if(!report_cublas(cublasGetStream(classic,&stream),"prefill Lt stream"))return false;
        auto found=std::find_if(plans_.begin(),plans_.end(),[&](const auto& p){
            return p->m==m&&p->n==n&&p->k==k&&p->lda==lda&&p->alignment==align;});
        if(found==plans_.end()){
            cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
            if(!report_cuda(cudaStreamIsCapturing(stream,&capture),"prefill Lt capture check"))return false;
            if(capture!=cudaStreamCaptureStatusNone||plans_.size()>=32){++fallbacks_;return true;}
            auto p=std::make_unique<Plan>();p->m=m;p->n=n;p->k=k;p->lda=lda;p->alignment=align;
            cublasLtMatmulPreference_t pref=nullptr;
            cublasOperation_t transpose=CUBLAS_OP_T;
            cublasStatus_t status=cublasLtMatmulDescCreate(&p->op,CUBLAS_COMPUTE_32F,CUDA_R_32F);
            if(status==CUBLAS_STATUS_SUCCESS)status=cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_TRANSA,&transpose,sizeof(transpose));
            if(status==CUBLAS_STATUS_SUCCESS)status=cublasLtMatrixLayoutCreate(&p->a,CUDA_R_16F,k,n,k);
            if(status==CUBLAS_STATUS_SUCCESS)status=cublasLtMatrixLayoutCreate(&p->b,CUDA_R_16F,k,m,lda);
            if(status==CUBLAS_STATUS_SUCCESS)status=cublasLtMatrixLayoutCreate(&p->c,CUDA_R_32F,n,m,n);
            if(status==CUBLAS_STATUS_SUCCESS)status=cublasLtMatmulPreferenceCreate(&pref);
            const size_t workspace=0;const uint32_t min_alignment=align;
            if(status==CUBLAS_STATUS_SUCCESS)status=cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace));
            for(auto attribute:{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
                if(status==CUBLAS_STATUS_SUCCESS)status=cublasLtMatmulPreferenceSetAttribute(pref,attribute,&min_alignment,sizeof(min_alignment));
            cublasLtMatmulHeuristicResult_t options[16];int count=0;
            if(status==CUBLAS_STATUS_SUCCESS)status=cublasLtMatmulAlgoGetHeuristic(handle_,p->op,p->a,p->b,p->c,p->c,pref,16,options,&count);
            const int desired_id=m>=896?67:21;
            const int desired_tile=m>=896?(n==13312?414:412):20;
            if(status==CUBLAS_STATUS_SUCCESS)for(int i=0;i<count;++i){
                if(options[i].state!=CUBLAS_STATUS_SUCCESS||options[i].workspaceSize)continue;
                int id=-1,tile=-1,split=-1;size_t written=0;
                if(cublasLtMatmulAlgoConfigGetAttribute(&options[i].algo,CUBLASLT_ALGO_CONFIG_ID,&id,sizeof(id),&written)!=CUBLAS_STATUS_SUCCESS||
                   cublasLtMatmulAlgoConfigGetAttribute(&options[i].algo,CUBLASLT_ALGO_CONFIG_TILE_ID,&tile,sizeof(tile),&written)!=CUBLAS_STATUS_SUCCESS||
                   cublasLtMatmulAlgoConfigGetAttribute(&options[i].algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split),&written)!=CUBLAS_STATUS_SUCCESS)continue;
                if(id==desired_id&&tile==desired_tile&&split==1){p->algo=options[i].algo;p->selected=true;break;}
            }
            // Copy the existing qualified configuration; change only its tile.
            // Validate geometry/alignment through the same layouts/preferences.
            // On unavailable/unsupported candidates retain the original plan.
            if(p->selected&&full_wide&&wide_tile23){
                auto candidate=p->algo;const int tile=23;
                cublasLtMatmulHeuristicResult_t checked{};
                bool aligned=cublasLtMatmulAlgoConfigSetAttribute(&candidate,CUBLASLT_ALGO_CONFIG_TILE_ID,&tile,sizeof(tile))==CUBLAS_STATUS_SUCCESS;
                for(auto attribute:{CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES,CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES,
                    CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES,CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES}){
                    uint32_t required=0;size_t written=0;
                    if(cublasLtMatmulAlgoCapGetAttribute(&candidate,attribute,&required,sizeof(required),&written)!=CUBLAS_STATUS_SUCCESS||
                       written!=sizeof(required)||required>uint32_t(align))aligned=false;
                }
                if(aligned&&
                   cublasLtMatmulAlgoCheck(handle_,p->op,p->a,p->b,p->c,p->c,&candidate,&checked)==CUBLAS_STATUS_SUCCESS&&
                   checked.state==CUBLAS_STATUS_SUCCESS&&!checked.workspaceSize){
                    p->algo=candidate;p->wide_tile23=true;
                }
            }
            if(pref)cublasLtMatmulPreferenceDestroy(pref);
            plans_.push_back(std::move(p));found=plans_.end()-1;
        }
        const auto& p=**found;if(!p.selected){++fallbacks_;return true;}
        const float alpha=1.f,beta=0.f;handled=true;
        if(!report_cublas(cublasLtMatmul(handle_,p.op,&alpha,weight,p.a,input,p.b,&beta,output,p.c,output,p.c,
            &p.algo,nullptr,0,stream),"prefill Lt matmul"))return false;
        ++calls_;if(p.wide_tile23)++wide_tile23_calls_;return true;
    }
};
} // namespace mollm_cuda
