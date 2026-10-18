#include "kernels/cuda/qsa.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {
void check(cudaError_t e) {
    if(e!=cudaSuccess){std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));std::exit(1);}
}
void require(bool value){if(!value){std::fprintf(stderr,"QSA assertion failed\n");std::exit(2);}}
struct Buffer {
    void* data=nullptr;
    explicit Buffer(size_t bytes){check(cudaMalloc(&data,bytes));}
    ~Buffer(){if(data)cudaFree(data);}
    Buffer(const Buffer&)=delete;
};

void selected_nan_underflow(bool half) {
    mollm_cuda::QsaSelectedAttentionParams p;
    p.heads=2;p.kv_heads=1;p.queries=1;p.keys=2;p.past=1;
    p.key_dim=p.value_dim=256;p.slots=2;p.scale=1.f/16;
    p.query_position_stride=p.output_position_stride=512;
    p.query_head_stride=p.output_head_stride=256;
    p.key_position_stride=p.value_position_stride=256;
    p.key_head_stride=p.value_head_stride=512;p.index_position_stride=2;
    std::vector<float> q(512,1.f),k(512,100.f),v(512,1.f),actual(512);
    std::fill_n(k.begin(),256,-100.f);v[0]=std::numeric_limits<float>::quiet_NaN();
    std::vector<__half> hk(512),hv(512);
    for(int i=0;i<512;++i){hk[i]=__float2half(k[i]);hv[i]=__float2half(v[i]);}
    int32_t ids[]={0,1};
    Buffer dq(2048),dk(half?1024:2048),dv(half?1024:2048),di(8),out(2048),scores(16);
    check(cudaMemcpy(dq.data,q.data(),2048,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dk.data,half?static_cast<void*>(hk.data()):static_cast<void*>(k.data()),half?1024:2048,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dv.data,half?static_cast<void*>(hv.data()):static_cast<void*>(v.data()),half?1024:2048,cudaMemcpyHostToDevice));
    check(cudaMemcpy(di.data,ids,8,cudaMemcpyHostToDevice));
    for(int mode:{0,1,2}) {
        p.reuse_gqa=mode==1;p.partitions=mode==2?2:1;p.selected_scores=static_cast<float*>(scores.data);
        check(mollm_cuda::launch_qsa_selected_attention(static_cast<float*>(dq.data),dk.data,dv.data,half,
            static_cast<int32_t*>(di.data),static_cast<float*>(out.data),p));
        check(cudaMemcpy(actual.data(),out.data,2048,cudaMemcpyDeviceToHost));
        for(int h=0;h<2;++h) {
            require(std::isnan(actual[h*256]));
            for(int d=1;d<256;++d)require(std::fabs(actual[h*256+d]-1.f)<3e-6);
        }
    }
}

void topk(int keys,int budget=2048,int ratio=4,int start=-1,bool force=false) {
    const int rows=std::min(keys,3),past=start<0?keys-rows:start,slots=budget+ratio-1;
    const size_t ss=keys/ratio+5,is=slots+3;
    std::vector<float> scores(rows*ss);
    Buffer ds(scores.size()*4),di((rows*is+16)*4);
    const size_t count=size_t(rows)*2*(budget/ratio);
    Buffer candidates((count+16)*4),candidate_ids((count+16)*4);
    auto* cs=static_cast<float*>(candidates.data)+8;
    auto* ci=static_cast<int32_t*>(candidate_ids.data)+8;
    std::vector<float> candidate_guard(count+16,12345.f);
    std::vector<int32_t> id_guard(count+16,12345);
    check(cudaMemcpy(candidates.data,candidate_guard.data(),candidate_guard.size()*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(candidate_ids.data,id_guard.data(),id_guard.size()*4,cudaMemcpyHostToDevice));
    cudaStream_t stream;check(cudaStreamCreate(&stream));
    cudaGraph_t graph;cudaGraphExec_t exec;
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    check(mollm_cuda::launch_qsa_causal_block_topk(static_cast<float*>(ds.data),
        static_cast<int32_t*>(di.data)+8,rows,keys,past,budget,ratio,ss,is,stream,cs,ci,force));
    check(cudaStreamEndCapture(stream,&graph));
    check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    for(int variant=0;variant<3;++variant) {
        for(int r=0;r<rows;++r)for(size_t b=0;b<ss;++b) {
            const bool future=int(b)>=(past+r+1)/ratio;
            scores[r*ss+b]=future?std::numeric_limits<float>::quiet_NaN():
                variant==0?float(b):variant==1?1.f:-float(b);
        }
        check(cudaMemcpy(ds.data,scores.data(),scores.size()*4,cudaMemcpyHostToDevice));
        std::vector<int32_t> actual(rows*is+16,12345);
        check(cudaMemcpy(di.data,actual.data(),actual.size()*4,cudaMemcpyHostToDevice));
        if(variant==0)check(mollm_cuda::launch_qsa_causal_block_topk(static_cast<float*>(ds.data),
            static_cast<int32_t*>(di.data)+8,rows,keys,past,budget,ratio,ss,is,stream,cs,ci,force));
        else check(cudaGraphLaunch(exec,stream));
        check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(actual.data(),di.data,actual.size()*4,cudaMemcpyDeviceToHost));
        for(int i=0;i<8;++i){require(actual[i]==12345);require(actual[rows*is+8+i]==12345);}
        for(int r=0;r<rows;++r) {
            const int visible=past+r+1,complete=visible/ratio;
            std::vector<int> blocks;
            for(int b=0;b<complete;++b)blocks.push_back(b);
            std::stable_sort(blocks.begin(),blocks.end(),[&](int a,int b){return scores[r*ss+a]>scores[r*ss+b];});
            blocks.resize(std::min(budget/ratio,complete));
            std::vector<int> expected;
            for(int b:blocks)for(int d=0;d<ratio;++d)expected.push_back(b*ratio+d);
            for(int id=complete*ratio;id<visible;++id)expected.push_back(id);
            std::sort(expected.begin(),expected.end());
            expected.resize(slots,-1);
            for(int s=0;s<slots;++s)require(actual[8+r*is+s]==expected[s]);
            for(size_t s=slots;s<is;++s)require(actual[8+r*is+s]==12345);
        }
    }
    check(cudaMemcpy(candidate_guard.data(),candidates.data,candidate_guard.size()*4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(id_guard.data(),candidate_ids.data,id_guard.size()*4,cudaMemcpyDeviceToHost));
    for(int i=0;i<8;++i){require(candidate_guard[i]==12345.f&&candidate_guard[count+8+i]==12345.f);
        require(id_guard[i]==12345&&id_guard[count+8+i]==12345);}
    require(mollm_cuda::launch_qsa_causal_block_topk(static_cast<float*>(ds.data),
        static_cast<int32_t*>(di.data),rows,32772,0,2048,4,ss,is,stream)==cudaErrorInvalidValue);
    check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));
}

void run(bool half,int keys,int rows,int heads,int kv,int kd,int vd,int slots,bool interleaved,bool use_mask=false,bool reuse=false,int partitions=1,float stress_scale=1.f) {
    mollm_cuda::QsaSelectedAttentionParams p{};
    p.reuse_gqa=reuse;
    p.partitions=partitions;
    Buffer scores(size_t(rows)*heads*slots*4);
    p.selected_scores=static_cast<float*>(scores.data);
    p.heads=heads;p.kv_heads=kv;p.keys=keys;p.queries=rows;p.past=keys-rows;
    p.key_dim=kd;p.value_dim=vd;p.slots=slots;p.scale=stress_scale/std::sqrt(float(kd));
    p.query_position_stride=kd*heads+3;p.query_head_stride=kd;
    p.output_position_stride=vd*heads+7;p.output_head_stride=vd;
    if(interleaved) {
        // Main graph's Q/output are head-major, unlike token-major index Q.
        p.query_position_stride=kd+3;p.query_head_stride=rows*p.query_position_stride+5;
        p.output_position_stride=vd+7;p.output_head_stride=rows*p.output_position_stride+3;
    }
    p.index_position_stride=slots+3;
    p.mask_column_stride=1;p.mask_row_stride=keys;
    p.key_position_stride=interleaved?kd*kv+5:kd+5;
    p.value_position_stride=interleaved?vd*kv+3:vd+3;
    p.key_head_stride=interleaved?kd:keys*p.key_position_stride+7;
    p.value_head_stride=interleaved?vd:keys*p.value_position_stride+7;
    const size_t kn=size_t(keys-1)*p.key_position_stride+(kv-1)*p.key_head_stride+kd;
    const size_t vn=size_t(keys-1)*p.value_position_stride+(kv-1)*p.value_head_stride+vd;
    const size_t on=size_t(rows-1)*p.output_position_stride+(heads-1)*p.output_head_stride+vd+3;
    const size_t qn=size_t(rows-1)*p.query_position_stride+(heads-1)*p.query_head_stride+kd+3;
    std::vector<float> query(qn,.125f);
    std::vector<bool> written(on,false);
    for(int r=0;r<rows;++r)for(int h=0;h<heads;++h)for(int d=0;d<vd;++d)
        written[r*p.output_position_stride+h*p.output_head_stride+d]=true;
    std::vector<float> key(kn,std::numeric_limits<float>::quiet_NaN());
    std::vector<float> value(vn,std::numeric_limits<float>::quiet_NaN());
    std::vector<int32_t> ids(rows*p.index_position_stride,-1);
    std::vector<float> mask(size_t(rows)*keys,-1e38f);
    for(int r=0;r<rows;++r)for(int id=0;id<=p.past+r;++id)
        mask[size_t(r)*keys+id]=use_mask&&id%6==1?-1e38f:0.f;
    // In full-size cases select the most recent 512 complete four-token
    // groups and the visible tail. This is a supplied selection, NOT indexer
    // qualification. All other KV values remain NaN to prove selective reads.
    for(int r=0;r<rows;++r) {
        const int visible=p.past+r+1;
        std::vector<int> selected;
        if(slots==2051) {
            const int end=visible/4*4;
            for(int id=std::max(0,end-2048);id<visible;++id)selected.push_back(id);
        } else {
            for(int id=1;id<visible&&int(selected.size())<slots-2;id+=3)selected.push_back(id);
        }
        for(size_t s=0;s<selected.size();++s)ids[r*p.index_position_stride+s]=selected[s];
        if(slots!=2051&&r==0)std::fill_n(ids.begin(),slots,-1); // entirely masked row
        for(int id:selected)for(int h=0;h<kv;++h) {
            if(use_mask&&id%6==1)continue; // masked selected KV remains NaN
            for(int d=0;d<kd;++d) {
                float x=std::sin(float((id%193)*7+h*3+d)*.017f)*.3f;
                key[id*p.key_position_stride+h*p.key_head_stride+d]=half?__half2float(__float2half(x)):x;
            }
            for(int d=0;d<vd;++d) {
                float x=std::cos(float((id%127)*13+h*11+d)*.031f)*.2f;
                value[id*p.value_position_stride+h*p.value_head_stride+d]=half?__half2float(__float2half(x)):x;
            }
        }
        if(slots!=2051) {
            ids[r*p.index_position_stride+slots-1]=keys+17; // invalid upper bound
            if(r<rows-1)ids[r*p.index_position_stride+slots-2]=keys-1; // future
        }
    }
    Buffer dq(query.size()*4),dk(kn*(half?2:4)),dv(vn*(half?2:4)),di(ids.size()*4),out((on+16)*4);
    Buffer dm(mask.size()*4);
    const float* device_mask=use_mask?static_cast<float*>(dm.data):nullptr;
    if(use_mask)check(cudaMemcpy(dm.data,mask.data(),mask.size()*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(di.data,ids.data(),ids.size()*4,cudaMemcpyHostToDevice));
    if(half) {
        std::vector<__half> k(kn),v(vn);
        for(size_t i=0;i<kn;++i)k[i]=__float2half(key[i]);
        for(size_t i=0;i<vn;++i)v[i]=__float2half(value[i]);
        check(cudaMemcpy(dk.data,k.data(),kn*2,cudaMemcpyHostToDevice));
        check(cudaMemcpy(dv.data,v.data(),vn*2,cudaMemcpyHostToDevice));
    } else {
        check(cudaMemcpy(dk.data,key.data(),kn*4,cudaMemcpyHostToDevice));
        check(cudaMemcpy(dv.data,value.data(),vn*4,cudaMemcpyHostToDevice));
    }
    cudaStream_t stream;check(cudaStreamCreate(&stream));
    cudaGraph_t graph;cudaGraphExec_t exec;
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    check(mollm_cuda::launch_qsa_selected_attention(static_cast<float*>(dq.data),dk.data,dv.data,half,
          static_cast<int32_t*>(di.data),static_cast<float*>(out.data)+8,p,stream,device_mask));
    check(cudaStreamEndCapture(stream,&graph));
    check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    for(int variant=0;variant<3;++variant) {
        for(size_t i=0;i<query.size();++i)query[i]=std::sin(float(i+17*variant)*.019f)*.4f;
        check(cudaMemcpy(dq.data,query.data(),query.size()*4,cudaMemcpyHostToDevice));
        std::vector<float> actual(on+16,12345.f);
        check(cudaMemcpy(out.data,actual.data(),actual.size()*4,cudaMemcpyHostToDevice));
        if(variant==0)
            check(mollm_cuda::launch_qsa_selected_attention(static_cast<float*>(dq.data),dk.data,dv.data,half,
                  static_cast<int32_t*>(di.data),static_cast<float*>(out.data)+8,p,stream,device_mask));
        else check(cudaGraphLaunch(exec,stream));
        check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(actual.data(),out.data,actual.size()*4,cudaMemcpyDeviceToHost));
        for(int i=0;i<8;++i){require(actual[i]==12345.f);require(actual[on+8+i]==12345.f);}
        if(p.reuse_gqa||p.partitions>1) {
            Buffer reference_output(actual.size()*4);
            std::vector<float> reference(actual.size(),12345.f);
            check(cudaMemcpy(reference_output.data,reference.data(),reference.size()*4,cudaMemcpyHostToDevice));
            auto legacy=p;legacy.reuse_gqa=false;legacy.partitions=1;legacy.selected_scores=nullptr;
            check(mollm_cuda::launch_qsa_selected_attention(static_cast<float*>(dq.data),dk.data,dv.data,half,
                static_cast<int32_t*>(di.data),static_cast<float*>(reference_output.data)+8,legacy,stream,device_mask));
            check(cudaStreamSynchronize(stream));
            check(cudaMemcpy(reference.data(),reference_output.data,reference.size()*4,cudaMemcpyDeviceToHost));
            require(std::memcmp(actual.data(),reference.data(),actual.size()*4)==0);
        }
        for(int r=0;r<rows;++r)for(int h=0;h<heads;++h) {
            std::vector<double> scores(slots,-std::numeric_limits<double>::infinity());
            double maximum=-std::numeric_limits<double>::infinity();
            const int kh=h/(heads/kv);
            for(int s=0;s<slots;++s) {
                const int id=ids[r*p.index_position_stride+s];
                if(id<0||id>=keys||id>p.past+r||(use_mask&&mask[size_t(r)*keys+id]!=0.f))continue;
                double dot=0;
                for(int d=0;d<kd;++d)dot+=double(query[r*p.query_position_stride+h*p.query_head_stride+d])*
                    key[id*p.key_position_stride+kh*p.key_head_stride+d];
                scores[s]=dot*p.scale;maximum=std::max(maximum,scores[s]);
            }
            double total=0;
            for(double& x:scores){x=std::isfinite(x)?std::exp(x-maximum):0.;total+=x;}
            for(int d=0;d<vd;++d) {
                double expected=0;
                if(total>0)for(int s=0;s<slots;++s)if(scores[s]>0) {
                    const int id=ids[r*p.index_position_stride+s];
                    expected+=scores[s]/total*value[id*p.value_position_stride+kh*p.value_head_stride+d];
                }
                const float got=actual[8+r*p.output_position_stride+h*p.output_head_stride+d];
                require(std::isfinite(got)&&std::fabs(got-expected)<3e-6);
            }
        }
        for(size_t i=0;i<on;++i)if(!written[i])require(actual[8+i]==12345.f);
    }
    auto bad=p;bad.slots=4097;
    require(mollm_cuda::launch_qsa_selected_attention(static_cast<float*>(dq.data),dk.data,dv.data,half,
         static_cast<int32_t*>(di.data),static_cast<float*>(out.data)+8,bad,stream)==cudaErrorInvalidValue);
    check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));
}
}
int main(int argc,char** argv) {
    int count=0;
    if(cudaGetDeviceCount(&count)!=cudaSuccess||count==0)return 77;
    if(argc==2&&!std::strcmp(argv[1],"--streaming-topk-only")) {
        for(int keys:{32772,32775,65536,131071,262144})topk(keys);
        topk(65536,2048,4,0);
        topk(65536,1024,1);topk(65536,2048,1);topk(65536,4096,1);
        topk(262143,4080,16);
        for(int keys:{1,3,2048,32768})topk(keys,2048,4,-1,true);
        std::puts("QSA streaming top-k: boundaries, all merge widths, ties, tails, guards and capture pass");return 0;
    }
    for(bool half:{false,true})selected_nan_underflow(half);
    if(argc==2&&!std::strcmp(argv[1],"--optimized-small-only")) {
        for(bool half:{false,true})for(bool interleaved:{false,true}) {
            run(half,131,3,24,2,256,256,131,interleaved,true,true);
            run(half,131,3,24,2,256,256,131,interleaved,true,false,16);
            run(half,19,3,6,2,256,256,9,interleaved,true,true);
            run(half,19,3,4,2,8,5,9,interleaved,true,false,16);
            run(half,131,3,24,2,256,256,131,interleaved,true,true,1,64.f);
            run(half,131,3,24,2,256,256,131,interleaved,true,false,16,64.f);
            run(half,32,32,24,2,256,256,131,interleaved,false,true);
        }
        std::puts("QSA optimized small-only: grouped/split, strides, masked poison and capture pass");
        return 0;
    }
    for(int keys:{1,3,2048,2049,2050,2051,2052,2054,8193,16387,32768,32772,65536,262144})topk(keys);
    for(bool half:{false,true})for(bool interleaved:{false,true}) {
        run(half,19,3,4,2,8,5,9,interleaved);
        run(half,19,3,4,2,8,5,9,interleaved,true);
        run(half,2054,3,4,2,17,13,2051,interleaved);
        run(half,32768,1,24,2,256,256,2051,interleaved);
        run(half,32768,1,24,2,256,256,2051,interleaved,false,true);
        run(half,2054,3,24,2,256,256,2051,interleaved,true,true);
        run(half,19,3,6,2,256,256,9,interleaved,true,true);
        run(half,32768,1,24,2,256,256,2051,interleaved,false,false,16);
        run(half,19,3,4,2,8,5,9,interleaved,true,false,16);
        run(half,131,3,24,2,256,256,131,interleaved,true,true,1,64.f);
        run(half,131,3,24,2,256,256,131,interleaved,true,false,16,64.f);
        run(half,64,64,24,2,256,256,2051,interleaved,false,true);
    }
    std::puts("QSA selected-key attention: FP16/FP32, GQA, poison, bounds, strides and capture pass");
}
