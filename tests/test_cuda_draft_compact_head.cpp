#include "backends/cuda/backend.h"
#include "kernels/cuda/draft_compact_head.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
static void require(bool ok,const char* m){if(!ok){std::fprintf(stderr,"draft head: %s\n",m);std::exit(1);}}
static void check(cudaError_t e){require(e==cudaSuccess,cudaGetErrorString(e));}
static void test_probability() {
    require(mollm_cuda::launch_draft_head_probability(nullptr,1,nullptr,nullptr,1,nullptr)==cudaErrorInvalidValue,"probability rejects null");
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    for(int n:{1,17,257,65537,248320})for(int distribution:{0,1,2}) {
        const int groups=std::min(256,(n+255)/256);
        float *logits=nullptr,*scratch=nullptr;
        mollm_cuda::DraftHeadResult* result=nullptr;
        check(cudaMalloc(&logits,size_t(n)*4));check(cudaMalloc(&scratch,size_t(groups+8)*4));
        check(cudaMalloc(&result,sizeof(*result)+16));check(cudaMemset(result,0x35,sizeof(*result)+16));
        std::vector<float> partials(groups+8,123.25f);check(cudaMemcpy(scratch,partials.data(),partials.size()*4,cudaMemcpyHostToDevice));
        require(mollm_cuda::launch_draft_head_probability(logits,n,result,scratch,0,stream)==cudaErrorInvalidValue,"zero probability groups");
        require(mollm_cuda::launch_draft_head_probability(logits,n,result,scratch,257,stream)==cudaErrorInvalidValue,"too many probability groups");
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        check(mollm_cuda::launch_draft_head_probability(logits,n,result,scratch,groups,stream));
        cudaGraph_t graph;cudaGraphExec_t executable;
        check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
        for(int replay=0;replay<2;++replay) {
            std::vector<float> values(n);
            for(int i=0;i<n;++i)values[i]=distribution==0?10000.f:
                distribution==1?(i==(replay?n-1:0)?80.f:-80.f):float((i*17+replay*31)%1001-500)*.031f+replay*1000.f;
            const int index=int(std::max_element(values.begin(),values.end())-values.begin());
            const float maximum=values[index];
            double denominator=0;for(float value:values)denominator+=std::exp(double(value)-maximum);
            const double expected=1.0/denominator;
            mollm_cuda::DraftHeadResult top{{maximum,index},-9.f,0.f},actual{};
            check(cudaMemcpy(logits,values.data(),size_t(n)*4,cudaMemcpyHostToDevice));
            check(cudaMemcpy(result,&top,sizeof(top),cudaMemcpyHostToDevice));
            check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
            check(cudaMemcpy(&actual,result,sizeof(actual),cudaMemcpyDeviceToHost));
            require(actual.top1.index==index&&actual.top1.value==maximum,"probability preserves argmax");
            require(std::abs(double(actual.probability)-expected)<=3e-6*expected,"GPU probability versus independent CPU double softmax");
            std::vector<unsigned char> guard(16);check(cudaMemcpy(guard.data(),reinterpret_cast<char*>(result)+sizeof(*result),16,cudaMemcpyDeviceToHost));
            require(std::all_of(guard.begin(),guard.end(),[](unsigned char x){return x==0x35;}),"probability result guard");
            check(cudaMemcpy(partials.data(),scratch,partials.size()*4,cudaMemcpyDeviceToHost));
            for(int i=groups;i<groups+8;++i)require(partials[i]==123.25f,"probability scratch guard");
        }
        check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));
        cudaFree(logits);cudaFree(scratch);cudaFree(result);
    }
    check(cudaStreamDestroy(stream));
}
int main(){
    int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||!count)return 77;
    test_probability();
    require(mollm_cuda::pack_q8_draft_head_compact(nullptr,nullptr,nullptr,1,32,cudaStreamPerThread)==cudaErrorInvalidValue,"null inputs rejected");
    for(int k:{32,64,2560})for(int n:{1,17,257}){
        std::vector<int8_t>w(size_t(n)*k);std::vector<float>s(n);
        for(int r=0;r<n;++r){s[r]=(r%3==0?-1.f:1.f)*(.001f*(1+r%7));for(int j=0;j<k;++j)w[size_t(r)*k+j]=r==0?0:int8_t((r*13+j*29)%256-128);}
        std::vector<mollm_cuda::Q4CompactG32>expected(size_t((n+7)/8)*(k/32)),actual(expected.size());
        for(int r=0;r<n;++r)for(int g=0;g<k/32;++g){
            int mx=0;for(int j=0;j<32;++j)mx=std::max(mx,std::abs(int(w[size_t(r)*k+g*32+j])));
            auto& b=expected[size_t(r/8)*(k/32)+g];float maximum=float(mx)*s[r];uint32_t bits;std::memcpy(&bits,&maximum,4);
            b.scales[r%8]=uint16_t((bits+0x7fffu+((bits>>16)&1u))>>16);
            for(int j=0;j<32;++j){int q=mx?int(std::nearbyint(float(w[size_t(r)*k+g*32+j])*(7.f/float(mx)))):0;q=std::clamp(q,-7,7)+8;b.q[r%8][j/2]|=uint8_t(q<<((j%2)*4));}
        }
        int8_t*dw=nullptr;float*ds=nullptr;mollm_cuda::Q4CompactG32*dq=nullptr;
        check(cudaMalloc(&dw,w.size()));check(cudaMalloc(&ds,s.size()*4));check(cudaMalloc(&dq,actual.size()*sizeof(actual[0])));
        check(cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice));check(cudaMemcpy(ds,s.data(),s.size()*4,cudaMemcpyHostToDevice));
        check(mollm_cuda::pack_q8_draft_head_compact(dw,ds,dq,n,k,cudaStreamPerThread));
        check(cudaMemcpy(actual.data(),dq,actual.size()*sizeof(actual[0]),cudaMemcpyDeviceToHost));
        require(!std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(actual[0])),"packed Q4 equals independent CPU quantizer, including padded rows");
        cudaFree(dw);cudaFree(ds);cudaFree(dq);
    }
    std::vector<std::vector<int8_t>> weights_keep;std::vector<std::vector<float>> scales_keep;
    CudaBackend backend;require(backend.available(),"backend available");
    for(int k:{32,36})for(int n:{17,257,17}){
        weights_keep.emplace_back(size_t(n)*k);scales_keep.emplace_back(n,1.f);
        auto& w=weights_keep.back();auto& s=scales_keep.back();
        for(int r=0;r<n;++r){w[size_t(r)*k]=100;w[size_t(r)*k+1]=r<2?int8_t(14+r):int8_t(-127);}
        Tensor weight=Tensor::create(Precision::INT8,MemoryType::EXTERNAL,n,k,1,1,w.data());
        weight.rowmajor_data=w.data();weight.scales=s.data();weight.group_size=k;weight.groups_per_row=1;backend.wrap_weight(weight);
        Tensor input=Tensor::create(Precision::FP32,MemoryType::NONE,k,1);require(backend.alloc_output(input,input.nbytes(),nullptr),"input allocation");
        std::vector<float>x(k);x[1]=1;require(backend.copy_from_host(x.data(),input,k*4),"upload input");
        const bool enabled=true;
        int target=backend.lm_head_argmax_device_and_end_graph(input,0,weight,n,k);
        if(target!=1)std::fprintf(stderr,"target mismatch n=%d k=%d id=%d failed=%d\n",n,k,target,int(backend.dispatch_failed()));
        require(target==1,"target head starts exact");
        require(backend.lm_head_draft_argmax_device_and_end_graph(input,0,weight,n,k)==((enabled&&k%32==0)?0:1),"only draft role may approximate");
        require(backend.lm_head_argmax_device_and_end_graph(input,0,weight,n,k)==1,"target remains exact after draft scope");
        std::fill(x.begin(),x.end(),0.f);require(backend.copy_from_host(x.data(),input,k*4),"changed input");
        require(backend.lm_head_draft_argmax_device_and_end_graph(input,0,weight,n,k)==0,"zero draft tie");
        float probability=99.f;
        require(backend.lm_head_draft_argmax_device_and_end_graph(input,0,weight,n,k,0,nullptr,&probability)==0,"probability keeps stable draft argmax");
        if(enabled&&k%32==0)require(std::abs(double(probability)-1.0/n)<1e-6/n,"backend uniform draft probability");
        else require(probability==-1.f,"unsupported draft confidence is explicit");
        require(backend.lm_head_argmax_device_and_end_graph(input,0,weight,n,k)==0,"probability request does not leak to target head");
        backend.free_output(input,nullptr);
    }
    std::puts("draft Q4 head: independent packing/probability oracles, guarded changed-input graph replay, signed/zero scales, padded rows, weight changes, exact target/draft isolation passed");
}
