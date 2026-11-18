// Numerical, graph-replay and performance checks for the greedy Q8 head screen.
#include "kernels/cuda/matmul.h"
#include "kernels/cuda/q8_head_screen.h"
#include "kernels/cuda/elementwise.h"
#include <limits>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <fstream>
#include <string>

static void check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(1);}}
static void require_at(bool ok,int line){if(!ok){std::fprintf(stderr,"Q6 screen assertion failed at line %d\n",line);std::exit(2);}}
#define require(ok) require_at((ok),__LINE__)
struct Buffer{void* p=nullptr;explicit Buffer(size_t bytes){check(cudaMalloc(&p,bytes));}~Buffer(){cudaFree(p);}template<class T>T* as(){return static_cast<T*>(p);}};
__device__ int8_t code8(size_t i){unsigned x=unsigned(i);x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;return int8_t(x&255);}
__global__ void initialize(int8_t* w,float* scales,int n,int k){
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(n)*k;i+=size_t(gridDim.x)*blockDim.x){
        size_t j=i/k==1?i-k:i;w[i]=code8(j); // Rows0/1 tie exactly.
    }
    for(int r=blockIdx.x*blockDim.x+threadIdx.x;r<n;r+=gridDim.x*blockDim.x)scales[r]=r<2?.001f:(r%13==0?-.001f:.001f);
}
int main(int argc,char** argv){
    const bool large_numeric=argc==2&&!std::strcmp(argv[1],"--large-numeric");
    const bool real=argc==3&&!std::strcmp(argv[1],"--real");
    const bool bench=real||(argc==2&&!std::strcmp(argv[1],"--bench"));
    if(argc!=1&&!bench&&!large_numeric){std::fprintf(stderr,"Usage: %s [--bench | --large-numeric | --real capture-prefix]\n",argv[0]);return 1;}
    std::vector<float> real_inputs;
    auto load=[&](const char* suffix,void* device,size_t bytes){
        std::ifstream file(std::string(argv[2])+suffix,std::ios::binary|std::ios::ate);
        require(file.good()&&file.tellg()==std::streamoff(bytes));file.seekg(0);
        std::vector<char> buffer(std::min(bytes,size_t(4)*1024*1024));
        for(size_t offset=0;offset<bytes;offset+=buffer.size()){
            const size_t chunk=std::min(buffer.size(),bytes-offset);
            require(bool(file.read(buffer.data(),chunk)));
            check(cudaMemcpy(static_cast<char*>(device)+offset,buffer.data(),chunk,cudaMemcpyHostToDevice));
        }
    };
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
    const int n=(bench||large_numeric)?248320:4099,k=2560;
    Buffer w(size_t(n)*k),packed(size_t(n)*k/4*3),scales(size_t(n)*4),x(k*4),a(n*4),out(n*4),reference(n*4),norm(8),best(257*4),count(4),ids(n*4);
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    if(real){
        load(".weights.i8",w.p,size_t(n)*k);load(".scales.f32",scales.p,size_t(n)*4);
        std::ifstream file(std::string(argv[2])+".inputs.f32",std::ios::binary|std::ios::ate);
        require(file.good());const size_t bytes=size_t(file.tellg());
        require(bytes>0&&bytes%(k*4)==0&&bytes<=size_t(k)*4*1024);
        real_inputs.resize(bytes/4);file.seekg(0);require(bool(file.read(reinterpret_cast<char*>(real_inputs.data()),bytes)));
    }else initialize<<<256,256,0,stream>>>(w.as<int8_t>(),scales.as<float>(),n,k);
    check(mollm_cuda::pack_q8_head_screen(w.as<int8_t>(),packed.as<uint8_t>(),n,stream));check(cudaStreamSynchronize(stream));
    auto control=[&]{check(mollm_cuda::launch_q8_tuned_gemv(x.as<float>(),w.as<int8_t>(),scales.as<float>(),reference.as<float>(),n,k,8,false,stream));};
    auto candidate=[&]{
        check(mollm_cuda::launch_q8_head_screen(x.as<float>(),w.as<int8_t>(),packed.as<uint8_t>(),scales.as<float>(),out.as<float>(),n,
            {a.as<float>(),best.as<float>(),norm.as<double>(),count.as<unsigned>(),ids.as<int>()},stream));
    };
    Buffer argmax_partial(256*sizeof(mollm_cuda::ArgMaxPair)),argmax_result(sizeof(mollm_cuda::ArgMaxPair));
    auto argmax=[&](const float* values){
        mollm_cuda::launch_argmax(values,n,argmax_partial.as<mollm_cuda::ArgMaxPair>(),256,argmax_result.as<mollm_cuda::ArgMaxPair>());
        mollm_cuda::ArgMaxPair result;check(cudaMemcpy(&result,argmax_result.p,sizeof(result),cudaMemcpyDeviceToHost));return result.index;
    };
    std::vector<float> original_scales(n);check(cudaMemcpy(original_scales.data(),scales.p,n*4,cudaMemcpyDeviceToHost));
    cudaGraph_t graph;cudaGraphExec_t executable;check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));candidate();check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    for(int fixture=0;fixture<(real?int(real_inputs.size()/k):(bench?3:15));++fixture){
        std::vector<float> input(k),got(n),ref(n),approx(n),scale(n);std::vector<int> selected(n);
        if(real)std::copy_n(real_inputs.data()+size_t(fixture)*k,k,input.data());
        else if(fixture==1){std::vector<int8_t> row(k);check(cudaMemcpy(row.data(),w.p,k,cudaMemcpyDeviceToHost));for(int i=0;i<k;++i)input[i]=float(row[i])/128.f;}
        else if(fixture>=2)for(int i=0;i<k;++i)input[i]=std::sin(float(i)*.31f);
        if(!real&&!bench){
            scale=original_scales;
            switch(fixture){
                case 3:for(float& v:input)v*=1e16f;break; // eligible large finite
                case 4:for(float& v:input)v*=1e20f;break; // out-of-range fallback
                case 5:for(float& v:input)v*=1e-40f;break; // subnormal input
                case 6:std::fill(scale.begin(),scale.end(),1e-40f);break;
                case 7:std::fill(scale.begin(),scale.end(),1e10f);break;
                case 8:std::fill(scale.begin(),scale.end(),1e11f);break;
                case 9:input[17]=std::numeric_limits<float>::quiet_NaN();break;
                case 10:input[17]=std::numeric_limits<float>::infinity();break;
                case 11:scale[0]=std::numeric_limits<float>::infinity();scale[1]=std::numeric_limits<float>::quiet_NaN();break;
                case 12:std::fill(input.begin(),input.end(),0.f);std::fill(scale.begin(),scale.end(),std::numeric_limits<float>::max());break;
                case 13:std::fill(scale.begin(),scale.end(),std::numeric_limits<float>::quiet_NaN());break;
                case 14:std::fill(scale.begin(),scale.end(),-0.f);break;
            }
            check(cudaMemcpyAsync(scales.p,scale.data(),n*4,cudaMemcpyHostToDevice,stream));
        }
        check(cudaMemcpyAsync(x.p,input.data(),k*4,cudaMemcpyHostToDevice,stream));control();check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
        unsigned retained=0;check(cudaMemcpy(&retained,count.p,4,cudaMemcpyDeviceToHost));require(retained>0&&retained<=unsigned(n));
        check(cudaMemcpy(got.data(),out.p,n*4,cudaMemcpyDeviceToHost));check(cudaMemcpy(ref.data(),reference.p,n*4,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(approx.data(),a.p,n*4,cudaMemcpyDeviceToHost));check(cudaMemcpy(scale.data(),scales.p,n*4,cudaMemcpyDeviceToHost));
        double l1=0;for(float v:input)l1+=std::fabs(double(v));
        check(cudaMemcpy(selected.data(),ids.p,retained*4,cudaMemcpyDeviceToHost));
        for(int r=0;r<n;++r)if(std::isfinite(approx[r])&&std::isfinite(l1)&&l1<=1e20&&std::isfinite(scale[r])&&std::fabs(scale[r])<=1e10f)
            require(std::fabs(double(approx[r])-ref[r])<=2.04*std::fabs(double(scale[r]))*l1+1e-5);
        for(unsigned i=0;i<retained;++i){int r=selected[i];require(r>=0&&r<n);
            require((std::isnan(got[r])&&std::isnan(ref[r]))||std::memcmp(&got[r],&ref[r],4)==0);}
        require(argmax(out.as<float>())==argmax(reference.as<float>()));
        std::printf("fixture=%d rows=%d retained=%u exact_argmax=pass\n",fixture,n,retained);
        if(bench){
            cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
            for(int mode:{0,1,1,0}){
                for(int i=0;i<3;++i){if(mode)candidate();else control();}check(cudaEventRecord(begin,stream));
                for(int i=0;i<20;++i){if(mode)candidate();else control();}check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));float ms=0;check(cudaEventElapsedTime(&ms,begin,end));
                std::printf("fixture=%d mode=%d ms=%.6f\n",fixture,mode,ms/20);
            }
            check(cudaEventDestroy(begin));check(cudaEventDestroy(end));
        }
    }
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));
}
