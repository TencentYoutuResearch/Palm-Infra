// Reuse private diagnostic kernels, never production backend source.
#define MOLLM_W4A16_KERNELS_ONLY
#include "bench_cuda_nvfp4_w4a16.cu"
#include <cub/device/device_select.cuh>
#include <numeric>

namespace {
struct ValidBatch {__host__ __device__ bool operator()(const mollm_cuda::Nvfp4MmaBatch& b)const{return b.first>=0&&b.size>0;}};
__global__ void bf16_inputs(const mollm_cuda::Nvfp4Projection* jobs,__nv_bfloat16* x,int inner){
    int k=blockIdx.x*256+threadIdx.x;if(k<inner)x[size_t(blockIdx.y)*inner+k]=__float2bfloat16_rn(jobs[blockIdx.y].input[k]);
}
}
int main(int argc,char** argv){
    bool gate=(argc==3&&!std::strcmp(argv[1],"--gate-counts"))||(argc==2&&!std::strcmp(argv[1],"--gate-checks-only"));
    bool small=argc==2&&(!std::strcmp(argv[1],"--checks-only")||gate);std::vector<int> counts{0,1,7,8,17,32,39};
    if(!small){
        if(argc!=3||(std::strcmp(argv[1],"--counts")&&!gate)){std::fprintf(stderr,"usage: %s --checks-only | --gate-checks-only | --counts CSV512 | --gate-counts CSV512 (sum=17430)\n",argv[0]);return 2;}
        counts.clear();const char* p=argv[2];
        while(*p){const char* start=p;while(*p>='0'&&*p<='9')++p;
            if(start==p||p-start>5||(*p&&*p!=','))return 2;
            long n=std::strtol(start,nullptr,10);if(n>17430)return 2;counts.push_back(n);
            if(!*p)break;++p;if(!*p)return 2;}
        if(counts.size()!=512||std::accumulate(counts.begin(),counts.end(),0)!=17430)return 2;
    }
    setenv("MOLLM_CUDA_HOST_HEADROOM_MB","102400",1);setenv("MOLLM_CUDA_NVFP4_MMA_BATCH_WARPS","4",1);
    cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,0));if(prop.major!=12||prop.minor!=1)return 77;
    int rows=gate?1280:(small?64:2560),inner=gate?2560:(small?64:640),experts=counts.size(),count=std::accumulate(counts.begin(),counts.end(),0);
    const int input_tokens=gate&&!small?1743:count;
    std::printf("diagnostic gate=%d routes=%d source_tokens=%d; gate candidate excludes activation/down quantization\n",gate,count,input_tokens);
    size_t codes=size_t(rows)*inner/2,sfs=size_t(rows)*inner/16,stride=codes+sfs+rows*4;
    size_t budget=2*experts*stride+size_t(count)*(inner*7+rows*8+3*sizeof(mollm_cuda::Nvfp4Projection)+3*sizeof(mollm_cuda::Nvfp4MmaBatch))+(size_t(2)<<20);
    if(gate)budget+=experts*sfs+size_t(count)*(inner/2+inner/16+640*4+320+40+sizeof(mollm_cuda::Nvfp4Projection))+64;
    if(!mollm_memory::cuda_host_allocation_allowed(budget,"bounded ragged W4A16 down diagnostic"))return 2;
    auto* w=alloc<uint8_t>(experts*stride);auto* fw=alloc<uint8_t>(experts*stride);
    auto* x=alloc<float>(size_t(count)*inner);auto* bx=alloc<__nv_bfloat16>(size_t(count)*inner);
    auto* y0=alloc<__nv_bfloat16>(size_t(count)*rows+16);auto* y1=alloc<__nv_bfloat16>(size_t(count)*rows+16);
    auto* yr=alloc<float>(size_t(count)*rows);
    auto* jobs=alloc<mollm_cuda::Nvfp4Projection>(count);auto* fjobs=alloc<mollm_cuda::Nvfp4Projection>(count);auto* rjobs=alloc<mollm_cuda::Nvfp4Projection>(count);
    auto* native_plan=alloc<mollm_cuda::Nvfp4MmaBatch>(count);auto* plan=alloc<mollm_cuda::Nvfp4MmaBatch>(count);auto* compact=alloc<mollm_cuda::Nvfp4MmaBatch>(count);
    auto* selected=alloc<int>(1);auto* packed=alloc<uint8_t>(size_t(count)*inner/2*(gate?2:1));auto* scales=alloc<uint8_t>(size_t(count)*inner/16*(gate?2:1));
    auto* view=alloc<uint8_t>(gate?experts*sfs:1);auto* down_jobs=alloc<mollm_cuda::Nvfp4Projection>(gate?count:1);
    auto* activated=alloc<float>(gate?size_t(count)*640+8:1);auto* down_packed=alloc<uint8_t>(gate?size_t(count)*320+32:1);auto* down_scales=alloc<uint8_t>(gate?size_t(count)*40+32:1);
    size_t temp_bytes=0;ck(cub::DeviceSelect::If(nullptr,temp_bytes,plan,compact,selected,count,ValidBatch{},cudaStreamPerThread));auto* temp=alloc<uint8_t>(temp_bytes);
    std::vector<uint8_t> payload(experts*stride),fragment(payload.size()),host_view(gate?experts*sfs:0);std::vector<int> owner(count);
    std::vector<mollm_cuda::Nvfp4Projection> h(count),fh(count),rh(count);
    int begin=0,expected_tiles=0;
    for(int e=0;e<experts;++e){auto* p=payload.data()+e*stride;
        for(int r=0;r<rows;++r){float global=(1+(r+e)%7)/(gate?4096.f:64.f);std::memcpy(p+codes+sfs+r*4,&global,4);
            for(int k=0;k<inner;k+=16){uint8_t sf[8]={0,1,7,8,48,56,64,126};p[codes+size_t(r)*(inner/16)+k/16]=sf[(r+k/16+e)%8];
                for(int b=0;b<8;++b)p[offset(r,k+2*b,inner,true)]=(r*29+k*17+b*13+e*7)&255;}}
        std::memcpy(fragment.data()+e*stride+codes,p+codes,sfs+rows*4);
        for(int m=0;m<rows;m+=16)for(int k=0;k<inner;k+=16)for(int lane=0;lane<32;++lane){int r=m+lane/4,t=lane%4;
            size_t dst=gate?e*stride+(size_t(m/32)*(inner/32)+k/32)*576+(((m/16)%2)*2+(k/16)%2)*128+lane*4:
                e*stride+((size_t(m/16)*(inner/16)+k/16)*32+lane)*4;
            fragment[dst]=p[offset(r,k+2*t,inner,true)];fragment[dst+1]=p[offset(r+8,k+2*t,inner,true)];
            fragment[dst+2]=p[offset(r,k+8+2*t,inner,true)];fragment[dst+3]=p[offset(r+8,k+8+2*t,inner,true)];}
        if(gate){
            for(int m=0;m<rows;m+=32)for(int k=0;k<inner;k+=32)for(int mh=0;mh<2;++mh)for(int kh=0;kh<2;++kh)for(int r=0;r<16;++r){
                size_t dst=e*stride+(size_t(m/32)*(inner/32)+k/32)*576+512+(mh*2+kh)*16+r;
                fragment[dst]=p[codes+size_t(m+mh*16+r)*(inner/16)+k/16+kh];}
            for(int r=0;r<rows;++r)for(int g=0;g<40;++g)for(int s=0;s<4;++s)
                host_view[e*sfs+(size_t(g)*rows+r)*4+s]=p[codes+size_t(r)*160+g*4+s];
        }
        for(int j=0;j<counts[e];++j){int i=begin+j;owner[i]=e;auto* wp=w+e*stride;
            h[i]={wp,gate?view+e*sfs:wp+codes,reinterpret_cast<float*>(wp+codes+sfs),x+size_t(i%input_tokens)*inner,reinterpret_cast<float*>(y0+size_t(i)*rows),true,false,1,1,gate?1.f:(j+1)/float(counts[e])};
            fh[i]=h[i];auto* fp=fw+e*stride;fh[i].weight=fp;fh[i].block_scales=fp+codes;fh[i].row_scales=reinterpret_cast<float*>(fp+codes+sfs);fh[i].output=reinterpret_cast<float*>(y1+size_t(i)*rows);
            rh[i]=fh[i];rh[i].output=yr+size_t(i)*rows;}
        begin+=counts[e];expected_tiles+=(counts[e]+15)/16;
    }
    ck(cudaMemcpy(w,payload.data(),payload.size(),cudaMemcpyHostToDevice));ck(cudaMemcpy(fw,fragment.data(),fragment.size(),cudaMemcpyHostToDevice));
    if(gate){ck(cudaMemcpy(view,host_view.data(),host_view.size(),cudaMemcpyHostToDevice));auto dh=h;
        for(int i=0;i<count;++i){dh[i].input_global=1024;dh[i].input=activated+size_t(i)*640;}
        ck(cudaMemcpy(down_jobs,dh.data(),count*sizeof(dh[0]),cudaMemcpyHostToDevice));}
    ck(cudaMemcpy(jobs,h.data(),count*sizeof(h[0]),cudaMemcpyHostToDevice));ck(cudaMemcpy(fjobs,fh.data(),count*sizeof(h[0]),cudaMemcpyHostToDevice));ck(cudaMemcpy(rjobs,rh.data(),count*sizeof(h[0]),cudaMemcpyHostToDevice));
    int bound=count/16+experts;dim3 grid((rows+127)/128,bound);
    auto candidate=[&](bool prepare_input=true){
        ck(mollm_cuda::launch_nvfp4_batch_plan_diagnostic(fjobs,count,plan,16,true,cudaStreamPerThread));
        ck(cub::DeviceSelect::If(temp,temp_bytes,plan,compact,selected,count,ValidBatch{},cudaStreamPerThread));
        if(prepare_input){bf16_inputs<<<dim3((inner+255)/256,count),256,0,cudaStreamPerThread>>>(fjobs,bx,inner);ck(cudaGetLastError());}
        if(gate)project_m32<false,2,true,false,true,true><<<grid,128,0,cudaStreamPerThread>>>(fjobs,compact,bx,rows,inner,selected);
        else project_m32<false,0,true,false,true,true><<<grid,128,0,cudaStreamPerThread>>>(fjobs,compact,bx,rows,inner,selected);ck(cudaGetLastError());};
    auto native=[&](bool prepared=false,bool publish=false){
        if(gate)ck(mollm_cuda::launch_nvfp4_mma_activation_diagnostic(jobs,down_jobs,count,native_plan,packed,scales,activated,down_packed,down_scales,publish,
            cudaStreamPerThread,false,32,false,false,nullptr,0,true,true));
        else ck(mollm_cuda::launch_nvfp4_mma_grouped(jobs,count,native_plan,rows,inner,0,cudaStreamPerThread,
            packed,scales,1,true,false,nullptr,0,8,false,prepared,false,true,false,true,false));};
    cudaGraph_t g;cudaGraphExec_t ex;ck(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));candidate();ck(cudaStreamEndCapture(cudaStreamPerThread,&g));ck(cudaGraphInstantiateWithFlags(&ex,g,0));
    std::vector<float> hx(size_t(count)*inner),ref(size_t(count)*rows);std::vector<__nv_bfloat16> result(size_t(count)*rows+16);
    size_t comparisons=0;double worst=0;
    for(int variant=0;variant<3;++variant){
        for(size_t i=0;i<hx.size();++i)hx[i]=__bfloat162float(__float2bfloat16(variant==2?0.f:(int((i*13+variant*11)%101)-50)/37.f));
        ck(cudaMemcpy(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));ck(cudaMemset(y1,0xff,result.size()*2));ck(cudaMemset(y1+size_t(count)*rows,0x5a,32));
        ck(cudaGraphLaunch(ex,cudaStreamPerThread));ck(cudaStreamSynchronize(cudaStreamPerThread));
        int actual=0;ck(cudaMemcpy(&actual,selected,4,cudaMemcpyDeviceToHost));if(actual!=expected_tiles||actual>bound)return 1;
        std::vector<mollm_cuda::Nvfp4MmaBatch> cp(actual);ck(cudaMemcpy(cp.data(),compact,actual*sizeof(cp[0]),cudaMemcpyDeviceToHost));int covered=0;
        for(auto b:cp){if(b.first!=covered||b.size<1||b.size>16||owner[b.first]!=owner[b.first+b.size-1])return 1;covered+=b.size;}if(covered!=count)return 1;
        if(gate)project_m32<false,2,true,false,true,false><<<grid,128,0,cudaStreamPerThread>>>(rjobs,compact,bx,rows,inner,selected);
        else project_m32<false,0,true,false,true,false><<<grid,128,0,cudaStreamPerThread>>>(rjobs,compact,bx,rows,inner,selected);ck(cudaGetLastError());ck(cudaStreamSynchronize(cudaStreamPerThread));
        ck(cudaMemcpy(result.data(),y1,result.size()*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(ref.data(),yr,ref.size()*4,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<ref.size();++i){__nv_bfloat16_raw expected=__float2bfloat16_rn(ref[i]),got=result[i];if(expected.x!=got.x||!std::isfinite(ref[i])){std::fprintf(stderr,"rounding mismatch gate%d variant%d index%zu fp32%g expected%04x got%04x\n",gate,variant,i,ref[i],expected.x,got.x);return 1;}}
        for(int i=0;i<32;++i)if(reinterpret_cast<uint8_t*>(result.data()+ref.size())[i]!=0x5a)return 1;
        size_t samples=small&&!gate?ref.size():1024;
        for(size_t s=0;s<samples;++s){size_t idx=small?s:(s*7919+variant*97)%ref.size();int route=idx/rows,r=idx%rows,e=owner[route];auto* p=payload.data()+e*stride;double sum=0,absum=0;
            for(int k=0;k<inner;++k){uint8_t c=p[offset(r,k,inner,true)];float weight=f4(k&1?c>>4:c&15)*f8(p[codes+size_t(r)*(inner/16)+k/16]);double v=double(__bfloat162float(__float2bfloat16(weight)))*hx[size_t(route%input_tokens)*inner+k];sum+=v;absum+=std::abs(v);}
            float global;std::memcpy(&global,p+codes+sfs+r*4,4);double factor=double(global)*h[route].output_scale;
            double error=std::abs(ref[idx]-sum*factor),tolerance=std::max(1e-7,3e-4*absum*factor);worst=std::max(worst,error/tolerance);if(error>tolerance){std::fprintf(stderr,"oracle mismatch gate%d variant%d index%zu got%g expected%g bound%g\n",gate,variant,idx,ref[idx],sum*factor,tolerance);return 1;}++comparisons;}
    }
    ck(cudaGraphExecDestroy(ex));ck(cudaGraphDestroy(g));
    if(!small||gate){
        for(size_t i=0;i<hx.size();++i)hx[i]=__bfloat162float(__float2bfloat16((int((i*13)%101)-50)/37.f));ck(cudaMemcpy(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));
        if(gate){
            ck(cudaMemset(activated,0xff,size_t(count)*640*4));ck(cudaMemset(activated+size_t(count)*640,0x5a,32));
            ck(cudaMemset(down_packed+size_t(count)*320,0x5a,32));ck(cudaMemset(down_scales+size_t(count)*40,0x5a,32));
            native(false,true);candidate();ck(cudaStreamSynchronize(cudaStreamPerThread));std::vector<float> a(size_t(count)*640+8);ck(cudaMemcpy(a.data(),activated,a.size()*4,cudaMemcpyDeviceToHost));
            for(size_t i=0;i<size_t(count)*640;++i)if(!std::isfinite(a[i])){std::fprintf(stderr,"native activated unwritten/nonfinite %zu\n",i);return 1;}
            for(int i=0;i<32;++i)if(reinterpret_cast<uint8_t*>(a.data()+size_t(count)*640)[i]!=0x5a)return 1;
            uint8_t guards[32];for(auto p:{down_packed+size_t(count)*320,down_scales+size_t(count)*40}){ck(cudaMemcpy(guards,p,32,cudaMemcpyDeviceToHost));for(auto b:guards)if(b!=0x5a)return 1;}
        }else{
            ck(cudaMemset(y0,0xff,result.size()*2));ck(cudaMemset(y0+size_t(count)*rows,0x5a,32));native();candidate();ck(cudaStreamSynchronize(cudaStreamPerThread));
            ck(cudaMemcpy(result.data(),y0,result.size()*2,cudaMemcpyDeviceToHost));
            for(size_t i=0;i<ref.size();++i)if(!std::isfinite(__bfloat162float(result[i])))return 1;
            for(int i=0;i<32;++i)if(reinterpret_cast<uint8_t*>(result.data()+ref.size())[i]!=0x5a)return 1;
        }
        if(!small)for(bool prepared:{false,true}){if(gate&&prepared)continue;for(int rep=0;rep<7;++rep){
            auto a_launch=[&](){native(prepared);};auto b_launch=[&](){candidate(!prepared);};float a,b;
            if(rep&1){b=timing(b_launch);a=timing(a_launch);}else{a=timing(a_launch);b=timing(b_launch);}
            std::printf("ragged %s prepared=%d routes=%d experts=%d compact_N16=%d native_us=%.3f W4A16_BF16_us=%.3f ratio=%.4f\n",gate?"fused-gate-vs-projection":"down",prepared,count,experts,expected_tiles,a,b,b/a);}}
    }
    std::printf("PASS compact coverage, captured input changes, full BF16-vs-FP32 rounding/tail checks; independent double comparisons=%zu worst_bound_fraction=%.6g; synthetic weights/input, no model-quality claim\n",comparisons,worst);
    for(void* p:{(void*)w,(void*)fw,(void*)x,(void*)bx,(void*)y0,(void*)y1,(void*)yr,(void*)jobs,(void*)fjobs,(void*)rjobs,(void*)native_plan,(void*)plan,(void*)compact,(void*)selected,(void*)packed,(void*)scales,(void*)temp,(void*)view,(void*)down_jobs,(void*)activated,(void*)down_packed,(void*)down_scales})ck(cudaFree(p));
    return 0;
}
