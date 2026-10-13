#include "kernels/cuda/recurrent.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void check(cudaError_t status) {
    if(status!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(status));std::exit(1);}
}
static bool run(int groups,int sequence,int kernel,int real) {
    const size_t state_count=size_t(groups)*(kernel-1);
    std::vector<float> input(size_t(groups)*sequence),weight(size_t(groups)*kernel),initial(state_count);
    for(size_t i=0;i<input.size();++i)input[i]=float(int(i*13%101)-50)/137.f;
    for(size_t i=0;i<weight.size();++i)weight[i]=float(int(i*7%31)-15)/53.f;
    for(size_t i=0;i<initial.size();++i)initial[i]=float(int(i*3%17)-8)/29.f;
    float *x,*w,*state,*output,*prefix;
    check(cudaMalloc(&x,input.size()*4));check(cudaMalloc(&w,weight.size()*4));
    check(cudaMalloc(&state,state_count*4));check(cudaMalloc(&output,(input.size()+8)*4));
    check(cudaMalloc(&prefix,(state_count*real+8)*4));
    check(cudaMemcpy(x,input.data(),input.size()*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(w,weight.data(),weight.size()*4,cudaMemcpyHostToDevice));
    std::vector<float> expected(state_count*real),expected_output(input.size(),0.f),one(groups);
    check(cudaMemcpy(state,initial.data(),state_count*4,cudaMemcpyHostToDevice));
    for(int t=0;t<real;++t) {
        if(!mollm_cuda::launch_shortconv(x+size_t(t)*groups,w,state,output,groups,1,kernel,1,1,groups))return false;
        check(cudaMemcpy(expected.data()+size_t(t)*state_count,state,state_count*4,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(one.data(),output,size_t(groups)*4,cudaMemcpyDeviceToHost));
        for(int g=0;g<groups;++g)expected_output[size_t(g)*sequence+t]=one[g];
    }
    std::vector<float> got(expected.size()+8,-123.f),got_output(input.size()+8,-123.f),final_state(state_count);
    cudaGraph_t graph;cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
    if(!mollm_cuda::launch_shortconv_prefixes(x,w,state,output,prefix,groups,sequence,kernel,real,1,groups))return false;
    check(cudaStreamEndCapture(cudaStreamPerThread,&graph));check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    for(int replay=0;replay<2;++replay) {
        if(replay) {
            for(float& value:input)value=-value;
            check(cudaMemcpy(x,input.data(),input.size()*4,cudaMemcpyHostToDevice));
            check(cudaMemcpy(state,initial.data(),state_count*4,cudaMemcpyHostToDevice));
            for(int t=0;t<real;++t) {
                if(!mollm_cuda::launch_shortconv(x+size_t(t)*groups,w,state,output,groups,1,kernel,1,1,groups))return false;
                check(cudaMemcpy(expected.data()+size_t(t)*state_count,state,state_count*4,cudaMemcpyDeviceToHost));
                check(cudaMemcpy(one.data(),output,size_t(groups)*4,cudaMemcpyDeviceToHost));
                for(int g=0;g<groups;++g)expected_output[size_t(g)*sequence+t]=one[g];
            }
        }
        check(cudaMemcpy(state,initial.data(),state_count*4,cudaMemcpyHostToDevice));
        std::fill(got.begin(),got.end(),-123.f);std::fill(got_output.begin(),got_output.end(),-123.f);
        check(cudaMemcpy(prefix,got.data(),got.size()*4,cudaMemcpyHostToDevice));
        check(cudaMemcpy(output,got_output.data(),got_output.size()*4,cudaMemcpyHostToDevice));
        check(cudaGraphLaunch(executable,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
        check(cudaMemcpy(got.data(),prefix,got.size()*4,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(got_output.data(),output,got_output.size()*4,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(final_state.data(),state,state_count*4,cudaMemcpyDeviceToHost));
        if(std::memcmp(got.data(),expected.data(),expected.size()*4)||
           std::memcmp(got_output.data(),expected_output.data(),expected_output.size()*4)||
           std::memcmp(final_state.data(),expected.data()+size_t(real-1)*state_count,state_count*4)) {
            std::fprintf(stderr,"prefix mismatch groups=%d seq=%d kernel=%d real=%d\n",groups,sequence,kernel,real);return false;
        }
        for(size_t i=expected.size();i<got.size();++i)if(got[i]!=-123.f)return false;
        for(size_t i=expected_output.size();i<got_output.size();++i)if(got_output[i]!=-123.f)return false;
        for(int kept=1;kept<real;++kept) {
            check(cudaMemcpy(state,prefix+size_t(kept-1)*state_count,state_count*4,cudaMemcpyDeviceToDevice));
            if(!mollm_cuda::launch_shortconv(x+size_t(kept)*groups,w,state,output,groups,1,kernel,1,1,groups))return false;
            check(cudaMemcpy(final_state.data(),state,state_count*4,cudaMemcpyDeviceToHost));
            check(cudaMemcpy(one.data(),output,size_t(groups)*4,cudaMemcpyDeviceToHost));
            if(std::memcmp(final_state.data(),expected.data()+size_t(kept)*state_count,state_count*4))return false;
            for(int group=0;group<groups;++group)
                if(std::memcmp(&one[group],&expected_output[size_t(group)*sequence+kept],4))return false;
        }
    }
    cudaGraphExecDestroy(executable);cudaGraphDestroy(graph);
    cudaFree(x);cudaFree(w);cudaFree(state);cudaFree(output);cudaFree(prefix);return true;
}
static bool gdn(int sequence,int real,int flags,int heads,int values) {
    const int width=2*heads*128+values*128,output_width=values*128;
    const size_t state_count=size_t(values)*128*128;
    std::vector<float> qkv(size_t(sequence)*width),a(sequence*values),b(a.size()),z(sequence*output_width);
    std::vector<float> alog(values,-1.f),bias(values,.1f),norm(128,1.f),initial(state_count);
    for(size_t i=0;i<qkv.size();++i)qkv[i]=float(int(i*7%61)-30)/83.f;
    for(size_t i=0;i<a.size();++i){a[i]=float(int(i%7)-3)/11.f;b[i]=float(int(i%11)-5)/13.f;}
    for(size_t i=0;i<z.size();++i)z[i]=float(int(i%13)-6)/17.f;
    for(size_t i=0;i<initial.size();++i)initial[i]=float(int(i%17)-8)/101.f;
    auto upload=[](const std::vector<float>& host){float*p;check(cudaMalloc(&p,host.size()*4));check(cudaMemcpy(p,host.data(),host.size()*4,cudaMemcpyHostToDevice));return p;};
    std::vector<float> one_qkv(width);
    float *q=upload(qkv),*single_q=upload(one_qkv),*da=upload(a),*db=upload(b),*dz=upload(z),*dl=upload(alog),*dt=upload(bias),*dn=upload(norm),*state=upload(initial),*output,*prefix;
    check(cudaMalloc(&output,(size_t(sequence)*output_width+8)*4));
    check(cudaMalloc(&prefix,(size_t(real)*state_count+8)*4));
    std::vector<float> expected(size_t(real)*state_count),expected_output(size_t(sequence)*output_width,0.f);
    for(int token=0;token<real;++token) {
        // GDN consumes convolution's feature-major QKV, not token-major input.
        for(int feature=0;feature<width;++feature)one_qkv[feature]=qkv[size_t(feature)*sequence+token];
        check(cudaMemcpy(single_q,one_qkv.data(),size_t(width)*4,cudaMemcpyHostToDevice));
        if(!mollm_cuda::launch_gdn(single_q,da+token*values,db+token*values,
            dz+token*output_width,dl,dt,dn,state,output,heads,values,128,128,1,1,flags,
            1e-6f,1e-6f,.125f,values,values,output_width))return false;
        check(cudaMemcpy(expected.data()+size_t(token)*state_count,state,state_count*4,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(expected_output.data()+size_t(token)*output_width,output,size_t(output_width)*4,cudaMemcpyDeviceToHost));
    }
    std::vector<float> got(expected.size()+8,-123.f),actual(expected_output.size()+8,-123.f),final_state(state_count);
    check(cudaMemcpy(state,initial.data(),state_count*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(prefix,got.data(),got.size()*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(output,actual.data(),actual.size()*4,cudaMemcpyHostToDevice));
    if(!mollm_cuda::launch_gdn_prefixes(q,da,db,dz,dl,dt,dn,state,output,prefix,
        heads,values,sequence,real,flags,1e-6f,1e-6f,.125f,values,values,output_width))return false;
    check(cudaMemcpy(got.data(),prefix,got.size()*4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(actual.data(),output,actual.size()*4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(final_state.data(),state,state_count*4,cudaMemcpyDeviceToHost));
    if(std::memcmp(expected.data(),got.data(),expected.size()*4)||
        std::memcmp(expected_output.data(),actual.data(),expected_output.size()*4)||
        std::memcmp(final_state.data(),expected.data()+size_t(real-1)*state_count,state_count*4)) {
        std::fprintf(stderr,"GDN prefix mismatch seq=%d real=%d flags=%d heads=%d values=%d\n",sequence,real,flags,heads,values);return false;
    }
    for(size_t i=expected.size();i<got.size();++i)if(got[i]!=-123.f)return false;
    for(size_t i=expected_output.size();i<actual.size();++i)if(actual[i]!=-123.f)return false;
    cudaGraph_t graph;cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
    if(!mollm_cuda::launch_gdn_prefixes(q,da,db,dz,dl,dt,dn,state,output,prefix,
        heads,values,sequence,real,flags,1e-6f,1e-6f,.125f,values,values,output_width))return false;
    check(cudaStreamEndCapture(cudaStreamPerThread,&graph));
    check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    for(int replay=0;replay<2;++replay) {
        if(replay) {
            for(float& value:qkv)value=-value;
            check(cudaMemcpy(q,qkv.data(),qkv.size()*4,cudaMemcpyHostToDevice));
            check(cudaMemcpy(state,initial.data(),state_count*4,cudaMemcpyHostToDevice));
            for(int token=0;token<real;++token) {
                for(int feature=0;feature<width;++feature)one_qkv[feature]=qkv[size_t(feature)*sequence+token];
                check(cudaMemcpy(single_q,one_qkv.data(),size_t(width)*4,cudaMemcpyHostToDevice));
                if(!mollm_cuda::launch_gdn(single_q,da+token*values,db+token*values,
                    dz+token*output_width,dl,dt,dn,state,output,heads,values,128,128,1,1,flags,
                    1e-6f,1e-6f,.125f,values,values,output_width))return false;
                check(cudaMemcpy(expected.data()+size_t(token)*state_count,state,state_count*4,cudaMemcpyDeviceToHost));
                check(cudaMemcpy(expected_output.data()+size_t(token)*output_width,output,size_t(output_width)*4,cudaMemcpyDeviceToHost));
            }
        }
        check(cudaMemcpy(state,initial.data(),state_count*4,cudaMemcpyHostToDevice));
        std::fill(got.begin(),got.end(),-123.f);std::fill(actual.begin(),actual.end(),-123.f);
        check(cudaMemcpy(prefix,got.data(),got.size()*4,cudaMemcpyHostToDevice));
        check(cudaMemcpy(output,actual.data(),actual.size()*4,cudaMemcpyHostToDevice));
        check(cudaGraphLaunch(executable,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
        check(cudaMemcpy(got.data(),prefix,got.size()*4,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(actual.data(),output,actual.size()*4,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(final_state.data(),state,state_count*4,cudaMemcpyDeviceToHost));
        if(std::memcmp(expected.data(),got.data(),expected.size()*4)||
            std::memcmp(expected_output.data(),actual.data(),expected_output.size()*4)||
            std::memcmp(final_state.data(),expected.data()+size_t(real-1)*state_count,state_count*4)) {
            std::fprintf(stderr,"GDN prefix graph mismatch replay=%d\n",replay);return false;
        }
        for(size_t i=expected.size();i<got.size();++i)if(got[i]!=-123.f)return false;
        for(size_t i=expected_output.size();i<actual.size();++i)if(actual[i]!=-123.f)return false;
        // Restore each accepted prefix, then consume the next token. Compare
        // against the independently generated serial post-prefix state.
        for(int kept=1;kept<real;++kept) {
            for(int feature=0;feature<width;++feature)one_qkv[feature]=qkv[size_t(feature)*sequence+kept];
            check(cudaMemcpy(single_q,one_qkv.data(),size_t(width)*4,cudaMemcpyHostToDevice));
            auto next=[&](){return mollm_cuda::launch_gdn(single_q,da+kept*values,db+kept*values,
                dz+kept*output_width,dl,dt,dn,state,output,heads,values,128,128,1,1,flags,
                1e-6f,1e-6f,.125f,values,values,output_width);};
            check(cudaMemcpy(state,expected.data()+size_t(kept-1)*state_count,state_count*4,cudaMemcpyHostToDevice));
            if(!next())return false;
            std::vector<float> next_state(state_count),next_output(output_width),restored_output(output_width);
            check(cudaMemcpy(next_state.data(),state,state_count*4,cudaMemcpyDeviceToHost));
            check(cudaMemcpy(next_output.data(),output,size_t(output_width)*4,cudaMemcpyDeviceToHost));
            check(cudaMemcpy(state,prefix+size_t(kept-1)*state_count,state_count*4,cudaMemcpyDeviceToDevice));
            if(!next())return false;
            check(cudaMemcpy(final_state.data(),state,state_count*4,cudaMemcpyDeviceToHost));
            check(cudaMemcpy(restored_output.data(),output,size_t(output_width)*4,cudaMemcpyDeviceToHost));
            if(std::memcmp(next_state.data(),final_state.data(),state_count*4)||
                std::memcmp(next_output.data(),restored_output.data(),size_t(output_width)*4))return false;
        }
    }
    cudaGraphExecDestroy(executable);cudaGraphDestroy(graph);
    for(float*p:{q,single_q,da,db,dz,dl,dt,dn,state,output,prefix})cudaFree(p);
    return true;
}
int main() {
    int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||!count)return 77;
    if(mollm_cuda::launch_shortconv_prefixes(nullptr,nullptr,nullptr,nullptr,nullptr,7,2,4,2,1,7))return 1;
    for(int groups:{7,257})for(int sequence:{2,3,4,5,7,8})for(int kernel:{2,4,8})
        for(int real:{sequence-1,sequence})if(!run(groups,sequence,kernel,real))return 1;
    for(int sequence:{2,3,4,5,7,8})for(int real:{sequence-1,sequence})for(int flags:{0,1,2,3})
        for(int heads:{1,2})if(!gdn(sequence,real,flags,heads,heads*2))return 1;
    std::puts("72 shortconv and 96 GDN prefix-state/output/guard fixtures passed; GDN graph replay/changed input exact");return 0;
}
