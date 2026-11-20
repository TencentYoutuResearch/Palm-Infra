#include "backends/cuda/fp16_expert_palette_cache.h"
#include "backends/cuda/backend.h"
#include "kernels/cuda/fp16_expert_palette.h"
#include "kernels/cuda/moe.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using Cache=mollm::cuda_backend_detail::Fp16ExpertPaletteCache;
using Job=mollm_cuda::Nvfp4Projection;
static void require(bool b,const char*s){if(!b){std::fprintf(stderr,"palette: %s\n",s);std::exit(1);}}
static void check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"palette CUDA: %s\n",cudaGetErrorString(e));std::exit(1);}}
template<class T>struct Buffer{T*p=nullptr;size_t n;explicit Buffer(size_t n):n(n){check(cudaMalloc(&p,n*sizeof(T)));}~Buffer(){cudaFree(p);}void put(const T*s){check(cudaMemcpy(p,s,n*sizeof(T),cudaMemcpyHostToDevice));}std::vector<T>get(){std::vector<T>v(n);check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}};
static Tensor tensor(std::vector<uint16_t>&v,int n,int k){Tensor t;t.prec=Precision::FP16;t.shape[0]=n;t.shape[1]=k;t.shape[2]=t.shape[3]=1;t.compute_strides();t.data=v.data();t.rowmajor_data=v.data();t.mem_type=MemoryType::EXTERNAL;return t;}
static void packing(){
 std::vector<uint16_t>w(256*256),different,bad(128*128);Cache c;
 for(size_t i=0;i<w.size();i++){
  unsigned q=unsigned((i/256*3+i%256)%256);
  const uint16_t edge[]={0,0x8000,0x7c00,0xfc00,0x7e01,0xfe02};
  w[i]=q<6?edge[q]:uint16_t(q*257);
 } // Every tile has256 distinct patterns, including signed zeros and NaNs.
 require(c.prepare(nullptr,128,128)==Cache::Result::Ineligible,"null");
 require(c.prepare(w.data(),127,128)==Cache::Result::Ineligible,"rows alignment");
 require(c.prepare(w.data(),128,129)==Cache::Result::Ineligible,"inner alignment");
 require(c.prepare(reinterpret_cast<const uint16_t*>(reinterpret_cast<const char*>(w.data())+1),128,128)==Cache::Result::Ineligible,"source alignment");
 require(c.prepare(w.data(),256,256)==Cache::Result::Ready,"256-values packing");
 const auto*p=c.find(w.data(),256,256);require(p&&c.bytes()==67584,"compressed size");
 require(c.prepare(w.data(),256,256)==Cache::Result::Ready&&c.size()==1&&c.find(w.data(),256,256)==p,"cache identity");
 require(!c.find(w.data(),128,256),"shape identity");
 std::vector<uint8_t>codes(w.size());std::vector<uint16_t>pal(4*256);
 check(cudaMemcpy(codes.data(),p->codes,codes.size(),cudaMemcpyDeviceToHost));check(cudaMemcpy(pal.data(),p->palette,pal.size()*2,cudaMemcpyDeviceToHost));
 for(int r=0;r<256;r++)for(int k=0;k<256;k++)require(pal[(r/128*2+k/128)*256+codes[r*256+k]]==w[r*256+k],"bit reconstruction");
 different=w;require(c.prepare(different.data(),256,256)==Cache::Result::Ready&&c.size()==2,"distinct source");
 for(int i=0;i<257;i++)bad[i]=i;
 require(c.prepare(bad.data(),128,128)==Cache::Result::Ineligible&&c.size()==2,"257-values fallback");
 cudaGraph_t graph;check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
 require(c.prepare(w.data(),256,256)==Cache::Result::Error,"load under capture");
 check(cudaStreamEndCapture(cudaStreamPerThread,&graph));check(cudaGraphDestroy(graph));
 // Device-memory admission failures must not publish an incomplete cache entry.
 setenv("MOLLM_CUDA_HOST_HEADROOM_MB","invalid",1);
 Cache denied;require(denied.prepare(w.data(),256,256)==Cache::Result::Error&&denied.size()==0&&denied.bytes()==0,"allocation failure atomicity");
 unsetenv("MOLLM_CUDA_HOST_HEADROOM_MB");
}
static void projections(int n,int k){
 constexpr int experts=3,routes=5;
 std::vector<uint16_t>w(size_t(experts)*n*k);
 for(size_t i=0;i<w.size();i++)w[i]=uint16_t(0x2800+(i%113)*4+(i/(n*k))*512);
 w[0]=0x8000;w[1]=0x7e01;w[2]=0xfc00; // Exact exceptional-weight conversion too.
 Cache cache;require(cache.prepare(w.data(),experts*n,k)==Cache::Result::Ready,"projection pack");const auto*p=cache.find(w.data(),experts*n,k);
 Buffer<uint16_t>original(w.size());original.put(w.data());Buffer<float>x(routes*k),base(routes*n+16),result(routes*n+16);Buffer<Job>j0(routes),j1(routes);
 std::vector<Job>a(routes),b(routes);std::vector<float>input(x.n),sentinel(base.n,-123.f);
 for(int r=0;r<routes;r++){int e=(r*2)%experts;a[r]={};a[r].weight=reinterpret_cast<const uint8_t*>(original.p+size_t(e)*n*k);a[r].input=x.p+r*k;a[r].output=base.p+8+r*n;b[r]=a[r];b[r].weight=p->codes+size_t(e)*n*k;b[r].block_scales=reinterpret_cast<const uint8_t*>(p->palette+size_t(e)*n*k/16384*256);b[r].output=result.p+8+r*n;}
 j0.put(a.data());j1.put(b.data());base.put(sentinel.data());result.put(sentinel.data());
 require(mollm_cuda::launch_fp16_palette_projections(nullptr,1,n,k)==cudaErrorInvalidValue,"null jobs");
 require(mollm_cuda::launch_fp16_palette_projections(j1.p,-1,n,k)==cudaErrorInvalidValue,"negative count");
 require(mollm_cuda::launch_fp16_palette_projections(j1.p,65536,n,k)==cudaErrorInvalidValue,"count limit");
 require(mollm_cuda::launch_fp16_palette_projections(j1.p,1,n-1,k)==cudaErrorInvalidValue,"invalid rows");
 require(mollm_cuda::launch_fp16_palette_projections(j1.p,1,n,k-1)==cudaErrorInvalidValue,"invalid inner");
 require(mollm_cuda::launch_fp16_palette_projections(reinterpret_cast<const Job*>(reinterpret_cast<const char*>(j1.p)+1),1,n,k)==cudaErrorInvalidValue,"unaligned jobs");
 check(mollm_cuda::launch_fp16_palette_projections(nullptr,0,n,k));
 cudaGraph_t graph;cudaGraphExec_t exec;
 check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));check(mollm_cuda::launch_fp16_palette_projections(j1.p,routes,n,k,cudaStreamPerThread));check(cudaStreamEndCapture(cudaStreamPerThread,&graph));check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
 for(int pass=0;pass<3;pass++){
  for(size_t i=0;i<input.size();i++)input[i]=float(int((i*37+pass*91)%8191)-4095)/1024.f;
  x.put(input.data());check(mollm_cuda::launch_fp16_expert_projections(j0.p,routes,n,k));
  if(pass)check(cudaGraphLaunch(exec,cudaStreamPerThread));else check(mollm_cuda::launch_fp16_palette_projections(j1.p,routes,n,k));
  auto ref=base.get(),got=result.get();require(!std::memcmp(ref.data(),got.data(),ref.size()*4),"projection bits/guards");
  for(int i=0;i<8;i++)require(got[i]==-123.f&&got[got.size()-1-i]==-123.f,"guards");
 }
 check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph)); // Retire readers before cache destruction.
}
static void backend(){
 std::vector<uint16_t>w(256*128,0x3c00),bad(128*128);for(int i=0;i<257;i++)bad[i]=i;
 cudaDeviceProp properties{};check(cudaGetDeviceProperties(&properties,0));const bool eligible=properties.major==12&&properties.minor==1;
 for(int repeat=0;repeat<2;repeat++){
  CudaBackend b;require(b.available(),"backend");auto t=tensor(w,256,128);b.wrap_routed_expert_weight(t,128);
  require(!b.dispatch_failed(),"backend prepare");
  if(eligible){require(!t.device.buffer&&b.fp16_expert_palette_bytes()==33792,"no fake FP16 device storage");b.wrap_routed_expert_weight(t,128);require(b.fp16_expert_palette_bytes()==33792,"backend reuse");}
  std::vector<uint16_t>copy(w.size());require(b.copy_to_host(t,copy.data(),copy.size()*2)&&copy==w,"host fallback read");
  auto mismatch=tensor(bad,128,128);b.wrap_routed_expert_weight(mismatch,128);require(mismatch.device.buffer&&!b.dispatch_failed(),"ineligible normal upload");
  auto ordinary=tensor(w,256,128);b.wrap_weight(ordinary);require(ordinary.device.buffer,"ordinary mixed consumer remains FP16");
 }
}
int main(int argc,char**){int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||!count)return 77;packing();projections(128,128);projections(256,256);if(argc>1){projections(1280,2560);projections(2560,640);}backend();std::puts("FP16 palette: passed");}
