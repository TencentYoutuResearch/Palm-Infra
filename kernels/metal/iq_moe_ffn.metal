#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
struct ExpertWeights { array<device const uchar*, 8> weights [[id(0)]]; };
kernel void iq_gemv(constant ExpertWeights& sources [[buffer(0)]], device const float* x [[buffer(1)]],
 device float* y [[buffer(2)]], constant uint4& shape [[buffer(3)]],
 constant uchar* grid [[buffer(4)]], constant uchar* signs [[buffer(5)]],
 uint2 tid [[thread_position_in_grid]], uint sl [[thread_index_in_simdgroup]]) {
 const uint row=tid.x,expert=tid.y,blocks=shape.x/256;
 if(row>=shape.y)return;
 const uint bytes=shape.w==13?66:98;
 float acc[8]={0,0,0,0,0,0,0,0};
 for(uint b=0;b<blocks;++b){
  device const uchar* p=sources.weights[expert]+(ulong(row)*blocks+b)*bytes;
  const float d=float(as_type<half>(ushort(ushort(p[0])|(ushort(p[1])<<8))));
  for(uint g=0;g<8;++g){
   device const uchar* q=p+(shape.w==13?6+g*8:66+g*4);
   const uint packed=uint(q[0])|(uint(q[1])<<8)|(uint(q[2])<<16)|(uint(q[3])<<24);
   const float scale=(d*(0.5f+float(packed>>28)))*(shape.w==13?0.25f:0.5f);
   for(uint part=0;part<4;++part){
    const uint sign=signs[(packed>>(7*part))&127];
    const uint code0=p[2+g*8+part*(shape.w==13?1:2)];
    const uint code1=shape.w==13?code0:p[2+g*8+part*2+1];
#pragma unroll
    for(uint lane=0;lane<8;++lane){
     const uint index=shape.w==13?code0*8+lane:(lane<4?code0:code1)*4+lane%4;
     const float magnitude=float(grid[index]);
     const float decoded=((sign&(1<<lane))?-magnitude:magnitude)*scale;
     acc[lane]=fma(x[expert*shape.x+b*256+g*32+part*8+lane],decoded,acc[lane]);
    }
   }
  }
 }
 const float p0=acc[0]+acc[4],p1=acc[1]+acc[5],p2=acc[2]+acc[6],p3=acc[3]+acc[7];
 y[expert*(shape.z+16)+row]=(p0+p1)+(p2+p3);

}

kernel void swiglu(device const float* gate [[buffer(0)]], device float* out [[buffer(1)]],
 constant uint& width [[buffer(2)]], uint2 tid [[thread_position_in_grid]]) {
 if(tid.x>=width)return;
 const uint i=tid.y*(width*2+16)+tid.x;
 const float x=min(gate[i],10.0f);
 out[tid.y*width+tid.x]=(x*(1.0f/(1.0f+exp(-x))))*clamp(gate[i+width],-10.0f,10.0f);
}
