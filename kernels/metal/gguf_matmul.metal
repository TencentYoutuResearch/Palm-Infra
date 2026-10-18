// Host Q5 delegation: native 176-byte blocks, eight threads per output row.
// Each lane preserves the CPU K order; the shuffle tree matches its final sum.
// This domain is compiled with -fno-fast-math. Only the explicit fma contracts.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
kernel void q5_gemv(device const uchar* weights [[buffer(0)]],
                    device const float* x [[buffer(1)]],
                    device float* y [[buffer(2)]],
                    constant uint2& shape [[buffer(3)]],
                    uint tid [[thread_position_in_grid]],
                    uint sl [[thread_index_in_simdgroup]]) {
    const uint row=tid/8, lane=tid%8, blocks=shape.x/256;
    if(row>=shape.y)return;
    float acc=0.0f;
    for(uint block=0;block<blocks;++block){
        device const uchar* p=weights+(ulong(row)*blocks+block)*176;
        const float d=float(as_type<half>(ushort(ushort(p[0])|(ushort(p[1])<<8))));
        const float dm=float(as_type<half>(ushort(ushort(p[2])|(ushort(p[3])<<8))));
        device const uchar* s=p+4;
        #pragma unroll
        for(uint g=0;g<8;++g){
            const uint sc=g<4 ? s[g]&63 : (s[g+4]&15)|((s[g-4]>>6)<<4);
            const uint mn=g<4 ? s[g+4]&63 : (s[g+4]>>4)|((s[g]>>6)<<4);
            const float scale=d*float(sc), minimum=dm*float(mn);
            #pragma unroll
            for(uint j=0;j<32;j+=8){
                const uint l=j+lane;
                const uint low=(p[48+(g/2)*32+l]>>((g&1)*4))&15;
                const uint q=low+((p[16+l]&(1<<g))?16:0);
                const float decoded=scale*float(q)-minimum;
                acc=fma(x[block*256+g*32+l],decoded,acc);
            }
        }
    }
    const uint base=sl&~7u;
    const float p0=simd_shuffle(acc,base)+simd_shuffle(acc,base+4);
    const float p1=simd_shuffle(acc,base+1)+simd_shuffle(acc,base+5);
    const float p2=simd_shuffle(acc,base+2)+simd_shuffle(acc,base+6);
    const float p3=simd_shuffle(acc,base+3)+simd_shuffle(acc,base+7);
    if(lane==0)y[row]=(p0+p1)+(p2+p3);
}
