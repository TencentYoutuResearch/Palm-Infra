#include "backends/cuda/backend.h"
#include "backends/cuda/prefill_batch_policy.h"
#include "graph/execute.h"
#include "runtime/expert_provider.h"
#include "kernels/cuda/moe.h"
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
constexpr int hidden=2560,inter=640,experts=2,shared=32;
void check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(1);}}
void require(bool ok){if(!ok){std::fprintf(stderr,"logical batch fixture assertion failed\n");std::exit(2);}}
struct Provider final:ExpertProvider {
    ExpertSource gate_source,down_source;
    std::vector<void*> buffers;
    Tensor gates[experts],downs[experts];
    int pins=0;
    void* bytes(size_t n,int value) {void* p=nullptr;check(cudaMalloc(&p,n));buffers.push_back(p);check(cudaMemset(p,value,n));return p;}
    float* scales(int n) {auto* p=static_cast<float*>(bytes(size_t(n)*4,0));std::vector<float> v(n,1.f/2560);check(cudaMemcpy(p,v.data(),size_t(n)*4,cudaMemcpyHostToDevice));return p;}
    Provider(bool code_view=false,bool vary_codes=false,bool scale_view=false) {
        for(auto* s:{&gate_source,&down_source}){s->provider=this;s->resident_device_views=true;}
        gate_source.nvfp4_input_scales_per_expert=2;gate_source.nvfp4_input_scales.assign(experts*2,1.f);
        down_source.nvfp4_input_scales_per_expert=1;down_source.nvfp4_input_scales.assign(experts,1.f);
        for(int e=0;e<experts;++e) {
            auto make=[&](int n,int k) {
                auto* codes=static_cast<uint8_t*>(bytes(size_t(n)*k/2,(e+1)*0x11));
                Tensor t=Tensor::create(Precision::NVFP4,MemoryType::EXTERNAL,n,k,1,1,codes);
                t.nvfp4_q8_pair_data=codes;
                t.nvfp4_scales=static_cast<uint8_t*>(bytes(size_t(n)*k/16,56));
                t.nvfp4_row_scales=scales(n);t.group_size=16;t.groups_per_row=k/16;return t;
            };
            gates[e]=make(2*inter,hidden);downs[e]=make(hidden,inter);
            if(vary_codes)for(auto* t:{&gates[e],&downs[e]}) {
                const int n=t->shape[0],k=t->shape[1];
                std::vector<uint8_t> payload(size_t(n)*k/2),sf(size_t(n)*k/16);
                for(int row=0;row<n;++row)for(int group=0;group<k/16;++group) {
                    sf[size_t(row)*(k/16)+group]=48+(row+group+e)%16;
                    for(int byte=0;byte<8;++byte)
                        payload[(size_t(row/4)*(k/16)+group)*32+(row%4)*8+byte]=(row*29+group*17+byte*13+e*37)&255;
                }
                check(cudaMemcpy(t->data,payload.data(),payload.size(),cudaMemcpyHostToDevice));
                check(cudaMemcpy(const_cast<uint8_t*>(t->nvfp4_scales),sf.data(),sf.size(),cudaMemcpyHostToDevice));
            }
            if(code_view) {
                auto* view=static_cast<uint8_t*>(bytes(size_t(2*inter)*hidden/2,0));
                mollm_cuda::Nvfp4Projection source{};
                source.weight=static_cast<const uint8_t*>(gates[e].data);source.pair_layout=true;
                check(mollm_cuda::launch_repack_nvfp4_gate_codes(source,view,2*inter,hidden));
                check(cudaDeviceSynchronize());
                gates[e].data=view;gates[e].nvfp4_q8_pair_data=view;gates[e].nvfp4_gate_codes_kmajor=true;
            }
            if(scale_view) {
                auto* view=static_cast<uint8_t*>(bytes(size_t(2*inter)*hidden/16,0));
                mollm_cuda::Nvfp4Projection source{};source.block_scales=gates[e].nvfp4_scales;
                check(mollm_cuda::launch_repack_nvfp4_gate_scales(source,view,2*inter,hidden));
                check(cudaDeviceSynchronize());
                gates[e].nvfp4_scales=view;gates[e].nvfp4_gate_scales_kmajor=true;
            }
        }
        gate_source.resident_nvfp4_gate_codes_kmajor=code_view;
        gate_source.resident_nvfp4_gate_scales_kmajor=scale_view;
    }
    ~Provider(){require(pins==0);for(void* p:buffers)cudaFree(p);}
    bool borrow(const ExpertSource*,const ExpertSource*,int,ExpertLease&,bool) override{return false;}
    bool borrow_resident_layer(const ExpertSource*,const ExpertSource*,std::vector<ExpertLease>& leases) override {
        leases.clear();leases.resize(experts);for(int e=0;e<experts;++e){leases[e]=ExpertLease(*this,this,gates[e],downs[e]);++pins;}return true;
    }
    bool request_many(const ExpertSource*,const ExpertSource*,const std::vector<int>&) override{return true;}
    bool retain_for_next_forward(const ExpertSource*,const ExpertSource*,const std::vector<int>&,bool) override{return true;}
    size_t resident_count(const ExpertSource*,const ExpertSource*,const std::vector<int>&) const override{return experts;}
    bool contains(const ExpertSource*,const ExpertSource*,int e) const override{return e>=0&&e<experts;}
    bool is_ready(const ExpertSource* g,const ExpertSource* d,int e) const override{return contains(g,d,e);}
    bool evict(const ExpertSource*,const ExpertSource*,int) override{return false;}
    void release_lease(void*) noexcept override{--pins;}
};
}

int main() {
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
    const char* option=std::getenv("MOLLM_TEST_NVFP4_RESIDENT_GATE_CODE_VIEW");
    const bool code_view=option&&!std::strcmp(option,"1");
    const char* scale_option=std::getenv("MOLLM_TEST_NVFP4_RESIDENT_GATE_SCALE_VIEW");
    const bool scale_view=scale_option&&!std::strcmp(scale_option,"1");
    require(!scale_view||code_view);
    const char* token_option=std::getenv("MOLLM_CUDA_NVFP4_PREFILL_TOKEN_CODES_KMAJOR");
    const bool token_codes_kmajor=!token_option||!std::strcmp(token_option,"1");
    const auto logical_option=[](const char* name) {
        return mollm_cuda::logical_prefill_option(std::getenv(name),true);
    };
    const bool input_kmajor=logical_option("MOLLM_CUDA_NVFP4_PREFILL_INPUT_SCALE_KMAJOR");
    const bool token_words=input_kmajor&&logical_option("MOLLM_CUDA_NVFP4_PREFILL_COMPACT_LOGITS")&&
        logical_option("MOLLM_CUDA_NVFP4_PREFILL_TOKEN_QUANTIZE")&&
        logical_option("MOLLM_CUDA_NVFP4_PREFILL_TOKEN_PAIR_WORDS");
    Provider provider(code_view,code_view,scale_view);
    std::unique_ptr<Provider> canonical;
    if(code_view)canonical=std::make_unique<Provider>(false,true);
    CudaBackend backend;
    if(!backend.available())return 77;
    backend.set_operator_fallback_policy(OperatorFallbackPolicy::REQUIRE_NATIVE);
    require(!backend.logical_prefill_batch_enabled());
    require(backend.set_logical_prefill_batch(true));
    cudaGraph_t empty=nullptr;
    check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
    require(!backend.set_logical_prefill_batch(false)&&backend.logical_prefill_batch_enabled());
    check(cudaStreamEndCapture(cudaStreamPerThread,&empty));size_t nodes=0;
    check(cudaGraphGetNodes(empty,nullptr,&nodes));require(nodes==0);check(cudaGraphDestroy(empty));
    require(backend.set_logical_prefill_batch(false));
    std::vector<float> router(experts*hidden,0),sg(shared*hidden,.001f),su(sg),sd(hidden*shared,.001f),ss(hidden,0);
    auto weight=[&](std::vector<float>& v,int n,int k){Tensor t=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,n,k,1,1,v.data());backend.wrap_weight(t);return t;};
    Tensor wr=weight(router,experts,hidden),wg=weight(sg,shared,hidden),wu=weight(su,shared,hidden),wd=weight(sd,hidden,shared),ws=weight(ss,1,hidden);
    Tensor gate=Tensor::create(Precision::NVFP4,MemoryType::EXTERNAL,2*inter,hidden,experts);
    Tensor down=Tensor::create(Precision::NVFP4,MemoryType::EXTERNAL,hidden,inter,experts);
    gate.moe_ssd_source=&provider.gate_source;down.moe_ssd_source=&provider.down_source;
    Tensor control_gate=gate,control_down=down;
    if(canonical){control_gate.moe_ssd_source=&canonical->gate_source;control_down.moe_ssd_source=&canonical->down_source;}
    GraphNode node;node.op_type=OpType::MOE;node.params.i32={hidden,experts,2,inter,shared,0,1,1,1,1,1};
    for(int rows:{15,16,31,32,512,1024,1025,1743,2048,2049,1}) {
        Tensor x=Tensor::create(Precision::FP32,MemoryType::NONE,hidden,rows),y=x;
        require(backend.alloc_output(x,x.nbytes(),nullptr)&&backend.alloc_output(y,y.nbytes(),nullptr));
        std::vector<float> input(x.nelements()),control(input.size()),actual(input.size());
        for(int variant=0;variant<3;++variant) {
            for(size_t i=0;i<input.size();++i)input[i]=float(int((i*13+variant*7)%97)-48)/128.f;
            require(backend.copy_from_host(input.data(),x,x.nbytes()));
            require(backend.set_logical_prefill_batch(false));
            backend.dispatch(node,{&x,&wr,&control_gate,&control_down,&wg,&wu,&wd,&ws},&y,nullptr);backend.end_graph();
            require(!backend.dispatch_failed()&&backend.copy_to_host(y,control.data(),y.nbytes()));
            require(backend.set_logical_prefill_batch(true));
            const auto fused=backend.nvfp4_activation_fused_calls(),kmajor=backend.nvfp4_input_kmajor_calls(),words=backend.nvfp4_token_pair_word_calls();
            const auto token_codes=backend.nvfp4_token_code_kmajor_calls();
            const auto scale_bytes=backend.nvfp4_gate_scale_cache_bytes();
            backend.dispatch(node,{&x,&wr,&gate,&down,&wg,&wu,&wd,&ws},&y,nullptr);backend.end_graph();
            require(!backend.dispatch_failed()&&backend.copy_to_host(y,actual.data(),y.nbytes()));
            if(scale_view||(code_view&&!token_words)) {
                require(backend.nvfp4_gate_scale_cache_bytes()==scale_bytes);
                std::printf("logical rejected fusion rows=%d scale_cache_growth=0\n",rows);
            }
            // This fixture enables BATCH_SMALL, whose native batch lower bound is32.
            const size_t selected=rows>=32&&rows<=2048&&(!code_view||token_words);
            require(backend.nvfp4_token_code_kmajor_calls()-token_codes==(selected&&code_view&&token_codes_kmajor&&token_words));
            std::printf("logical selection rows=%d fused=%zu kmajor=%zu words=%zu expected=%zu\n",rows,
                backend.nvfp4_activation_fused_calls()-fused,backend.nvfp4_input_kmajor_calls()-kmajor,
                backend.nvfp4_token_pair_word_calls()-words,selected);std::fflush(stdout);
            require(backend.nvfp4_activation_fused_calls()-fused==selected&&
                backend.nvfp4_input_kmajor_calls()-kmajor==selected*input_kmajor&&
                backend.nvfp4_token_pair_word_calls()-words==selected*token_words);
            require(!std::memcmp(control.data(),actual.data(),y.nbytes()));
            std::printf("NVFP4 logical batch rows=%d variant=%d selected=%zu output_byte_exact=1\n",rows,variant,selected);std::fflush(stdout);
        }
        backend.free_output(x,nullptr);backend.free_output(y,nullptr);
    }
    require(backend.set_logical_prefill_batch(false));
    std::printf("capture policy immutable; default/large/decode boundaries; two distinct resident experts code_view=%d versus_canonical=%d PASS\n",code_view,canonical!=nullptr);
}
