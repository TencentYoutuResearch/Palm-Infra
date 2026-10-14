#include "engine/engine.h"
#include "backends/cuda/backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {
void require(bool x,const char* s){if(!x){std::fprintf(stderr,"QSA engine: %s\n",s);std::exit(1);}}
EngineConfig config(const char* path,int chunk) {
    EngineConfig c;c.package_path=path;c.device=Device::CUDA;
    c.device_fallback=DeviceFallbackPolicy::REQUIRE_REQUESTED;
    c.operator_fallback=OperatorFallbackPolicy::ALLOW_REFERENCE;
    c.n_ctx=64;c.prefill_chunk_size=chunk;c.num_threads=1;
    c.sampling.temperature=0.f;return c;
}
std::vector<float> hidden(LLMEngine& e,const std::vector<int>& ids) {
    Tensor t=e.prefill_hidden(ids);require(t.data&&t.prec==Precision::FP32,"hidden output");
    std::vector<float> result(t.ptr<float>(),t.ptr<float>()+t.nelements());
    require(std::all_of(result.begin(),result.end(),[](float x){return std::isfinite(x);}),"finite hidden");
    return result;
}
}
int main(int argc,char** argv) {
    require(argc==2,"package argument");
    CudaBackend probe;if(!probe.available())return 77;
    for(int chunk:{3,17}) {
        LLMEngine e;require(e.load(config(argv[1],chunk)),"load serialized QSA model");
        std::vector<int> prompt;for(int i=0;i<29;++i)prompt.push_back(1+i%31);
        require(e.prefill(prompt)>=0&&e.past_len()==29,"sparse prefill beyond budget");
        require(e.decode(17)>=0&&e.past_len()==30,"decode shares prefill QSA state");
        require(e.prefill({3,7,11,13,17})>=0&&e.past_len()==35,"multi-turn appends history");
        e.reset();require(e.past_len()==0,"reset position");
        auto reset=hidden(e,{2,5,7,11,13,17,19,23,29,31,3,4,6});
        LLMEngine fresh;require(fresh.load(config(argv[1],chunk)),"fresh comparison load");
        auto baseline=hidden(fresh,{2,5,7,11,13,17,19,23,29,31,3,4,6});
        require(reset==baseline,"reset exact fresh hidden");
        require(e.decode(9)>=0&&fresh.decode(9)>=0,"decode after reset/fresh");
    }
    LLMEngine cpu;auto unsupported=config(argv[1],3);unsupported.device=Device::CPU;
    require(!cpu.load(unsupported),"unsupported CPU QSA fails instead of dense substitution");
    std::puts("Serialized QSA engine: runtime context growth, sparse prefill, decode, multi-turn and reset pass");
}
