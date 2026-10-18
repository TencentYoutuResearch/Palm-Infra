#include "engine/engine.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

struct Result {
    std::vector<float> hidden, logits, all_hidden, all_logits;
    int token = -1;
};
static bool evaluate(LLMEngine& engine, bool decode, const std::vector<int>& inputs, Result& result) {
    std::fprintf(stderr, "PROBE_BEGIN mode=%s tokens=%zu\n", decode ? "decode" : "prefill", inputs.size());
    engine.reset();
    if(engine.prefill(std::vector<int>(16,0))<0)return false;
    size_t width=0;
    const size_t calls=decode?inputs.size():1;
    for(size_t index=0;index<calls;++index) {
        Tensor hidden=decode?engine.decode_hidden(inputs[index]):engine.prefill_hidden(inputs);
        if(!hidden.data || hidden.prec!=Precision::FP32 || hidden.shape[2]!=1 || hidden.shape[3]!=1)return false;
        width=hidden.shape[0];
        const size_t tokens=decode?1:inputs.size();
        if(hidden.nelements()!=static_cast<int64_t>(width*tokens))return false;
        const float* data=static_cast<const float*>(hidden.data);
        result.all_hidden.insert(result.all_hidden.end(),data,data+width*tokens);
        auto logits=engine.run_lmhead_raw(hidden,tokens,true);
        if(logits.empty() || logits.size()%tokens)return false;
        result.all_logits.insert(result.all_logits.end(),logits.begin(),logits.end());
    }
    result.hidden.assign(result.all_hidden.begin(),result.all_hidden.begin()+width);
    result.logits.assign(result.all_logits.begin(),result.all_logits.begin()+result.all_logits.size()/inputs.size());
    result.token=std::max_element(result.logits.begin(),result.logits.end())-result.logits.begin();
    return true;
}
static bool compare(const char* name,Result a,Result b,bool full=false,bool strict=false) {
    if(full) {a.hidden=std::move(a.all_hidden);b.hidden=std::move(b.all_hidden);
              a.logits=std::move(a.all_logits);b.logits=std::move(b.all_logits);}
    if(a.hidden.size()!=b.hidden.size() || a.logits.size()!=b.logits.size())return false;
    double hidden_error=0,logit_error=0,hidden_square=0,error_square=0;
    for(size_t i=0;i<a.hidden.size();++i) {
        if(!std::isfinite(a.hidden[i]) || !std::isfinite(b.hidden[i]))return false;
        hidden_error=std::max(hidden_error,static_cast<double>(std::fabs(a.hidden[i]-b.hidden[i])));
        hidden_square+=static_cast<double>(a.hidden[i])*a.hidden[i];
        const double delta=static_cast<double>(a.hidden[i])-b.hidden[i];error_square+=delta*delta;
    }
    for(size_t i=0;i<a.logits.size();++i) {
        if(!std::isfinite(a.logits[i]) || !std::isfinite(b.logits[i]))return false;
        logit_error=std::max(logit_error,static_cast<double>(std::fabs(a.logits[i]-b.logits[i])));
    }
    std::printf("%s token_a=%d token_b=%d max_hidden_error=%.9g relative_hidden_l2=%.9g max_logit_error=%.9g\n",
        name,a.token,b.token,hidden_error,std::sqrt(error_square/std::max(hidden_square,1e-30)),logit_error);
    return !strict || (hidden_error==0 && logit_error==0);
}
int main(int argc,char** argv) {
    if(argc!=2 && (argc!=3 || std::strcmp(argv[2],"--strict")))return 1;
    const bool strict=argc==3;
    EngineConfig config;config.package_path=argv[1];config.device=Device::CUDA;
    config.device_fallback=DeviceFallbackPolicy::REQUIRE_REQUESTED;
    config.offload_ple=true;config.n_ctx=512;config.num_threads=8;config.sampling.temperature=0;
    LLMEngine engine;if(!engine.load(config))return 1;
    const int first=engine.prefill(std::vector<int>(16,0));
    if(first<0)return 1;
    std::printf("prompt_first_token=%d\n",first);
    Result serial,single,batch_a,batch_b,serial_four;
    if(!evaluate(engine,true,{first},serial) || !evaluate(engine,false,{first},single) ||
       !evaluate(engine,false,{first,24,15,1088},batch_a) ||
       !evaluate(engine,false,{first,11,24,24},batch_b) ||
       !evaluate(engine,true,{first,24,15,1088},serial_four))return 1;
    bool matches=compare("decode_vs_prefill1",serial,single,false,strict);
    matches=compare("prefill1_vs_prefill4",single,batch_a,false,strict)&&matches;
    matches=compare("prefill4_different_future",batch_a,batch_b,false,strict)&&matches;
    matches=compare("serial4_vs_prefill4_all_positions",serial_four,batch_a,true,strict)&&matches;
    if(!matches)return 2;
    // Default diagnostic; --strict requires exact hidden/logit equality for
    // this fixture only, not broader model quality or arbitrary prompts.
    return 0;
}
