#include "cli_common.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace {
void require(bool ok,const char* reason){if(!ok)throw std::runtime_error(reason);}
struct Snapshot {
    int past=0,token=0;
    std::vector<uint32_t> hidden;
    bool operator==(const Snapshot& other)const{return past==other.past&&token==other.token&&hidden==other.hidden;}
};
std::vector<Snapshot> run(LLMEngine& engine,const Tokenizer& tokenizer,const std::vector<int>& ids) {
    engine.reset();require(engine.past_len()==0,"reset must clear position");
    std::vector<Snapshot> result;
    int consumed=0;
    auto record=[&](const Tensor& hidden,int count,const char* phase) {
        consumed+=count;require(engine.past_len()==consumed,"persistent position mismatch");
        require(hidden.data&&hidden.prec==Precision::FP32&&hidden.is_contiguous()&&
                hidden.shape[2]==1&&hidden.shape[3]==1,"invalid host hidden state");
        const auto* last=hidden.ptr<float>()+(hidden.shape[1]-1)*hidden.shape[0];
        Snapshot snapshot;snapshot.past=consumed;snapshot.hidden.resize(hidden.shape[0]);
        for(int d=0;d<hidden.shape[0];++d){require(std::isfinite(last[d]),"nonfinite hidden");std::memcpy(&snapshot.hidden[d],last+d,4);}
        auto logits=engine.run_lmhead_raw(hidden,int(hidden.shape[1]),false);
        require(!logits.empty()&&std::all_of(logits.begin(),logits.end(),[](float x){return std::isfinite(x);}),"invalid logits");
        snapshot.token=int(std::max_element(logits.begin(),logits.end())-logits.begin());
        std::printf("state stage=%zu phase=%s past=%d token=%d hidden_bits=",result.size(),phase,snapshot.past,snapshot.token);
        for(auto bits:snapshot.hidden)std::printf("%08x",bits);
        std::printf("\n");result.push_back(std::move(snapshot));
        return result.back().token;
    };
    auto prefill=[&](const std::vector<int>& input,const char* phase) {
        int token=-1;
        const int limit=engine.config().prefill_chunk_size>0?engine.config().prefill_chunk_size:2048;
        for(size_t start=0;start<input.size();start+=limit) {
            const size_t end=std::min(start+size_t(limit),input.size());
            std::vector<int> chunk(input.begin()+start,input.begin()+end);
            token=record(engine.prefill_hidden(chunk),int(chunk.size()),phase);
        }
        return token;
    };
    // Cross a partial pooled group, switch to decode, then back to prefill.
    prefill(std::vector<int>(ids.begin(),ids.begin()+4097),"first-prefill");
    for(int i=4097;i<4104;++i)record(engine.decode_hidden(ids[i]),1,"teacher-decode");
    int token=prefill(std::vector<int>(ids.begin()+4104,ids.end()),"continued-prefill");
    for(int i=0;i<8;++i)token=record(engine.decode_hidden(token),1,"generated-decode");
    const auto followup=tokenizer.encode("\n</think>\n<|im_end|>\n<|im_start|>user\n请继续说明缓存状态如何保持一致。<|im_end|>\n<|im_start|>assistant\n<think>\n");
    token=prefill(followup,"followup-prefill");
    for(int i=0;i<8;++i)token=record(engine.decode_hidden(token),1,"followup-decode");
    return result;
}
}
int main(int argc,char** argv) {
    try {
        CliCommonOptions opts;std::string error;
        if(!parse_common_args(argc,argv,opts,error))throw std::runtime_error(error);
        require(opts.device==Device::CUDA&&!opts.qsa_companion_dir.empty(),"requires CUDA QSA companion");
        std::string text=opts.prompt;
        if(!opts.prompt_file.empty()){std::ifstream input(opts.prompt_file);require(bool(input),"missing prompt file");text.assign(std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>());}
        require(!text.empty(),"long prompt is required");
        Tokenizer tokenizer;LLMEngine engine;
        if(!load_runtime(opts,tokenizer,engine,error))throw std::runtime_error(error);
        const auto ids=tokenizer.apply_chat(text);
        require(ids.size()>4104&&ids.size()+256<size_t(opts.n_ctx),"prompt must exceed4104 tokens and leave256 context slots");
        std::vector<Snapshot> reference;
        for(int repeat=0;repeat<std::max(2,opts.repeat);++repeat) {
            std::printf("state_repeat=%d\n",repeat);
            auto actual=run(engine,tokenizer,ids);
            if(repeat==0)reference=std::move(actual);
            else require(reference==actual,"reset versus fresh sequence differs");
        }
        std::printf("qsa_state_pass=1 snapshots=%zu prompt_tokens=%zu final_past=%d\n",reference.size(),ids.size(),reference.back().past);
        return 0;
    } catch(const std::exception& error){std::fprintf(stderr,"QSA state check: %s\n",error.what());return 1;}
}
