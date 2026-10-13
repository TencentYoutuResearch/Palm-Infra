#define main mollm_ppl_entry_for_test
#include "examples/ppl.cpp"
#undef main

int main() {
    auto parse = [](std::vector<std::string> args, bool expected) {
        std::vector<char*> argv;
        for (auto& arg : args) argv.push_back(arg.data());
        Options options;
        std::string error;
        return parse_args(int(argv.size()), argv.data(), options, error) == expected &&
            (expected || !error.empty());
    };
    const std::vector<std::string> base{"ppl", "--package", "unused", "--text", "test"};
    auto logical=base;
    logical.insert(logical.end(),{"--device","cuda","--offload-ple","--cuda-prefill-reclaim",
        "--cuda-prefill-logical-batch","--n-ctx","2048","--chunk-size","2048"});
#ifdef MOLLM_CUDA
    if(!parse(logical,true))return 1;
    for(const char* chunk:{"128","512","1024","2048"}) {
        auto supported=logical;
        *(std::find(supported.begin(),supported.end(),"--n-ctx")+1)="16384";
        *(std::find(supported.begin(),supported.end(),"--chunk-size")+1)=chunk;
        if(!parse(supported,true))return 1;
    }
    for(const char* flag:{"--cuda-prefill-reclaim","--offload-ple"}) {
        auto bad=logical;bad.erase(std::find(bad.begin(),bad.end(),flag));if(!parse(bad,false))return 1;
    }
    for(const char* option:{"--chunk-size"}) {
        auto bad=logical;auto it=std::find(bad.begin(),bad.end(),option);*(it+1)="2049";
        if(!parse(bad,false))return 1;
    }
#else
    if(!parse(logical,false))return 1;
#endif
    for (const char* value : {"0", "-1", "x", "2x", "2147483648"}) {
        auto args = base;
        args.insert(args.end(), {"--decode-after-prefill", value});
        if (!parse(args, false)) return 1;
    }
    auto args = base;
    args.push_back("--decode-after-prefill");
    if (!parse(args, false)) return 1;
    args.push_back("256");
    if (!parse(args, true)) return 1;
    args.push_back("--decode-token-by-token");
    if (!parse(args, false)) return 1;
    for (int tokens : {2, 7, 257, 357}) {
        for (int prefix : {0, 1, 4, 256, 512}) {
            Options options;
            options.decode_after_prefill = prefix;
            options.chunk_size = 32;
            int targets = 0, prefill_calls = 0, decode_calls = 0;
            for (int offset = 0; offset < tokens - 1;) {
                const auto step = scoring_step(options, offset, tokens - offset);
                if (step.count <= 0 || (step.decode && step.count != 1)) return 1;
                targets += std::min(step.count, tokens - 1 - offset);
                prefill_calls += !step.decode;
                decode_calls += step.decode;
                offset += step.count;
            }
            if (targets != tokens - 1) return 1;
            if (prefix && (prefill_calls != 1 || decode_calls != std::max(0, tokens - 1 - prefix))) return 1;
            if (!prefix && decode_calls) return 1;
        }
        Options options;
        options.decode_token_by_token = true;
        for (int offset = 0; offset < tokens - 1; ++offset) {
            const auto step = scoring_step(options, offset, tokens - offset);
            if (step.count != 1 || step.decode != (offset > 0)) return 1;
        }
    }
    return 0;
}
