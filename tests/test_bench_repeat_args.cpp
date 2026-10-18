#include "examples/cli_common.h"
#include "examples/bench_prompt_prefix.h"
#include "engine/logical_prefill.h"

#include <cstdio>
#include <string>
#include <vector>

static bool parse(std::vector<std::string> args, int expected, bool valid) {
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    CliCommonOptions options;
    std::string error;
    const bool accepted = parse_common_args(static_cast<int>(argv.size()), argv.data(), options, error);
    if (accepted != valid || (accepted && options.repeat != expected) || (!accepted && error.empty())) {
        std::fprintf(stderr, "repeat argument regression: %s\n", error.c_str());
        return false;
    }
    return true;
}

int main() {
    {
        int prefix=-1;std::string error;
        if(!mollm_bench::parse_prompt_prefix(nullptr,false,prefix,error)||prefix!=0)return 1;
        for(const char* invalid:{"","0","-1","+1","1x","2147483648","99999999999999999"})
            if(mollm_bench::parse_prompt_prefix(invalid,false,prefix,error)||error.empty())return 1;
        if(mollm_bench::parse_prompt_prefix("2",true,prefix,error))return 1;
        if(!mollm_bench::parse_prompt_prefix("2",false,prefix,error)||prefix!=2)return 1;
        std::vector<int> ids{41,17,93};
        if(!mollm_bench::apply_prompt_prefix(ids,0,error)||ids.size()!=3)return 1;
        if(mollm_bench::apply_prompt_prefix(ids,4,error)||ids.size()!=3)return 1;
        if(mollm_bench::apply_prompt_prefix(ids,-1,error)||ids.size()!=3)return 1;
        if(!mollm_bench::apply_prompt_prefix(ids,prefix,error)||ids!=std::vector<int>({41,17}))return 1;
    }
    CliCommonOptions defaults;
    if(defaults.cuda_prefill_logical_batch||make_engine_config(defaults).cuda_prefill_logical_batch)return 1;
    if (defaults.cuda_prefill_reclaim || make_engine_config(defaults).cuda_prefill_reclaim) return 1;
    defaults.cuda_prefill_reclaim = true;
    if (!make_engine_config(defaults).cuda_prefill_reclaim) return 1;
    if (!parse({"bench", "--package", "fixture.mollm", "--cuda-prefill-reclaim"}, 1, false)) return 1;
    if (!parse({"bench", "--package", "fixture.mollm", "--offload-ple", "--cuda-prefill-reclaim"}, 1, false)) return 1;
#ifdef MOLLM_CUDA
    for(const char* context:{"1","2048","16384","32768"})for(const char* chunk:{"0","1","128","512","1024","1743","2048"}) {
        std::vector<std::string> args{"bench","--package","fixture.mollm","--device","cuda",
            "--offload-ple","--cuda-prefill-reclaim","--cuda-prefill-logical-batch",
            "--n-ctx",context,"--prefill-chunk-size",chunk};
        std::vector<char*> argv;for(auto& arg:args)argv.push_back(arg.data());
        CliCommonOptions opts;std::string error;
        if(!parse_common_args(argv.size(),argv.data(),opts,error)||!opts.cuda_prefill_logical_batch||
            !make_engine_config(opts).cuda_prefill_logical_batch)return 1;
        const auto config=make_engine_config(opts);
        if(config.n_ctx!=std::stoi(context)||config.prefill_chunk_size!=std::stoi(chunk)||
           !mollm_prefill::logical_config_supported(config)||
           mollm_prefill::logical_chunk_limit(config)!=(std::stoi(chunk)>0?std::stoi(chunk):2048))return 1;
    }
    for(const char* chunk:{"2049","8192"})
        if(!parse({"bench","--package","fixture.mollm","--device","cuda","--offload-ple",
            "--cuda-prefill-reclaim","--cuda-prefill-logical-batch","--n-ctx","16384",
            "--prefill-chunk-size",chunk},1,false))return 1;
    for (const char* chunk : {"256", "512", "1024"}) {
        std::vector<std::string> reclaim_args{"bench", "--package", "fixture.mollm",
            "--device", "cuda", "--offload-ple", "--cuda-prefill-reclaim",
            "--n-ctx", "4096", "--prefill-chunk-size", chunk};
        std::vector<char*> reclaim_argv;
        for (auto& arg : reclaim_args) reclaim_argv.push_back(arg.data());
        CliCommonOptions opts; std::string error;
        if (!parse_common_args(reclaim_argv.size(), reclaim_argv.data(), opts, error) ||
            !opts.cuda_prefill_reclaim || !make_engine_config(opts).cuda_prefill_reclaim) return 1;
    }
    for (const char* bad : {"--mmap", "--n-ctx", "--prefill-chunk-size"}) {
        std::vector<std::string> args{"bench", "--package", "fixture.mollm", "--device", "cuda",
            "--offload-ple", "--cuda-prefill-reclaim", "--n-ctx", "4096", bad};
        if (std::string(bad) != "--mmap") args.push_back("8192");
        if (!parse(args, 1, false)) return 1;
    }
    if (!parse({"bench", "--package", "fixture.mollm", "--device", "cuda",
        "--cuda-prefill-reclaim", "--n-ctx", "4096"}, 1, false)) return 1;
#endif
    for (const char* value : {"0", "128", "512", "1024"}) {
        std::vector<std::string> chunk_args{"bench", "--package", "fixture.mollm",
            "--prefill-chunk-size", value};
        std::vector<char*> chunk_argv;
        for (auto& arg : chunk_args) chunk_argv.push_back(arg.data());
        CliCommonOptions chunk_options;
        std::string chunk_error;
        if (!parse_common_args(chunk_argv.size(), chunk_argv.data(), chunk_options, chunk_error)
            || chunk_options.prefill_chunk_size != std::stoi(value)
            || make_engine_config(chunk_options).prefill_chunk_size != std::stoi(value)) return 1;
    }
    for (const char* value : {"-1", "2x", "", "2147483648"})
        if (!parse({"bench", "--package", "fixture.mollm", "--prefill-chunk-size", value}, 0, false)) return 1;
    if (!parse({"bench", "--package", "fixture.mollm", "--prefill-chunk-size"}, 0, false)) return 1;
    if (!parse({"bench", "--package", "fixture.mollm"}, 1, true)) return 1;
    for (const char* value : {"1", "4", "10000"})
        if (!parse({"bench", "--package", "fixture.mollm", "--repeat", value}, std::stoi(value), true)) return 1;
    for (const char* value : {"0", "-1", "10001", "four", "2x", "", "2147483648"})
        if (!parse({"bench", "--package", "fixture.mollm", "--repeat", value}, 0, false)) return 1;
    if (!parse({"bench", "--package", "fixture.mollm", "--repeat"}, 0, false)) return 1;
    if (!parse({"bench", "--package", "fixture.mollm", "--mtp-companion"}, 0, false)) return 1;
    if (!parse({"bench", "--package", "fixture.mollm", "--mtp-companion", ""}, 0, false)) return 1;
    std::vector<std::string> args{"bench", "--package", "fixture.mollm",
        "--mtp-companion", "/local/draft", "--mtp-draft-tokens", "3", "--print-token-ids"};
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    CliCommonOptions options;
    std::string error;
    if (!parse_common_args(argv.size(), argv.data(), options, error)) return 1;
    if (!options.print_token_ids) return 1;
    const EngineConfig config = make_engine_config(options);
    if (config.mtp_companion_dir != "/local/draft" || config.mtp_draft_tokens != 3) return 1;
    return 0;
}
