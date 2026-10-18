#pragma once
#include <climits>
#include <string>
#include <vector>

namespace mollm_bench {
// Benchmark-only diagnostic; default leaves the tokenized prompt unchanged.
inline bool parse_prompt_prefix(const char* value,bool dummy,int& limit,std::string& error) {
    limit=0;
    if(!value)return true;
    if(dummy||!*value) {error="prompt-prefix diagnostic requires a positive length and real prompt";return false;}
    int parsed=0;
    for(const char* p=value;*p;++p) {
        if(*p<'0'||*p>'9'||parsed>(INT_MAX-(*p-'0'))/10) {
            error="invalid MOLLM_BENCH_PROMPT_PREFIX_TOKENS";return false;
        }
        parsed=parsed*10+(*p-'0');
    }
    if(!parsed){error="prompt-prefix length must be positive";return false;}
    limit=parsed;return true;
}
inline bool apply_prompt_prefix(std::vector<int>& ids,int limit,std::string& error) {
    if(limit<0||size_t(limit)>ids.size()) {
        error="prompt-prefix length exceeds tokenized prompt";return false;
    }
    if(limit)ids.resize(size_t(limit));
    return true;
}
}
