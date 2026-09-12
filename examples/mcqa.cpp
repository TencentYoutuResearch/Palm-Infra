#include "engine/engine.h"
#include "engine/tokenizer.h"
#include "runtime/threading.h"

#include <json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

struct Options {
    std::string package_path;
    std::string data_path;
    std::string output_path;
    int limit = 0;
    int n_ctx = 4096;
    int threads = default_worker_threads();
    Device device = Device::CPU;
    WeightLoadingMode weight_loading = WeightLoadingMode::RESIDENT;
};

bool parse_int(const char* text, int& value) {
    if (!text || !*text)
        return false;
    char* end = nullptr;
    long parsed = std::strtol(text, &end, 10);
    if (!end || *end || parsed < 0 ||
        parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

void usage(const char* argv0) {
    std::fprintf(
        stderr,
        "Usage: %s --package model.mollm --data questions.jsonl [options]\n"
        "Each JSONL row needs {\"prompt\": string, \"gold\": \"A\"..\"J\"}.\n"
        "An optional \"choices\" string selects the allowed labels "
        "(default: \"ABCD\").\n"
        "Options:\n"
        "  --output results.jsonl\n"
        "  --limit N\n"
        "  --threads N\n"
        "  --n-ctx N\n"
        "  --device cpu|metal\n"
        "  --mmap\n",
        argv0);
}

bool parse_args(int argc, char** argv, Options& opts) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "mcqa: %s needs a value\n", flag);
                return nullptr;
            }
            return argv[++i];
        };
        if (arg == "--package") {
            const char* v = value("--package");
            if (!v) return false;
            opts.package_path = v;
        } else if (arg == "--data") {
            const char* v = value("--data");
            if (!v) return false;
            opts.data_path = v;
        } else if (arg == "--output") {
            const char* v = value("--output");
            if (!v) return false;
            opts.output_path = v;
        } else if (arg == "--limit") {
            const char* v = value("--limit");
            if (!v || !parse_int(v, opts.limit)) return false;
        } else if (arg == "--threads") {
            const char* v = value("--threads");
            if (!v || !parse_int(v, opts.threads) || opts.threads < 1)
                return false;
        } else if (arg == "--n-ctx") {
            const char* v = value("--n-ctx");
            if (!v || !parse_int(v, opts.n_ctx) || opts.n_ctx < 2)
                return false;
        } else if (arg == "--device") {
            const char* v = value("--device");
            if (!v) return false;
            const std::string device = v;
            if (device == "cpu") {
                opts.device = Device::CPU;
            } else if (device == "metal") {
#ifdef MOLLM_METAL
                opts.device = Device::METAL;
#else
                std::fprintf(stderr,
                             "mcqa: Metal requires -DMOLLM_METAL=ON\n");
                return false;
#endif
            } else {
                std::fprintf(stderr, "mcqa: unknown device %s\n", v);
                return false;
            }
        } else if (arg == "--mmap") {
            opts.weight_loading = WeightLoadingMode::MMAP;
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "mcqa: unknown option %s\n", arg.c_str());
            return false;
        }
    }
    if (opts.package_path.empty() || opts.data_path.empty()) {
        usage(argv[0]);
        return false;
    }
    return true;
}

std::string parse_choices(const json& doc) {
    const std::string choices = doc.value("choices", std::string("ABCD"));
    if (choices.empty() || choices.size() > 10)
        return {};
    for (size_t i = 0; i < choices.size(); ++i) {
        if (choices[i] != static_cast<char>('A' + i))
            return {};
    }
    return choices;
}

char parse_gold(const json& doc, const std::string& choices) {
    const std::string gold = doc.at("gold").get<std::string>();
    for (char ch : gold) {
        if (choices.find(ch) != std::string::npos)
            return ch;
    }
    return '\0';
}

}  // namespace

int main(int argc, char** argv) {
    Options opts;
    if (!parse_args(argc, argv, opts))
        return 1;

    std::ifstream input(opts.data_path);
    if (!input) {
        std::fprintf(stderr, "mcqa: failed to open %s\n", opts.data_path.c_str());
        return 1;
    }
    std::vector<json> docs;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty())
            continue;
        try {
            docs.push_back(json::parse(line));
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "mcqa: invalid JSONL row: %s\n", ex.what());
            return 1;
        }
        if (opts.limit > 0 && static_cast<int>(docs.size()) >= opts.limit)
            break;
    }
    if (docs.empty()) {
        std::fprintf(stderr, "mcqa: no questions found\n");
        return 1;
    }

    size_t max_choice_count = 0;
    for (size_t index = 0; index < docs.size(); ++index) {
        const std::string choices = parse_choices(docs[index]);
        if (choices.empty()) {
            std::fprintf(stderr, "mcqa: row %zu has invalid choices\n", index);
            return 1;
        }
        if (!parse_gold(docs[index], choices)) {
            std::fprintf(stderr, "mcqa: row %zu has invalid gold label\n",
                         index);
            return 1;
        }
        max_choice_count = std::max(max_choice_count, choices.size());
    }

    LLMEngine engine;
    EngineConfig cfg;
    cfg.package_path = opts.package_path;
    cfg.n_ctx = opts.n_ctx;
    cfg.num_threads = opts.threads;
    cfg.device = opts.device;
    cfg.weight_loading = opts.weight_loading;
    cfg.sampling.temperature = 0.0f;
    if (!engine.load(cfg)) {
        std::fprintf(stderr, "mcqa: failed to load package\n");
        return 1;
    }

    Tokenizer tokenizer;
    const auto architecture_it =
        engine.package_metadata().find("architecture");
    const std::string architecture =
        architecture_it == engine.package_metadata().end()
            ? std::string()
            : architecture_it->second;
    if (!tokenizer.load(engine.config().tokenizer_path,
                        engine.config().chat_template_path, architecture)) {
        std::fprintf(stderr, "mcqa: failed to load tokenizer\n");
        return 1;
    }

    std::vector<int> choice_ids;
    for (size_t choice = 0; choice < max_choice_count; ++choice) {
        const char label = static_cast<char>('A' + choice);
        const std::vector<int> ids = tokenizer.encode(std::string(1, label));
        if (ids.size() != 1) {
            std::fprintf(stderr,
                         "mcqa: choice %c is not a single tokenizer token\n",
                         label);
            return 1;
        }
        choice_ids.push_back(ids[0]);
    }
    std::printf("choice_token_ids=");
    for (size_t i = 0; i < choice_ids.size(); ++i) {
        std::printf("%s%c:%d", i ? "," : "",
                    static_cast<char>('A' + i), choice_ids[i]);
    }
    std::printf("\n");

    int prefill_chunk = 256;
    const auto prefill_it =
        engine.package_metadata().find("prefill_seq_len");
    if (prefill_it != engine.package_metadata().end()) {
        int parsed = 0;
        if (parse_int(prefill_it->second.c_str(), parsed) && parsed > 0)
            prefill_chunk = parsed;
    }

    std::ofstream output;
    if (!opts.output_path.empty()) {
        output.open(opts.output_path);
        if (!output) {
            std::fprintf(stderr, "mcqa: failed to open %s\n",
                         opts.output_path.c_str());
            return 1;
        }
    }

    int correct = 0;
    int answered = 0;
    auto eval_start = std::chrono::steady_clock::now();
    for (size_t index = 0; index < docs.size(); ++index) {
        const json& doc = docs[index];
        const std::string choices = parse_choices(doc);
        if (choices.empty()) {
            std::fprintf(stderr, "mcqa: row %zu has invalid choices\n", index);
            return 1;
        }
        const char gold = parse_gold(doc, choices);
        if (!gold) {
            std::fprintf(stderr, "mcqa: row %zu has invalid gold label\n", index);
            return 1;
        }
        const std::string prompt = doc.at("prompt").get<std::string>();
        const std::vector<int> tokens = tokenizer.apply_chat(
            std::vector<ChatMessage>{{"user", prompt}},
            /*enable_thinking=*/false);
        if (tokens.empty() || static_cast<int>(tokens.size()) > opts.n_ctx) {
            std::fprintf(stderr, "mcqa: row %zu prompt has %zu tokens\n",
                         index, tokens.size());
            return 1;
        }

        engine.reset();
        Tensor hidden;
        int last_chunk = 0;
        for (size_t offset = 0; offset < tokens.size();
             offset += static_cast<size_t>(last_chunk)) {
            last_chunk = std::min(
                prefill_chunk, static_cast<int>(tokens.size() - offset));
            std::vector<int> chunk(
                tokens.begin() + static_cast<ptrdiff_t>(offset),
                tokens.begin() + static_cast<ptrdiff_t>(offset + last_chunk));
            hidden = engine.prefill_hidden(chunk);
            if (!hidden.data) {
                std::fprintf(stderr, "mcqa: prefill failed on row %zu\n", index);
                return 1;
            }
        }
        std::vector<float> logits =
            engine.run_lmhead_raw(hidden, last_chunk, false);
        if (logits.empty()) {
            std::fprintf(stderr, "mcqa: lm_head failed on row %zu\n", index);
            return 1;
        }

        int selected = 0;
        std::vector<float> scores(choices.size());
        std::vector<int> row_choice_ids(choices.size());
        for (size_t choice = 0; choice < choices.size(); ++choice) {
            const int token = choice_ids[choice];
            if (token < 0 || token >= static_cast<int>(logits.size())) {
                std::fprintf(stderr, "mcqa: choice token outside vocabulary\n");
                return 1;
            }
            row_choice_ids[choice] = token;
            scores[choice] = logits[token];
            if (scores[choice] > scores[selected])
                selected = static_cast<int>(choice);
        }
        const char answer = static_cast<char>('A' + selected);
        const bool is_correct = answer == gold;
        correct += is_correct ? 1 : 0;
        answered++;
        std::printf("%3zu gold=%c answer=%c correct=%s scores=",
                    index, gold, answer, is_correct ? "true" : "false");
        for (size_t choice = 0; choice < scores.size(); ++choice)
            std::printf("%s%.6f", choice ? "," : "", scores[choice]);
        std::printf("\n");
        std::fflush(stdout);
        if (output) {
            json row = {
                {"index", doc.value("index", static_cast<int>(index))},
                {"gold", std::string(1, gold)},
                {"answer", std::string(1, answer)},
                {"correct", is_correct},
                {"choice_token_ids", row_choice_ids},
                {"choice_logits", scores},
                {"prompt_tokens", tokens.size()},
            };
            output << row.dump() << '\n';
            output.flush();
        }
    }
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - eval_start).count();
    std::printf("score=%d/%d\n", correct, answered);
    std::printf("accuracy=%.6f\n",
                answered ? static_cast<double>(correct) / answered : 0.0);
    std::printf("elapsed_seconds=%.3f\n", seconds);
    return 0;
}
