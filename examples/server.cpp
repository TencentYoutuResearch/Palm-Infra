#include "examples/cli_common.h"

#include <json.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

struct ServerOptions {
    CliCommonOptions runtime;
    std::string host = "127.0.0.1";
    int port = 8080;
};

bool parse_int(const char *s, int &out) {
    char *end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (!s[0] || !end || *end || v < 1 || v > 65535)
        return false;
    out = (int)v;
    return true;
}

void usage(const char *argv0) {
    std::fprintf(
        stderr,
        "Usage: %s --package model.mollm [--host 127.0.0.1] [--port 8080]\n"
        "       [--device cpu|metal] [--threads N] [--n-ctx N]\n"
        "Endpoints: GET /v1/models, POST /v1/chat/completions\n",
        argv0);
}

bool parse_args(int argc, char **argv, ServerOptions &o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto value = [&](const char *flag) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "server: %s needs a value\n", flag);
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--package") {
            const char *v = value("--package");
            if (!v)
                return false;
            o.runtime.package_path = v;
        } else if (a == "--host") {
            const char *v = value("--host");
            if (!v)
                return false;
            o.host = v;
        } else if (a == "--port") {
            const char *v = value("--port");
            if (!v || !parse_int(v, o.port))
                return false;
        } else if (a == "--threads") {
            const char *v = value("--threads");
            if (!v || !parse_int(v, o.runtime.num_threads))
                return false;
        } else if (a == "--n-ctx") {
            const char *v = value("--n-ctx");
            if (!v || !parse_int(v, o.runtime.n_ctx))
                return false;
        } else if (a == "--device") {
            const char *v = value("--device");
            if (!v)
                return false;
            if (!std::strcmp(v, "cpu"))
                o.runtime.device = Device::CPU;
#ifdef MOLLM_METAL
            else if (!std::strcmp(v, "metal"))
                o.runtime.device = Device::METAL;
#endif
            else {
                std::fprintf(stderr, "server: unsupported device %s\n", v);
                return false;
            }
        } else if (a == "--mmap")
            o.runtime.weight_loading = WeightLoadingMode::MMAP;
        else if (a == "--help" || a == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "server: unknown option %s\n", a.c_str());
            return false;
        }
    }
    if (o.runtime.package_path.empty()) {
        usage(argv[0]);
        return false;
    }
    o.runtime.temperature = 0.0f;
    o.runtime.top_k = 0;
    o.runtime.top_p = 0.0f;
    return true;
}

bool send_all(int fd, const std::string &s) {
    size_t off = 0;
    while (off < s.size()) {
        ssize_t n = ::send(fd, s.data() + off, s.size() - off, 0);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
            return false;
        off += (size_t)n;
    }
    return true;
}

void send_json(int fd, int status, const json &body) {
    std::string payload = body.dump();
    std::string reason = status == 200   ? "OK"
                         : status == 404 ? "Not Found"
                                         : "Bad Request";
    std::string h = "HTTP/1.1 " + std::to_string(status) + " " + reason +
                    "\r\nContent-Type: application/json\r\nContent-Length: " +
                    std::to_string(payload.size()) +
                    "\r\nConnection: close\r\n\r\n";
    send_all(fd, h + payload);
}

void send_error(int fd, int status, const std::string &message,
                const char *type) {
    send_json(fd, status, {{"error", {{"message", message}, {"type", type}}}});
}

bool begin_sse_response(int fd) {
    static constexpr char kHeader[] =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n\r\n";
    return send_all(fd, kHeader);
}

bool send_sse_event(int fd, const json &event) {
    return send_all(fd, "data: " + event.dump() + "\n\n");
}

json make_stream_chunk(const std::string &id, const std::string &model,
                       const json &delta, const json &finish_reason) {
    return {{"id", id},
            {"object", "chat.completion.chunk"},
            {"model", model},
            {"choices", json::array({{{"index", 0},
                                       {"delta", delta},
                                       {"finish_reason", finish_reason}}})}};
}

bool read_request(int fd, std::string &method, std::string &path,
                  std::string &body) {
    std::string data;
    char buf[8192];
    size_t header_end = std::string::npos;
    while ((header_end = data.find("\r\n\r\n")) == std::string::npos) {
        ssize_t n = recv(fd, buf, sizeof(buf), 0);
        if (n <= 0)
            return false;
        data.append(buf, (size_t)n);
        if (data.size() > 1024 * 1024)
            return false;
    }
    size_t line_end = data.find("\r\n");
    if (line_end == std::string::npos)
        return false;
    std::string first = data.substr(0, line_end);
    size_t a = first.find(' '), b = first.find(' ', a + 1);
    if (a == std::string::npos || b == std::string::npos)
        return false;
    method = first.substr(0, a);
    path = first.substr(a + 1, b - a - 1);
    size_t content_length = 0, pos = line_end + 2;
    while (pos < header_end) {
        size_t e = data.find("\r\n", pos);
        std::string h = data.substr(pos, e - pos);
        size_t c = h.find(':');
        if (c != std::string::npos) {
            std::string k = h.substr(0, c);
            std::transform(k.begin(), k.end(), k.begin(), ::tolower);
            if (k == "content-length")
                content_length =
                    (size_t)std::strtoull(h.c_str() + c + 1, nullptr, 10);
        }
        pos = e + 2;
    }
    if (content_length > 1024 * 1024)
        return false;
    body = data.substr(header_end + 4);
    while (body.size() < content_length) {
        ssize_t n = recv(fd, buf, sizeof(buf), 0);
        if (n <= 0)
            return false;
        body.append(buf, (size_t)n);
    }
    if (body.size() > content_length)
        body.resize(content_length);
    return true;
}

size_t common_prefix(const std::vector<int> &a, const std::vector<int> &b) {
    size_t n = std::min(a.size(), b.size()), i = 0;
    while (i < n && a[i] == b[i])
        ++i;
    return i;
}

class Server {
  public:
    Server(LLMEngine &e, const Tokenizer &t, std::string model)
        : engine(e), tokenizer(t), model_id(std::move(model)) {}

    void handle(int fd, const std::string &method, const std::string &path,
                const std::string &body) {
        if (method == "GET" && path == "/v1/models") {
            send_json(fd, 200,
                      {{"object", "list"},
                       {"data", json::array({{{"id", model_id},
                                              {"object", "model"},
                                              {"owned_by", "mollm"}}})}});
            return;
        }
        if (method != "POST" || path != "/v1/chat/completions") {
            send_error(fd, 404, "route not found", "not_found");
            return;
        }
        json req;
        try {
            req = json::parse(body);
        } catch (const std::exception &ex) {
            send_error(fd, 400, ex.what(), "invalid_request_error");
            return;
        }
        if (req.contains("temperature") &&
            (!req["temperature"].is_number() ||
             req["temperature"].get<double>() != 0.0)) {
            send_error(fd, 400,
                       "this server currently supports temperature=0 only",
                       "invalid_request_error");
            return;
        }
        std::vector<ChatMessage> messages;
        try {
            for (const auto &m : req.at("messages")) {
                std::string role = m.at("role").get<std::string>();
                if (role != "system" && role != "user" && role != "assistant")
                    throw std::runtime_error("unsupported message role: " +
                                             role);
                messages.push_back({role, m.at("content").get<std::string>()});
            }
        } catch (const std::exception &ex) {
            send_error(fd, 400, ex.what(), "invalid_request_error");
            return;
        }
        std::vector<int> full = tokenizer.apply_chat(messages);
        if (full.empty()) {
            send_error(fd, 400, "empty prompt", "invalid_request_error");
            return;
        }
        size_t prefix = common_prefix(cached_tokens, full);
        if (prefix < cached_tokens.size()) {
            engine.reset();
            cached_tokens.clear();
            prefix = 0;
        }
        std::vector<int> delta(full.begin() + (ptrdiff_t)prefix, full.end());
        if (delta.empty()) {
            engine.reset();
            cached_tokens.clear();
            delta = full;
        }
        int max_tokens = 256;
        try {
            max_tokens = req.value("max_tokens",
                                   req.value("max_completion_tokens", 256));
        } catch (const std::exception &ex) {
            send_error(fd, 400, ex.what(), "invalid_request_error");
            return;
        }
        int available = engine.config().n_ctx - engine.past_len();
        if (max_tokens < 1 || available < 1) {
            send_error(fd, 400,
                       "max_tokens/context limit leaves no generation capacity",
                       "invalid_request_error");
            return;
        }
        max_tokens = std::min(max_tokens, available);
        bool stream = req.value("stream", false);
        GenerationResult result;
        std::string error;
        const std::string id = "chatcmpl-mollm";
        if (stream) {
            if (!begin_sse_response(fd))
                return;
            send_sse_event(fd, make_stream_chunk(id, model_id,
                                                  {{"role", "assistant"}}, nullptr));
        }
        std::function<void(int, const std::string &)> on_token;
        if (stream) {
            on_token = [&](int, const std::string &piece) {
                send_sse_event(fd, make_stream_chunk(id, model_id,
                                                      {{"content", piece}}, nullptr));
            };
        }
        bool ok = generate_greedy(engine, tokenizer, delta, max_tokens,
                                  tokenizer.eos_id(), result, error, on_token, false);
        if (!ok) {
            engine.reset();
            cached_tokens.clear();
            if (!stream)
                send_error(fd, 400, error, "server_error");
            else
                send_sse_event(fd, {{"error", {{"message", error},
                                               {"type", "server_error"}}}});
            return;
        }
        cached_tokens.insert(cached_tokens.end(), delta.begin(), delta.end());
        if (result.token_ids.size() > 1)
            cached_tokens.insert(cached_tokens.end(), result.token_ids.begin(),
                                 result.token_ids.end() - 1);
        if ((int)cached_tokens.size() != engine.past_len()) {
            engine.reset();
            cached_tokens.clear();
        }
        if (stream) {
            send_sse_event(fd, make_stream_chunk(id, model_id, json::object(),
                                                  result.hit_eos ? "stop" : "length"));
            send_all(fd, "data: [DONE]\n\n");
            return;
        }
        send_json(
            fd, 200,
            {{"id", id},
             {"object", "chat.completion"},
             {"model", model_id},
             {"choices",
              json::array(
                  {{{"index", 0},
                    {"message",
                     {{"role", "assistant"}, {"content", result.text}}},
                    {"finish_reason", result.hit_eos ? "stop" : "length"}}})},
             {"usage",
              {{"prompt_tokens", full.size()},
               {"completion_tokens", result.token_ids.size()},
               {"total_tokens", full.size() + result.token_ids.size()}}}});
    }

  private:
    LLMEngine &engine;
    const Tokenizer &tokenizer;
    std::string model_id;
    std::vector<int> cached_tokens;
};

} // namespace

int main(int argc, char **argv) {
    ServerOptions opts;
    if (!parse_args(argc, argv, opts))
        return 1;
    Tokenizer tokenizer;
    LLMEngine engine;
    int prefill = 0;
    std::string error;
    if (!load_runtime(opts.runtime, tokenizer, engine, prefill, error)) {
        std::fprintf(stderr, "server: %s\n", error.c_str());
        return 1;
    }
    std::string model = "mollm";
    auto it = engine.package_metadata().find("model_name");
    if (it != engine.package_metadata().end())
        model = it->second;
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) {
        std::perror("socket");
        return 1;
    }
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)opts.port);
    if (inet_pton(AF_INET, opts.host.c_str(), &addr.sin_addr) != 1) {
        std::fprintf(stderr, "server: host must be an IPv4 address\n");
        return 1;
    }
    if (bind(s, (sockaddr *)&addr, sizeof(addr)) < 0 || listen(s, 16) < 0) {
        std::perror("bind/listen");
        return 1;
    }
    std::fprintf(stderr, "mollm server listening on http://%s:%d (model=%s)\n",
                 opts.host.c_str(), opts.port, model.c_str());
    Server server(engine, tokenizer, model);
    for (;;) {
        int fd = accept(s, nullptr, nullptr);
        if (fd < 0) {
            if (errno == EINTR)
                continue;
            std::perror("accept");
            break;
        }
        std::string method, path, body;
        if (read_request(fd, method, path, body))
            server.handle(fd, method, path, body);
        else
            send_json(fd, 400,
                      {{"error",
                        {{"message", "invalid HTTP request"},
                         {"type", "invalid_request_error"}}}});
        close(fd);
    }
    close(s);
    return 0;
}
