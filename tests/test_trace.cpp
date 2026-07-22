#include "kernels/trace.h"

#include <cstdio>
#include <fstream>
#include <string>

int main() {
    const std::string path = "/tmp/mollm_test_trace.json";
    mollm_trace::start(path);
    mollm_trace::set_thread_name("trace-test-main");
    mollm_trace::record_duration("unit", "complete_event", 1, 3,
                                 "{\"value\":7}");
    mollm_trace::record_flow("unit", "queued_event", 2, 42, true);
    mollm_trace::record_flow("unit", "queued_event", 4, 42, false);
    if (!mollm_trace::write()) {
        std::fprintf(stderr, "failed to write trace\n");
        return 1;
    }
    std::ifstream input(path);
    std::string json((std::istreambuf_iterator<char>(input)),
                     std::istreambuf_iterator<char>());
    std::remove(path.c_str());
    if (json.find("traceEvents") == std::string::npos ||
        json.find("complete_event") == std::string::npos ||
        json.find("queued_event") == std::string::npos ||
        json.find("trace-test-main") == std::string::npos ||
        json.find("\"value\":7") == std::string::npos) {
        std::fprintf(stderr, "trace JSON lacks expected events\n");
        return 1;
    }
    std::printf("Chrome trace test passed!\n");
    return 0;
}
