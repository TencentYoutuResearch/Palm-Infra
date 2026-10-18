#include "storage/sha256.h"
#include <cstdio>
#include <exception>
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    try { std::puts(mollm::detail::sha256_file(argv[1]).c_str()); }
    catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
