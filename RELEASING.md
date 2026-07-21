# Release checklist

Before publishing a release:

- Verify that bundled third-party code and model conversion dependencies are
  compatible with the Apache-2.0 project license.
- Build and run CTest on ARM64 CPU and Apple Metal configurations.
- Run the strict `pp256 + tg64`, warmup 3, 4-thread, five-process median suite
  for Youtu-LLM-2B and Qwen3.5-4B in FP16/W8/W4.
- Run the documented 512-token PPL checks and deterministic CPU/Metal text
  comparisons for supported Metal architectures.
- Test `mollm_server` models, non-streaming chat, streaming chat, exact-prefix
  reuse, and divergent-history invalidation.
- Confirm the repository contains no model packages, generated weights, build
  directories, calibration corpora without redistribution permission, tokens,
  local paths, or benchmark logs.
- Update README support/performance tables and document the exact hardware,
  commands, package format, and known limitations.
- Tag only after a clean clone passes configure, build, tests, and the README
  quick-start commands.

The HTTP server is currently a serialized local server. Do not describe it as
production-ready until authentication, TLS/reverse-proxy guidance, request
limits, cancellation, concurrency, and multi-session cache isolation exist.
