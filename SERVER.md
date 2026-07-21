# HTTP server

`mollm_server` is a small, dependency-free (beyond the bundled JSON parser)
OpenAI-compatible HTTP server. It is intended as a local serving baseline, not
as an internet-facing production proxy.

## Build and run

```bash
cmake -G Ninja -B build_i8mm -DCMAKE_BUILD_TYPE=Release
cmake --build build_i8mm -j

./build_i8mm/mollm_server \
  --package qwen35_4b_w4g128.mollm \
  --host 127.0.0.1 --port 8080 --threads 4
```

Metal builds accept `--device metal`. Metal package weights are resident.

## Endpoints

- `GET /v1/models`
- `POST /v1/chat/completions`

Both regular JSON and server-sent event streaming are supported:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mollm",
    "messages": [{"role": "user", "content": "Reply with OK only."}],
    "temperature": 0,
    "max_tokens": 32,
    "stream": true
  }'
```

The first version deliberately supports deterministic `temperature=0`
generation only. Authentication, TLS, tool calls, logprobs, parallel requests,
and continuous batching are not implemented. Bind to loopback unless a trusted
reverse proxy provides the missing production controls.

## Cache behavior

The server owns one engine and serializes requests. It retains the engine KV/GDN
state plus the exact token prefix from the previous request. If the next request's
rendered chat prompt extends that prefix, only the suffix is prefetched. If it
diverges, the engine resets before generation. This is the same correctness rule
used by the interactive multi-turn REPL.

This single-entry exact-prefix cache is intentionally conservative:

- token equality is required; text equality is not assumed;
- cache length is checked against `engine.past_len()` after every response;
- divergent histories and generation failures invalidate the cache;
- it does not copy KV tensors or retain multiple sessions.

A future multi-session cache should add explicit engine-state snapshot/restore,
byte accounting, capacity limits, and LRU eviction rather than sharing mutable
cache buffers across requests.
