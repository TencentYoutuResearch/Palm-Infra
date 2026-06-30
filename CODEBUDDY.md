# mollm

Mobile-oriented LLM inference engine. ARM NEON FP16FML kernels.
Currently benchmarked on Apple Silicon; targeting mobile/edge ARM on roadmap.

## Context recovery

**Always read `docs/CURRENT_STATE.md` first** — it contains the current project
status, performance benchmarks, architecture summary, and next steps. This
allows quick context recovery after clearing conversation history.

## Key files

- `docs/CURRENT_STATE.md` — current status (read this first)
- `docs/OPTIMIZATION_LOG_QWEN35.md` — Qwen3.5 optimization journey (attempts 1-12)
- `docs/OPTIMIZATION_LOG.md` — Youtu-LLM optimization log
- `README.md` — project overview, build/run instructions, roadmap
- `docs/ARCHITECTURE.md` — architecture design

## Conventions

- 4B is the primary benchmark target (0.8B has measurement instability)
- Bench protocol: pp256 + tg64, warmup=3, 5 runs median, fan max
- Update `docs/OPTIMIZATION_LOG_QWEN35.md` after each optimization step
- Update `docs/CURRENT_STATE.md` when project status changes
