#!/usr/bin/env python3
"""Offline, container-isolated original-checkpoint reference with host RAM guard.

Never loads a .mollm package. Uses explicit token IDs, not a second chat template.
The container stage imports the separately installed vLLM implementation. The
host kills/removes only the exact container it creates, never other workloads.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def read_tokens(path):
    lines = Path(path).read_text().splitlines()
    if len(lines) != 2 or not lines[0].startswith("tokens=") or not lines[1].startswith("token_ids="):
        raise ValueError("expected mollm_ppl --tokenize-only output")
    count = int(lines[0][7:])
    tokens = [int(value) for value in lines[1][10:].split(",")]
    if count != len(tokens) or count < 2 or any(value < 0 for value in tokens):
        raise ValueError("invalid token count/IDs")
    return tokens


def loss_rows(tokens, prompt_logprobs):
    if prompt_logprobs is None or len(prompt_logprobs) != len(tokens):
        raise ValueError("reference did not return every prompt position")
    rows = []
    for position in range(1, len(tokens)):
        target = tokens[position]
        item = prompt_logprobs[position]
        if item is None or target not in item:
            raise ValueError(f"target missing at position {position}")
        ce = -float(item[target].logprob)
        if not math.isfinite(ce) or ce < 0:
            raise ValueError(f"invalid reference CE at position {position}")
        rows.append((position-1, target, ce))
    return rows


def available_kib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise ValueError("MemAvailable unavailable")


def stage(args):
    import importlib.metadata
    import hashlib
    from vllm import LLM, SamplingParams
    tokens = read_tokens(args.tokens)
    if len(tokens)+1 > args.n_ctx:
        raise ValueError("tokens exceed reference context")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "vllm")}
    print("reference_versions="+json.dumps(versions), flush=True)
    config_bytes = (Path(args.model)/"config.json").read_bytes()
    config = json.loads(config_bytes)
    print("reference_config_sha256="+hashlib.sha256(config_bytes).hexdigest(), flush=True)
    print("reference_architectures="+json.dumps(config.get("architectures")), flush=True)
    llm = LLM(model=args.model, dtype="bfloat16", trust_remote_code=False,
              skip_tokenizer_init=True, max_model_len=args.n_ctx,
              max_num_batched_tokens=args.batch, max_num_seqs=1,
              gpu_memory_utilization=0.70, kv_cache_memory_bytes=512*1024**2,
              enforce_eager=True, enable_prefix_caching=False,
              limit_mm_per_prompt={"image": 0, "video": 0},
              seed=42)
    lengths = sorted(set([min(args.prefix, len(tokens)), len(tokens)]))
    for length in lengths:
        if length < 2:
            raise ValueError("reference prefix must exceed one token")
        output = llm.generate([{"prompt_token_ids": tokens[:length]}],
                              SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1,
                                             detokenize=False), use_tqdm=False)
        if len(output) != 1 or not output[0].finished:
            raise ValueError("reference request did not finish")
        rows = loss_rows(tokens[:length], output[0].prompt_logprobs)
        path = Path(args.output)/f"reference-{length}.csv"
        with path.open("x", newline="") as destination:
            writer = csv.writer(destination)
            writer.writerow(["position", "target", "ce"])
            writer.writerows(rows)
        print("reference_result="+json.dumps(dict(tokens=length, scored=len(rows),
              mean_ce=math.fsum(row[2] for row in rows)/len(rows),
              generated_ids=output[0].outputs[0].token_ids, output=str(path))), flush=True)
    return 0


def host(args):
    model, tokens_path, output = (Path(value).resolve() for value in (args.model, args.tokens, args.output))
    if not (model/"model.safetensors.index.json").is_file():
        raise ValueError("local indexed original checkpoint required")
    tokens = read_tokens(tokens_path)
    if len(tokens)+1 > args.n_ctx or not 2 <= args.prefix <= args.n_ctx:
        raise ValueError("tokens/prefix exceed context")
    if output.exists():
        raise ValueError("output directory must be new; never overwrite reference evidence")
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise ValueError("GPU is busy")
    if available_kib() < 100*1024**2:
        raise ValueError("reference requires100GiB MemAvailable at startup")
    image = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", args.image], text=True).strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("unresolved image identity")
    output.mkdir()
    (output/"tmp").mkdir()
    environment = dict(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HOME="/output/hf",
                       HF_HUB_DISABLE_TELEMETRY="1", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1",
                       VLLM_PLE_MMAP="1", VLLM_PLE_MMAP_PREWARM="0", VLLM_PLE_MMAP_PREFETCH="0",
                       VLLM_PLE_MMAP_WORKERS="4", OMP_NUM_THREADS="8", TMPDIR="/output/tmp",
                       XDG_CACHE_HOME="/output/xdg", XDG_CONFIG_HOME="/output/xdg-config",
                       TORCHINDUCTOR_CACHE_DIR="/output/inductor",
                       TRITON_CACHE_DIR="/output/triton", CUDA_CACHE_PATH="/output/cuda",
                       VLLM_CACHE_ROOT="/output/vllm", FLASHINFER_WORKSPACE_BASE="/output/flashinfer")
    command = ["docker", "create", "--network", "none", "--read-only", "--gpus", "all",
               "--memory", "96g", "--memory-swap", "96g", "--pids-limit", "512", "--shm-size", "256m"]
    for source, target, readonly in ((model, "/model", True), (tokens_path, "/tokens.txt", True),
                                    (Path(__file__).resolve(), "/reference.py", True), (output, "/output", False)):
        # Docker's mount grammar cannot represent commas in source paths.
        if "," in str(source):
            raise ValueError("mount source path contains comma")
        command += ["--mount", f"type=bind,source={source},target={target}"+(",readonly" if readonly else "")]
    for name, value in environment.items():
        command += ["--env", f"{name}={value}"]
    command += ["--entrypoint", "python3", image, "/reference.py", "--container-stage",
                "--model", "/model", "--tokens", "/tokens.txt", "--output", "/output",
                "--n-ctx", str(args.n_ctx), "--batch", str(args.batch), "--prefix", str(args.prefix)]
    container = subprocess.check_output(command, text=True).strip()
    if not re.fullmatch(r"[0-9a-f]{64}", container):
        raise ValueError("invalid created container identity")
    with (output/"provenance.json").open("x") as destination:
        json.dump(dict(image=image, container=container, model=str(model), tokens=len(tokens),
                       n_ctx=args.n_ctx, batch=args.batch, environment=environment), destination, indent=2)
    print(f"reference_container={container} image={image}", flush=True)
    client = None
    stopped = False
    try:
        with (output/"model.log").open("x") as log, (output/"memory.log").open("x") as memory:
            client = subprocess.Popen(["docker", "start", "--attach", container], stdout=log, stderr=subprocess.STDOUT)
            while client.poll() is None:
                try:
                    free = available_kib()
                except (OSError, ValueError):
                    free = 0
                memory.write(f"available_kib={free}\n"); memory.flush()
                if free < 24*1024**2:
                    stopped = True
                    print("reference_memory_guard=stop", flush=True)
                    subprocess.run(["docker", "kill", container], check=False, timeout=15)
                    break
                time.sleep(.25)
            client.wait(timeout=30)
        code = int(subprocess.check_output(["docker", "inspect", "--format", "{{.State.ExitCode}}", container], text=True))
        print(f"reference_exit={code} memory_guard={int(stopped)}", flush=True)
        return 137 if stopped else code
    finally:
        # The id came only from our successful docker create. SIGKILL applies
        # only to this test container, including on interrupted client/guard.
        running = subprocess.check_output(["docker", "inspect", "--format", "{{.State.Running}}", container], text=True).strip()
        if running == "true":
            subprocess.run(["docker", "kill", container], check=False, timeout=15)
        if client is not None and client.poll() is None:
            client.wait(timeout=30)
        subprocess.run(["docker", "rm", container], check=True, timeout=15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container-stage", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--image", default="qwen38-flash-dgx:latest")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--prefix", type=int, default=256)
    args = parser.parse_args()
    if not 256 <= args.n_ctx <= 8192 or not 1 <= args.batch <= 2048:
        parser.error("bounded reference requires ctx256..8192, batch1..2048")
    try:
        return stage(args) if args.container_stage else host(args)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
