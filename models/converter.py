#!/usr/bin/env python3
"""mollm model converter — auto-detect model type from config.json and dispatch.

Usage:
    python3 models/converter.py <model_dir|checkpoint.pth> <output.mollm> [quant]

The converter reads <model_dir>/config.json to determine the model type:
    - model_type "qwen3"    → Qwen3 decoder-only converter (qwen3.py)
    - model_type "qwen3_moe" → Qwen3 MoE-compatible converter (qwen3_moe.py)
    - model_type "qwen3_5"  → Qwen3.5 converter (qwen35.py)
    - model_type "qwen3_5_moe" → Qwen3.5-MoE text converter (qwen35_moe.py)
    - model_type "youtu"    → Youtu-LLM MLA converter (mla.py)
    Others → error with supported types list.
"""

import sys
import os
import json
from pathlib import Path


# Supported model types: model_type → (converter_module, converter_func)
SUPPORTED_MODELS = {
    "qwen3":       ("qwen3",      "convert_qwen3"),
    "qwen3_moe":   ("qwen3_moe",  "convert_qwen3_moe"),
    "qwen3_5":     ("qwen35",     "convert_qwen35"),
    "qwen3_5_moe": ("qwen35_moe", "convert_qwen35_moe"),
    "youtu":       ("mla",        "convert_mla"),
}

DEFAULT_PREFILL_SEQ_LEN = 256


def detect_model_type(model_dir: str) -> str:
    """Read config.json and return model_type."""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    with open(config_path) as f:
        cfg = json.load(f)
    model_type = cfg.get("model_type", "")
    if model_type not in SUPPORTED_MODELS and _looks_like_qwen3_moe(cfg):
        return "qwen3_moe"
    return model_type


def _looks_like_qwen3_moe(cfg: dict) -> bool:
    """Detect Qwen3-style decoder MoE configs without naming private forks."""
    required = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "moe_intermediate_size",
        "num_experts_per_tok",
    )
    has_experts = "n_routed_experts" in cfg or "num_experts" in cfg
    return has_experts and all(k in cfg for k in required)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_dir> <output.mollm> [quant]")
        print("Quant modes: fp16, w8pc, w4g128, w4mixg128")
        print()
        print("Supported model types:")
        for mt, (mod, func) in SUPPORTED_MODELS.items():
            print(f"  {mt:20s}  (module: {mod}.py)")
        sys.exit(1)

    model_dir = sys.argv[1]
    output_path = sys.argv[2]

    # RWKV v7 is distributed primarily as an official .pth state_dict, not a
    # Hugging Face directory. It has a dedicated converter and no HF config.
    if model_dir.lower().endswith(".pth"):
        if len(sys.argv) > 3:
            print("RWKV v7 converter accepts no quant mode yet; use rwkv7.py --help for options")
            sys.exit(1)
        from rwkv7 import convert_rwkv7
        convert_rwkv7(model_dir, output_path)
        return
    prefill_seq_len = DEFAULT_PREFILL_SEQ_LEN
    quant = "fp16"

    extra = sys.argv[3:]
    if len(extra) == 1:
        quant = extra[0]
    elif len(extra) == 3 and extra[0].isdigit() and extra[1].isdigit():
        print("Warning: legacy positional layer/chunk/quant form is "
              "deprecated. Layer count is read from config.json; use the "
              "model-specific converter directly for truncated debug packages.")
        prefill_seq_len = int(extra[1])
        quant = extra[2]
    elif len(extra) != 0:
        print(f"Usage: {sys.argv[0]} <model_dir> <output.mollm> [quant]")
        print("Quant modes: fp16, w8pc, w4g128, w4mixg128")
        sys.exit(1)

    # Detect model type
    model_type = detect_model_type(model_dir)
    if model_type not in SUPPORTED_MODELS:
        supported = ", ".join(SUPPORTED_MODELS.keys())
        print(f"Error: unsupported model_type '{model_type}' in {model_dir}/config.json")
        print(f"Supported types: {supported}")
        sys.exit(1)

    mod_name, func_name = SUPPORTED_MODELS[model_type]

    print(f"Detected model_type='{model_type}' → {mod_name}.convert_{mod_name}()")
    print(f"  model_dir:       {model_dir}")
    print(f"  output:          {output_path}")
    print(f"  prefill_chunk:   {prefill_seq_len} (internal)")
    print(f"  quant:           {quant}")
    print()

    # Import and dispatch
    module = __import__(mod_name)
    convert_fn = getattr(module, func_name)
    convert_fn(model_dir, output_path, prefill_seq_len=prefill_seq_len, quant=quant)


if __name__ == "__main__":
    # Ensure models/ dir is in path for imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
