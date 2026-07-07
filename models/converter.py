#!/usr/bin/env python3
"""mollm model converter — auto-detect model type from config.json and dispatch.

Usage:
    python3 models/converter.py <model_dir> <output.mollm> [num_layers] [prefill_seq_len] [quant]

The converter reads <model_dir>/config.json to determine the model type:
    - model_type "qwen3_5"  → Qwen3.5 converter (qwen35.py)
    - model_type "qwen3_5_moe" → Qwen3.5-MoE text converter (qwen35_moe.py)
    - model_type "youtu"    → Youtu-LLM MLA converter (mla.py)
    Others → error with supported types list.
"""

import sys
import os
import json
from pathlib import Path


# Supported model types: model_type → (converter_module, converter_func, default_num_layers)
SUPPORTED_MODELS = {
    "qwen3_5":     ("qwen35",     "convert_qwen35",     24),
    "qwen3_5_moe": ("qwen35_moe", "convert_qwen35_moe", 1),
    "youtu":       ("mla",        "convert_mla",        32),
}


def detect_model_type(model_dir: str) -> str:
    """Read config.json and return model_type."""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    with open(config_path) as f:
        cfg = json.load(f)
    model_type = cfg.get("model_type", "")
    return model_type


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_dir> <output.mollm> [num_layers] [prefill_seq_len] [quant]")
        print("Quant modes: none, w8pc, w8gN, w4gN, w4mixgN")
        print()
        print("Supported model types:")
        for mt, (mod, func, default_layers) in SUPPORTED_MODELS.items():
            print(f"  {mt:20s}  (module: {mod}.py, default layers: {default_layers})")
        sys.exit(1)

    model_dir = sys.argv[1]
    output_path = sys.argv[2]
    num_layers_arg = int(sys.argv[3]) if len(sys.argv) > 3 else None
    prefill_seq_len = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    quant = sys.argv[5] if len(sys.argv) > 5 else "none"

    # Detect model type
    model_type = detect_model_type(model_dir)
    if model_type not in SUPPORTED_MODELS:
        supported = ", ".join(SUPPORTED_MODELS.keys())
        print(f"Error: unsupported model_type '{model_type}' in {model_dir}/config.json")
        print(f"Supported types: {supported}")
        sys.exit(1)

    mod_name, func_name, default_layers = SUPPORTED_MODELS[model_type]
    num_layers = num_layers_arg if num_layers_arg is not None else default_layers

    print(f"Detected model_type='{model_type}' → {mod_name}.convert_{mod_name}()")
    print(f"  model_dir:       {model_dir}")
    print(f"  output:          {output_path}")
    print(f"  num_layers:      {num_layers}")
    print(f"  prefill_seq_len: {prefill_seq_len}")
    print(f"  quant:           {quant}")
    print()

    # Import and dispatch
    module = __import__(mod_name)
    convert_fn = getattr(module, func_name)
    convert_fn(model_dir, output_path, num_layers, prefill_seq_len, quant=quant)


if __name__ == "__main__":
    # Ensure models/ dir is in path for imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
