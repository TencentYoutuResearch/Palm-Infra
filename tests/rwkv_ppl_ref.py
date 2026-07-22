#!/usr/bin/env python3
"""Compute chunked RWKV-7 PPL with the rwkv-mobile PyTorch reference."""

import argparse
import math
import sys
import time
import types
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--token-ids", required=True,
                    help="mollm_ppl --tokenize-only output")
    ap.add_argument("--rwkv-mobile", required=True)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--chunk-size", type=int, default=256)
    args = ap.parse_args()

    converter = Path(args.rwkv_mobile) / "converter"
    sys.path.insert(0, str(converter))
    from rwkv_src.model_utils import get_dummy_state
    from rwkv_src.rwkv_modeling import RWKV_RNN

    line = next(x for x in Path(args.token_ids).read_text().splitlines()
                if x.startswith("token_ids="))
    ids = [int(x) for x in line.removeprefix("token_ids=").split(",")]
    ids = ids[:args.max_tokens]

    cfg = types.SimpleNamespace(
        MODEL_NAME=str(Path(args.checkpoint).with_suffix("")),
        USE_CUDA=False,
        fp16=False,
        wkv_customop=False,
        USE_CUSTOM_WKV=False,
        USE_EMBEDDING=True,
        RESCALE_LAYER=0,
        USE_ONNX_REDUCE_L2=True,
        USE_ONNX_L2NORM=False,
    )
    torch.set_grad_enabled(False)
    model = RWKV_RNN(cfg).eval()

    # The package folds blocks.0.ln0 into the embedding table and stores the
    # The mollm package stores matrix weights (including embedding and head)
    # as FP16 while retaining vectors, activations, and recurrent state FP32.
    for parameter in model.parameters():
        if parameter.ndim == 2:
            parameter.data = parameter.data.half().float()

    # rwkv-mobile truncates the final FFN to the last position when exporting
    # generation graphs. Disable only that export optimization so every input
    # position has logits for next-token CE.
    for block in model.blocks:
        block.ffn.num_layers = 0

    state = get_dummy_state(1, model.args, model.device)
    total_ce = 0.0
    n_targets = 0
    start = time.perf_counter()
    for offset in range(0, len(ids) - 1, args.chunk_size):
        chunk = ids[offset:offset + args.chunk_size]
        x = torch.tensor([chunk], dtype=torch.long, device=model.device)
        logits, state = model(x, state)
        # The mobile generation wrapper squeezes the sequence dimension from
        # token-shift states on multi-token calls; its next call normally has
        # seq=1. Restore the equivalent [1, 1, hidden] view for chunked PPL.
        for layer in range(model.args.n_layer):
            for index in (3 * layer, 3 * layer + 2):
                if state[index].ndim == 2:
                    state[index] = state[index].unsqueeze(1)
        usable = min(len(chunk), len(ids) - 1 - offset)
        targets = torch.tensor(ids[offset + 1:offset + 1 + usable],
                               dtype=torch.long, device=logits.device)
        ce = torch.nn.functional.cross_entropy(
            logits[0, :usable].float(), targets, reduction="sum")
        total_ce += float(ce)
        n_targets += usable
        running_ce = total_ce / n_targets
        print(f"chunk={offset // args.chunk_size + 1} offset={offset} "
              f"targets={n_targets} ce={running_ce:.6f} "
              f"ppl={math.exp(running_ce):.6f}", flush=True)

    mean_ce = total_ce / n_targets
    elapsed = time.perf_counter() - start
    print(f"tokens={len(ids)}")
    print(f"targets={n_targets}")
    print(f"ce={mean_ce:.6f}")
    print(f"ppl={math.exp(mean_ce):.6f}")
    print(f"elapsed_s={elapsed:.2f}")


if __name__ == "__main__":
    main()
