"""Dump Qwen3.5 prefill + decode reference using HuggingFace transformers.

GROUND TRUTH for C++ engine comparison.
Uses full model forward (not manual layer-by-layer) to ensure
exact match with HF's internal computation path.
"""
import os, sys
import numpy as np
import torch

if len(sys.argv) < 2:
    raise SystemExit(f"usage: {sys.argv[0]} MODEL_DIR")
MODEL_DIR = sys.argv[1]
OUT_DIR = "/tmp/qwen35_full_ref"
DEC_OUT_DIR = "/tmp/qwen35_decode_ref"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DEC_OUT_DIR, exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading model (FP16)...")
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16, trust_remote_code=True)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

tc = model.config.get_text_config() if hasattr(model.config, 'get_text_config') else model.config.text_config
NUM_LAYERS = tc.num_hidden_layers
print(f"hidden={tc.hidden_size}, layers={NUM_LAYERS}, vocab={tc.vocab_size}")

# ---- Prefill ----
text = "Hello, world!"
input_ids = tokenizer.encode(text, return_tensors='pt')
n_tokens = input_ids.shape[1]
print(f"Prefill tokens: {input_ids[0].tolist()} (n={n_tokens})")

with torch.no_grad():
    out = model(input_ids, output_hidden_states=True)
    argmax = int(out.logits[0, -1].argmax().item())
    print(f"Prefill argmax: {argmax}")

    for layer in range(NUM_LAYERS):
        h = out.hidden_states[layer + 1][0, n_tokens - 1].float().numpy()
        h.tofile(f"{OUT_DIR}/layer_{layer:02d}_hidden.f32")

    final_h = out.hidden_states[NUM_LAYERS][0, n_tokens - 1].float().numpy()
    final_h.tofile(f"{OUT_DIR}/final_normed.f32")
    last_logits = out.logits[0, -1].float().numpy()
    last_logits.tofile(f"{OUT_DIR}/logits_last.f32")

    with open(f"{OUT_DIR}/argmax.txt", "w") as f:
        f.write(f"{argmax}\n")
    del out

print(f"Prefill dumps saved to {OUT_DIR}/")

# ---- Decode ----
decode_token = argmax
print(f"Decode with token {decode_token}...")
decode_ids = torch.cat([input_ids, torch.tensor([[decode_token]])], dim=1)

with torch.no_grad():
    out2 = model(decode_ids, output_hidden_states=True)
    dec_argmax = int(out2.logits[0, -1].argmax().item())
    print(f"Decode argmax: {dec_argmax}")

    for layer in range(NUM_LAYERS):
        h = out2.hidden_states[layer + 1][0, -1].float().numpy()
        h.tofile(f"{DEC_OUT_DIR}/layer_{layer:02d}_decode_hidden.f32")

    final_h2 = out2.hidden_states[NUM_LAYERS][0, -1].float().numpy()
    final_h2.tofile(f"{DEC_OUT_DIR}/decode_hidden.f32")
    dec_logits = out2.logits[0, -1].float().numpy()
    dec_logits.tofile(f"{DEC_OUT_DIR}/decode_logits.f32")

    with open(f"{DEC_OUT_DIR}/argmax.txt", "w") as f:
        f.write(f"{dec_argmax}\n")
    del out2

print(f"Decode dumps saved to {DEC_OUT_DIR}/")
