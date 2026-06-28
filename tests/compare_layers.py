"""Layer-by-layer comparison between HF transformers reference and C++ engine dumps.

Loads per-layer hidden states from:
  - /tmp/qwen35_full_ref/ (HF transformers, last real token)
  - /tmp/qwen35_cpp/ (C++ engine node dumps, d0 innermost)

Auto-detects node IDs from dump filenames.
"""
import os
import numpy as np
import re

REF_DIR = "/tmp/qwen35_full_ref"
CPP_DIR = "/tmp/qwen35_cpp"

def load_f32(path):
    return np.fromfile(path, dtype=np.float32)

def get_post_mlp_ids():
    """Auto-detect post-MLP ADD node IDs from C++ dump filenames."""
    nodes = []
    pat = re.compile(r'node_(\d+)_add_d(\d+)x(\d+)x\d+x\d+\.f32')
    for fname in sorted(os.listdir(CPP_DIR)):
        m = pat.match(fname)
        if m and int(m.group(3)) == 4:  # prefill seq_len=4
            nodes.append(int(m.group(1)))
    nodes.sort()
    if len(nodes) < 48:
        raise RuntimeError(f"Expected >= 48 ADD nodes, found {len(nodes)}")
    return [nodes[i] for i in range(1, len(nodes), 2)]  # every other = post-MLP

def load_cpp_add(node_id, seq_len):
    """Load a C++ ADD node dump for given seq_len."""
    pat = re.compile(rf"node_{node_id:04d}_add_d(\d+)x(\d+)x\d+x\d+\.f32")
    for fname in sorted(os.listdir(CPP_DIR)):
        m = pat.match(fname)
        if m and int(m.group(2)) == seq_len:
            d = load_f32(os.path.join(CPP_DIR, fname))
            d0, d1 = int(m.group(1)), int(m.group(2))
            return d.reshape(d1, d0)
    return None

def cos_sim(a, b):
    a, b = a.ravel(), b.ravel()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def main():
    post_mlp = get_post_mlp_ids()
    print(f"Detected {len(post_mlp)} post-MLP nodes")

    print("\n=== Prefill (C++ vs HF) ===")
    for i in range(min(24, len(post_mlp))):
        hf_path = f"{REF_DIR}/layer_{i:02d}_hidden.f32"
        if not os.path.exists(hf_path):
            print(f"  L{i:2d}: no HF reference")
            continue
        hf = load_f32(hf_path)
        cpp = load_cpp_add(post_mlp[i], 4)
        if cpp is None:
            print(f"  L{i:2d} n{post_mlp[i]}: no C++ dump")
            continue
        cos = cos_sim(hf, cpp[-1, :])
        flag = "✓" if cos > 0.999 else f"✗ {cos:.6f}"
        print(f"  L{i:2d} n{post_mlp[i]}: cos={cos:.6f} {flag}")

    print("\n=== Decode (C++ vs HF) ===")
    dec_dir = "/tmp/qwen35_decode_ref"
    for i in range(min(24, len(post_mlp))):
        hf_path = f"{dec_dir}/layer_{i:02d}_decode_hidden.f32"
        if not os.path.exists(hf_path):
            print(f"  L{i:2d}: no HF decode reference")
            continue
        hf = load_f32(hf_path)
        cpp = load_cpp_add(post_mlp[i], 1)
        if cpp is None:
            print(f"  L{i:2d}: no C++ dump")
            continue
        cos = cos_sim(hf, cpp[0, :])
        flag = "✓" if cos > 0.999 else f"✗ {cos:.6f}"
        print(f"  L{i:2d}: cos={cos:.6f} {flag}")

if __name__ == "__main__":
    main()
