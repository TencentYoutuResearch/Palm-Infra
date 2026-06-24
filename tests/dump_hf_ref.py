"""
Dump intermediate values from HF Youtu-LLM-2B model for debugging.
Compares against C++ mlllm engine values at each stage.
"""
import json, struct, math, sys
import numpy as np

MODEL_DIR = "/Users/molly/workspace-youtulm-ncnn/Youtu-LLM-2B"

def load_safetensors(path):
    with open(path, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
    tensors = {}
    with open(path, 'rb') as f:
        f.seek(8 + header_len)
        for name, meta in header.items():
            if name == '__metadata__':
                continue
            dtype_str = meta['dtype']
            shape = meta['shape']
            offsets = meta['data_offsets']
            f.seek(8 + header_len + offsets[0])
            data = f.read(offsets[1] - offsets[0])
            if dtype_str == 'F32':
                arr = np.frombuffer(data, dtype=np.float32).reshape(shape)
            elif dtype_str == 'F16':
                arr = np.frombuffer(data, dtype=np.float16).astype(np.float32).reshape(shape)
            elif dtype_str == 'BF16':
                as_u32 = np.frombuffer(data, dtype=np.uint16).astype(np.uint32) << 16
                arr = as_u32.view(np.float32).reshape(shape)
            else:
                raise ValueError(f"Unknown dtype: {dtype_str}")
            tensors[name] = arr
    return tensors

def rms_norm(x, weight, eps=1e-6):
    # x: [hidden_dim, seq_len]
    # weight: [hidden_dim]
    hidden_dim, seq_len = x.shape
    out = np.zeros_like(x)
    for s in range(seq_len):
        ss = np.mean(x[:, s] ** 2)
        rms = 1.0 / np.sqrt(ss + eps)
        out[:, s] = x[:, s] * rms * weight
    return out

def rope(x, cos, sin, rope_dim, interleave=True):
    # x: [D, N] or [D, N, H] (multi-head). cos/sin: [rope_dim/2, N]
    half = rope_dim // 2
    out = x.copy()
    N = x.shape[1]
    for n in range(N):
        if interleave:
            for i in range(half):
                x0 = x[2*i, n]
                x1 = x[2*i+1, n]
                c = cos[i, n]
                s = sin[i, n]
                out[2*i, n]   = x0 * c - x1 * s
                out[2*i+1, n] = x0 * s + x1 * c
        else:
            for i in range(half):
                x0 = x[i, n]
                x1 = x[i+half, n]
                c = cos[i, n]
                s = sin[i, n]
                out[i, n]       = x0 * c - x1 * s
                out[i+half, n]  = x0 * s + x1 * c
    return out

def generate_rope_cache(seq_len, start_pos, rope_dim, rope_theta):
    half = rope_dim // 2
    cos = np.zeros((half, seq_len), dtype=np.float32)
    sin = np.zeros((half, seq_len), dtype=np.float32)
    for n in range(seq_len):
        pos = start_pos + n
        for i in range(half):
            theta = 1.0 / (rope_theta ** (2.0 * i / rope_dim))
            angle = pos * theta
            cos[i, n] = math.cos(angle)
            sin[i, n] = math.sin(angle)
    return cos, sin

def sdpa(q, k, v, causal=True, scale=None, head_dim=None, v_head_dim=None):
    """
    q, k, v: tensors shaped [head_dim, seq_len, num_heads]
    Returns: [v_head_dim, seq_len, num_heads]
    """
    if head_dim is None:
        head_dim = q.shape[0]
    if v_head_dim is None:
        v_head_dim = v.shape[0]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    seq_len = q.shape[1]
    num_heads = q.shape[2]

    out = np.zeros((v_head_dim, seq_len, num_heads), dtype=np.float32)

    for h in range(num_heads):
        for s in range(seq_len):
            # QK^T
            scores = np.zeros(seq_len, dtype=np.float32)
            for j in range(seq_len):
                scores[j] = np.dot(q[:, s, h], k[:, j, h])
            scores *= scale

            # causal mask
            if causal:
                for j in range(s + 1, seq_len):
                    scores[j] = -float('inf')

            # softmax
            max_s = scores.max()
            scores = np.exp(scores - max_s)
            scores /= scores.sum()

            # attn * V
            for j in range(seq_len):
                out[:, s, h] += scores[j] * v[:, j, h]

    return out

def main():
    with open(f"{MODEL_DIR}/config.json") as f:
        cfg = json.load(f)

    weights = load_safetensors(f"{MODEL_DIR}/model.safetensors")

    hidden_size = cfg['hidden_size']
    num_heads = cfg['num_attention_heads']
    qk_nope_dim = cfg['qk_nope_head_dim']
    qk_rope_dim = cfg['qk_rope_head_dim']
    qk_head_dim = qk_nope_dim + qk_rope_dim
    v_head_dim = cfg['v_head_dim']
    q_lora_rank = cfg['q_lora_rank']
    kv_lora_rank = cfg['kv_lora_rank']
    eps = cfg.get('rms_norm_eps', 1e-6)
    scale = qk_head_dim ** -0.5
    rope_theta = cfg['rope_parameters']['rope_theta']

    # Tokenize "Hello, world!"
    # Use the mlllm tokenizer IDs we know: [9926, 11, 1922, 0]
    token_ids = [9926, 11, 1922, 0]
    seq_len = len(token_ids)

    # Embed
    embed_w = weights['model.embed_tokens.weight'].astype(np.float32)
    hidden = np.zeros((hidden_size, seq_len), dtype=np.float32)
    for s, tid in enumerate(token_ids):
        hidden[:, s] = embed_w[tid, :]

    print(f"=== HF EMBED ===")
    print(f"  hidden[0..2, 0] = {hidden[0,0]:.4f} {hidden[1,0]:.4f} {hidden[2,0]:.4f}")

    # Layer 0
    pfx = 'model.layers.0.self_attn'

    # Input RMSNorm
    w_ln = weights['model.layers.0.input_layernorm.weight'].astype(np.float32)
    x_normed = rms_norm(hidden, w_ln, eps)

    print(f"\n=== HF RMSNorm (layer 0 input) ===")
    print(f"  x_normed[0..2, 0] = {x_normed[0,0]:.4f} {x_normed[1,0]:.4f} {x_normed[2,0]:.4f}")

    # Q path
    # Also print matmul outputs for first layer
    w_q_a = weights[f'{pfx}.q_a_proj.weight'].astype(np.float32)  # [q_lora_rank, hidden_size]
    w_q_a_norm = weights[f'{pfx}.q_a_layernorm.weight'].astype(np.float32)
    w_q_b = weights[f'{pfx}.q_b_proj.weight'].astype(np.float32)  # [num_heads*qk_head_dim, q_lora_rank]

    q_comp = w_q_a @ x_normed  # [q_lora_rank, seq_len]
    print(f"  MATMUL q_a_proj out[0..2,0] = {q_comp[0,0]:.4f} {q_comp[1,0]:.4f} {q_comp[2,0]:.4f}")
    q_comp = rms_norm(q_comp, w_q_a_norm, eps)
    q = w_q_b @ q_comp  # [num_heads*qk_head_dim, seq_len]
    print(f"  MATMUL q_b_proj out[0..2,0] = {q[0,0]:.4f} {q[1,0]:.4f} {q[2,0]:.4f}")

    # Reshape to [qk_head_dim, num_heads, seq_len] -> [qk_head_dim, seq_len, num_heads]
    q = q.reshape(num_heads, qk_head_dim, seq_len).transpose(1, 2, 0)  # [qk_head_dim, seq_len, num_heads]
    q_nope = q[:qk_nope_dim, :, :]
    q_rope = q[qk_nope_dim:, :, :]

    print(f"\n=== HF Q PROJECTION ===")
    print(f"  q.shape = {q.shape}")
    print(f"  q_nope[0..2, 0, 0] = {q_nope[0,0,0]:.4f} {q_nope[1,0,0]:.4f} {q_nope[2,0,0]:.4f}")
    print(f"  q_rope[0..2, 0, 0] = {q_rope[0,0,0]:.4f} {q_rope[1,0,0]:.4f} {q_rope[2,0,0]:.4f}")

    # KV path
    w_kv_a = weights[f'{pfx}.kv_a_proj_with_mqa.weight'].astype(np.float32)
    w_kv_a_norm = weights[f'{pfx}.kv_a_layernorm.weight'].astype(np.float32)
    w_kv_b = weights[f'{pfx}.kv_b_proj.weight'].astype(np.float32)

    kv = w_kv_a @ x_normed  # [kv_lora_rank + qk_rope_dim, seq_len]
    kv_compressed = kv[:kv_lora_rank, :]
    k_rope_raw = kv[kv_lora_rank:, :]

    print(f"\n=== HF KV COMPRESSION ===")
    print(f"  kv.shape = {kv.shape}")
    print(f"  kv_compressed[0..2, 0] = {kv_compressed[0,0]:.4f} {kv_compressed[1,0]:.4f} {kv_compressed[2,0]:.4f}")
    print(f"  k_rope_raw[0..2, 0] = {k_rope_raw[0,0]:.4f} {k_rope_raw[1,0]:.4f} {k_rope_raw[2,0]:.4f}")

    kv_normed = rms_norm(kv_compressed, w_kv_a_norm, eps)
    kv_expanded = w_kv_b @ kv_normed  # [num_heads*(qk_nope_dim+v_head_dim), seq_len]
    kv_expanded = kv_expanded.reshape(num_heads, qk_nope_dim+v_head_dim, seq_len).transpose(1, 2, 0)
    k_nope = kv_expanded[:qk_nope_dim, :, :]
    v = kv_expanded[qk_nope_dim:, :, :]

    # Reshape k_rope_raw for RoPE
    k_rope_raw_3d = k_rope_raw.reshape(qk_rope_dim, seq_len, 1)

    print(f"\n=== HF K/V SPLIT ===")
    print(f"  k_nope[0..2, 0, 0] = {k_nope[0,0,0]:.4f} {k_nope[1,0,0]:.4f} {k_nope[2,0,0]:.4f}")
    print(f"  v[0..2, 0, 0] = {v[0,0,0]:.4f} {v[1,0,0]:.4f} {v[2,0,0]:.4f}")

    # RoPE cache
    cos, sin = generate_rope_cache(seq_len, 0, qk_rope_dim, rope_theta)

    print(f"\n=== HF RoPE CACHE ===")
    print(f"  cos[0..3, 0] = {cos[0,0]:.8f} {cos[1,0]:.8f} {cos[2,0]:.8f} {cos[3,0]:.8f}")
    print(f"  sin[0..3, 0] = {sin[0,0]:.8f} {sin[1,0]:.8f} {sin[2,0]:.8f} {sin[3,0]:.8f}")

    # Apply RoPE
    q_rope_rotated = rope(q_rope, cos, sin, qk_rope_dim, interleave=True)
    k_rope_rotated = rope(k_rope_raw_3d, cos, sin, qk_rope_dim, interleave=True)

    print(f"\n=== HF AFTER RoPE ===")
    print(f"  q_rope_rotated[0..2, 0, 0] = {q_rope_rotated[0,0,0]:.4f} {q_rope_rotated[1,0,0]:.4f} {q_rope_rotated[2,0,0]:.4f}")
    print(f"  k_rope_rotated[0..2, 0, 0] = {k_rope_rotated[0,0,0]:.4f} {k_rope_rotated[1,0,0]:.4f} {k_rope_rotated[2,0,0]:.4f}")

    # Concat nope + rope for Q and K
    q_full = np.concatenate([q_nope, q_rope_rotated], axis=0)  # [qk_head_dim, seq_len, num_heads]
    k_full = np.concatenate([k_nope, np.tile(k_rope_rotated, (1, 1, num_heads))], axis=0)

    print(f"\n=== HF Q/K FULL (after concat) ===")
    print(f"  q_full[0..2, 0, 0] = {q_full[0,0,0]:.4f} {q_full[1,0,0]:.4f} {q_full[2,0,0]:.4f}")
    print(f"  q_full[125..130, 0, 0] = {q_full[125,0,0]:.4f} {q_full[126,0,0]:.4f} {q_full[127,0,0]:.4f} {q_full[128,0,0]:.4f} {q_full[129,0,0]:.4f} {q_full[130,0,0]:.4f}")
    print(f"  k_full[0..2, 0, 0] = {k_full[0,0,0]:.4f} {k_full[1,0,0]:.4f} {k_full[2,0,0]:.4f}")
    print(f"  k_full[125..130, 0, 0] = {k_full[125,0,0]:.4f} {k_full[126,0,0]:.4f} {k_full[127,0,0]:.4f} {k_full[128,0,0]:.4f} {k_full[129,0,0]:.4f} {k_full[130,0,0]:.4f}")
    print(f"  q_full.shape = {q_full.shape}, k_full.shape = {k_full.shape}")

    # SDPA
    attn_out = sdpa(q_full, k_full, v, causal=True, scale=scale, head_dim=qk_head_dim, v_head_dim=v_head_dim)

    print(f"\n=== HF SDPA OUTPUT ===")
    print(f"  attn_out[0..2, 0, 0] = {attn_out[0,0,0]:.4f} {attn_out[1,0,0]:.4f} {attn_out[2,0,0]:.4f}")

    # QK scores for first head, first query
    scores_h0_s0 = np.zeros(seq_len, dtype=np.float32)
    for j in range(seq_len):
        scores_h0_s0[j] = np.dot(q_full[:, 0, 0], k_full[:, j, 0])
    scores_h0_s0 *= scale
    print(f"\n=== HF QK scores (head 0, query 0) ===")
    print(f"  raw scores[0..3] = {scores_h0_s0[0]:.4f} {scores_h0_s0[1]:.4f} {scores_h0_s0[2]:.4f} {scores_h0_s0[3]:.4f}")

    # After softmax
    scores_h0_s0_causal = scores_h0_s0.copy()
    for j in range(1, seq_len):
        scores_h0_s0_causal[j] = -float('inf')
    max_s = scores_h0_s0_causal.max()
    scores_h0_s0_causal = np.exp(scores_h0_s0_causal - max_s)
    scores_h0_s0_causal /= scores_h0_s0_causal.sum()
    print(f"  after causal+softmax[0..3] = {scores_h0_s0_causal[0]:.4f} {scores_h0_s0_causal[1]:.4f} {scores_h0_s0_causal[2]:.4f} {scores_h0_s0_causal[3]:.4f}")

    # Output projection
    w_o = weights[f'{pfx}.o_proj.weight'].astype(np.float32)  # [hidden_size, num_heads*v_head_dim]
    attn_out_flat = attn_out.transpose(2, 0, 1).reshape(num_heads * v_head_dim, seq_len)  # [num_heads*v_head_dim, seq_len]
    out_proj = w_o @ attn_out_flat  # [hidden_size, seq_len]

    # Residual
    out_residual = hidden + out_proj

    print(f"\n=== HF AFTER OUTPUT PROJ + RESIDUAL ===")
    print(f"  out_proj[0..2, 0] = {out_proj[0,0]:.4f} {out_proj[1,0]:.4f} {out_proj[2,0]:.4f}")
    print(f"  residual[0..2, 0] = {out_residual[0,0]:.4f} {out_residual[1,0]:.4f} {out_residual[2,0]:.4f}")


if __name__ == '__main__':
    main()
