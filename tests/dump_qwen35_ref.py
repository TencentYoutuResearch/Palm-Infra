"""Dump Qwen3.5 layer 0 (linear attention) intermediate tensors for comparison."""
import json
import numpy as np
import torch
import sys

model_dir = sys.argv[1] if len(sys.argv) > 1 else "/Users/molly/workspace-youtulm-ncnn/Qwen3.5-0.8B"

from transformers import AutoModelForCausalLM, AutoTokenizer
import safetensors

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Get text config
tc = model.config.get_text_config() if hasattr(model.config, 'get_text_config') else model.config.text_config
print(f"hidden_size={tc.hidden_size}, num_layers={tc.num_hidden_layers}")
print(f"layer_types[0:4]={tc.layer_types[0:4]}")

# Create a simple input: 4 tokens
input_ids = torch.tensor([[1, 2, 3, 4]])
print(f"input_ids: {input_ids}")

# Run model up to layer 0 and dump intermediates
with torch.no_grad():
    # Embed
    embed_weight = model.model.embed_tokens.weight
    hidden = embed_weight[input_ids]  # [1, 4, 1024]
    print(f"hidden shape: {hidden.shape}")

    # Layer 0 input_layernorm
    layer0 = model.model.layers[0]
    x_normed = layer0.input_layernorm(hidden)
    print(f"x_normed shape: {x_normed.shape}")

    # Linear attention
    linear_attn = layer0.linear_attn
    print(f"linear_attn type: {type(linear_attn)}")

    # Dump weights
    print(f"in_proj_qkv.weight: {linear_attn.in_proj_qkv.weight.shape}")
    print(f"in_proj_z.weight: {linear_attn.in_proj_z.weight.shape}")
    print(f"in_proj_a.weight: {linear_attn.in_proj_a.weight.shape}")
    print(f"in_proj_b.weight: {linear_attn.in_proj_b.weight.shape}")
    print(f"A_log: {linear_attn.A_log.shape} = {linear_attn.A_log.data}")
    print(f"dt_bias: {linear_attn.dt_bias.shape} = {linear_attn.dt_bias.data}")
    print(f"conv1d.weight: {linear_attn.conv1d.weight.shape}")
    print(f"norm.weight: {linear_attn.norm.weight.shape}")
    print(f"out_proj.weight: {linear_attn.out_proj.weight.shape}")

    # Run linear attention manually
    # 1. in_proj_qkv
    mixed_qkv = linear_attn.in_proj_qkv(x_normed)  # [1, 4, 6144]
    mixed_qkv = mixed_qkv.transpose(1, 2)  # [1, 6144, 4]
    print(f"\nmixed_qkv: {mixed_qkv.shape}")

    # 2. in_proj_a, in_proj_b
    a = linear_attn.in_proj_a(x_normed)  # [1, 4, 16]
    b = linear_attn.in_proj_b(x_normed)  # [1, 4, 16]
    print(f"a: {a.shape}, b: {b.shape}")

    # 3. in_proj_z
    z = linear_attn.in_proj_z(x_normed)  # [1, 4, 2048]
    z = z.reshape(1, 4, -1, linear_attn.head_v_dim)  # [1, 4, 16, 128]
    print(f"z: {z.shape}")

    # 4. Short conv (no cache, first run)
    # conv1d expects [batch, channels, seq]
    conv_out = torch.nn.functional.silu(
        linear_attn.conv1d(mixed_qkv)[:, :, :mixed_qkv.shape[-1]]
    )
    print(f"conv_out: {conv_out.shape}")

    # 5. Split qkv
    key_dim = linear_attn.key_dim  # 2048
    value_dim = linear_attn.value_dim  # 2048
    conv_out_t = conv_out.transpose(1, 2)  # [1, 4, 6144]
    query, key, value = torch.split(conv_out_t, [key_dim, key_dim, value_dim], dim=-1)
    print(f"query: {query.shape}, key: {key.shape}, value: {value.shape}")

    # 6. Reshape to heads
    batch_size, seq_len = 1, 4
    query = query.reshape(batch_size, seq_len, -1, linear_attn.head_k_dim)  # [1, 4, 16, 128]
    key = key.reshape(batch_size, seq_len, -1, linear_attn.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, linear_attn.head_v_dim)
    print(f"query reshaped: {query.shape}")

    # 7. beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias)
    beta = b.sigmoid()  # [1, 4, 16]
    g = -linear_attn.A_log.float().exp() * torch.nn.functional.softplus(a.float() + linear_attn.dt_bias)
    print(f"beta: {beta.shape}, g: {g.shape}")
    print(f"beta[0,0,:4]: {beta[0,0,:4]}")
    print(f"g[0,0,:4]: {g[0,0,:4]}")

    # 8. GDN recurrence (use torch_recurrent_gated_delta_rule)
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule
    core_attn_out, last_state = torch_recurrent_gated_delta_rule(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    print(f"core_attn_out: {core_attn_out.shape}")  # [1, 4, 16, 128]
    print(f"last_state: {last_state.shape}")  # [1, 16, 128, 128]

    # 9. Norm + gate
    core_attn_out_2d = core_attn_out.reshape(-1, linear_attn.head_v_dim)  # [4, 128]
    z_2d = z.reshape(-1, linear_attn.head_v_dim)  # [4, 128]
    normed_gated = linear_attn.norm(core_attn_out_2d, z_2d)  # [4, 128]
    print(f"normed_gated: {normed_gated.shape}")

    # 10. out_proj
    normed_gated_3d = normed_gated.reshape(batch_size, seq_len, -1)  # [1, 4, 2048]
    output = linear_attn.out_proj(normed_gated_3d)  # [1, 4, 1024]
    print(f"output: {output.shape}")

    # Save all intermediates
    np.savez("/tmp/qwen35_layer0_ref.npz",
             x_normed=x_normed.numpy(),
             mixed_qkv=mixed_qkv.numpy(),
             a=a.numpy(),
             b=b.numpy(),
             z=z.numpy(),
             conv_out=conv_out.numpy(),
             query=query.numpy(),
             key=key.numpy(),
             value=value.numpy(),
             beta=beta.numpy(),
             g=g.numpy(),
             core_attn_out=core_attn_out.numpy(),
             last_state=last_state.numpy(),
             normed_gated=normed_gated.numpy(),
             output=output.numpy(),
             A_log=linear_attn.A_log.data.numpy(),
             dt_bias=linear_attn.dt_bias.data.numpy(),
             norm_weight=linear_attn.norm.weight.data.numpy(),
             conv1d_weight=linear_attn.conv1d.weight.data.numpy(),
             in_proj_qkv_weight=linear_attn.in_proj_qkv.weight.data.numpy(),
             in_proj_z_weight=linear_attn.in_proj_z.weight.data.numpy(),
             in_proj_a_weight=linear_attn.in_proj_a.weight.data.numpy(),
             in_proj_b_weight=linear_attn.in_proj_b.weight.data.numpy(),
             out_proj_weight=linear_attn.out_proj.weight.data.numpy(),
             embed_weight=embed_weight.data.numpy(),
    )
    print("\nSaved to /tmp/qwen35_layer0_ref.npz")
    print(f"core_attn_out[0,0,0,:5]: {core_attn_out[0,0,0,:5]}")
    print(f"output[0,0,:5]: {output[0,0,:5]}")
