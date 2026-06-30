"""Dump Youtu-LLM-2B per-layer hidden states for comparison with C++ engine."""
import torch, sys, os, json
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

model_path = '/Users/molly/workspace-youtulm-ncnn/Youtu-LLM-2B'
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Use the same text as PPL test
text = """The quick brown fox jumps over the lazy dog. This is a common pangram used in typography and font display. It contains every letter of the English alphabet at least once, making it useful for testing typefaces. However, there are many other pangrams in various languages that serve similar purposes. In computing, pangrams are often used to verify that a font displays all characters correctly. They can also be used to test keyboard layouts and text rendering engines. The most famous English pangram is probably the one about the quick brown fox, but there are alternatives like "Sphinx of black quartz, judge my vow" or "How vexingly quick daft zebras jump!" Each has its own charm and history. Beyond English, many languages have their own traditional pangrams. For example, in French, one might use a phrase about port wine. In German, there are sentences featuring every letter including the special characters like umlauts. In Chinese, the concept is different because each character represents a word or morpheme rather than a letter, so the idea of a pangram does not directly apply. Instead, one might use a passage that exercises a wide range of characters and stroke types. In Japanese, similar considerations apply, with the additional complexity of three writing systems: hiragana, katakana, and kanji. A well-designed text rendering system should handle all of these correctly, including proper line breaking, justification, and vertical text layout."""

ids = tokenizer.encode(text, add_special_tokens=True)[:256]
print(f'Tokens: {len(ids)}', file=sys.stderr)

outdir = '/tmp/youtu_ref'
os.makedirs(outdir, exist_ok=True)

# Hook to capture per-layer output
layer_outputs = {}
def make_hook(i):
    def hook(module, input, output):
        # output is a tuple (hidden_states, ...)
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        # h shape: [1, seq, hidden]
        # Save last token's hidden state
        last = h[0, -1, :].detach().float().numpy()
        layer_outputs[i] = last
    return hook

# Register hooks on each layer
for i, layer in enumerate(model.model.layers):
    layer.register_forward_hook(make_hook(i))

# Run forward pass
input_ids = torch.tensor([ids])
with torch.no_grad():
    out = model(input_ids)

# Save layer outputs
for i in sorted(layer_outputs.keys()):
    layer_outputs[i].tofile(f'{outdir}/layer_{i:02d}_hidden.f32')
    print(f'Layer {i}: shape={layer_outputs[i].shape}', file=sys.stderr)

# Save final logits (last token)
logits = out.logits[0, -1, :].detach().float().numpy()
logits.tofile(f'{outdir}/final_logits.f32')
print(f'Logits shape: {logits.shape}', file=sys.stderr)

# Also save input embeddings (after embed_tokens)
embed = model.model.embed_tokens(torch.tensor([ids]))
# Save first 4 tokens of embedding for short-prompt comparison
embed[:, :4, :].detach().float().numpy().tofile(f'{outdir}/embed_4tok.f32')

print(f'\nDumped {len(layer_outputs)} layers + logits to {outdir}/', file=sys.stderr)
print(f'Expected: layer_00_hidden.f32 ... layer_31_hidden.f32 + final_logits.f32')
