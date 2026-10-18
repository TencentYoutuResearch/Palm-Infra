"""CPU selection oracle for Qwen4Exp QSA, not an inference implementation.

The official indexer ranks complete compress_ratio-token blocks by the sum
of rectified query-head/pooled-key dot products. This helper consumes those
scores; it does NOT replace projection, pooling/RMSNorm/RoPE or indexer caches.
The mask is a set of selected token IDs, so block ranking order is immaterial.
Equal scores use ascending block IDs for deterministic tests. PyTorch topk
does not specify that tie order: conformance tests must avoid cutoff ties or
compare admissible tied sets rather than asserting identical tied indices.
Reference: transformers/models/qwen4_exp/modeling_qwen4_exp.py,
Qwen4ExpTextQSAIndexer (Apache-2.0).
"""
import math


def select_qsa_tokens(visible, block_scores, budget=2048, compress_ratio=4):
    """Return chronological IDs from selected complete blocks and visible tail."""
    if (not isinstance(budget, int) or not isinstance(compress_ratio, int)
            or budget <= 0 or compress_ratio <= 0 or budget % compress_ratio):
        raise ValueError("budget must be a positive multiple of compress_ratio")
    visible = list(visible)
    if any(not isinstance(token, int) or token < 0 for token in visible):
        raise ValueError("visible token IDs must be nonnegative integers")
    if any(a >= b for a, b in zip(visible, visible[1:])):
        raise ValueError("visible token IDs must be unique and chronological")
    blocks = len(visible) // compress_ratio
    scores = list(block_scores)
    if len(scores) != blocks or any(not math.isfinite(x) for x in scores):
        raise ValueError("one finite score per complete visible block required")
    chosen = sorted(range(blocks), key=lambda i: (-scores[i], i))[:budget // compress_ratio]
    selected = [visible[block * compress_ratio + offset]
                for block in chosen for offset in range(compress_ratio)]
    selected.extend(visible[blocks * compress_ratio:])
    return sorted(selected)
