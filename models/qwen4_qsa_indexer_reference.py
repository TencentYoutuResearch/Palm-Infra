"""Small NumPy QSA computation/cache oracle; not a runtime backend.

Implements the single-sequence, single indexer KV-head case used by the
released Flash Next config. RoPE width is supplied explicitly: QSA shares
the MAIN attention position embeddings, not indexer_dim * rotary_fraction.
Raw projected keys must survive chunk boundaries. Pooling is per query's
visible tokens, before centered RMSNorm and group-start RoPE.
Algorithm reference (Apache-2.0): Hugging Face Transformers,
models/qwen4_exp/modeling_qwen4_exp.py, Qwen4ExpTextQSAIndexer.
"""
import math
import numpy as np

from qwen4_qsa_reference import select_qsa_tokens


def centered_rms_norm(x, weight, eps):
    value = x.astype(np.float32)
    normalized = value / np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + eps)
    return (normalized * (1 + weight.astype(np.float32))).astype(x.dtype)


def partial_rope(x, positions, rope_dim, theta):
    """Half-split rotation with cos/sin rounded to the activation dtype."""
    frequency = 1 / np.power(np.float32(theta),
                            np.arange(0, rope_dim, 2, dtype=np.float32) / rope_dim)
    angles = np.asarray(positions, dtype=np.float32)[:, None] * frequency
    cosine = np.concatenate((np.cos(angles), np.cos(angles)), axis=-1).astype(x.dtype)
    sine = np.concatenate((np.sin(angles), np.sin(angles)), axis=-1).astype(x.dtype)
    while cosine.ndim < x.ndim:
        cosine, sine = cosine[:, None], sine[:, None]
    prefix = x[..., :rope_dim]
    rotated = np.concatenate((-prefix[..., rope_dim // 2:], prefix[..., :rope_dim // 2]), axis=-1)
    return np.concatenate((prefix * cosine + rotated * sine, x[..., rope_dim:]), axis=-1)


class QsaIndexerReference:
    def __init__(self, projection, q_norm, k_norm, *, heads=4, budget=2048,
                 compress_ratio=4, rope_dim=64, theta=10000000., eps=1e-6):
        projection, q_norm, k_norm = map(np.asarray, (projection, q_norm, k_norm))
        dim = q_norm.size
        select_qsa_tokens([], [], budget, compress_ratio)
        if (heads < 1 or q_norm.ndim != 1 or k_norm.shape != q_norm.shape
                or projection.ndim != 2 or projection.shape[0] != (heads + 1) * dim
                or projection.shape[1] < 1 or not 0 < rope_dim <= dim or rope_dim % 2
                or not math.isfinite(theta) or theta <= 0 or not math.isfinite(eps) or eps <= 0):
            raise ValueError("invalid QSA indexer geometry or normalization")
        if not all(np.isfinite(value).all() for value in (projection, q_norm, k_norm)):
            raise ValueError("nonfinite QSA weights")
        self.projection, self.q_norm, self.k_norm = [value.copy() for value in (projection, q_norm, k_norm)]
        self.heads, self.dim, self.budget, self.ratio = heads, dim, budget, compress_ratio
        self.rope_dim, self.theta, self.eps = rope_dim, theta, eps
        self.reset()

    def reset(self):
        self.raw_keys = None
        self.positions = np.empty(0, dtype=np.int64)

    def append(self, hidden, positions=None, visible=None):
        """Append [tokens, hidden] and return selected cache IDs per query.

        Optional boolean visibility is [new_tokens, total_cached_tokens];
        future-key visibility is rejected. Position IDs may differ from cache
        IDs, e.g. text following left padding. A failed call leaves state intact.
        """
        hidden = np.asarray(hidden)
        if (hidden.ndim != 2 or hidden.shape[1] != self.projection.shape[1]
                or hidden.dtype not in (np.dtype('float16'), np.dtype('float32'))
                or not np.isfinite(hidden).all()):
            raise ValueError("finite FP16/FP32 hidden rows required")
        past, rows = len(self.positions), len(hidden)
        if self.raw_keys is not None and self.raw_keys.dtype != hidden.dtype:
            raise ValueError("cache activation dtype cannot change")
        if positions is None:
            positions = np.arange(past, past + rows, dtype=np.int64)
        positions = np.asarray(positions)
        if (positions.shape != (rows,) or not np.issubdtype(positions.dtype, np.integer)
                or np.any(positions < 0)):
            raise ValueError("nonnegative integer position IDs required")
        causal = np.arange(past + rows)[None, :] <= np.arange(past, past + rows)[:, None]
        if visible is None:
            visible = causal
        else:
            visible = np.asarray(visible)
            if visible.shape != causal.shape or visible.dtype != np.bool_ or np.any(visible & ~causal):
                raise ValueError("boolean causal visibility required")
        # FP32 accumulate then cast back, also for pooling and centered norm.
        # Explicit per-row reduction keeps this small oracle independent of
        # BLAS GEMV/GEMM selection when chunk size changes. Not a fast kernel.
        projected = np.sum(hidden.astype(np.float32)[:, None, :] *
                           self.projection.astype(np.float32)[None, :, :],
                           axis=-1, dtype=np.float32).astype(hidden.dtype)
        query = projected[:, :self.heads * self.dim].reshape(rows, self.heads, self.dim)
        query = partial_rope(centered_rms_norm(query, self.q_norm, self.eps),
                             positions, self.rope_dim, self.theta)
        new_keys = projected[:, self.heads * self.dim:]
        raw_keys = new_keys if self.raw_keys is None else np.concatenate((self.raw_keys, new_keys))
        all_positions = np.concatenate((self.positions, positions))
        selections = []
        for row in range(rows):
            ids = np.flatnonzero(visible[row])
            blocks = len(ids) // self.ratio
            grouped = ids[:blocks * self.ratio].reshape(blocks, self.ratio)
            if blocks:
                pooled = raw_keys[grouped].astype(np.float32).mean(axis=1).astype(hidden.dtype)
                keys = partial_rope(centered_rms_norm(pooled, self.k_norm, self.eps),
                                    all_positions[grouped[:, 0]], self.rope_dim, self.theta)
                dots = query[row].astype(np.float32) @ keys.astype(np.float32).T
                scores = np.maximum(dots, 0).sum(axis=0) / math.sqrt(self.dim)
            else:
                scores = []
            selections.append(select_qsa_tokens(ids.tolist(), scores, self.budget, self.ratio))
        self.raw_keys, self.positions = raw_keys.copy(), all_positions.copy()
        return selections
