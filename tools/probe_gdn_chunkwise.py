#!/usr/bin/env python3
"""Offline chunkwise algebra probe, NOT a CUDA backend or speed benchmark.

Derives coefficients from this repository's scalar gated-delta recurrence.
Block arithmetic reassociates FP32 operations and is not byte-exact with the
production warp/FMA reduction. No precision/quality policy is changed here.
Uses direct decay products, never ratios of cumulative products: zero decay
and underflow must not turn a mathematically finite update into 0/0.
"""
import argparse
import json
import numpy as np


def validate(q, k, v, decay, beta, state):
    if q.dtype not in (np.dtype('float32'), np.dtype('float64')) or any(
            x.dtype != q.dtype for x in (k, v, decay, beta, state)):
        raise ValueError("matching FP32/FP64 arrays required")
    if (q.ndim != 2 or k.shape != q.shape or v.ndim != 2 or
            len(v) != len(q) or decay.shape != (len(q),) or
            beta.shape != decay.shape or state.shape != (q.shape[1], v.shape[1]) or
            not q.shape[1] or not v.shape[1]):
        raise ValueError("invalid recurrence dimensions")
    if not all(np.isfinite(x).all() for x in (q, k, v, decay, beta, state)):
        raise ValueError("nonfinite recurrence input")
    if np.any(decay < 0) or np.any(decay > 1) or np.any(beta < 0) or np.any(beta > 1):
        raise ValueError("decay/beta outside [0,1]")


def sequential(q, k, v, decay, beta, state, scale=.125):
    validate(q, k, v, decay, beta, state)
    state = state.copy()
    output = np.empty_like(v)
    for t in range(len(q)):
        state *= decay[t]
        delta = (v[t] - k[t] @ state) * beta[t]
        # Same mathematical readout as the CUDA optimized recurrence, not
        # a claim to reproduce its per-lane FMA or shuffle reduction order.
        output[t] = (q[t] @ state + delta * (q[t] @ k[t])) * scale
        state += k[t, :, None] * delta[None, :]
    return output, state


def coefficients(q, k, v, decay, beta):
    """Chunk-local matrices; all are independent of the incoming state."""
    count = len(q)
    if count < 1 or count > 128:
        raise ValueError("coefficient block must have 1..128 tokens")
    dtype = q.dtype
    ratios = np.zeros((count, count), dtype=dtype)
    for t in range(count):
        product = dtype.type(1)
        for j in range(t, -1, -1):
            ratios[t, j] = product
            product *= decay[j]
    prefix = np.cumprod(decay, dtype=dtype)
    lower = np.tril(beta[:, None] * ratios * (k @ k.T), -1)
    inverse = np.eye(count, dtype=dtype)
    for t in range(1, count):
        inverse[t, :t] = -(lower[t, :t] @ inverse[:t, :t])
    # H = U - W @ S_in; S_out = p_end*S_in + K_tail.T @ H.
    u = inverse @ (beta[:, None] * v)
    w = inverse @ ((beta * prefix)[:, None] * k)
    attention = np.tril(ratios * (q @ k.T))
    return u, w, prefix[:, None] * q, attention, ratios[-1, :, None] * k, prefix[-1]


def chunkwise(q, k, v, decay, beta, state, chunk=32, scale=.125):
    validate(q, k, v, decay, beta, state)
    if type(chunk) is not int or chunk < 1 or chunk > 128:
        raise ValueError("chunk must be an integer in 1..128")
    state = state.copy()
    output = np.empty_like(v)
    for first in range(0, len(q), chunk):
        end = min(first + chunk, len(q))
        u, w, qp, attention, tail, decay_end = coefficients(
            q[first:end], k[first:end], v[first:end], decay[first:end], beta[first:end])
        delta = u - w @ state
        output[first:end] = (qp @ state + attention @ delta) * scale
        state = decay_end * state + tail.T @ delta
    return output, state


def fixture(sequence, key_dim, value_dim, variant, dtype):
    rng = np.random.default_rng(4701 + variant + sequence)
    q = rng.normal(size=(sequence, key_dim)).astype(dtype)
    k = rng.normal(size=q.shape).astype(dtype)
    for x in (q, k):
        x /= np.sqrt((x*x).sum(axis=1, keepdims=True) + dtype(1e-6))
    v = (rng.normal(size=(sequence, value_dim)) * .25).astype(dtype)
    state = (rng.normal(size=(key_dim, value_dim)) * .03125).astype(dtype)
    decay = rng.uniform(.9, 1, size=sequence).astype(dtype)
    beta = rng.uniform(0, 1, size=sequence).astype(dtype)
    if variant == 1:
        decay[::17] = 0
        decay[1::19] = 1
        beta[::11] = 0
        beta[1::13] = 1
    elif variant == 2:
        decay.fill(dtype(1e-4))  # Prefix underflows inside ordinary blocks.
    elif variant == 3:
        decay.fill(1)
        k[:] = k[0]  # Highly correlated keys, not just random near-orthogonal keys.
        q[:] = q[0]
        beta.fill(1)
    return q, k, v, decay, beta, state


def probe(sequence=1743, chunk=32):
    records = []
    for dtype in (np.float64, np.float32):
        for variant in range(4):
            args = fixture(sequence, 128, 128, variant, dtype)
            expected, state = sequential(*args)
            actual, final = chunkwise(*args, chunk=chunk)
            out_error = float(np.max(np.abs(expected-actual)))
            state_error = float(np.max(np.abs(state-final)))
            if not np.isfinite(actual).all() or not np.isfinite(final).all():
                raise RuntimeError("nonfinite block result")
            tolerance = 1e-11 if dtype == np.float64 else 1e-5
            if max(out_error, state_error) > tolerance:
                raise RuntimeError(f"algebra probe error {out_error}/{state_error}")
            records.append(dict(dtype=np.dtype(dtype).name, variant=variant,
                                output_max_abs=out_error, state_max_abs=state_error,
                                output_byteexact=expected.tobytes() == actual.tobytes(),
                                state_byteexact=state.tobytes() == final.tobytes()))
    return dict(scope="synthetic_CPU_algebra_not_CUDA_model_quality_or_TPS",
                sequence=sequence, chunk=chunk, records=records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', type=int, default=1743)
    parser.add_argument('--chunk', type=int, default=32)
    args = parser.parse_args()
    if args.sequence < 1 or args.sequence > 2048 or args.chunk not in (16, 32, 64, 128):
        parser.error("sequence1..2048; chunk16/32/64/128 only")
    print(json.dumps(probe(args.sequence, args.chunk), indent=2))
