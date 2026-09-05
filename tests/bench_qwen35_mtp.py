#!/usr/bin/env python3
"""Optional official Qwen3.5-4B transactional-MTP performance gate."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


PACKAGE_ENV = "MOLLM_QWEN35_4B_MTP_PACKAGE"
MIN_SPEEDUP_ENV = "MOLLM_MTP_MIN_SPEEDUP"
DEFAULT_MIN_SPEEDUP = 1.05
RUNS_PER_ARM = 5


def parse_kv_output(output: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in output.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key:
            values[key] = value
    return values


def required_value(result: dict[str, str], key: str, arm: str, pair: int) -> str:
    if key not in result:
        raise AssertionError(
            f"pair {pair + 1} {arm} output is missing {key}")
    return result[key]


def integer_value(result: dict[str, str], key: str, arm: str, pair: int) -> int:
    value = required_value(result, key, arm, pair)
    try:
        return int(value)
    except ValueError as error:
        raise AssertionError(
            f"pair {pair + 1} {arm} has invalid {key}={value!r}") from error


def float_value(result: dict[str, str], key: str, arm: str, pair: int) -> float:
    value = required_value(result, key, arm, pair)
    try:
        number = float(value)
    except ValueError as error:
        raise AssertionError(
            f"pair {pair + 1} {arm} has invalid {key}={value!r}") from error
    if not math.isfinite(number):
        raise AssertionError(
            f"pair {pair + 1} {arm} has non-finite {key}={value!r}")
    return number


def run_arm(bench: Path, package: Path, draft_tokens: int) -> dict[str, str]:
    command = [
        str(bench),
        "--package", str(package),
        "--device", "cpu",
        "--prompt-tokens", "256",
        "--max-new-tokens", "65",
        "--warmup", "3",
        "--threads", "4",
        "--temperature", "0",
        "--output", "kv",
        "--dump-token-ids",
        "--mtp-draft-tokens", str(draft_tokens),
    ]
    completed = subprocess.run(
        command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        arm = "mtp" if draft_tokens else "plain"
        raise RuntimeError(
            f"{arm} benchmark exited {completed.returncode}\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}")
    return parse_kv_output(completed.stdout)


def validate_run(result: dict[str, str], arm: str, pair: int) -> None:
    expected = {
        "threads": 4,
        "prompt_tokens": 256,
        "generated_tokens": 65,
        "decode_tokens": 64,
    }
    for key, expected_value in expected.items():
        actual = integer_value(result, key, arm, pair)
        if actual != expected_value:
            raise AssertionError(
                f"pair {pair + 1} {arm} expected {key}={expected_value}, "
                f"got {actual}")
    if required_value(result, "hit_eos", arm, pair) != "false":
        raise AssertionError(f"pair {pair + 1} {arm} unexpectedly hit EOS")
    token_ids = required_value(result, "generated_token_ids", arm, pair)
    if not token_ids:
        raise AssertionError(
            f"pair {pair + 1} {arm} emitted no generated token IDs")

    float_value(result, "decode_tps", arm, pair)
    float_value(result, "tpot_ms", arm, pair)
    float_value(result, "peak_rss_mb", arm, pair)
    if arm == "mtp":
        for key in ("mtp_steps", "mtp_drafted", "mtp_verify_tokens"):
            if integer_value(result, key, arm, pair) <= 0:
                raise AssertionError(
                    f"pair {pair + 1} MTP expected nonzero {key}")
        integer_value(result, "mtp_accepted", arm, pair)
        for key in (
                "mtp_draft_ms", "mtp_draft_model_ms", "mtp_verify_ms",
                "mtp_sample_ms", "mtp_sync_ms"):
            float_value(result, key, arm, pair)


def package_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as package_file:
        while True:
            block = package_file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def minimum_speedup() -> float:
    raw = os.environ.get(MIN_SPEEDUP_ENV)
    if raw is None:
        return DEFAULT_MIN_SPEEDUP
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"invalid {MIN_SPEEDUP_ENV}={raw!r}") from error
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"invalid {MIN_SPEEDUP_ENV}={raw!r}")
    return value


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: bench_qwen35_mtp.py <mollm_bench>", file=sys.stderr)
        return 2

    package_raw = os.environ.get(PACKAGE_ENV)
    if not package_raw:
        print(f"SKIP: {PACKAGE_ENV} is not set")
        return 77
    package = Path(package_raw)
    if not package.is_file():
        print(f"SKIP: {PACKAGE_ENV} does not name an existing file: {package}")
        return 77
    bench = Path(sys.argv[1])
    if not bench.is_file():
        print(f"benchmark executable does not exist: {bench}", file=sys.stderr)
        return 2

    try:
        threshold = minimum_speedup()
        paired_results: list[dict[str, dict[str, str]]] = []
        execution_samples: list[tuple[int, str, float]] = []
        for pair in range(RUNS_PER_ARM):
            order = (("plain", 0), ("mtp", 1))
            if pair % 2:
                order = tuple(reversed(order))
            results: dict[str, dict[str, str]] = {}
            for arm, draft_tokens in order:
                result = run_arm(bench, package, draft_tokens)
                validate_run(result, arm, pair)
                results[arm] = result
                execution_samples.append((
                    pair + 1, arm,
                    float_value(result, "decode_tps", arm, pair)))

            plain_ids = required_value(results["plain"],
                                       "generated_token_ids", "plain", pair)
            mtp_ids = required_value(results["mtp"],
                                     "generated_token_ids", "mtp", pair)
            if plain_ids != mtp_ids:
                raise AssertionError(
                    f"pair {pair + 1} generated token IDs differ between "
                    "plain and MTP")
            paired_results.append(results)

        plain_results = [pair["plain"] for pair in paired_results]
        mtp_results = [pair["mtp"] for pair in paired_results]
        plain_decode = [float(result["decode_tps"])
                        for result in plain_results]
        mtp_decode = [float(result["decode_tps"])
                      for result in mtp_results]
        median_plain = statistics.median(plain_decode)
        median_mtp = statistics.median(mtp_decode)
        if median_plain <= 0.0:
            raise AssertionError("plain median decode_tps must be positive")
        speedup = median_mtp / median_plain

        total_drafted = sum(int(result["mtp_drafted"])
                            for result in mtp_results)
        total_accepted = sum(int(result["mtp_accepted"])
                             for result in mtp_results)
        acceptance = total_accepted / total_drafted
        peak_rss = max(
            float(result["peak_rss_mb"])
            for result in plain_results + mtp_results)

        print("decode_tps_samples:")
        for pair, arm, decode_tps in execution_samples:
            print(f"  pair={pair} arm={arm} decode_tps={decode_tps:.2f}")
        print(f"median_plain_decode_tps={median_plain:.2f}")
        print(f"median_mtp_decode_tps={median_mtp:.2f}")
        print(
            "median_plain_tpot_ms="
            f"{statistics.median(float(r['tpot_ms']) for r in plain_results):.2f}")
        print(
            "median_mtp_tpot_ms="
            f"{statistics.median(float(r['tpot_ms']) for r in mtp_results):.2f}")
        print(f"mtp_aggregate_acceptance={acceptance:.6f}")
        for key in (
                "mtp_draft_ms", "mtp_draft_model_ms", "mtp_verify_ms",
                "mtp_sample_ms", "mtp_sync_ms"):
            median_phase = statistics.median(
                float(result[key]) for result in mtp_results)
            print(f"median_{key}={median_phase:.2f}")
        print(f"peak_rss_mb={peak_rss:.1f}")
        print(f"package_sha256={package_sha256(package)}")
        print(f"required_speedup={threshold:.6f}")
        print(f"speedup={speedup:.6f}")

        if speedup < threshold:
            print(
                f"FAIL: median MTP speedup {speedup:.6f} is below "
                f"required {threshold:.6f}", file=sys.stderr)
            return 1
        return 0
    except (AssertionError, OSError, RuntimeError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
