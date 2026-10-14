#!/usr/bin/env python3
"""Compare aligned teacher-forced CE rows; lower loss alone is not equivalence."""
import argparse
import csv
import json
import math


def read_rows(path):
    with open(path, newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["position", "target", "ce"]:
            raise ValueError(f"{path}: invalid token-loss columns")
        rows = []
        for row in reader:
            position, target, ce = int(row["position"]), int(row["target"]), float(row["ce"])
            if position != len(rows) or target < 0 or not math.isfinite(ce) or ce < 0:
                raise ValueError(f"{path}: invalid position/target/CE at row {len(rows)}")
            rows.append((position, target, ce))
        if not rows:
            raise ValueError(f"{path}: no scored tokens")
        return rows


def compare(control, candidate):
    a, b = read_rows(control), read_rows(candidate)
    if len(a) != len(b) or any(x[:2] != y[:2] for x, y in zip(a, b)):
        raise ValueError("token-loss positions/targets are not aligned")
    errors = [abs(x[2] - y[2]) for x, y in zip(a, b)]
    ordered = sorted(errors)
    worst = max(range(len(errors)), key=errors.__getitem__)
    count = len(a)
    ce_a = math.fsum(x[2] for x in a) / count
    ce_b = math.fsum(x[2] for x in b) / count
    return dict(tokens=count, exact_tokens=sum(x == 0 for x in errors),
                mean_abs=math.fsum(errors) / count, max_abs=errors[worst],
                worst_position=a[worst][0], p99_abs=ordered[min(count-1, int(.99*count))],
                control_ce=ce_a, candidate_ce=ce_b, ce_delta=ce_b-ce_a)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control")
    parser.add_argument("candidate")
    parser.add_argument("--max-mean-abs", type=float, default=1e-4)
    parser.add_argument("--max-abs", type=float, default=1e-3)
    args = parser.parse_args()
    if any(not math.isfinite(x) or x < 0 for x in (args.max_mean_abs, args.max_abs)):
        parser.error("tolerances must be finite and nonnegative")
    try:
        report = compare(args.control, args.candidate)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    report["equivalent"] = report["mean_abs"] <= args.max_mean_abs and report["max_abs"] <= args.max_abs
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["equivalent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
