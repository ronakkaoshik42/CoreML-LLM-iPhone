#!/usr/bin/env python3
"""Summarize benchmark logs without modifying them."""

import argparse
import re
import statistics
from pathlib import Path


FIELDS = ("ttft_sec", "total_sec", "decode_sec", "tokps", "peak_gb")


def values(line: str) -> dict[str, str]:
    return dict(re.findall(r"(?:^|\s)([A-Za-z_]+)=([^\s]+)", line))


def parse(path: Path) -> list[dict[str, str]]:
    runs: list[dict[str, str]] = []
    current: dict[str, object] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("\t", 1)[-1]
        if "[BENCH_START]" in line:
            current = {**values(line), "results": []}
        elif current is not None and "[RESULT]" in line:
            results = current["results"]
            assert isinstance(results, list)
            results.append(line)
        elif current is not None and "[BENCH_DONE]" in line:
            current.update(values(line))
            results = current.pop("results")
            assert isinstance(results, list)
            if len(results) >= 2:
                current.update(values(results[1]))
                runs.append({key: str(value) for key, value in current.items()})
            current = None
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize completed wrapper-level benchmark results.")
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()
    runs = [run for path in args.logs for run in parse(path)]
    if not runs:
        parser.error("no completed runs with a second [RESULT] line found")

    print("Wrapper-level runs (second [RESULT] line):")
    for run in runs:
        metrics = " ".join(f"{field}={run.get(field, 'NA')}" for field in FIELDS)
        print(f"  tag={run.get('tag', 'legacy')} {metrics}")

    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for run in runs:
        key = tuple(run.get(field, "legacy")
                    for field in ("suite_tag", "model", "mode", "state"))
        groups.setdefault(key, []).append(run)

    print("Summary by suite/model/mode/state:")
    for key, group in groups.items():
        print("  " + " ".join(
            f"{field}={value}" for field, value in zip(
                ("suite_tag", "model", "mode", "state"), key)))
        for field in FIELDS:
            samples = [float(run[field]) for run in group
                       if run.get(field) not in (None, "NA")]
            if samples:
                print(f"    {field}: median={statistics.median(samples):.3f} "
                      f"range={min(samples):.3f}..{max(samples):.3f} n={len(samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
