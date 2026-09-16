#!/usr/bin/env python
"""Summarize GRZO_PROFILE_OUT JSON files into a per-method table.

    python scripts/summarize_profile.py profiling/time profiling/memory

Each directory holds ``<method>_rank<k>.json`` files written by the trainer:
a list of ``{"name", "step", "t_ms"[, "peak_MB", "delta_MB"]}`` records, one
per instrumented stage per step. For every method this prints the mean
wall-clock per optimization step (stages summed within a step, averaged over
steps and ranks) and, when recorded, the mean per-step peak GPU memory.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict


def summarize_dir(directory):
    per_method = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(directory, "*_rank*.json"))):
        method = re.sub(r"_rank\d+\.json$", "", os.path.basename(path))
        by_step = defaultdict(list)
        for rec in json.load(open(path)):
            by_step[rec["step"]].append(rec)
        for recs in by_step.values():
            entry = {"t_ms": sum(r["t_ms"] for r in recs)}
            if all("peak_MB" in r for r in recs):
                entry["peak_MB"] = max(r["peak_MB"] for r in recs)
            per_method[method].append(entry)
    return per_method


def main():
    dirs = sys.argv[1:] or ["profiling/time", "profiling/memory"]
    for d in dirs:
        rows = summarize_dir(d)
        if not rows:
            print(f"{d}: no *_rank*.json files")
            continue
        print(f"\n## {d}\n")
        print("| method | samples | ms/step | peak GB |")
        print("|---|---|---|---|")
        for method, steps in sorted(rows.items()):
            n = len(steps)
            t = sum(s["t_ms"] for s in steps) / n
            peaks = [s["peak_MB"] for s in steps if "peak_MB" in s]
            peak = f"{sum(peaks) / len(peaks) / 1024:.2f}" if peaks else "-"
            print(f"| {method} | {n} | {t:.0f} | {peak} |")


if __name__ == "__main__":
    main()
