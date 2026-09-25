#!/usr/bin/env python3
"""Aggregate evaluation JSON by mode, reporting seeds separately, not pooled pseudo-replication."""
import argparse
import glob
import json
import statistics
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("pattern", help="Quoted glob, e.g. 'runs/eval-*.json'")
a = p.parse_args()
groups = {}
for path in sorted(glob.glob(a.pattern)):
    item = json.loads(Path(path).read_text())
    key = item["mode"] if item["action_selection"] != "uniform_random" else "uniform_random"
    groups.setdefault(key, []).append((path, item["success_rate"], item["seed"]))
if not groups:
    raise SystemExit("No matching evaluation files.")
for key, rows in groups.items():
    values = [x[1] for x in rows]
    print(json.dumps({"mode": key, "runs": len(rows), "mean_success_rate": statistics.mean(values),
        "between_run_sample_std": statistics.stdev(values) if len(values)>1 else None,
        "files_and_scores": rows, "warning": "Check identical evaluation seeds and matched training budgets."}))
