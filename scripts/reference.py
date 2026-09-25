#!/usr/bin/env python3
"""Run unmodified Shiu v630 model with sugar inputs and MN9 readout.

Notebook constants are parsed with ast.literal_eval, never notebook execution.
This wrapper is source-checked, not executed against full biological data in this delivery.
"""
import argparse
import ast
import copy
import csv
import json
import sys
from pathlib import Path


def notebook_constant(path: Path, name: str):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell["source"])
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                    return ast.literal_eval(node.value)
    raise KeyError(f"Notebook constant not found: {name}; inspect the pinned source.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="upstream/Drosophila_brain_model")
    p.add_argument("--out", default="runs/reference-smoke.csv")
    p.add_argument("--seconds", type=float, default=.1)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--rates", type=float, nargs="+", default=[0, 100, 150])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--codegen", choices=["numpy", "cython"], default="numpy")
    a = p.parse_args()
    if a.seconds <= 0 or a.trials < 1 or min(a.rates) < 0:
        raise ValueError("Invalid duration/trials/rates.")
    import brian2 as b2
    import pandas as pd
    repo = Path(a.repo).resolve()
    sys.path.insert(0, str(repo))
    import model
    comp = repo / "2023_03_23_completeness_630_final.csv"
    con = repo / "2023_03_23_connectivity_630_final.parquet"
    for path in [comp, con, repo / "example.ipynb"]:
        if not path.is_file():
            raise FileNotFoundError(path)
    sugar = notebook_constant(repo / "example.ipynb", "neu_sugar")
    mn9 = notebook_constant(repo / "example.ipynb", "id_mn9")
    ids = pd.read_csv(comp, dtype=str).iloc[:, 0].tolist()
    index = {int(value): i for i, value in enumerate(ids)}
    missing = [value for value in [*sugar, mn9] if value not in index]
    if missing:
        raise ValueError(f"v630 neuron IDs not in completeness table: {missing}")
    b2.prefs.codegen.target = a.codegen
    b2.defaultclock.dt = .1 * b2.ms
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input_hz", "trial", "seconds", "mn9_spikes", "mn9_hz"])
        writer.writeheader()
        for rate in a.rates:
            for trial in range(a.trials):
                b2.start_scope()
                b2.seed(a.seed + trial)  # paired seeds across stimulation rates
                params = copy.deepcopy(model.default_params)
                params["t_run"] = a.seconds * b2.second
                params["r_poi"] = rate * b2.Hz
                activity = model.run_trial([index[x] for x in sugar], [], [],
                                           str(comp), str(con), params)
                count = len(activity.get(index[mn9], []))
                result = {"input_hz": rate, "trial": trial, "seconds": a.seconds,
                          "mn9_spikes": count, "mn9_hz": count / a.seconds}
                writer.writerow(result); f.flush(); print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
