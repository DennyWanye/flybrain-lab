#!/usr/bin/env bash
# CPU/software validation inside the core container or an existing compatible CPU environment.
# This is not a full-data/GPU/four-physical-node validation.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-"runs/local-validation-$(date +%Y%m%d-%H%M%S)"}
mkdir -p "$OUT"
python -m compileall -q flylab.py scripts tests
for file in scripts/*.sh; do bash -n "$file"; done
python -m pytest -q tests | tee "$OUT/pytest.txt"
python flylab.py --threads 1 prepare --source synthetic --neurons 512 --out "$OUT/synthetic.npz"
python flylab.py --threads 1 gpu-check --device cpu | tee "$OUT/cpu-spmm.txt"
python flylab.py --threads 1 train --graph "$OUT/synthetic.npz" --device cpu \
  --envs 4 --steps 16 --minibatch 32 --epochs 2 --updates 3 --out "$OUT/train"
python flylab.py --threads 1 eval --graph "$OUT/synthetic.npz" --device cpu \
  --checkpoint "$OUT/train/policy.pt" --episodes 5 --out "$OUT/eval.json"
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  flylab.py --threads 1 dist-check | tee "$OUT/gloo-check.txt"
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  flylab.py --threads 1 train --graph "$OUT/synthetic.npz" --device cpu \
  --envs 2 --steps 16 --minibatch 16 --epochs 1 --updates 2 --out "$OUT/distributed"
echo "Software-only validation completed: $OUT"
