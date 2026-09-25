#!/usr/bin/env bash
# All paths passed to the contained command should be relative to this project root.
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT"/{data,runs,reports,cache,tmp,upstream}
ARGS=(--rm --init --gpus all --network host --shm-size 2g
      --user "$(id -u):$(id -g)" -e HOME=/work/tmp -e PYTHONUNBUFFERED=1
      -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
      -v "$ROOT:/work" -w /work)
for NAME in GLOO_SOCKET_IFNAME OMP_NUM_THREADS FLY_NODE_RANK FLY_NNODES \
            FLY_MASTER_ADDR FLY_MASTER_PORT; do
  if [[ -n "${!NAME:-}" ]]; then ARGS+=(-e "$NAME=${!NAME}"); fi
done
exec docker run "${ARGS[@]}" "${FLY_IMAGE:-flybrain-lab:0.1}" "$@"
