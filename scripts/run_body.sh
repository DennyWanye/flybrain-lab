#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT"/{runs,reports,cache/body-home}
exec docker run --rm --init --gpus all --shm-size 2g \
  --user "$(id -u):$(id -g)" -e HOME=/work/cache/body-home \
  -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -v "$ROOT:/work" -w /work flybrain-body:2.1 "$@"
