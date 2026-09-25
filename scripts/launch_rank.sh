#!/usr/bin/env bash
set -Eeuo pipefail
: "${FLY_NODE_RANK:?Set node rank 0,1,2,3}"
: "${FLY_NNODES:?Set number of participating nodes}"
: "${FLY_MASTER_ADDR:?Set rank0 private, routable IP}"
: "${FLY_MASTER_PORT:=29610}"
: "${GLOO_SOCKET_IFNAME:?Set local reachable private interface}"
exec torchrun --nnodes="$FLY_NNODES" --nproc-per-node=1 \
  --node-rank="$FLY_NODE_RANK" --master-addr="$FLY_MASTER_ADDR" \
  --master-port="$FLY_MASTER_PORT" --max-restarts=0 flylab.py "$@"
