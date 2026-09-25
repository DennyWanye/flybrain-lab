#!/usr/bin/env bash
# Read-only inspection. No drivers, network configuration, firewall, or services are changed.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p reports
OUT="reports/preflight-$(hostname)-$(date +%Y%m%d-%H%M%S).txt"
{
  date -Is
  hostname
  uname -a
  cat /etc/os-release
  echo '--- memory/disk ---'
  free -h
  df -h "$PWD"
  echo '--- GPU ---'
  command -v nvidia-smi && nvidia-smi
  echo '--- Docker ---'
  command -v docker && docker version
  docker info --format '{{json .Runtimes}}'
  docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
  echo '--- network (inspection only) ---'
  ip -br address
  ip route
  echo '--- occupied listening ports ---'
  ss -lnt
} 2>&1 | tee "$OUT"
printf 'Report: %s\n' "$OUT"
