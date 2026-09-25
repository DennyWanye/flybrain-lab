#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
SHA=91bdd1e7dcf193f3e7ca5a8933497fcef63b7960
DIR=upstream/Drosophila_brain_model
mkdir -p upstream
if [[ ! -d "$DIR/.git" ]]; then
  git init "$DIR"
  git -C "$DIR" remote add origin https://github.com/philshiu/Drosophila_brain_model.git
fi
if [[ -n "$(git -C "$DIR" status --porcelain)" ]]; then
  echo "Upstream directory has local changes. Back them up; refusing checkout." >&2
  exit 1
fi
git -C "$DIR" fetch --depth 1 origin "$SHA"
git -C "$DIR" checkout --detach "$SHA"
test "$(git -C "$DIR" rev-parse HEAD)" = "$SHA"
printf '\nPinned upstream: %s\n' "$SHA"
find "$DIR" -maxdepth 2 -type f \( -name '*.parquet' -o -name '*completeness*.csv' -o -name 'Completeness*.csv' \) -printf '%p\n'
