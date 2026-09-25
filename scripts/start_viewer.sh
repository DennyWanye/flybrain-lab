#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
PYTHONPATH="$PROJECT" exec python3 -m flyview serve --project-root "$PROJECT" --host 127.0.0.1 --port 8765
