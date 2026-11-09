#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
if [[ ! -f .venv/doish-ready-v1 ]]; then
  .venv/bin/python -m pip install -r requirements.txt
  touch .venv/doish-ready-v1
fi
exec .venv/bin/python launcher.py "$@"
