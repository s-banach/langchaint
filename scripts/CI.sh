#!/usr/bin/env bash
# CI.sh: the check gate. pyrefly, ruff check, ruff format --check, and pytest must all
# pass with zero errors before a commit is Done. The formatter runs in --check mode, so
# a formatting difference fails the gate instead of being rewritten under the author.
# Uses `uv run` so each tool resolves from the locked dev
# dependency group rather than a hand-activated .venv, and takes no arguments so the
# configuration is committed, not assembled at the invocation.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== pyrefly check ==="
uv run pyrefly check

echo "=== ruff check ==="
uv run ruff check

echo "=== ruff format --check ==="
uv run ruff format --check

echo "=== pytest ==="
uv run pytest
