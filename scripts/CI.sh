#!/usr/bin/env bash
# CI.sh fixes Ruff findings and runs every required check.
# Any fix or failing check makes this run fail.
# `uv run` uses the locked dev dependency group.
# This script accepts no arguments.
set -e

cd "$(dirname "$0")/.."

ci_exit_status=0

echo "=== uv lock --upgrade ==="
uv lock --upgrade || ci_exit_status=1

echo "=== ruff check --fix ==="
uv run ruff check --fix --exit-non-zero-on-fix || ci_exit_status=1

echo "=== ruff format ==="
uv run ruff format --exit-non-zero-on-format || ci_exit_status=1

echo "=== pyrefly check ==="
uv run pyrefly check || ci_exit_status=1

echo "=== pyrefly coverage check ==="
uv run pyrefly coverage check || ci_exit_status=1

echo "=== pytest ==="
uv run pytest || ci_exit_status=1

exit "$ci_exit_status"
