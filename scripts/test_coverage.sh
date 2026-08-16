#!/usr/bin/env bash
# test_coverage.sh runs the configured test suite and reports line and branch coverage.
# This script accepts no arguments.
set -e

cd "$(dirname "$0")/.."

uv run coverage run -m pytest
uv run coverage report
