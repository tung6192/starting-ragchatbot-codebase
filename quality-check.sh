#!/bin/bash

# Run code quality checks

set -e

echo "=== Code Quality Checks ==="
echo ""

echo "Checking code formatting with black..."
uv run --extra dev black --check backend/ main.py

echo ""
echo "=== All quality checks passed! ==="
