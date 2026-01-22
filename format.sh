#!/bin/bash

# Format Python code using black

set -e

echo "Running black formatter..."

# Run black on the backend directory and main.py
uv run --extra dev black backend/ main.py

echo "Formatting complete!"
