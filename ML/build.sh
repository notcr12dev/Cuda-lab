#!/usr/bin/env bash
set -euo pipefail

# Build the tiny MLP (CUDA). Run from anywhere.
cd "$(dirname "$0")"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "Error: nvcc not found. Install CUDA Toolkit 12.8." >&2
  exit 1
fi

mkdir -p build

echo "Compiling mlp.cu..."
nvcc -O2 -o build/mlp mlp.cu

echo "OK: ./build/mlp"
