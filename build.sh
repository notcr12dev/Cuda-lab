#!/usr/bin/env bash
set -euo pipefail

# Ir a la carpeta del script
cd "$(dirname "$0")"

# Verificar nvcc
if ! command -v nvcc >/dev/null 2>&1; then
  echo "Error: nvcc no encontrado. Instala CUDA Toolkit." >&2
  exit 1
fi

mkdir -p build

echo "Compilando main.cu..."
nvcc -O2 -o build/main main.cu

echo "OK: ./build/main"
