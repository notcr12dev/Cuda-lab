# How to Contribute

Thank you for your interest in contributing to Cuda Lab Horizon!

## Requirements

- CUDA Toolkit 12.8 (`nvcc` in PATH)
- MSVC (on Windows) or GCC (on Linux)
- Make (optional)
- Git + Bash

Verify your environment:

```bash
nvcc --version
```

## Workflow

1. Fork the repository and clone it:

```bash
git clone <your-fork-url>
cd Cuda
```

2. Create a branch for your change:

```bash
git checkout -b feature/my-change
```

3. Build and test your change:

```bash
sh build.sh
./build/main
```

Expected output:

```
Error máximo: 0.000000
```

## Code Style

- CUDA/C++ code in `.cu` / `.cuh` files.
- Use `-O2` for compilation (this is what `build.sh` uses).
- Avoid `nvcc` warnings. If adding new flags, document the reason.
- Clear formatting and comments only where they add value (kernels, launches, H2D/D2H copies).

## Pull Requests

1. Commit with clear messages (e.g., `fix: correct index in saxpy kernel`).
2. Ensure `sh build.sh` compiles without errors or new warnings.
3. If changing behavior, explain which kernel is affected and how you tested it (GPU used, output obtained).
4. Open the PR against `master`/`main` describing:
   - The problem it solves
   - How to test it
   - GPU / driver / CUDA version used

## Reporting Issues

Include:

- Operating system and NVIDIA driver version
- Output of `nvcc --version`
- Command executed and full error log
- GPU (`nvidia-smi`)

## License

By contributing, you agree that your contribution be distributed under the MIT license found in the `LICENSE` file.