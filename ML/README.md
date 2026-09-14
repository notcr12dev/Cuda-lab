# ML — Tiny Neural Network from Scratch

This folder (`ML/`) is a self-contained mini-container for a small neural
network built from zero: a **2 → 16 → 1 MLP** (ReLU hidden layer) that learns
the regression task `y = x1² + x2²` on synthetic data.

## Stack (who does what)

| Layer      | Files                        | Role                                              |
|------------|------------------------------|---------------------------------------------------|
| C++        | `mlp.cu` (host code)         | Network definition, host memory (`new`/`delete`), CSV I/O, training loop, SGD + backward pass on CPU |
| CUDA C++   | `mlp.cu` (kernels)           | Forward pass on the GPU (`fc_relu_forward`, `fc_forward`), device memory (`cudaMalloc` / `cudaMemcpy` / `cudaFree`) |
| Python     | `make_dataset.py`, `plot.py` | Dataset generation, loading results and plotting losses |

The split is intentionally simple: the GPU runs the math-heavy forward pass,
the CPU keeps the (tiny) backward pass readable, and Python handles data and
visualization where it shines.

## Files

- `mlp.cu` — the whole network + training in one file (~250 lines).
- `make_dataset.py` — generates `data_train.csv` / `data_test.csv` with numpy.
- `plot.py` — loads `losses.csv` + `preds.csv`, prints test RMSE, saves `results.png`.
- `build.sh` — compiles with `nvcc -O2 -o build/mlp mlp.cu`.
- `requirements.txt` — `numpy`, `matplotlib`.

Generated (git-ignored) artifacts: `data_*.csv`, `losses.csv`, `preds.csv`,
`results.png`, `build/`.

## Quickstart

```bash
cd ML
pip install -r requirements.txt

python make_dataset.py   # create data_train.csv + data_test.csv
sh build.sh              # compile -> ./build/mlp
./build/mlp data_train.csv data_test.csv 500 0.05
python plot.py           # prints RMSE, writes results.png
```

Expected output: training MSE drops from ~0.5 to ~0.01 and test RMSE lands
around ~0.1 (noise floor is 0.05).

## Model details

- Architecture: Linear(2→16) + ReLU + Linear(16→1), full-batch gradient descent.
- Init: He uniform (seed 42). Default hyperparams: 500 epochs, lr 0.05.
- Loss: mean squared error. CLI: `./build/mlp [train] [test] [epochs] [lr]`.
