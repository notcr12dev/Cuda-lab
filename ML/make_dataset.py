"""Generate synthetic regression datasets: y = x1^2 + x2^2 + noise."""
import numpy as np

def make(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2)).astype(np.float64)
    y = X[:, 0] ** 2 + X[:, 1] ** 2 + rng.normal(0, 0.05, size=n)
    return X, y

def save(path, X, y):
    data = np.column_stack([X, y])
    np.savetxt(path, data, delimiter=",", header="x1,x2,y", comments="", fmt="%.6f")
    print(f"Wrote {path} ({len(y)} rows)")

if __name__ == "__main__":
    Xtr, ytr = make(2000, 0)
    Xte, yte = make(500, 1)
    save("data_train.csv", Xtr, ytr)
    save("data_test.csv", Xte, yte)
