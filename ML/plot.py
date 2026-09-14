"""Load training losses + test predictions and plot them."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

losses = np.loadtxt("losses.csv", delimiter=",", skiprows=1)
preds = np.loadtxt("preds.csv", delimiter=",", skiprows=1)  # x1,x2,y_true,y_pred

rmse = float(np.sqrt(np.mean((preds[:, 2] - preds[:, 3]) ** 2)))
print(f"Test RMSE: {rmse:.4f}  (n={len(preds)})")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].plot(losses[:, 0], losses[:, 1])
ax[0].set_xlabel("epoch")
ax[0].set_ylabel("train MSE")
ax[0].set_title("Training loss")
ax[0].grid(True, alpha=0.3)

ax[1].scatter(preds[:, 2], preds[:, 3], s=8, alpha=0.5)
lo, hi = preds[:, 2:4].min(), preds[:, 2:4].max()
ax[1].plot([lo, hi], [lo, hi], "r--", label="ideal")
ax[1].set_xlabel("y_true")
ax[1].set_ylabel("y_pred")
ax[1].set_title(f"Test: true vs pred (RMSE={rmse:.4f})")
ax[1].legend()
ax[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("results.png", dpi=120)
print("Wrote results.png")
