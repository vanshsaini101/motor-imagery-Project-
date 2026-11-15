
from scipy.io import loadmat
import numpy as np

mat = loadmat("data/A01T.mat")
data = mat["data"]

record = data[0, 3][0, 0]
X = record["X"]

print("Signal shape:", X.shape)
print("\nColumn-wise statistics:\n")

for col in range(X.shape[1]):
    signal = X[:, col]

    print(
        f"Column {col + 1:02d} | "
        f"Mean: {np.mean(signal):10.3f} | "
        f"Std: {np.std(signal):10.3f} | "
        f"Min: {np.min(signal):10.3f} | "
        f"Max: {np.max(signal):10.3f}"
    )

print("\nFirst 5 rows:")
print(X[:5, :])
