import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# File paths (Windows에서 WSL 파일 접근)
# =========================
base = Path(r"\\wsl$\Ubuntu\home\mskim\abtem_project")

depths_A_path = base / "O_K_300kV_50mrad_depths_A.npy"
depths_nm_path = base / "O_K_300kV_50mrad_depths_nm.npy"
cum_path = base / "O_K_300kV_50mrad_cumulative_maps.npy"
diff_path = base / "O_K_300kV_50mrad_differential_maps.npy"

# =========================
# Load arrays
# =========================
depths_A = np.load(depths_A_path)
depths_nm = np.load(depths_nm_path)
cum = np.load(cum_path)
diff = np.load(diff_path)

# =========================
# Basic info
# =========================
print("depths_A:")
print(depths_A)
print()

print("depths_nm:")
print(depths_nm)
print()

print("cumulative_maps shape:", cum.shape)
print("differential_maps shape:", diff.shape)

# =========================
# Plot cumulative maps
# =========================
n = cum.shape[0]
ncols = min(5, n)
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
axes = np.atleast_1d(axes).ravel()

for i in range(n):
    ax = axes[i]
    im = ax.imshow(cum[i], origin="lower")
    ax.set_title(f"Cumulative: {depths_nm[i]:.2f} nm")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for j in range(n, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()

# =========================
# Plot differential maps
# =========================
n = diff.shape[0]
ncols = min(5, n)
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
axes = np.atleast_1d(axes).ravel()

for i in range(n):
    ax = axes[i]
    im = ax.imshow(diff[i], origin="lower")
    if i == 0:
        title = f"Differential: 0–{depths_nm[i]:.2f} nm"
    else:
        title = f"Differential: {depths_nm[i-1]:.2f}–{depths_nm[i]:.2f} nm"
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for j in range(n, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()