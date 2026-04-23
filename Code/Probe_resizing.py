import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter
from ase.io import read
import abtem


def fmt_num(x, ndigits=2):
    return f"{x:.{ndigits}f}".replace(".", "p")


def gaussian_blur_stack_fwhm(stack, fwhm_A, pixel_size_x_A, pixel_size_y_A):
    """
    stack: (n, ny, nx) or (ny, nx)
    fwhm_A: Gaussian FWHM in Å
    pixel_size_x_A, pixel_size_y_A: image pixel size in Å/pixel
    """
    if fwhm_A <= 0:
        return np.array(stack, copy=True)

    sigma_A = fwhm_A / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_x_pix = sigma_A / pixel_size_x_A
    sigma_y_pix = sigma_A / pixel_size_y_A

    arr = np.asarray(stack, dtype=float)

    if arr.ndim == 2:
        return gaussian_filter(arr, sigma=(sigma_y_pix, sigma_x_pix), mode="nearest")
    elif arr.ndim == 3:
        out = np.empty_like(arr)
        for i in range(arr.shape[0]):
            out[i] = gaussian_filter(arr[i], sigma=(sigma_y_pix, sigma_x_pix), mode="nearest")
        return out
    else:
        raise ValueError("stack must be 2D or 3D")


# =========================================================
# USER INPUT
# =========================================================
# 이미 계산 끝난 npy 파일
npy_file = Path("./Total28_Frozenphonon10_0.08_200kV_20mrad/Total28_STO12BMO4STO12_ADF_200kV_conv20mrad_det60to200mrad_thick10p956nm_dfstep1p50nm_rep2x2_fp10_sig0p08A_samp0p030A_devgpu_focusSeries.npy")

# 원래 시뮬레이션에 사용한 CIF
cif_file = Path("./Total28_STO12BMO4STO12.cif")

# 원래 시뮬레이션 조건과 동일하게 입력
repeat_x = 2
repeat_y = 2

# scan_center_tile_only 를 원래 코드와 동일하게
scan_center_tile_only = False

# 60 pm = 0.60 Å
effective_probe_fwhm_A = 0.40

# 원래 코드에서 사용한 scan 영역과 동일하게
# scan_center_tile_only=False 였다면 그대로 두면 됨
# True였으면 중앙 타일 fraction만 자동 반영
abtem.config.set({"device": "cpu"})


# =========================================================
# 1) Load stack
# =========================================================
if not npy_file.exists():
    raise FileNotFoundError(f"npy file not found: {npy_file.resolve()}")

adf_stack = np.load(npy_file)
print("Loaded stack shape:", adf_stack.shape)

if adf_stack.ndim != 3:
    raise ValueError("Expected focus-series stack with shape (n_focus, ny, nx)")

n_focus, ny, nx = adf_stack.shape


# =========================================================
# 2) Reconstruct pixel size from CIF + repeat info
# =========================================================
if not cif_file.exists():
    raise FileNotFoundError(f"CIF file not found: {cif_file.resolve()}")

atoms = read(cif_file)
atoms = abtem.orthogonalize_cell(atoms)

cell_x_A = atoms.cell.lengths()[0]
cell_y_A = atoms.cell.lengths()[1]

sim_cell_x_A = cell_x_A * repeat_x
sim_cell_y_A = cell_y_A * repeat_y

if scan_center_tile_only and (repeat_x > 1 or repeat_y > 1):
    start_x = (repeat_x // 2) / repeat_x
    end_x = (repeat_x // 2 + 1) / repeat_x
    start_y = (repeat_y // 2) / repeat_y
    end_y = (repeat_y // 2 + 1) / repeat_y
else:
    start_x, end_x = 0.0, 1.0
    start_y, end_y = 0.0, 1.0

scan_width_x_A = (end_x - start_x) * sim_cell_x_A
scan_width_y_A = (end_y - start_y) * sim_cell_y_A

pixel_size_x_A = scan_width_x_A / nx
pixel_size_y_A = scan_width_y_A / ny

print(f"Scan width x = {scan_width_x_A:.4f} Å, y = {scan_width_y_A:.4f} Å")
print(f"Pixel size x = {pixel_size_x_A:.4f} Å/pix, y = {pixel_size_y_A:.4f} Å/pix")


# =========================================================
# 3) Apply Gaussian blur to the already-saved stack
# =========================================================
adf_stack_blurred = gaussian_blur_stack_fwhm(
    adf_stack,
    fwhm_A=effective_probe_fwhm_A,
    pixel_size_x_A=pixel_size_x_A,
    pixel_size_y_A=pixel_size_y_A,
)

print("Blurred stack shape:", adf_stack_blurred.shape)


# =========================================================
# 4) Save
# =========================================================
out_stem = (
    npy_file.stem
    + f"_blurFWHM{fmt_num(effective_probe_fwhm_A, 2)}A"
)

np.save(f"{out_stem}.npy", adf_stack_blurred)

with open(f"{out_stem}_metadata.txt", "w", encoding="utf-8") as f:
    f.write(f"input_npy: {npy_file}\n")
    f.write(f"cif_file: {cif_file}\n")
    f.write(f"repeat_x: {repeat_x}\n")
    f.write(f"repeat_y: {repeat_y}\n")
    f.write(f"scan_center_tile_only: {scan_center_tile_only}\n")
    f.write(f"cell_x_A: {cell_x_A}\n")
    f.write(f"cell_y_A: {cell_y_A}\n")
    f.write(f"scan_width_x_A: {scan_width_x_A}\n")
    f.write(f"scan_width_y_A: {scan_width_y_A}\n")
    f.write(f"pixel_size_x_A: {pixel_size_x_A}\n")
    f.write(f"pixel_size_y_A: {pixel_size_y_A}\n")
    f.write(f"effective_probe_fwhm_A: {effective_probe_fwhm_A}\n")
    f.write(f"stack_shape: {adf_stack.shape}\n")


# =========================================================
# 5) Plot blurred stack
# =========================================================
ncols = 5
nrows = int(np.ceil(n_focus / ncols))

vmin = adf_stack_blurred.min()
vmax = adf_stack_blurred.max()

fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows))
axes = np.atleast_1d(axes).ravel()

for i in range(n_focus):
    ax = axes[i]
    im = ax.imshow(adf_stack_blurred[i], origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(f"focus index = {i}")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.06)

for j in range(n_focus, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig(f"{out_stem}.png", dpi=200)
plt.show()