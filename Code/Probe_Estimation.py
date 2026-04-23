import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LogNorm

# =========================================================
# USER INPUT: 파일 5개를 직접 지정
# =========================================================
cross_section_file = Path("./Total28_Frozenphonon10_0.08_200kV_20mrad/Total28_STO12BMO4STO12_ProbeVacuum_200kV_conv20mrad_ADFdet60to200mrad_thick10p956nm_dfstep1p50nm_extent20p0A_samp0p020A_devgpu_crossSection_xz.npy")
axial_profile_file = Path("./Total28_Frozenphonon10_0.08_200kV_20mrad/Total28_STO12BMO4STO12_ProbeVacuum_200kV_conv20mrad_ADFdet60to200mrad_thick10p956nm_dfstep1p50nm_extent20p0A_samp0p020A_devgpu_axialProfile.npy")
focus0_profile_file = Path("./Total28_Frozenphonon10_0.08_200kV_20mrad/Total28_STO12BMO4STO12_ProbeVacuum_200kV_conv20mrad_ADFdet60to200mrad_thick10p956nm_dfstep1p50nm_extent20p0A_samp0p020A_devgpu_focus0_lateralProfile.npy")
defocus_values_file = Path("./Total28_Frozenphonon10_0.08_200kV_20mrad/Total28_STO12BMO4STO12_ProbeVacuum_200kV_conv20mrad_ADFdet60to200mrad_thick10p956nm_dfstep1p50nm_extent20p0A_samp0p020A_devgpu_defocusValues_A.npy")
x_axis_file = Path("./Total28_Frozenphonon10_0.08_200kV_20mrad/Total28_STO12BMO4STO12_ProbeVacuum_200kV_conv20mrad_ADFdet60to200mrad_thick10p956nm_dfstep1p50nm_extent20p0A_samp0p020A_devgpu_xAxis_A.npy")

# -----------------------------
# 표시 범위
# -----------------------------
defocus_min_nm = -10.0
defocus_max_nm = 10.0

x_min_A = -5.0
x_max_A = 5.0

intensity_mode = "linear"   # "linear" or "log"
vmin = 0
vmax = 1.0
log_floor = 1e-4

profile_ymin = 0.0
profile_ymax = 1.05

save_figure = True
output_png = "probevacuum_adjusted.png"

# =========================================================
# Load
# =========================================================
probe_xz = np.load(cross_section_file)          # (n_defocus, nx)
probe_axial = np.load(axial_profile_file)       # (n_defocus,)
probe_focus0 = np.load(focus0_profile_file)     # (nx,)
defocus_values_A = np.load(defocus_values_file) # (n_defocus,)
x_axis_A = np.load(x_axis_file)                 # (nx,)

defocus_values_nm = defocus_values_A / 10.0

# =========================================================
# Range selection
# =========================================================
mask_defocus = (defocus_values_nm >= defocus_min_nm) & (defocus_values_nm <= defocus_max_nm)
mask_x = (x_axis_A >= x_min_A) & (x_axis_A <= x_max_A)

if not np.any(mask_defocus):
    raise ValueError("선택한 defocus 범위 안에 데이터가 없습니다.")
if not np.any(mask_x):
    raise ValueError("선택한 x 범위 안에 데이터가 없습니다.")

probe_xz_sel = probe_xz[mask_defocus][:, mask_x]
probe_axial_sel = probe_axial[mask_defocus]
defocus_sel_nm = defocus_values_nm[mask_defocus]
x_sel_A = x_axis_A[mask_x]
probe_focus0_sel = probe_focus0[mask_x]

# =========================================================
# Plot
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

if intensity_mode.lower() == "log":
    img = np.clip(probe_xz_sel, log_floor, None)
    im = axes[0].imshow(
        img,
        origin="lower",
        aspect="auto",
        extent=[x_sel_A[0], x_sel_A[-1], defocus_sel_nm[0], defocus_sel_nm[-1]],
        norm=LogNorm(vmin=max(log_floor, img.min()), vmax=img.max()),
    )
else:
    im = axes[0].imshow(
        probe_xz_sel,
        origin="lower",
        aspect="auto",
        extent=[x_sel_A[0], x_sel_A[-1], defocus_sel_nm[0], defocus_sel_nm[-1]],
        vmin=vmin,
        vmax=vmax,
    )

axes[0].set_title("Vacuum probe cross-section")
axes[0].set_xlabel("Lateral position x (Å)")
axes[0].set_ylabel("Defocus (nm)")
plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

axes[1].plot(defocus_sel_nm, probe_axial_sel, marker="o", ms=4)
axes[1].axhline(0.5, ls="--", lw=1)
axes[1].set_title("Optical-axis profile")
axes[1].set_xlabel("Defocus (nm)")
axes[1].set_ylabel("Normalized probe intensity")
axes[1].set_xlim(defocus_min_nm, defocus_max_nm)
axes[1].set_ylim(profile_ymin, profile_ymax)

axes[2].plot(x_sel_A, probe_focus0_sel)
axes[2].set_title("Lateral profile at Δf = 0")
axes[2].set_xlabel("Lateral position x (Å)")
axes[2].set_ylabel("Normalized probe intensity")
axes[2].set_xlim(x_min_A, x_max_A)
axes[2].set_ylim(profile_ymin, profile_ymax)

plt.tight_layout()

if save_figure:
    plt.savefig(output_png, dpi=200)
    print(f"Saved: {output_png}")

plt.show()