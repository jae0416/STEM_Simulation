import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
import abtem


# =========================================================
# Helper functions
# =========================================================
def fmt_num(x, ndigits=2):
    return f"{x:.{ndigits}f}".replace(".", "p")


def fwhm_from_profile(x, y):
    """
    x : 1D axis
    y : 1D profile
    return FWHM in x-units
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    if y.size == 0 or np.max(y) <= 0:
        return np.nan

    y = y / np.max(y)
    half = 0.5
    above = y >= half

    if not np.any(above):
        return np.nan

    idx = np.where(above)[0]
    i1, i2 = idx[0], idx[-1]

    # left interpolation
    if i1 == 0:
        x_left = x[0]
    else:
        x0, x1 = x[i1 - 1], x[i1]
        y0, y1 = y[i1 - 1], y[i1]
        x_left = x0 + (half - y0) * (x1 - x0) / (y1 - y0)

    # right interpolation
    if i2 == len(y) - 1:
        x_right = x[-1]
    else:
        x0, x1 = x[i2], x[i2 + 1]
        y0, y1 = y[i2], y[i2 + 1]
        x_right = x0 + (half - y0) * (x1 - x0) / (y1 - y0)

    return x_right - x_left


# =========================================================
# Input
# =========================================================
cif_file = Path("./Total28_STO12BMO4STO12.cif")

energy_eV = 300e3
semiangle_mrad = 50

adf_inner_mrad = 60
adf_outer_mrad = 195

repeat_x = 2
repeat_y = 2

potential_sampling_A = 0.03
slice_thickness_A = 1.0

# specimen focal-series step
defocus_step_A = 15.0

# frozen phonon
num_frozen_configs = 2
sigma_displacement_A = 0.08

# vacuum probe profile settings
probe_extent_A = 20.0
probe_sampling_profile_A = 0.02

# True로 바꾸면 repeat_x x repeat_y supercell 중 중앙 1 tile만 스캔
scan_center_tile_only = False

abtem.config.set({"device": "cpu"})


# =========================================================
# 1) Read CIF and get specimen thickness automatically
# =========================================================
if not cif_file.exists():
    raise FileNotFoundError(f"CIF file not found: {cif_file.resolve()}")

atoms = read(cif_file)
atoms = abtem.orthogonalize_cell(atoms)

print("Loaded atoms:", atoms)
print("Cell lengths (Å):", atoms.cell.lengths())

# beam direction = z 가정
sample_thickness_A = atoms.cell.lengths()[2]
sample_thickness_nm = sample_thickness_A / 10.0

print(f"Sample thickness = {sample_thickness_A:.4f} Å ({sample_thickness_nm:.3f} nm)")

# x, y only repeat
atoms_sim = atoms * (repeat_x, repeat_y, 1)
print("Simulated atoms:", len(atoms_sim))

# specimen defocus values: 0 ~ sample thickness
defocus_values_A = np.arange(-10.0, sample_thickness_A + 1e-9, defocus_step_A)
if abs(defocus_values_A[-1] - sample_thickness_A) > 1e-6:
    defocus_values_A = np.append(defocus_values_A, sample_thickness_A)

print("ADF defocus values (Å):", defocus_values_A)
print("ADF defocus values (nm):", defocus_values_A / 10.0)


# =========================================================
# 2) Save basenames
# =========================================================
cif_stem = cif_file.stem
defocus_step_nm = defocus_step_A / 10.0

base_name = (
    f"{cif_stem}"
    f"_ADF"
    f"_{int(energy_eV/1000)}kV"
    f"_conv{int(semiangle_mrad)}mrad"
    f"_det{int(adf_inner_mrad)}to{int(adf_outer_mrad)}mrad"
    f"_thick{fmt_num(sample_thickness_nm, 3)}nm"
    f"_dfstep{fmt_num(defocus_step_nm, 2)}nm"
    f"_rep{repeat_x}x{repeat_y}"
    f"_fp{num_frozen_configs}"
    f"_sig{fmt_num(sigma_displacement_A, 2)}A"
    f"_samp{fmt_num(potential_sampling_A, 3)}A"
)

probe_base_name = (
    f"{cif_stem}"
    f"_ProbeVacuum"
    f"_{int(energy_eV/1000)}kV"
    f"_conv{int(semiangle_mrad)}mrad"
    f"_ADFdet{int(adf_inner_mrad)}to{int(adf_outer_mrad)}mrad"
    f"_thick{fmt_num(sample_thickness_nm, 3)}nm"
    f"_dfstep{fmt_num(defocus_step_nm, 2)}nm"
    f"_extent{fmt_num(probe_extent_A, 1)}A"
    f"_samp{fmt_num(probe_sampling_profile_A, 3)}A"
)

print("ADF save basename  :", base_name)
print("Probe save basename:", probe_base_name)


# =========================================================
# 3) Vacuum probe profile (Fig. 2 style)
# =========================================================
# specimen thickness 기준 대칭 defocus 범위
probe_defocus_values_A = np.arange(
    -sample_thickness_A,
    sample_thickness_A + 1e-9,
    defocus_step_A
)
if abs(probe_defocus_values_A[-1] - sample_thickness_A) > 1e-6:
    probe_defocus_values_A = np.append(probe_defocus_values_A, sample_thickness_A)

probe_gpts = int(round(probe_extent_A / probe_sampling_profile_A))
if probe_gpts % 2 == 0:
    probe_gpts += 1

print("Vacuum probe defocus values (Å):", probe_defocus_values_A)
print("Vacuum probe defocus values (nm):", probe_defocus_values_A / 10.0)
print("Vacuum probe gpts:", probe_gpts)

probe_images = []

for df in probe_defocus_values_A:
    print(f"[Probe] Calculating vacuum probe at defocus = {df:.1f} Å")

    probe_vac = abtem.Probe(
        energy=energy_eV,
        semiangle_cutoff=semiangle_mrad,
        defocus=float(df),
        extent=(probe_extent_A, probe_extent_A),
        gpts=(probe_gpts, probe_gpts),
    )

    probe_intensity = probe_vac.build().intensity().compute()
    arr = np.asarray(probe_intensity.array)
    probe_images.append(arr)

probe_images = np.asarray(probe_images)   # (n_defocus, ny, nx)

cy = probe_images.shape[1] // 2
cx = probe_images.shape[2] // 2

# Fig. 2 style cross-section: x vs defocus
probe_xz = probe_images[:, cy, :]

# optical-axis profile: center intensity vs defocus
probe_axial = probe_images[:, cy, cx]

# lateral profile at Gaussian focus (Δf ~ 0)
i0 = np.argmin(np.abs(probe_defocus_values_A))
probe_focus0 = probe_images[i0, cy, :]

# normalize
probe_xz_norm = probe_xz / np.max(probe_xz)
probe_axial_norm = probe_axial / np.max(probe_axial)
probe_focus0_norm = probe_focus0 / np.max(probe_focus0)

x_axis_A = np.linspace(-probe_extent_A / 2, probe_extent_A / 2, probe_images.shape[2])
defocus_axis_nm_probe = probe_defocus_values_A / 10.0

axial_fwhm_A = fwhm_from_profile(probe_defocus_values_A, probe_axial_norm)
axial_fwhm_nm = axial_fwhm_A / 10.0 if np.isfinite(axial_fwhm_A) else np.nan

print(f"Vacuum probe axial FWHM = {axial_fwhm_A:.3f} Å ({axial_fwhm_nm:.3f} nm)")

# save probe arrays
np.save(f"{probe_base_name}_crossSection_xz.npy", probe_xz_norm)
np.save(f"{probe_base_name}_axialProfile.npy", probe_axial_norm)
np.save(f"{probe_base_name}_focus0_lateralProfile.npy", probe_focus0_norm)
np.save(f"{probe_base_name}_defocusValues_A.npy", probe_defocus_values_A)
np.save(f"{probe_base_name}_xAxis_A.npy", x_axis_A)

with open(f"{probe_base_name}_metadata.txt", "w", encoding="utf-8") as f:
    f.write(f"cif_file: {cif_file}\n")
    f.write(f"energy_eV: {energy_eV}\n")
    f.write(f"semiangle_mrad: {semiangle_mrad}\n")
    f.write(f"adf_inner_mrad: {adf_inner_mrad}\n")
    f.write(f"adf_outer_mrad: {adf_outer_mrad}\n")
    f.write(f"sample_thickness_A: {sample_thickness_A}\n")
    f.write(f"sample_thickness_nm: {sample_thickness_nm}\n")
    f.write(f"probe_extent_A: {probe_extent_A}\n")
    f.write(f"probe_sampling_profile_A: {probe_sampling_profile_A}\n")
    f.write(f"probe_gpts: {probe_gpts}\n")
    f.write(f"probe_defocus_values_A: {probe_defocus_values_A.tolist()}\n")
    f.write(f"axial_fwhm_A: {axial_fwhm_A}\n")
    f.write(f"axial_fwhm_nm: {axial_fwhm_nm}\n")

fig_probe, axes_probe = plt.subplots(1, 3, figsize=(16, 4.5))

im0 = axes_probe[0].imshow(
    probe_xz_norm,
    origin="lower",
    aspect="auto",
    extent=[x_axis_A[0], x_axis_A[-1], defocus_axis_nm_probe[0], defocus_axis_nm_probe[-1]],
)
axes_probe[0].set_title("Vacuum probe cross-section")
axes_probe[0].set_xlabel("Lateral position x (Å)")
axes_probe[0].set_ylabel("Defocus (nm)")
plt.colorbar(im0, ax=axes_probe[0], fraction=0.046, pad=0.04)

axes_probe[1].plot(defocus_axis_nm_probe, probe_axial_norm, marker="o", ms=3)
axes_probe[1].axhline(0.5, ls="--", lw=1)
axes_probe[1].set_title(f"Optical-axis profile\nFWHM = {axial_fwhm_A:.2f} Å")
axes_probe[1].set_xlabel("Defocus (nm)")
axes_probe[1].set_ylabel("Normalized probe intensity")
axes_probe[1].set_ylim(0, 1.05)

axes_probe[2].plot(x_axis_A, probe_focus0_norm)
axes_probe[2].set_title("Lateral profile at Δf = 0")
axes_probe[2].set_xlabel("Lateral position x (Å)")
axes_probe[2].set_ylabel("Normalized probe intensity")
axes_probe[2].set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(f"{probe_base_name}_crossSection.png", dpi=200)
plt.show()
plt.close(fig_probe)


# =========================================================
# 4) Frozen phonons + potential
# =========================================================
frozen_phonons = abtem.FrozenPhonons(
    atoms_sim,
    num_configs=num_frozen_configs,
    sigmas=sigma_displacement_A,
)

potential = abtem.Potential(
    frozen_phonons,
    sampling=potential_sampling_A,
    slice_thickness=slice_thickness_A,
)


# =========================================================
# 5) Probe / scan / detector for specimen ADF
# =========================================================
probe0 = abtem.Probe(
    energy=energy_eV,
    semiangle_cutoff=semiangle_mrad,
    defocus=0.0,
)
probe0.grid.match(potential)

probe_waves = probe0.build()
print("Probe cutoff angles (mrad):", probe_waves.cutoff_angles)

max_angle = min(probe_waves.cutoff_angles)
if adf_outer_mrad >= max_angle:
    raise RuntimeError(
        f"ADF outer angle ({adf_outer_mrad} mrad) exceeds/saturates "
        f"maximum simulated angle ({max_angle:.3f} mrad). "
        f"Reduce outer angle or decrease potential_sampling_A."
    )

if scan_center_tile_only and (repeat_x > 1 or repeat_y > 1):
    start_x = (repeat_x // 2) / repeat_x
    end_x = (repeat_x // 2 + 1) / repeat_x
    start_y = (repeat_y // 2) / repeat_y
    end_y = (repeat_y // 2 + 1) / repeat_y
else:
    start_x, end_x = 0.0, 1.0
    start_y, end_y = 0.0, 1.0

scan = abtem.GridScan(
    start=(start_x, start_y),
    end=(end_x, end_y),
    sampling=probe0.aperture.nyquist_sampling,
    fractional=True,
    potential=potential,
)

detector = abtem.AnnularDetector(
    inner=adf_inner_mrad,
    outer=adf_outer_mrad,
)


# =========================================================
# 6) Defocus-series ADF
# =========================================================
adf_stack = []

for df in defocus_values_A:
    print(f"[ADF] Calculating defocus = {df:.1f} Å ({df/10:.2f} nm focus depth)")

    probe = abtem.Probe(
        energy=energy_eV,
        semiangle_cutoff=semiangle_mrad,
        defocus=float(df),
    )
    probe.grid.match(potential)

    adf_measurement = probe.scan(
        potential=potential,
        scan=scan,
        detectors=detector,
    )

    adf_mean = adf_measurement.reduce_ensemble().compute()
    adf_stack.append(np.asarray(adf_mean.array))

adf_stack = np.asarray(adf_stack)
print("ADF focus-series shape:", adf_stack.shape)


# =========================================================
# 7) Save ADF results
# =========================================================
np.save(f"{base_name}_focusDepths_A.npy", defocus_values_A)
np.save(f"{base_name}_focusSeries.npy", adf_stack)

metadata = {
    "cif_file": str(cif_file),
    "energy_eV": energy_eV,
    "semiangle_mrad": semiangle_mrad,
    "adf_inner_mrad": adf_inner_mrad,
    "adf_outer_mrad": adf_outer_mrad,
    "sample_thickness_A": sample_thickness_A,
    "sample_thickness_nm": sample_thickness_nm,
    "repeat_x": repeat_x,
    "repeat_y": repeat_y,
    "potential_sampling_A": potential_sampling_A,
    "slice_thickness_A": slice_thickness_A,
    "defocus_values_A": defocus_values_A.tolist(),
    "num_frozen_configs": num_frozen_configs,
    "sigma_displacement_A": sigma_displacement_A,
    "scan_center_tile_only": scan_center_tile_only,
    "scan_start_fractional": (start_x, start_y),
    "scan_end_fractional": (end_x, end_y),
}

with open(f"{base_name}_metadata.txt", "w", encoding="utf-8") as f:
    for k, v in metadata.items():
        f.write(f"{k}: {v}\n")

# plot ADF focus-series
n = len(defocus_values_A)
ncols = 5
nrows = math.ceil(n / ncols)

vmin = adf_stack.min()
vmax = adf_stack.max()

fig_adf, axes_adf = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows))
axes_adf = np.atleast_1d(axes_adf).ravel()

for i, (img, df) in enumerate(zip(adf_stack, defocus_values_A)):
    ax = axes_adf[i]
    im = ax.imshow(img, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(f"focus = {df/10:.2f} nm")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.06)

for j in range(i + 1, len(axes_adf)):
    axes_adf[j].axis("off")

plt.tight_layout()
plt.savefig(f"{base_name}_focusSeries.png", dpi=200)
plt.show()
plt.close(fig_adf)