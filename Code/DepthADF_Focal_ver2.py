import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
import abtem
def fmt_num(x, ndigits=2):
    return f"{x:.{ndigits}f}".replace(".", "p")
# -----------------------------
# Input
# -----------------------------
cif_file = Path("./Total28_STO12BMO4STO12.cif")

energy_eV = 300e3
semiangle_mrad = 80

adf_inner_mrad = 60
adf_outer_mrad = 195

repeat_x = 2
repeat_y = 2

potential_sampling_A = 0.03
slice_thickness_A = 1.0

# focus step: 10 Å = 1 nm
defocus_step_A = 15.0

num_frozen_configs = 2
sigma_displacement_A = 0.08

abtem.config.set({"device": "cpu"})
# -----------------------------
# Read CIF
# -----------------------------
if not cif_file.exists():
    raise FileNotFoundError(f"CIF file not found: {cif_file.resolve()}")

atoms = read(cif_file)
atoms = abtem.orthogonalize_cell(atoms)

print("Loaded atoms:", atoms)
print("Cell lengths (Å):", atoms.cell.lengths())

# beam direction = z 라고 가정
sample_thickness_A = atoms.cell.lengths()[2]
sample_thickness_nm = sample_thickness_A / 10.0

print(f"Sample thickness = {sample_thickness_A:.4f} Å ({sample_thickness_nm:.3f} nm)")

# x, y 만 반복
atoms_sim = atoms * (repeat_x, repeat_y, 1)

print("Simulated atoms:", len(atoms_sim))

# defocus values: 0 ~ sample thickness
defocus_values_A = np.arange(-10.0, sample_thickness_A + 1e-9, defocus_step_A)

# 마지막 thickness가 step에 딱 안 맞으면 마지막 점 추가
if abs(defocus_values_A[-1] - sample_thickness_A) > 1e-6:
    defocus_values_A = np.append(defocus_values_A, sample_thickness_A)

print("Defocus values (Å):", defocus_values_A)
print("Defocus values (nm):", defocus_values_A / 10.0)
print("Cell lengths (Å):", atoms.cell.lengths())

# -----------------------------
# Frozen phonons + potential
# -----------------------------
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

# -----------------------------
# Probe / scan / detector
# -----------------------------
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
        f"maximum simulated angle ({max_angle:.3f} mrad)."
    )

scan = abtem.GridScan(
    start=(0.0, 0.0),
    end=(1.0, 1.0),
    sampling=probe0.aperture.nyquist_sampling,
    fractional=True,
    potential=potential,
)

detector = abtem.AnnularDetector(
    inner=adf_inner_mrad,
    outer=adf_outer_mrad,
)

# -----------------------------
# Defocus series
# -----------------------------
adf_stack = []

for df in defocus_values_A:
    print(f"Calculating defocus = {df:.1f} Å ({df/10:.2f} nm focus depth)")

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
# -----------------------------
# Save name
# -----------------------------
cif_stem = cif_file.stem
sample_thickness_nm = sample_thickness_A / 10.0
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

print("Save basename:", base_name)

# -----------------------------
# Save arrays
# -----------------------------
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
}

with open(f"{base_name}_metadata.txt", "w", encoding="utf-8") as f:
    for k, v in metadata.items():
        f.write(f"{k}: {v}\n")

# -----------------------------
# Plot
# -----------------------------
n = len(defocus_values_A)
ncols = 5
nrows = math.ceil(n / ncols)

vmin = adf_stack.min()
vmax = adf_stack.max()

fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows))
axes = np.atleast_1d(axes).ravel()

for i, (img, df) in enumerate(zip(adf_stack, defocus_values_A)):
    ax = axes[i]
    im = ax.imshow(img, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(f"focus = {df/10:.2f} nm")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.06)

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig(f"{base_name}_focusSeries.png", dpi=200)
plt.show()