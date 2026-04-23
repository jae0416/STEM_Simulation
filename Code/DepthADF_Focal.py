import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
import abtem

# -----------------------------
# Input
# -----------------------------
cif_file = Path("./STO7LNO4STO7.cif")

energy_eV = 300e3
semiangle_mrad = 50

adf_inner_mrad = 60
adf_outer_mrad = 195

target_thickness_A = 100.0   # 10 nm
repeat_x = 1
repeat_y = 1

potential_sampling_A = 0.03
slice_thickness_A = 1.0

# Focus positions relative to entrance surface
# 0, 1, 2, ... 10 nm  ->  0, 10, 20, ... 100 Å
defocus_values_A = np.arange(0.0, target_thickness_A + 1e-9, 20.0)

# Frozen phonon settings
num_frozen_configs = 2
sigma_displacement_A = 0.08   # start value; tune later if needed

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

# Fixed 10 nm sample thickness
c_axis_A = atoms.cell.lengths()[1]
nz = math.ceil(target_thickness_A / c_axis_A)

atoms_10nm = atoms * (repeat_x, repeat_y, nz)

print(f"Unit-cell c = {c_axis_A:.4f} Å")
print(f"z repetitions = {nz}")
print(f"Total simulated thickness = {nz * c_axis_A:.4f} Å")
print("10 nm supercell atoms:", len(atoms_10nm))

# -----------------------------
# Frozen phonons + potential
# -----------------------------
# One scalar sigma is the simplest starting point.
# Later you can replace with a dict by element if you want.
frozen_phonons = abtem.FrozenPhonons(
    atoms_10nm,
    num_configs=num_frozen_configs,
    sigmas=sigma_displacement_A,
)

potential = abtem.Potential(
    frozen_phonons,
    sampling=potential_sampling_A,
    slice_thickness=slice_thickness_A,
)

# -----------------------------
# Scan / detector
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
        f"maximum simulated angle ({max_angle:.3f} mrad). "
        f"Reduce outer angle or decrease potential_sampling_A."
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
# Defocus series with frozen-phonon averaging
# -----------------------------
adf_stack = []

for df in defocus_values_A:
    print(f"Calculating defocus = {df:.1f} Å ({df/10:.1f} nm focus depth)")

    probe = abtem.Probe(
        energy=energy_eV,
        semiangle_cutoff=semiangle_mrad,
        defocus=float(df),
    )
    probe.grid.match(potential)

    # This measurement carries a frozen-phonon ensemble axis
    adf_measurement = probe.scan(
        potential=potential,
        scan=scan,
        detectors=detector,
    )

    # Average over frozen-phonon configurations before compute
    adf_mean = adf_measurement.reduce_ensemble().compute()

    adf_stack.append(np.asarray(adf_mean.array))

adf_stack = np.asarray(adf_stack)

print("ADF focus-series shape:", adf_stack.shape)

# -----------------------------
# Save
# -----------------------------
np.save("ADF_focus_depths_A.npy", defocus_values_A)
np.save("ADF_focus_series_frozen_phonon.npy", adf_stack)

# -----------------------------
# Plot with common color scale
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
    ax.set_title(f"focus = {df/10:.1f} nm")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig("ADF_focus_series_frozen_phonon.png", dpi=200)
plt.show()