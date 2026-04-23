import math
from pathlib import Path

import numpy as np
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

import matplotlib.pyplot as plt
from ase.io import read
import abtem
from abtem.inelastic.core_loss import SubshellTransitions
from abtem.multislice import transition_potential_multislice_and_detect


# ============================================================
# User input
# ============================================================
# ---- structure ----
cif_file = Path("./Total14_STO6BMO3STO6.cif")
zone_axis = (0, 0, 1)

# ---- microscope ----
energy_eV = 300e3
semiangle_mrad = 50.0

# ---- detector / EELS integration ----
collection_inner_mrad = 0.0
collection_outer_mrad = 70.0

# ---- specimen thickness / depth sampling ----
target_thickness_A = 20.0
depth_step_A = 10.0
slice_thickness_A = 1.0

# ---- lateral supercell ----
repeat_x = 2
repeat_y = 2

# ---- scan control ----
scan_mode = "projected_unit_cell"
scan_gpts = (32, 32)

# ---- potential grid ----
potential_gpts = 128

# ---- computation ----
device = "cpu"
num_workers = 2
site_threshold = 0.95
max_batch = 2
double_channel = False

# ---- target edge ----
transition_Z = 8
transition_n = 1
transition_l = 0
transition_label = "O_K"

# ---- subshell transition model ----
xc = "PBE"
order = 1
epsilon = 10

# ---- output prefix ----
output_prefix = f"{transition_label}_300kV_50mrad"


# ============================================================
# Helper functions
# ============================================================
def orient_atoms_to_zone_axis(atoms, zone_axis=(0, 0, 1)):
    atoms = atoms.copy()

    if zone_axis == (0, 0, 1):
        return atoms
    if zone_axis == (1, 0, 0):
        atoms.rotate(90, "y", rotate_cell=True)
        return atoms
    if zone_axis == (0, 1, 0):
        atoms.rotate(-90, "x", rotate_cell=True)
        return atoms
    if zone_axis == (1, 1, 0):
        atoms.rotate(45, "z", rotate_cell=True)
        atoms.rotate(-90, "x", rotate_cell=True)
        return atoms

    raise NotImplementedError(
        f"zone_axis={zone_axis} is not implemented in this helper."
    )


def make_scan(potential, repeat_x, repeat_y, scan_mode="projected_unit_cell", scan_gpts=(64, 64)):
    if scan_mode == "projected_unit_cell":
        end = (1 / repeat_x, 1 / repeat_y)
        fractional = True
    elif scan_mode == "full_supercell":
        end = (1.0, 1.0)
        fractional = True
    else:
        raise ValueError("scan_mode must be 'projected_unit_cell' or 'full_supercell'")

    scan = abtem.GridScan(
        start=(0.0, 0.0),
        end=end,
        gpts=scan_gpts,
        fractional=fractional,
        potential=potential,
        endpoint=False,
    )
    return scan


def save_depth_series_png(stack, titles, out_png, cmap="inferno"):
    n = len(stack)
    ncols = min(5, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, (img, title) in enumerate(zip(stack, titles)):
        ax = axes[i]
        im = ax.imshow(img, origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def normalize_measurement_to_depth_yx(measurement_array, exit_thicknesses, keep_mask):
    arr = np.asarray(measurement_array)
    print("Raw cumulative array shape:", arr.shape)

    exit_count = len(exit_thicknesses)

    thickness_axis = None
    for ax, size in enumerate(arr.shape):
        if size == exit_count:
            thickness_axis = ax
            break

    if thickness_axis is None:
        raise RuntimeError(
            f"Could not identify thickness axis in array shape {arr.shape}; "
            f"expected one axis with length {exit_count}."
        )

    if thickness_axis != 0:
        arr = np.moveaxis(arr, thickness_axis, 0)

    arr = np.squeeze(arr)
    print("Thickness-first squeezed shape:", arr.shape)

    if arr.ndim < 3:
        raise RuntimeError(
            f"Expected at least 3 dimensions after squeeze, got shape {arr.shape}."
        )

    if arr.ndim != 3:
        raise RuntimeError(
            f"Expected final array shape (depth, y, x), got {arr.shape}."
        )

    arr = arr[keep_mask]
    print("Kept cumulative maps shape:", arr.shape)
    return arr


# ============================================================
# Basic checks
# ============================================================
if not cif_file.exists():
    raise FileNotFoundError(f"CIF file not found: {cif_file.resolve()}")

ratio = depth_step_A / slice_thickness_A
if not np.isclose(ratio, round(ratio)):
    raise ValueError("depth_step_A must be an integer multiple of slice_thickness_A.")

exit_planes_every_n_slices = int(round(ratio))

abtem.config.set({"device": device})


# ============================================================
# Read CIF, orient, orthogonalize
# ============================================================
atoms_uc = read(cif_file)
atoms_uc = orient_atoms_to_zone_axis(atoms_uc, zone_axis=zone_axis)
atoms_uc = abtem.orthogonalize_cell(atoms_uc)

print("Loaded and oriented unit cell:")
print(atoms_uc)
print("Cell lengths (Å):", atoms_uc.cell.lengths())
print("Beam direction after orientation = z")
print("Requested zone axis:", zone_axis)

c_axis_A = atoms_uc.cell.lengths()[2]
repeat_z = math.ceil(target_thickness_A / c_axis_A)

atoms = atoms_uc * (repeat_x, repeat_y, repeat_z)

print(f"Unit-cell c = {c_axis_A:.4f} Å")
print(f"repeat_z = {repeat_z}")
print(f"Total simulated thickness = {repeat_z * c_axis_A:.4f} Å")


# ============================================================
# Potential with exit planes at requested depth intervals
# ============================================================
potential = abtem.Potential(
    atoms,
    gpts=potential_gpts,
    slice_thickness=slice_thickness_A,
    exit_planes=exit_planes_every_n_slices,
    device=device,
)

exit_thicknesses = np.asarray(potential.exit_thicknesses)
print("Exit thicknesses (Å):", exit_thicknesses)

keep = (exit_thicknesses > 0.0) & (exit_thicknesses <= target_thickness_A + 1e-9)
depths_A = exit_thicknesses[keep]
depths_nm = depths_A / 10.0

if len(depths_A) == 0:
    raise RuntimeError("No exit planes retained. Check target_thickness_A and depth_step_A.")


# ============================================================
# Core-loss transitions
# ============================================================
transitions = SubshellTransitions(
    Z=transition_Z,
    n=transition_n,
    l=transition_l,
    xc=xc,
    order=order,
    epsilon=epsilon,
)

print(f"{transition_label}: number of transitions = {len(transitions)}")

transition_potentials = transitions.get_transition_potentials(energy=energy_eV)
transition_potentials.grid.match(potential)
transition_potentials = transition_potentials.build()


# ============================================================
# Probe, scan, detector
# ============================================================
probe = abtem.Probe(
    energy=energy_eV,
    semiangle_cutoff=semiangle_mrad,
    device=device,
)
probe.grid.match(potential)

scan = make_scan(
    potential=potential,
    repeat_x=repeat_x,
    repeat_y=repeat_y,
    scan_mode=scan_mode,
    scan_gpts=scan_gpts,
)

detector = abtem.FlexibleAnnularDetector(to_cpu=True)


# ============================================================
# Core-loss scan (abTEM 1.0.9 compatible)
# ============================================================
probes = probe.build(
    scan=scan,
    max_batch=max_batch,
    lazy=False,
)

print("Probe waves shape:", probes.shape)

measurements = transition_potential_multislice_and_detect(
    waves=probes,
    potential=potential,
    transition_potential=transition_potentials,
    detectors=[detector],
    double_channel=double_channel,
    threshold=site_threshold,
    sites=atoms,
    pbar=True,
)

if isinstance(measurements, (list, tuple)):
    if len(measurements) != 1:
        raise RuntimeError(f"Expected one detector output, got {len(measurements)}")
    measurements = measurements[0]

if hasattr(measurements, "compute"):
    measurements = measurements.compute(
        scheduler="threads",
        num_workers=num_workers,
    )


# ============================================================
# Convert radial detector data -> cumulative ionization maps
# ============================================================
cumulative_maps_measurement = measurements.integrate_radial(
    inner=collection_inner_mrad,
    outer=collection_outer_mrad,
)

cumulative_maps = normalize_measurement_to_depth_yx(
    cumulative_maps_measurement.array,
    exit_thicknesses=exit_thicknesses,
    keep_mask=keep,
)

if cumulative_maps.shape[0] != len(depths_A):
    raise RuntimeError(
        f"Mismatch: cumulative_maps.shape[0]={cumulative_maps.shape[0]} "
        f"but len(depths_A)={len(depths_A)}"
    )


# ============================================================
# Convert cumulative -> differential depth slice maps
# ============================================================
differential_maps = np.empty_like(cumulative_maps)
differential_maps[0] = cumulative_maps[0]
differential_maps[1:] = cumulative_maps[1:] - cumulative_maps[:-1]
differential_maps = np.clip(differential_maps, 0, None)

print("Differential maps shape:", differential_maps.shape)


# ============================================================
# Save arrays
# ============================================================
np.save(f"{output_prefix}_depths_A.npy", depths_A)
np.save(f"{output_prefix}_depths_nm.npy", depths_nm)
np.save(f"{output_prefix}_cumulative_maps.npy", cumulative_maps)
np.save(f"{output_prefix}_differential_maps.npy", differential_maps)

with open(f"{output_prefix}_metadata.txt", "w", encoding="utf-8") as f:
    f.write(f"cif_file = {cif_file}\n")
    f.write(f"zone_axis = {zone_axis}\n")
    f.write(f"energy_eV = {energy_eV}\n")
    f.write(f"semiangle_mrad = {semiangle_mrad}\n")
    f.write(f"collection_inner_mrad = {collection_inner_mrad}\n")
    f.write(f"collection_outer_mrad = {collection_outer_mrad}\n")
    f.write(f"target_thickness_A = {target_thickness_A}\n")
    f.write(f"depth_step_A = {depth_step_A}\n")
    f.write(f"slice_thickness_A = {slice_thickness_A}\n")
    f.write(f"repeat_x = {repeat_x}\n")
    f.write(f"repeat_y = {repeat_y}\n")
    f.write(f"repeat_z = {repeat_z}\n")
    f.write(f"scan_mode = {scan_mode}\n")
    f.write(f"scan_gpts = {scan_gpts}\n")
    f.write(f"potential_gpts = {potential_gpts}\n")
    f.write(f"transition_label = {transition_label}\n")
    f.write(f"transition_Z = {transition_Z}\n")
    f.write(f"transition_n = {transition_n}\n")
    f.write(f"transition_l = {transition_l}\n")
    f.write(f"double_channel = {double_channel}\n")


# ============================================================
# Plot cumulative maps
# ============================================================
cum_titles = [f"cum. {dnm:.1f} nm" for dnm in depths_nm]
save_depth_series_png(
    cumulative_maps,
    cum_titles,
    f"{output_prefix}_cumulative_depth_series.png",
)

# ============================================================
# Plot differential maps
# ============================================================
diff_titles = []
for i in range(len(depths_nm)):
    if i == 0:
        diff_titles.append(f"0–{depths_nm[i]:.1f} nm")
    else:
        diff_titles.append(f"{depths_nm[i-1]:.1f}–{depths_nm[i]:.1f} nm")

save_depth_series_png(
    differential_maps,
    diff_titles,
    f"{output_prefix}_differential_depth_series.png",
)

print("Done.")
