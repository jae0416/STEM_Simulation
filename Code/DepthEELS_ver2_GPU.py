import math
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")

import numpy as np
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

import matplotlib.pyplot as plt
from ase.io import read
import abtem

from abtem.inelastic.core_loss import SubshellTransitions
from abtem.multislice import transition_potential_multislice_and_detect

try:
    import dask
except Exception:
    dask = None


# =========================================================
# Optional GPU detection
# =========================================================
# RTX 5060 + current abTEM EELS detector path is unstable with Numba CUDA.
# Safer default: CPU
USE_GPU = False

try:
    import cupy as cp
    _gpu_count = cp.cuda.runtime.getDeviceCount()
    if _gpu_count < 1:
        USE_GPU = False
except Exception as e:
    print(f"[INFO] CuPy / GPU not available, fallback to CPU. Reason: {e}")
    USE_GPU = False

if USE_GPU:
    abtem.config.set({
        "device": "gpu",
        "dask.chunk-size-gpu": "512 MB",
    })
    if dask is not None:
        dask.config.set({"num_workers": 1})
    DEVICE = "gpu"
    print("[INFO] Running on GPU")
else:
    abtem.config.set({"device": "cpu"})
    DEVICE = "cpu"
    print("[INFO] Running on CPU")


# =========================================================
# Helper functions
# =========================================================
def fmt_num(x, ndigits=2):
    return f"{x:.{ndigits}f}".replace(".", "p")


def sanitize_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("_")


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


def extract_final_cumulative_map(measurement_array, exit_count):
    """
    Return ONE final cumulative EELS image of shape (ny, nx).

    Handles:
      - single exit plane -> 2D array (ny, nx)
      - explicit depth axis -> 3D array (depth, ny, nx) or equivalent
    """
    arr = np.asarray(measurement_array)
    print("Raw cumulative array shape:", arr.shape)

    # already one image
    if arr.ndim == 2:
        return arr

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
    print("Squeezed cumulative array shape:", arr.shape)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        return arr[-1]

    raise RuntimeError(
        f"Could not reduce cumulative array to final 2D image. Final shape={arr.shape}"
    )


def build_scan(potential, repeat_x, repeat_y, scan_mode, scan_center_tile_only, scan_gpts):
    """
    scan_center_tile_only=True:
        scan only central 1 tile of repeated supercell
    scan_mode='projected_unit_cell':
        scan first projected unit cell only
    scan_mode='full_supercell':
        scan full repeated supercell
    """
    if scan_center_tile_only and (repeat_x > 1 or repeat_y > 1):
        start_x = (repeat_x // 2) / repeat_x
        end_x = (repeat_x // 2 + 1) / repeat_x
        start_y = (repeat_y // 2) / repeat_y
        end_y = (repeat_y // 2 + 1) / repeat_y
    else:
        if scan_mode == "projected_unit_cell":
            start_x, end_x = 0.0, 1.0 / repeat_x
            start_y, end_y = 0.0, 1.0 / repeat_y
        elif scan_mode == "full_supercell":
            start_x, end_x = 0.0, 1.0
            start_y, end_y = 0.0, 1.0
        else:
            raise ValueError("scan_mode must be 'projected_unit_cell' or 'full_supercell'")

    scan = abtem.GridScan(
        start=(start_x, start_y),
        end=(end_x, end_y),
        gpts=scan_gpts,
        fractional=True,
        potential=potential,
        endpoint=False,
    )

    return scan, (start_x, start_y), (end_x, end_y)


# =========================================================
# Input
# =========================================================
cif_file = Path("./Total28_STO12BMO4STO12.cif")
zone_axis = (0, 0, 1)

energy_eV = 300e3
semiangle_mrad = 50.0

# EELS collection angle
collection_inner_mrad = 0.0
collection_outer_mrad = 70.0

# x, y repeat -> changes FOV when scan_mode="full_supercell"
repeat_x = 2
repeat_y = 2

# Multislice slice thickness - 얼마나 촘촘히 계산할지
slice_thickness_A = 5.0

# Defocus series
defocus_step_A = 15.0
defocus_start_A = -10

# Scan / potential
scan_mode = "full_supercell"      # "projected_unit_cell" or "full_supercell"
scan_center_tile_only = False     # if True, scan only central tile
scan_gpts = (32, 32)              # output image pixel grid
potential_gpts = 128              # wave/potential resolution

# Computation
site_threshold = 0.95
max_batch = 1
double_channel = False
num_workers = 1

# Transition model
xc = "PBE"
order = 1
epsilon = 10

# Choose edges - 더 다양한 edge simulation하고싶으면 아래에서 추가
EDGES = [
    {"label": "Mn_L", "Z": 25, "n": 2, "l": 1},
]

# Example:
# EDGES = [
#     {"label": "Ba_M", "Z": 56, "n": 3, "l": 2},
#     {"label": "Mn_L", "Z": 25, "n": 2, "l": 1},
#     {"label": "Sr_M", "Z": 38, "n": 3, "l": 2},
#     {"label": "Ti_L", "Z": 22, "n": 2, "l": 1},
# ]


# =========================================================
# 1) Read CIF and get specimen thickness automatically
# =========================================================
if not cif_file.exists():
    raise FileNotFoundError(f"CIF file not found: {cif_file.resolve()}")

atoms = read(cif_file)
atoms = orient_atoms_to_zone_axis(atoms, zone_axis=zone_axis)
atoms = abtem.orthogonalize_cell(atoms)

print("Loaded atoms:", atoms)
print("Cell lengths (Å):", atoms.cell.lengths())

sample_thickness_A = atoms.cell.lengths()[2]
sample_thickness_nm = sample_thickness_A / 10.0

print(f"Sample thickness = {sample_thickness_A:.4f} Å ({sample_thickness_nm:.3f} nm)")

atoms_sim = atoms * (repeat_x, repeat_y, 1)
print("Simulated atoms:", len(atoms_sim))

defocus_values_A = np.arange(defocus_start_A, sample_thickness_A + 1e-9, defocus_step_A)
if abs(defocus_values_A[-1] - sample_thickness_A) > 1e-6:
    defocus_values_A = np.append(defocus_values_A, sample_thickness_A)

print("EELS defocus values (Å):", defocus_values_A)
print("EELS defocus values (nm):", defocus_values_A / 10.0)


# =========================================================
# 2) Exit-plane control: final cumulative image only
# =========================================================
n_slices_total = int(np.ceil(sample_thickness_A / slice_thickness_A))
exit_planes_every_n_slices = n_slices_total

print(f"Using final-only exit plane: every {exit_planes_every_n_slices} slices")


# =========================================================
# 3) Potential
# =========================================================
potential = abtem.Potential(
    atoms_sim,
    gpts=potential_gpts,
    slice_thickness=slice_thickness_A,
    exit_planes=exit_planes_every_n_slices,
    device=DEVICE,
)

exit_thicknesses = np.asarray(potential.exit_thicknesses)
print("Exit thicknesses (Å):", exit_thicknesses)

if len(exit_thicknesses) == 0:
    raise RuntimeError("No exit planes were generated.")


# =========================================================
# 4) Common scan
# =========================================================
scan, scan_start_fractional, scan_end_fractional = build_scan(
    potential=potential,
    repeat_x=repeat_x,
    repeat_y=repeat_y,
    scan_mode=scan_mode,
    scan_center_tile_only=scan_center_tile_only,
    scan_gpts=scan_gpts,
)

print("Scan start (fractional):", scan_start_fractional)
print("Scan end   (fractional):", scan_end_fractional)

# Pre-check maximum simulated angle
probe0 = abtem.Probe(
    energy=energy_eV,
    semiangle_cutoff=semiangle_mrad,
    defocus=0.0,
    device=DEVICE,
)
probe0.grid.match(potential)

probe0_waves = probe0.build()
if hasattr(probe0_waves, "to_cpu"):
    probe0_waves = probe0_waves.to_cpu()

if hasattr(probe0_waves, "cutoff_angles"):
    max_angle = float(min(probe0_waves.cutoff_angles))
    print("Probe cutoff angles (mrad):", probe0_waves.cutoff_angles)
    if collection_outer_mrad >= max_angle:
        raise RuntimeError(
            f"EELS outer angle ({collection_outer_mrad} mrad) exceeds/saturates "
            f"maximum simulated angle ({max_angle:.3f} mrad). "
            f"Reduce collection_outer_mrad or increase potential_gpts."
        )


# =========================================================
# 5) Save basenames and run per edge
# =========================================================
cif_stem = sanitize_name(cif_file.stem)
defocus_step_nm = defocus_step_A / 10.0

for edge in EDGES:
    edge_label = edge["label"]
    Z = edge["Z"]
    n_shell = edge["n"]
    l_shell = edge["l"]

    base_name = (
        f"{cif_stem}"
        f"_{edge_label}"
        f"_EELS"
        f"_{int(energy_eV/1000)}kV"
        f"_conv{int(semiangle_mrad)}mrad"
        f"_col{int(collection_inner_mrad)}to{int(collection_outer_mrad)}mrad"
        f"_thick{fmt_num(sample_thickness_nm, 3)}nm"
        f"_dfstep{fmt_num(defocus_step_nm, 2)}nm"
        f"_rep{repeat_x}x{repeat_y}"
        f"_scan{scan_gpts[0]}x{scan_gpts[1]}"
        f"_gpts{potential_gpts}"
        f"_dev{DEVICE}"
    )

    print("=" * 80)
    print(f"[EDGE] {edge_label}")
    print("EELS save basename:", base_name)

    transitions = SubshellTransitions(
        Z=Z,
        n=n_shell,
        l=l_shell,
        xc=xc,
        order=order,
        epsilon=epsilon,
    )

    print(f"{edge_label}: number of transitions = {len(transitions)}")

    transition_potentials = transitions.get_transition_potentials(energy=energy_eV)
    transition_potentials.grid.match(potential)
    transition_potentials = transition_potentials.build()

    sites = atoms_sim[atoms_sim.numbers == Z]
    print(f"{edge_label}: number of matching sites = {len(sites)}")

    if len(sites) == 0:
        print(f"[WARN] No {edge_label} atoms found in structure. Skipping.")
        continue

    eels_stack = []

    for df in defocus_values_A:
        print(f"[{edge_label}] Calculating defocus = {df:.1f} Å ({df/10:.2f} nm focus depth)")

        probe = abtem.Probe(
            energy=energy_eV,
            semiangle_cutoff=semiangle_mrad,
            defocus=float(df),
            device=DEVICE,
        )
        probe.grid.match(potential)

        detector = abtem.FlexibleAnnularDetector(to_cpu=True)

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
            sites=sites,
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

        cumulative_maps_measurement = measurements.integrate_radial(
            inner=collection_inner_mrad,
            outer=collection_outer_mrad,
        )

        single_eels_map = extract_final_cumulative_map(
            cumulative_maps_measurement.array,
            exit_count=len(exit_thicknesses),
        )

        eels_stack.append(single_eels_map)

    eels_stack = np.asarray(eels_stack)
    print(f"{edge_label} focus-series shape:", eels_stack.shape)

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------
    np.save(f"{base_name}_focusDepths_A.npy", defocus_values_A)
    np.save(f"{base_name}_focusSeries.npy", eels_stack)

    metadata = {
        "cif_file": str(cif_file),
        "zone_axis": zone_axis,
        "edge_label": edge_label,
        "Z": Z,
        "n": n_shell,
        "l": l_shell,
        "energy_eV": energy_eV,
        "semiangle_mrad": semiangle_mrad,
        "collection_inner_mrad": collection_inner_mrad,
        "collection_outer_mrad": collection_outer_mrad,
        "sample_thickness_A": sample_thickness_A,
        "sample_thickness_nm": sample_thickness_nm,
        "repeat_x": repeat_x,
        "repeat_y": repeat_y,
        "slice_thickness_A": slice_thickness_A,
        "exit_planes_every_n_slices": exit_planes_every_n_slices,
        "potential_gpts": potential_gpts,
        "scan_mode": scan_mode,
        "scan_center_tile_only": scan_center_tile_only,
        "scan_gpts": scan_gpts,
        "scan_start_fractional": scan_start_fractional,
        "scan_end_fractional": scan_end_fractional,
        "defocus_values_A": defocus_values_A.tolist(),
        "site_threshold": site_threshold,
        "max_batch": max_batch,
        "double_channel": double_channel,
        "device": DEVICE,
        "num_matching_sites": len(sites),
    }

    with open(f"{base_name}_metadata.txt", "w", encoding="utf-8") as f:
        for k, v in metadata.items():
            f.write(f"{k}: {v}\n")

    # -----------------------------------------------------
    # Plot EELS focus-series
    # -----------------------------------------------------
    n = len(defocus_values_A)
    ncols = 5
    nrows = math.ceil(n / ncols)

    vmin = eels_stack.min()
    vmax = eels_stack.max()

    fig_eels, axes_eels = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows))
    axes_eels = np.atleast_1d(axes_eels).ravel()

    for i, (img, df) in enumerate(zip(eels_stack, defocus_values_A)):
        ax = axes_eels[i]
        im = ax.imshow(img, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(f"focus = {df/10:.2f} nm")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.06)

    for j in range(i + 1, len(axes_eels)):
        axes_eels[j].axis("off")

    plt.tight_layout()
    plt.savefig(f"{base_name}_focusSeries.png", dpi=200)
    plt.close(fig_eels)

    print(f"[{edge_label}] Figure saved. Done.")