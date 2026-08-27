#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Force-embedding / padding utilities + photoluminescence workflow.

Improvements vs your current version:
1) Uses the `tolerance` argument (not hardcoded).
2) PBC-aware distance using the supercell minimum-image convention (optional).
3) Nearest-neighbour mapping (choose closest match within tolerance).
4) Optional 1-to-1 (bijective) mapping so the same small-cell atom cannot be reused.
5) Optional writing of a "synthetic OUTCAR-like" TOTAL-FORCE block for downstream tools.
6) Safer structure reader (returns lattice + cartesian positions).
7) Optional consistency checks for species order via POSCAR (recommended).

Notes / assumptions:
- Your unit forces/positions come from the unit OUTCAR TOTAL-FORCE block (cartesian Å, eV/Å).
- The positions in unit OUTCAR are assumed to be in the SAME global coordinate frame
  as the target supercell positions (common in embedding workflows where you keep the defect region aligned).
- PBC minimum-image is applied in the SUPERcell lattice only (usually what you want for mapping near boundaries).
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Helpers: POSCAR/CONTCAR reader
# =============================================================================

def read_poscar(path):
    """
    Read a VASP POSCAR/CONTCAR.
    Returns:
        lattice (3,3) in Å
        species (list[str])
        counts (list[int])
        positions_cart (N,3) in Å (cartesian)
    """
    with open(path, "r") as f:
        lines = [ln.rstrip() for ln in f]

    scale = float(lines[1].split()[0])
    lattice = np.array([lines[i].split() for i in range(2, 5)], dtype=float) * scale

    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    n_atoms = sum(counts)

    coord_line = lines[7].strip().lower()
    # Handle optional "Selective dynamics"
    selective = False
    if coord_line.startswith("s"):
        selective = True
        coord_line = lines[8].strip().lower()
        pos_start = 9
    else:
        pos_start = 8

    is_direct = coord_line.startswith("d")
    positions = []
    for i in range(pos_start, pos_start + n_atoms):
        parts = lines[i].split()
        positions.append([float(parts[0]), float(parts[1]), float(parts[2])])
    positions = np.array(positions, dtype=float)

    if is_direct:
        # Wrap slightly >1 due to numerical noise
        positions = positions - np.floor(positions)
        positions_cart = positions @ lattice
    else:
        positions_cart = positions

    return lattice, species, counts, positions_cart


# =============================================================================
# Helpers: OUTCAR readers (forces + positions, ions per type)
# =============================================================================

def extract_ions_per_type(outcar_path):
    with open(outcar_path, "r") as f:
        for line in f:
            if "ions per type" in line:
                return [int(x) for x in line.split("=")[1].split()]
    raise ValueError("'ions per type' line not found in OUTCAR.")


def read_last_total_force_block(outcar_path):
    """
    Reads the *last* TOTAL-FORCE block from OUTCAR.

    Returns:
        positions_cart (N,3) in Å
        forces (N,3) in eV/Å
    """
    lines_buffer = []
    collecting = False

    with open(outcar_path, "r") as f:
        for line in f:
            if "TOTAL-FORCE" in line:
                lines_buffer = []
                collecting = True
                continue
            if collecting:
                if "total drift:" in line:
                    collecting = False
                    continue
                lines_buffer.append(line.rstrip("\n"))

    # Parse the last collected block
    # Typical format:
    #  (header line)
    #  x y z fx fy fz
    #  ...
    #  (blank)
    positions = []
    forces = []

    # skip first header line; ignore final blank if present
    for line in lines_buffer[1:]:
        parts = line.split()
        if len(parts) == 6:
            positions.append([float(parts[0]), float(parts[1]), float(parts[2])])
            forces.append([float(parts[3]), float(parts[4]), float(parts[5])])

    if len(positions) == 0:
        raise ValueError(f"No parsable TOTAL-FORCE block found in {outcar_path}")

    return np.array(positions, dtype=float), np.array(forces, dtype=float)


# =============================================================================
# Minimum-image distance under supercell PBC
# =============================================================================

def minimum_image_displacement(super_lattice, dr_cart):
    """
    Apply minimum image convention for displacement vectors under supercell lattice.

    Inputs:
        super_lattice: (3,3) lattice in Å
        dr_cart: (...,3) cart displacement in Å

    Returns:
        dr_mic_cart: (...,3) cart displacement after wrapping in fractional space
    """
    invA = np.linalg.inv(super_lattice)
    dr_frac = dr_cart @ invA  # (...,3)
    dr_frac_wrapped = dr_frac - np.round(dr_frac)  # wrap to [-0.5, 0.5)
    return dr_frac_wrapped @ super_lattice


# =============================================================================
# Force embedding / padding
# =============================================================================

def build_unit_to_target_force_mapping(
    unit_outcar,
    target_poscar,
    tolerance=9e-2,
    pbc=True,
    bijective=True,
    unit_poscar_for_species_check=None,
):
    """
    Build one atom-index mapping from a small/unit OUTCAR into a large target POSCAR.

    Returns:
        target_to_unit: (N_target,) array.
                        target_to_unit[i] is the matching unit atom index, or -1 if unmatched.
        target_positions: (N_target,3) cartesian positions from target POSCAR, in target order.
        mapping_info: dict with matching statistics.

    Important:
      - The returned target_to_unit array is in the target POSCAR ordering.
      - Reusing this same mapping for ES and GS forces guarantees that each
        F_es[i] - F_gs[i] compares the same small-cell atom index.
    """
    super_lattice, super_species, super_counts, super_pos = read_poscar(target_poscar)

    if unit_poscar_for_species_check is not None:
        _, unit_species, _, _ = read_poscar(unit_poscar_for_species_check)
        if unit_species != super_species:
            raise ValueError(
                "Species order mismatch between unit POSCAR and target POSCAR.\n"
                f"Unit:   {unit_species}\n"
                f"Target: {super_species}\n"
                "Fix: ensure POSCAR species lines are consistent, or disable this check."
            )

    unit_counts = extract_ions_per_type(unit_outcar)
    unit_pos, _ = read_last_total_force_block(unit_outcar)

    if sum(unit_counts) != unit_pos.shape[0]:
        raise ValueError(
            f"Unit OUTCAR ions-per-type sum ({sum(unit_counts)}) != "
            f"TOTAL-FORCE atom count ({unit_pos.shape[0]})."
        )
    if sum(super_counts) != super_pos.shape[0]:
        raise ValueError("Target POSCAR atom count inconsistent.")
    if len(unit_counts) != len(super_counts):
        raise ValueError(
            "Different number of species types between unit OUTCAR and target POSCAR.\n"
            f"Unit (ions per type): {unit_counts}\n"
            f"Target (POSCAR):      {super_counts}\n"
            "Fix: ensure same species ordering/types in both."
        )

    target_to_unit = np.full(super_pos.shape[0], -1, dtype=int)
    super_offsets = np.cumsum([0] + super_counts)
    unit_offsets = np.cumsum([0] + unit_counts)

    used_target = np.zeros(super_pos.shape[0], dtype=bool) if bijective else None
    matched = 0
    unmatched = 0
    reused_prevented = 0
    max_distance = 0.0

    for k in range(len(super_counts)):
        i0, i1 = super_offsets[k], super_offsets[k + 1]
        j0, j1 = unit_offsets[k], unit_offsets[k + 1]

        super_block = super_pos[i0:i1]
        unit_block_pos = unit_pos[j0:j1]

        for jj in range(unit_block_pos.shape[0]):
            r_unit = unit_block_pos[jj]
            dr = super_block - r_unit[None, :]

            if pbc:
                dr = minimum_image_displacement(super_lattice, dr)

            d2 = np.einsum("ij,ij->i", dr, dr)
            order = np.argsort(d2)

            found_target = False
            for cand_local in order:
                d_best = float(np.sqrt(d2[cand_local]))
                if d_best > tolerance:
                    break

                cand_global = i0 + int(cand_local)
                if bijective and used_target[cand_global]:
                    reused_prevented += 1
                    continue

                target_to_unit[cand_global] = j0 + jj
                matched += 1
                max_distance = max(max_distance, d_best)
                if bijective:
                    used_target[cand_global] = True
                found_target = True
                break

            if not found_target:
                unmatched += 1

    info = {
        "matched_unit_atoms": matched,
        "unmatched_unit_atoms": unmatched,
        "zero_force_target_atoms": int(np.sum(target_to_unit < 0)),
        "tolerance_A": tolerance,
        "pbc": pbc,
        "bijective": bijective,
        "reused_prevented": reused_prevented,
        "max_match_distance_A": max_distance,
        "n_target": super_pos.shape[0],
        "n_unit": unit_pos.shape[0],
    }

    return target_to_unit, super_pos, info


def apply_force_embedding_mapping(unit_outcar, target_to_unit):
    """
    Apply an existing target->unit mapping to one OUTCAR force array.

    The returned force array is ordered exactly like the target POSCAR used to
    build target_to_unit. Unmatched target atoms receive zero force.
    """
    _, unit_forces = read_last_total_force_block(unit_outcar)

    matched_unit_indices = target_to_unit[target_to_unit >= 0]
    if matched_unit_indices.size and np.max(matched_unit_indices) >= unit_forces.shape[0]:
        raise ValueError(
            f"Mapping refers to unit atom index {int(np.max(matched_unit_indices))}, "
            f"but {unit_outcar} only has {unit_forces.shape[0]} force rows."
        )

    forces_target = np.zeros((target_to_unit.shape[0], 3), dtype=float)
    matched_targets = target_to_unit >= 0
    forces_target[matched_targets] = unit_forces[target_to_unit[matched_targets]]
    return forces_target


def embed_forces_from_unit_outcar(
    unit_outcar,
    target_poscar,
    tolerance=9e-2,
    pbc=True,
    bijective=True,
    unit_poscar_for_species_check=None,
    write_synthetic_outcar_path=None,
):
    """
    Map forces from a small "unit" OUTCAR into a larger target POSCAR/CONTCAR.

    Algorithm:
      - Build a target-POSCAR-order mapping from matched target atoms to unit atoms.
      - Apply that mapping to the unit OUTCAR forces.
      - Matched target atoms inherit the corresponding unit force.
      - Unmatched target atoms keep force = 0.

    Parameters:
      tolerance: Å
      pbc: apply minimum-image on the supercell lattice for distance evaluation
      bijective: prevent assigning multiple unit atoms to the same target atom
      unit_poscar_for_species_check: if provided, verify species labels and ordering match target
      write_synthetic_outcar_path: if set, write a minimal OUTCAR-like TOTAL-FORCE block with
                                   positions (target) and embedded forces.

    Returns:
      forces_target (N_target, 3)
      mapping_info dict with stats
    """
    target_to_unit, super_pos, info = build_unit_to_target_force_mapping(
        unit_outcar=unit_outcar,
        target_poscar=target_poscar,
        tolerance=tolerance,
        pbc=pbc,
        bijective=bijective,
        unit_poscar_for_species_check=unit_poscar_for_species_check,
    )
    forces_target = apply_force_embedding_mapping(unit_outcar, target_to_unit)

    if write_synthetic_outcar_path is not None:
        write_minimal_outcar_total_force_block(
            write_synthetic_outcar_path,
            super_pos,
            forces_target,
            title="SYNTHETIC OUTCAR (force-embedded)",
        )

    return forces_target, info


def write_minimal_outcar_total_force_block(path, positions_cart, forces, title="SYNTHETIC OUTCAR"):
    """
    Write a minimal "OUTCAR-like" file containing a TOTAL-FORCE block.
    Many post-processing scripts only search for 'TOTAL-FORCE' and then parse x y z fx fy fz.
    """
    with open(path, "w") as f:
        f.write(f"{title}\n")
        f.write(" TOTAL-FORCE (eV/Angst)\n")
        f.write(" -------------------------------------------------------------------\n")
        f.write("    POSITION                                       TOTAL-FORCE\n")
        f.write(" -------------------------------------------------------------------\n")
        for r, ff in zip(positions_cart, forces):
            f.write(f" {r[0]:16.8f} {r[1]:16.8f} {r[2]:16.8f} {ff[0]:16.8f} {ff[1]:16.8f} {ff[2]:16.8f}\n")
        f.write(" -------------------------------------------------------------------\n")
        f.write(" total drift:   0.000000  0.000000  0.000000\n")


# =============================================================================
# Analytic monopole tail for charged (Delta q != 0) transitions
# =============================================================================
#
# Physics
# -------
# For a transition that moves charge between a localised defect level and the
# delocalised host bands (Delta q = +-1, e.g. C_B or C_N in h-BN) the far field
# of the transition force field dF = F_ex - F_gs is the Coulomb field of the
# transferred charge acting on the Born effective charges of the host,
#
#     dF_a(r)  ~=  [ Delta q * Z*_a * e^2 / (4 pi eps0 eps) ] * r_hat / r^2 ,
#
# with Z*_B ~= -Z*_N in h-BN, so the field is radial, decays as 1/r^2 and is
# equal and opposite on the two sublattices. A 1/r^2 field is never negligible
# at any affordable reference supercell size, so truncating dF at the reference
# boundary -- which is exactly what apply_force_embedding_mapping does when it
# zeroes the unmatched target atoms -- leaves a step discontinuity.
#
# That discontinuity does NOT break the q = 0 acoustic sum rule: with
# bijective=True the sum over the target equals the sum over the reference, and
# the unmatched atoms contribute exactly zero, so |sum_a dF_a| stays at the
# ~1e-5 eV/A that VASP delivers. The damage is at the smallest NON-ZERO q of the
# target, in the structure factor
#
#     F~(q) = | sum_a m_a^(-1/2) dF_a exp(i q.R_a) | ,
#
# and the 1/E^2 in ConfigCoordinatesF (giving S_k ~ proj^2 / E^3 once
# S_k = 2 pi f q_k^2 is folded in) amplifies that into spurious weight in the
# acoustic / flexural region of S(E).
#
# The functions below continue dF analytically past the reference boundary
# instead of truncating it. Re-imposing sum_a dF_a = 0 afterwards is necessary
# hygiene -- the tail breaks a condition that currently holds -- but it is not
# by itself the fix.
#
# The neutral C_B-C_N dimer (Delta q = 0) has no monopole term and must not be
# corrected this way.


def species_labels_from_counts(species, counts):
    """
    Expand a POSCAR species/counts pair into a per-atom label array.

    The Born effective charge, and hence the monopole amplitude, is a property
    of the species, so every step of the tail fit works per sublattice.

    Returns:
        labels: (N,) object array of species strings, in POSCAR order.
    """
    if len(species) != len(counts):
        raise ValueError(
            f"species/counts length mismatch: {len(species)} species vs {len(counts)} counts."
        )
    labels = []
    for s, n in zip(species, counts):
        labels.extend([s] * int(n))
    return np.array(labels, dtype=object)


def locate_minority_species_atom(species, counts):
    """
    Locate the defect as the unique atom of the least abundant species.

    For a substitutional carbon monomer in h-BN the POSCAR carries a single C
    among many B and N, so the minority species is the defect. A dimer (two C)
    is rejected here, which is correct: Delta q = 0 and there is no monopole.

    Returns:
        (index, label) with index in POSCAR order.
    """
    counts_i = [int(c) for c in counts]
    n_min = min(counts_i)
    winners = [k for k, c in enumerate(counts_i) if c == n_min]
    if len(winners) != 1:
        raise ValueError(
            "Cannot locate the defect automatically: several species tie for least "
            f"abundant ({[species[k] for k in winners]}, count {n_min}). "
            "Pass tail_defect_index explicitly."
        )
    k = winners[0]
    if counts_i[k] != 1:
        raise ValueError(
            f"Cannot locate the defect automatically: minority species '{species[k]}' has "
            f"{counts_i[k]} atoms, not 1. A monopole tail is defined for a single charged "
            "point defect; a C-C dimer has Delta q = 0 and needs no tail. "
            "Pass tail_defect_index explicitly if this really is a monomer."
        )
    return int(np.sum(counts_i[:k])), species[k]


def minimum_image_radius(lattice, positions=None):
    """
    Largest radius at which the periodic field has not yet wrapped onto itself.

    Half the smallest perpendicular width of the cell. Beyond this radius the
    monopole field of the defect overlaps its own periodic images and neither
    the fit nor the analytic fill is meaningful there.

    If positions are given, lattice directions along which the atoms span less
    than half the cell are excluded: no pair of atoms can then be more than half
    a cell apart along that direction, so it never limits anything. This is what
    removes the vacuum direction of a slab, which would otherwise dominate the
    minimum and report a spuriously small radius.
    """
    lattice = np.asarray(lattice, dtype=float)
    volume = abs(float(np.linalg.det(lattice)))

    populated = [True, True, True]
    if positions is not None:
        frac = np.asarray(positions, dtype=float) @ np.linalg.inv(lattice)
        span = frac.max(axis=0) - frac.min(axis=0)
        populated = [bool(s >= 0.5) for s in span]

    widths = []
    for i in range(3):
        if not populated[i]:
            continue
        j, k = (i + 1) % 3, (i + 2) % 3
        area = float(np.linalg.norm(np.cross(lattice[j], lattice[k])))
        widths.append(volume / area)

    if not widths:
        raise ValueError(
            "No lattice direction is populated by atoms; cannot define a minimum-image radius."
        )
    return 0.5 * min(widths)


def is_orthogonal_lattice(lattice, tolerance=1e-8):
    """
    True if the lattice vectors are mutually orthogonal.

    Checked on the normalised off-diagonal terms of the metric tensor, so it is
    scale free. The production h-BN cells are the rectangular 4-atom
    representation (b = sqrt(3) a) and are strictly orthorhombic; the primitive
    hexagonal-vector representation is not.
    """
    lattice = np.asarray(lattice, dtype=float)
    metric = lattice @ lattice.T
    norms = np.sqrt(np.diag(metric))
    for i in range(3):
        for j in range(i + 1, 3):
            if abs(metric[i, j]) / (norms[i] * norms[j]) > float(tolerance):
                return False
    return True


def defect_displacements(lattice, positions, defect_index, exact_minimum_image=True):
    """
    Displacements of every atom from the defect, under the true minimum image.

    This starts from the existing minimum_image_displacement, which wraps the
    fractional coordinates by rounding. For an ORTHOGONAL lattice that rule is
    provably exact: the axes do not mix, so minimising |dx|, |dy| and |dz|
    independently minimises the norm. The production cells are orthorhombic, so
    the refinement below is a no-op on them and is asserted to be one.

    It is not exact for a general cell, where the Voronoi region does not
    coincide with the fractional parallelepiped. In the primitive
    hexagonal-vector representation of the same material, rounding misassigns
    about a fifth of the atoms in a 6-16 A window, and where it errs the two
    candidate r_hat are close to perpendicular. That would not matter for atom
    matching, which only asks whether a distance is below ~0.09 A, but the
    monopole fit and fill need r and r_hat themselves. So the wrapped
    displacement is refined by searching the 27 nearest periodic images and
    keeping the shortest, which costs nothing and is correct for either cell
    choice. Set exact_minimum_image=False to reproduce the plain rounding rule.

    Returns:
        R (N,3) displacements in A, r (N,) their norms, n_refined (int) the
        number of atoms whose distance the refinement actually shortened.
    """
    lattice = np.asarray(lattice, dtype=float)
    positions = np.asarray(positions, dtype=float)

    R = minimum_image_displacement(lattice, positions - positions[int(defect_index)])

    n_refined = 0
    if exact_minimum_image:
        r_rounded = np.linalg.norm(R, axis=1)
        shifts = np.array(
            [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
            dtype=float,
        ) @ lattice
        candidates = R[:, None, :] + shifts[None, :, :]
        d2 = np.einsum("nsx,nsx->ns", candidates, candidates)
        best = np.argmin(d2, axis=1)
        R = candidates[np.arange(R.shape[0]), best]
        r_exact = np.linalg.norm(R, axis=1)

        # Count only genuine shortenings. An atom exactly on a cell face has two
        # equidistant images and the search may return the other one; that is a
        # tie, not a refinement.
        scale = max(float(np.max(r_rounded)), 1.0)
        n_refined = int(np.sum(r_rounded - r_exact > 1e-9 * scale))

        if is_orthogonal_lattice(lattice):
            worst = float(np.max(np.abs(r_rounded - r_exact)))
            if worst > 1e-9 * scale:
                raise ValueError(
                    "Internal error: on an orthogonal lattice the 27-image search must agree "
                    "with fractional rounding to machine precision, because minimising each "
                    f"coordinate independently minimises the norm. Largest disagreement "
                    f"{worst:.3e} A over {n_refined} atoms. This is a bug in the image search, "
                    "not a correction to the rounding rule."
                )

    return R, np.linalg.norm(R, axis=1), n_refined


def image_ambiguity(lattice, positions, defect_index, atom_mask=None, tolerance=0.02):
    """
    Count atoms for which "the" nearest periodic image of the defect is ambiguous.

    Beyond the minimum-image radius two or more images of the defect can be
    equidistant to within numerical noise. The fill points away from one of them,
    so its direction r_hat there is arbitrary at the level of the degeneracy: on
    a cell face the two candidates are related by a symmetry operation and their
    fields differ by a large angle. This does not affect atoms inside the
    minimum-image radius, where the nearest image is unique by construction.

    Returns:
        n_ambiguous: atoms whose second-nearest image is within `tolerance`
                     (relative) of the nearest.
    """
    lattice = np.asarray(lattice, dtype=float)
    positions = np.asarray(positions, dtype=float)

    R = minimum_image_displacement(lattice, positions - positions[int(defect_index)])
    if atom_mask is not None:
        R = R[np.asarray(atom_mask, dtype=bool)]
    if R.shape[0] == 0:
        return 0

    shifts = np.array(
        [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float
    ) @ lattice
    candidates = R[:, None, :] + shifts[None, :, :]
    d = np.sort(np.linalg.norm(candidates, axis=2), axis=1)
    nearest, second = d[:, 0], d[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(nearest > 0, (second - nearest) / nearest, np.inf)
    return int(np.sum(rel < float(tolerance)))


def enforce_force_sum_rule(dF):
    """
    Re-impose the acoustic (translational) sum rule sum_a dF_a = 0.

    Phonopy's band.yaml holds eigenvectors of the mass-weighted dynamical
    matrix, so the Gamma acoustic modes have e_a ~ sqrt(m_a). Since
    ConfigCoordinatesF forms sum_a m_a^(-1/2) dF_a . e_a, the acoustic
    projection is proportional to sum_a dF_a and subtracting the plain (not
    mass-weighted) mean annihilates it exactly.

    Returns:
        dF_corrected (N,3), stats dict with |sum dF| before and after.
    """
    dF = np.asarray(dF, dtype=float)
    net_before = dF.sum(axis=0)
    dF_corrected = dF - net_before / dF.shape[0]
    net_after = dF_corrected.sum(axis=0)
    stats = {
        "net_force_before": net_before.copy(),
        "net_force_after": net_after.copy(),
        "abs_net_force_before_eVA": float(np.linalg.norm(net_before)),
        "abs_net_force_after_eVA": float(np.linalg.norm(net_after)),
    }
    return dF_corrected, stats


def _fit_radial_amplitude(dF_rad, r):
    """
    One zero-intercept least squares fit of dF_rad against r^-2.

    A = sum(dF_rad * r^-2) / sum(r^-4), plus the R^2 of that fit and an
    independent log-log slope, which is the honest test of the assumed
    exponent. Shared by the full-window fit and the sub-window drift test so
    the two can never disagree on method.
    """
    dF_rad = np.asarray(dF_rad, dtype=float)
    r = np.asarray(r, dtype=float)
    n = int(r.size)
    if n == 0:
        return {
            "n_atoms": 0, "amplitude_eVA": 0.0, "r_squared": float("nan"),
            "loglog_exponent": float("nan"), "n_sign_consistent": 0,
        }

    x = r ** -2.0
    amplitude = float(np.sum(dF_rad * x) / np.sum(x ** 2))

    ss_res = float(np.sum((dF_rad - amplitude * x) ** 2))
    ss_tot = float(np.sum((dF_rad - np.mean(dF_rad)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    same_sign = (
        dF_rad * np.sign(amplitude) > 0
        if amplitude != 0.0
        else np.zeros_like(dF_rad, dtype=bool)
    )
    if int(np.sum(same_sign)) >= 3:
        slope, _ = np.polyfit(np.log(r[same_sign]), np.log(np.abs(dF_rad[same_sign])), 1)
        exponent = float(slope)
    else:
        exponent = float("nan")

    return {
        "n_atoms": n,
        "amplitude_eVA": amplitude,
        "r_squared": r_squared,
        "loglog_exponent": exponent,
        "n_sign_consistent": int(np.sum(same_sign)),
    }


def fit_monopole_tail(
    dF,
    positions,
    target_to_unit,
    defect_index,
    species,
    lattice,
    r_fit=(6.0, 16.0),
    delta_q=None,
    min_atoms_per_species=20,
    min_matched_r_max=18.0,
    exponent_bounds=(-2.4, -1.6),
    sub_windows=((6.0, 9.0), (9.0, 12.0), (12.0, 16.0)),
    sub_window_min_atoms=8,
    drift_tolerance_percent=5.0,
    drift_noise_floor_percent=1.0,
    exact_minimum_image=True,
    verbose=True,
):
    """
    Fit the monopole amplitude A_s of the 1/r^2 far field of dF, per species.

    Physics: dF_rad,a = dF_a . R_hat_a is expected to follow A_s / r^2 with
    A_s proportional to Delta q times the Born effective charge Z*_s, so the
    fit is done separately on each sublattice and A_B / A_N should come out
    near -1. The fit uses only MATCHED atoms (those the reference actually
    supplied) inside a window where the field is monopole dominated: far enough
    from the defect that multipoles and the localised relaxation have died off,
    close enough to the defect that the reference's own periodic cutoff has not
    yet bitten.

    A least squares fit of dF_rad against r^-2 with no intercept gives
    A_s = sum(dF_rad * r^-2) / sum(r^-4). An independent log-log fit of
    log|dF_rad| against log r returns the observed exponent, which is the
    honest test of whether the window is clean: it should come out near -2.

    Returns:
        amplitudes: dict species -> A_s in eV*A (so that dF = A_s * r_hat / r^2).
        fit_info: dict of diagnostics (R^2, atom counts, exponents, A_B/A_N,
                  the effective window, matched r_max, and any warnings).
    """
    dF = np.asarray(dF, dtype=float)
    positions = np.asarray(positions, dtype=float)
    target_to_unit = np.asarray(target_to_unit, dtype=int)
    species = np.asarray(species, dtype=object)
    lattice = np.asarray(lattice, dtype=float)

    n_atoms = positions.shape[0]
    if dF.shape != (n_atoms, 3):
        raise ValueError(f"dF has shape {dF.shape}, expected {(n_atoms, 3)}.")
    if target_to_unit.shape != (n_atoms,):
        raise ValueError(
            f"target_to_unit has shape {target_to_unit.shape}, expected {(n_atoms,)}."
        )
    if species.shape != (n_atoms,):
        raise ValueError(f"species has shape {species.shape}, expected {(n_atoms,)}.")
    if not (0 <= int(defect_index) < n_atoms):
        raise ValueError(f"defect_index {defect_index} out of range for {n_atoms} atoms.")

    R, r, n_refined = defect_displacements(
        lattice, positions, defect_index, exact_minimum_image=exact_minimum_image
    )

    matched = target_to_unit >= 0
    not_defect = np.arange(n_atoms) != int(defect_index)
    if not np.any(matched):
        raise ValueError("No matched atoms: the mapping produced an empty reference region.")

    matched_r_max = float(np.max(r[matched]))
    mic_radius = minimum_image_radius(lattice, positions)

    if matched_r_max < float(min_matched_r_max):
        raise ValueError(
            "Reference supercell is too small to support a monopole fit.\n"
            f"  matched-region r_max = {matched_r_max:.2f} A "
            f"(need >= {float(min_matched_r_max):.2f} A)\n"
            "  A reference this small is contaminated by its own periodic cutoff over the\n"
            "  whole radial range, so no choice of fit window rescues it: measured log-log\n"
            "  exponents over 5-13 A are -2.69 (7x5, r_max 14.35 A) against -1.97 (10x8,\n"
            "  r_max 21.67 A). Shrinking r_fit here would fit the cutoff, not the monopole.\n"
            "  Use a 10x8 or larger reference, or lower min_matched_r_max deliberately."
        )

    r_lo = float(r_fit[0])
    r_hi_allowed = 0.9 * matched_r_max
    if r_fit[1] is None:
        r_hi = min(16.0, r_hi_allowed)
    else:
        r_hi = float(r_fit[1])
        if r_hi > r_hi_allowed:
            raise ValueError(
                f"Fit window upper bound r_fit[1] = {r_hi:.2f} A exceeds 0.9 * matched r_max "
                f"= {r_hi_allowed:.2f} A (matched r_max = {matched_r_max:.2f} A).\n"
                "  The outer 10% of the matched region is already distorted by the reference's\n"
                "  own periodic cutoff. Lower r_fit[1], or pass r_fit=(r_lo, None) to let it\n"
                "  default to min(16.0, 0.9 * matched r_max)."
            )
    if r_hi <= r_lo:
        raise ValueError(f"Empty fit window: r_fit resolved to ({r_lo:.2f}, {r_hi:.2f}) A.")

    warnings = []
    in_window = matched & not_defect & (r >= r_lo) & (r < r_hi)
    if r_hi > mic_radius:
        n_beyond = int(np.sum(in_window & (r > mic_radius)))
        warnings.append(
            f"fit window upper bound {r_hi:.2f} A exceeds the minimum-image radius "
            f"{mic_radius:.2f} A of the target; {n_beyond} of the "
            f"{int(np.sum(in_window))} window atoms lie beyond it, where the defect's "
            "monopole field already overlaps its own periodic images."
        )

    R_hat = np.zeros_like(R)
    nonzero = r > 0
    R_hat[nonzero] = R[nonzero] / r[nonzero, None]
    dF_rad = np.einsum("ij,ij->i", dF, R_hat)

    unmatched = ~matched
    unique_species = [s for s in dict.fromkeys(species.tolist())]

    amplitudes = {}
    per_species = {}
    for s in unique_species:
        is_s = species == s
        sel = in_window & is_s
        n_sel = int(np.sum(sel))
        n_unmatched_s = int(np.sum(unmatched & is_s))

        if n_sel < int(min_atoms_per_species):
            if n_unmatched_s > 0:
                raise ValueError(
                    f"Species '{s}' has only {n_sel} matched atoms in the fit window "
                    f"[{r_lo:.2f}, {r_hi:.2f}) A (need >= {int(min_atoms_per_species)}), "
                    f"but {n_unmatched_s} of its atoms are unmatched and would need an "
                    "analytic tail fitted from that window. Widen r_fit or use a larger "
                    "reference; do not fit an amplitude from this few atoms."
                )
            amplitudes[s] = 0.0
            per_species[s] = {
                "n_atoms_in_window": n_sel,
                "n_unmatched": n_unmatched_s,
                "amplitude_eVA": 0.0,
                "r_squared": float("nan"),
                "loglog_exponent": float("nan"),
                "skipped": True,
            }
            continue

        fit = _fit_radial_amplitude(dF_rad[sel], r[sel])
        amplitudes[s] = fit["amplitude_eVA"]
        per_species[s] = {
            "n_atoms_in_window": n_sel,
            "n_unmatched": n_unmatched_s,
            "amplitude_eVA": fit["amplitude_eVA"],
            "r_squared": fit["r_squared"],
            "loglog_exponent": fit["loglog_exponent"],
            "n_sign_consistent": fit["n_sign_consistent"],
            "skipped": False,
        }

    # ---- sub-window drift -------------------------------------------------
    # A single amplitude A only describes the field if A comes out the same
    # wherever inside the window it is measured. Refitting A independently over
    # consecutive shells tests exactly that, and the direction of any drift says
    # which way the assumption fails. |A| growing with r means the effective
    # screening is weakening outwards, which is what a q-dependent
    # (Rytova-Keldysh) dielectric response does in a layered 2D system, and the
    # single-parameter fill will then under-shoot at large r. |A| shrinking with
    # r means the reference cell's own periodic images are pulling the outer
    # shells down, and the window must move inward.
    fill_r_max = float(np.max(r[unmatched])) if np.any(unmatched) else 0.0
    requested_sub = [
        (float(lo), float(hi)) for lo, hi in (sub_windows or ()) if lo >= r_lo and hi <= r_hi
    ]
    if len(requested_sub) < 2:
        edges = np.linspace(r_lo, r_hi, 4)
        requested_sub = [(float(edges[i]), float(edges[i + 1])) for i in range(3)]
        sub_windows_are_default = False
    else:
        sub_windows_are_default = True

    drift_percent = {}
    for s in unique_species:
        if per_species[s]["skipped"]:
            per_species[s]["sub_windows"] = []
            per_species[s]["drift_percent"] = float("nan")
            continue

        is_s = species == s
        shells = []
        for lo, hi in requested_sub:
            sel_sub = matched & not_defect & is_s & (r >= lo) & (r < hi)
            n_sub = int(np.sum(sel_sub))
            entry = {"r_lo_A": lo, "r_hi_A": hi, "n_atoms": n_sub}
            if n_sub >= int(sub_window_min_atoms):
                entry.update(_fit_radial_amplitude(dF_rad[sel_sub], r[sel_sub]))
                entry["sufficient"] = True
            else:
                entry.update(
                    {"amplitude_eVA": float("nan"), "r_squared": float("nan"),
                     "loglog_exponent": float("nan"), "sufficient": False}
                )
            shells.append(entry)

        usable = [e for e in shells if e["sufficient"]]
        A_global = per_species[s]["amplitude_eVA"]
        monotone = False
        extrapolated = float("nan")
        if len(usable) >= 2 and A_global != 0.0:
            # Positive means |A| grows with r, for either sign of A.
            values = np.array([e["amplitude_eVA"] for e in usable]) / A_global
            d = 100.0 * (values[-1] - values[0])

            # Monotone only counts as a trend once the drift clears the noise
            # floor; three amplitudes agreeing to 1e-6 are "monotone" by accident.
            steps = np.diff(values)
            monotone = bool(
                (np.all(steps > 0) or np.all(steps < 0))
                and abs(d) >= float(drift_noise_floor_percent)
            )

            # The fill region reaches further out than the fit window, so a drift
            # measured inside the window understates the error at the far edge.
            # A straight line through the shell amplitudes, extrapolated to the
            # outermost filled atom, bounds it.
            if fill_r_max > 0 and len(usable) >= 2:
                centres = np.array([0.5 * (e["r_lo_A"] + e["r_hi_A"]) for e in usable])
                slope, intercept = np.polyfit(centres, values, 1)
                extrapolated = 100.0 * (slope * fill_r_max + intercept - 1.0)
        else:
            d = float("nan")

        per_species[s]["sub_windows"] = shells
        per_species[s]["drift_percent"] = d
        per_species[s]["drift_monotone"] = monotone
        per_species[s]["drift_extrapolated_percent"] = extrapolated
        drift_percent[s] = d

    finite_drift = {s: d for s, d in drift_percent.items() if np.isfinite(d)}
    if not finite_drift:
        drift_verdict = "not determined (too few atoms per sub-window)"
    else:
        worst_species = max(finite_drift, key=lambda s: abs(finite_drift[s]))
        worst = finite_drift[worst_species]
        monotone = per_species[worst_species].get("drift_monotone", False)
        extrap = per_species[worst_species].get("drift_extrapolated_percent", float("nan"))
        extrap_note = (
            f"; extrapolated to the outermost filled atom at {fill_r_max:.1f} A that is "
            f"{extrap:+.1f}%"
            if np.isfinite(extrap)
            else ""
        )
        if abs(worst) <= float(drift_tolerance_percent) and not monotone:
            drift_verdict = (
                f"flat to within {float(drift_tolerance_percent):.0f}% (largest |drift| "
                f"{abs(worst):.1f}% on {worst_species}, no monotone trend above the "
                f"{float(drift_noise_floor_percent):.0f}% noise floor): the single-parameter "
                "A/r^2 form is sound"
            )
        elif abs(worst) <= float(drift_tolerance_percent):
            drift_verdict = (
                f"small but MONOTONE: {worst:+.1f}% on {worst_species} across the sub-windows"
                f"{extrap_note}. Below the {float(drift_tolerance_percent):.0f}% tolerance, but a "
                "monotone trend is a trend, not scatter -- quote the extrapolated figure as the "
                "residual error rather than treating the fit as exact"
            )
        elif worst > 0:
            drift_verdict = (
                f"|A| GROWS with r by {worst:+.1f}% on {worst_species}"
                f"{' (monotone)' if monotone else ''}: consistent with weakening screening, i.e. a "
                f"q-dependent dielectric response; the fill under-shoots at large r{extrap_note}"
            )
            warnings.append(
                f"sub-window drift {worst:+.1f}% on species {worst_species}: |A| grows outward, so "
                "a single amplitude under-shoots the far field"
                f"{extrap_note}. Quote that as the residual error, or move to a Rytova-Keldysh form."
            )
        else:
            drift_verdict = (
                f"|A| SHRINKS with r by {worst:+.1f}% on {worst_species}"
                f"{' (monotone)' if monotone else ''}: consistent with the reference cell's own "
                "periodic images pulling the outer shells down; move the fit window inward"
                f"{extrap_note}"
            )
            warnings.append(
                f"sub-window drift {worst:+.1f}% on species {worst_species}: |A| shrinks outward, "
                "which is the signature of reference-image contamination. Move r_fit inward."
            )

    fitted = [s for s in unique_species if not per_species[s]["skipped"]]
    sel_all = in_window & np.isin(species.astype(str), np.array([str(s) for s in fitted]))
    positive = sel_all & (np.abs(dF_rad) > 0)
    if int(np.sum(positive)) >= 3:
        slope, _ = np.polyfit(
            np.log(r[positive]), np.log(np.abs(dF_rad[positive])), 1
        )
        exponent_all = float(slope)
    else:
        exponent_all = float("nan")

    if np.isfinite(exponent_all) and not (
        float(exponent_bounds[0]) <= exponent_all <= float(exponent_bounds[1])
    ):
        warnings.append(
            f"combined log-log exponent {exponent_all:.2f} is outside "
            f"[{exponent_bounds[0]}, {exponent_bounds[1]}]: the fit window is contaminated "
            "or the transition is not monopole dominated."
        )

    ratio_B_over_N = float("nan")
    if "B" in amplitudes and "N" in amplitudes and amplitudes["N"] != 0.0:
        ratio_B_over_N = amplitudes["B"] / amplitudes["N"]
        if abs(ratio_B_over_N + 1.0) > 0.2:
            warnings.append(
                f"A_B / A_N = {ratio_B_over_N:.3f}, expected near -1 "
                "(Z*_B ~= -Z*_N); the two sublattices disagree."
            )

    sign_check = None
    if delta_q is not None and "B" in amplitudes and not per_species["B"]["skipped"]:
        expected = float(np.sign(delta_q))
        observed = float(np.sign(amplitudes["B"]))
        sign_check = {"expected_sign_A_B": expected, "observed_sign_A_B": observed}
        if observed != expected:
            warnings.append(
                "*** SIGN CHECK FAILED *** sign(A_B) = "
                f"{observed:+.0f} but Delta q = {delta_q:+g} requires {expected:+.0f}. "
                "The B-sublattice radial component is positive for C_B (Delta q = +1) and "
                "negative for C_N (Delta q = -1). A flip means either tail_defect_index "
                "points at the wrong atom, or dF is being formed as F_gs - F_ex instead of "
                "F_ex - F_gs. DO NOT TRUST THE TAIL UNTIL THIS IS RESOLVED."
            )

    fit_info = {
        "defect_index": int(defect_index),
        "defect_species": str(species[int(defect_index)]),
        "r_fit_requested_A": (float(r_fit[0]), r_fit[1]),
        "r_fit_effective_A": (r_lo, r_hi),
        "matched_r_max_A": matched_r_max,
        "minimum_image_radius_A": mic_radius,
        "n_matched": int(np.sum(matched)),
        "n_unmatched": int(np.sum(unmatched)),
        "n_atoms_in_window": int(np.sum(in_window)),
        "exact_minimum_image": bool(exact_minimum_image),
        "orthogonal_lattice": bool(is_orthogonal_lattice(lattice)),
        "n_images_refined": n_refined,
        "sub_windows_A": requested_sub,
        "sub_windows_are_requested": sub_windows_are_default,
        "drift_percent": drift_percent,
        "sub_window_drift_verdict": drift_verdict,
        "per_species": per_species,
        "combined_loglog_exponent": exponent_all,
        "ratio_A_B_over_A_N": ratio_B_over_N,
        "delta_q": delta_q,
        "sign_check": sign_check,
        "warnings": warnings,
    }

    if verbose:
        for w in warnings:
            print(f"[monopole tail WARNING] {w}")

    return amplitudes, fit_info


def apply_monopole_tail(
    dF,
    positions,
    target_to_unit,
    defect_index,
    species,
    lattice,
    amplitudes,
    enforce_sum_rule=True,
    exact_minimum_image=True,
):
    """
    Replace the truncation zeros of dF with the fitted analytic monopole tail.

    Every atom the reference did not supply (target_to_unit < 0) currently
    carries dF = 0, which is a step discontinuity in a field that physically
    decays as 1/r^2. Here those atoms instead receive

        dF_a = A_species * R_hat_a / r_a^2

    with R_a the minimum-image displacement from the defect. Matched atoms keep
    their DFT values untouched. Adding the tail injects a net force, so the
    acoustic sum rule sum_a dF_a = 0 is re-imposed afterwards; this is hygiene
    that restores a condition truncation already satisfied, not the correction
    itself.

    Returns:
        dF_corrected (N,3), stats dict.
    """
    dF = np.asarray(dF, dtype=float)
    positions = np.asarray(positions, dtype=float)
    target_to_unit = np.asarray(target_to_unit, dtype=int)
    species = np.asarray(species, dtype=object)
    lattice = np.asarray(lattice, dtype=float)

    n_atoms = positions.shape[0]
    if dF.shape != (n_atoms, 3):
        raise ValueError(f"dF has shape {dF.shape}, expected {(n_atoms, 3)}.")

    R, r, _ = defect_displacements(
        lattice, positions, defect_index, exact_minimum_image=exact_minimum_image
    )

    unmatched = target_to_unit < 0
    n_filled = int(np.sum(unmatched))

    if n_filled and np.any(r[unmatched] <= 0.0):
        raise ValueError(
            "An unmatched atom sits at r = 0 from the defect; the analytic tail diverges "
            "there. Check tail_defect_index."
        )

    missing = sorted({str(s) for s in species[unmatched]} - {str(k) for k in amplitudes})
    if missing:
        raise ValueError(
            f"No fitted amplitude for species {missing}, which have unmatched atoms to fill."
        )

    mic_radius = minimum_image_radius(lattice, positions)
    n_beyond_mic = int(np.sum(unmatched & (r > mic_radius)))
    n_ambiguous = image_ambiguity(lattice, positions, defect_index, atom_mask=unmatched)

    abs_sum_before = float(np.sum(np.abs(dF)))
    net_before = dF.sum(axis=0)

    dF_corrected = dF.copy()
    if n_filled:
        idx = np.where(unmatched)[0]
        A = np.array([amplitudes[str(species[i])] for i in idx], dtype=float)
        R_hat = R[idx] / r[idx, None]
        dF_corrected[idx] = (A / r[idx] ** 2)[:, None] * R_hat

    abs_sum_tail = float(np.sum(np.abs(dF_corrected[unmatched]))) if n_filled else 0.0
    abs_sum_after = float(np.sum(np.abs(dF_corrected)))
    net_after_fill = dF_corrected.sum(axis=0)

    sum_rule_stats = None
    if enforce_sum_rule:
        dF_corrected, sum_rule_stats = enforce_force_sum_rule(dF_corrected)

    stats = {
        "n_atoms_filled": n_filled,
        "filled_r_min_A": float(np.min(r[unmatched])) if n_filled else float("nan"),
        "filled_r_max_A": float(np.max(r[unmatched])) if n_filled else float("nan"),
        "abs_sum_dF_before_eVA": abs_sum_before,
        "abs_sum_dF_after_eVA": abs_sum_after,
        "abs_sum_dF_analytic_region_eVA": abs_sum_tail,
        "analytic_fraction_of_abs_sum": (abs_sum_tail / abs_sum_after) if abs_sum_after > 0 else 0.0,
        "abs_net_force_before_eVA": float(np.linalg.norm(net_before)),
        "abs_net_force_after_fill_eVA": float(np.linalg.norm(net_after_fill)),
        "abs_net_force_final_eVA": (
            sum_rule_stats["abs_net_force_after_eVA"]
            if sum_rule_stats is not None
            else float(np.linalg.norm(net_after_fill))
        ),
        "sum_rule_enforced": bool(enforce_sum_rule),
        "minimum_image_radius_A": mic_radius,
        "n_filled_beyond_minimum_image_radius": n_beyond_mic,
        "n_filled_image_ambiguous": n_ambiguous,
        "amplitudes_eVA": {str(k): float(v) for k, v in amplitudes.items()},
    }
    return dF_corrected, stats


def apply_analytic_tail_to_difference(
    F_es,
    F_gs,
    target_positions,
    target_to_unit,
    target_poscar,
    delta_q,
    defect_index=None,
    fit_window=(6.0, 16.0),
    enforce_sum_rule=True,
    exact_minimum_image=True,
    verbose=True,
):
    """
    Form dF = F_es - F_gs on the target and continue its monopole tail analytically.

    The tail is a property of the DIFFERENCE force field only. Each individual
    state force at large r is dominated by terms that cancel in the difference,
    so the tail must never be added to F_es or F_gs separately. This function is
    the single place where dF is built and corrected.

    Returns:
        dF_corrected (N,3), info dict {fit_info, apply_stats, defect_index, ...}.
    """
    if delta_q is None:
        raise ValueError(
            "analytic_tail=True requires an explicit tail_delta_q of +1 or -1 "
            "(+1 for C_B, -1 for C_N). The amplitude itself is fitted, not derived from "
            "Delta q; the value is used to check the sign of the fitted field."
        )
    if float(delta_q) == 0.0:
        raise ValueError(
            "tail_delta_q = 0 is a neutral (localised-to-localised) transition, such as the "
            "C_B-C_N dimer. A Delta q = 0 transition has no monopole term: its transition "
            "force field decays as ~r^-3 and converges properly inside the reference cell, "
            "so truncation is already the right boundary condition and no analytic tail is "
            "needed or defined. Run with analytic_tail=False."
        )
    if abs(abs(float(delta_q)) - 1.0) > 1e-9:
        raise ValueError(
            f"tail_delta_q = {delta_q} is not +1 or -1. Only single-electron charged "
            "transitions are wired here."
        )

    F_es = np.asarray(F_es, dtype=float)
    F_gs = np.asarray(F_gs, dtype=float)
    if F_es.shape != F_gs.shape:
        raise ValueError(f"F_es shape {F_es.shape} != F_gs shape {F_gs.shape}.")

    lattice, species_names, counts, _ = read_poscar(target_poscar)
    labels = species_labels_from_counts(species_names, counts)
    if labels.shape[0] != F_es.shape[0]:
        raise ValueError(
            f"Target POSCAR {target_poscar} has {labels.shape[0]} atoms but the embedded "
            f"force arrays have {F_es.shape[0]} rows."
        )

    if defect_index is None:
        defect_index, defect_label = locate_minority_species_atom(species_names, counts)
        auto_located = True
    else:
        defect_index = int(defect_index)
        defect_label = str(labels[defect_index])
        auto_located = False

    dF = F_es - F_gs

    amplitudes, fit_info = fit_monopole_tail(
        dF=dF,
        positions=target_positions,
        target_to_unit=target_to_unit,
        defect_index=defect_index,
        species=labels,
        lattice=lattice,
        r_fit=fit_window,
        delta_q=delta_q,
        exact_minimum_image=exact_minimum_image,
        verbose=verbose,
    )

    dF_corrected, apply_stats = apply_monopole_tail(
        dF=dF,
        positions=target_positions,
        target_to_unit=target_to_unit,
        defect_index=defect_index,
        species=labels,
        lattice=lattice,
        amplitudes=amplitudes,
        enforce_sum_rule=enforce_sum_rule,
        exact_minimum_image=exact_minimum_image,
    )

    info = {
        "defect_index": defect_index,
        "defect_species": defect_label,
        "defect_auto_located": auto_located,
        "delta_q": float(delta_q),
        "amplitudes_eVA": {str(k): float(v) for k, v in amplitudes.items()},
        "fit_info": fit_info,
        "apply_stats": apply_stats,
    }

    if verbose:
        print_monopole_tail_summary(info)

    return dF_corrected, info


def print_monopole_tail_summary(info):
    """Print the fit and application diagnostics as one block."""
    fit = info["fit_info"]
    app = info["apply_stats"]
    lo, hi = fit["r_fit_effective_A"]

    print("")
    print("=" * 74)
    print(" ANALYTIC MONOPOLE TAIL  (dF = F_es - F_gs, charged transition)")
    print("=" * 74)
    print(
        f"  defect atom            : index {info['defect_index']} ({info['defect_species']}), "
        f"{'auto-located' if info['defect_auto_located'] else 'user-specified'}"
    )
    print(f"  Delta q                : {info['delta_q']:+g}")
    print(f"  matched / unmatched    : {fit['n_matched']} / {fit['n_unmatched']} atoms")
    print(f"  matched region r_max   : {fit['matched_r_max_A']:.2f} A")
    print(f"  minimum-image radius   : {fit['minimum_image_radius_A']:.2f} A")
    print(
        f"  minimum image          : {'exact (27-image search)' if fit['exact_minimum_image'] else 'fractional rounding'}"
        f", {fit['n_images_refined']} atoms refined"
    )
    print(f"  fit window (effective) : [{lo:.2f}, {hi:.2f}) A, {fit['n_atoms_in_window']} atoms")
    print("")
    print("  per-species fit of dF_rad = A / r^2")
    print(f"    {'species':<9}{'N_fit':>7}{'A [eV*A]':>15}{'R^2':>10}{'exponent':>11}")
    for s, d in fit["per_species"].items():
        if d["skipped"]:
            print(f"    {str(s):<9}{d['n_atoms_in_window']:>7}{'skipped (no unmatched atoms)':>36}")
        else:
            print(
                f"    {str(s):<9}{d['n_atoms_in_window']:>7}{d['amplitude_eVA']:>15.6f}"
                f"{d['r_squared']:>10.4f}{d['loglog_exponent']:>11.3f}"
            )
    print("")
    print("  sub-window drift of A  (is one amplitude enough?)")
    print(f"    {'species':<9}{'window [A]':>14}{'N':>6}{'A [eV*A]':>15}{'R^2':>9}{'exponent':>11}")
    for s, d in fit["per_species"].items():
        if d["skipped"]:
            continue
        for e in d.get("sub_windows", []):
            win = f"{e['r_lo_A']:.1f} - {e['r_hi_A']:.1f}"
            if e["sufficient"]:
                print(
                    f"    {str(s):<9}{win:>14}{e['n_atoms']:>6}{e['amplitude_eVA']:>15.6f}"
                    f"{e['r_squared']:>9.4f}{e['loglog_exponent']:>11.3f}"
                )
            else:
                print(f"    {str(s):<9}{win:>14}{e['n_atoms']:>6}{'too few atoms':>35}")
        if np.isfinite(d.get("drift_percent", float("nan"))):
            tag = "monotone" if d.get("drift_monotone") else "scatter"
            extra = d.get("drift_extrapolated_percent", float("nan"))
            line = f"    {str(s):<9}{'drift':>14}{'':>6}{d['drift_percent']:>14.1f}%  ({tag}"
            if np.isfinite(extra):
                line += f", {extra:+.1f}% extrapolated to the fill edge"
            print(line + ")")
    print(f"    verdict: {fit['sub_window_drift_verdict']}")
    print("")
    print(f"  combined log-log exponent : {fit['combined_loglog_exponent']:.3f}  (expect ~ -2)")
    print(f"  A_B / A_N                 : {fit['ratio_A_B_over_A_N']:.3f}  (expect ~ -1)")
    if fit["sign_check"] is not None:
        ok = fit["sign_check"]["observed_sign_A_B"] == fit["sign_check"]["expected_sign_A_B"]
        print(f"  sign(A_B) vs Delta q      : {'OK' if ok else 'FAILED'}")
    print("")
    print("  application")
    print(f"    atoms filled analytically : {app['n_atoms_filled']}")
    print(
        f"    their radial range        : {app['filled_r_min_A']:.2f} - "
        f"{app['filled_r_max_A']:.2f} A"
    )
    print(f"    sum|dF| before / after    : {app['abs_sum_dF_before_eVA']:.6f} / "
          f"{app['abs_sum_dF_after_eVA']:.6f} eV/A")
    print(f"    carried by analytic region: {100.0 * app['analytic_fraction_of_abs_sum']:.2f} %")
    if app["n_atoms_filled"]:
        print(
            f"    beyond minimum-image radius: {app['n_filled_beyond_minimum_image_radius']} of "
            f"{app['n_atoms_filled']}  (radius {app['minimum_image_radius_A']:.2f} A)"
        )
        print(
            f"    nearest image ambiguous   : {app['n_filled_image_ambiguous']} of "
            f"{app['n_atoms_filled']}"
        )
    print(f"    |sum dF| before fill      : {app['abs_net_force_before_eVA']:.3e} eV/A")
    print(f"    |sum dF| after fill       : {app['abs_net_force_after_fill_eVA']:.3e} eV/A")
    print(
        f"    |sum dF| final            : {app['abs_net_force_final_eVA']:.3e} eV/A"
        f"  (sum rule {'enforced' if app['sum_rule_enforced'] else 'NOT enforced'})"
    )
    extra = list(fit["warnings"])
    if app["n_atoms_filled"] and app["n_filled_beyond_minimum_image_radius"] > 0:
        extra.append(
            f"{app['n_filled_beyond_minimum_image_radius']} of {app['n_atoms_filled']} filled "
            f"atoms lie beyond the minimum-image radius {app['minimum_image_radius_A']:.2f} A, "
            "where the target's own periodicity would matter if it were physical. It is not: "
            "the target supercell is a device for generating a dense phonon spectrum, and the "
            "field being continued is that of one isolated defect, so a single-image fill is "
            "what is wanted and a lattice sum would reintroduce the defect-image interaction "
            f"this correction exists to remove. For {app['n_filled_image_ambiguous']} of them "
            "two images are equidistant to within 2%, so which one the fill points away from "
            "is arbitrary."
        )
    if extra:
        print("")
        print("  warnings")
        for w in extra:
            print(f"    - {w}")
    print("=" * 74)
    print("")


# =============================================================================
# Criterion C3: the difference-force structure factor at small finite q
# =============================================================================

def smallest_nonzero_q(lattice, n_q=5, include_out_of_plane=False, n_max=3):
    """
    The n_q smallest non-zero reciprocal vectors of the target supercell.

    Truncating dF at the reference boundary leaves F~(q = 0) intact (bijective
    mapping preserves the sum rule) while distorting F~(q) at the smallest
    non-zero q the target supports. Those are the q that matter, so they are the
    ones to look at.

    For a slab the third lattice vector is vacuum and its reciprocal vector is
    not a physical wavevector, so out-of-plane components are excluded by
    default.

    Returns:
        q_vectors (n_q,3) in 1/A, hkl (n_q,3) integer indices.
    """
    lattice = np.asarray(lattice, dtype=float)
    b = 2.0 * np.pi * np.linalg.inv(lattice).T  # rows b[i], b[i].a[j] = 2 pi delta_ij

    rng = range(-int(n_max), int(n_max) + 1)
    third = rng if include_out_of_plane else [0]
    # q and -q give the same |F~(q)|, so only one of each pair is kept: the one
    # whose first non-zero index is positive.
    hkl = np.array(
        [
            [h, k, l]
            for h in rng
            for k in rng
            for l in third
            if (h, k, l) != (0, 0, 0) and next(v for v in (h, k, l) if v != 0) > 0
        ],
        dtype=int,
    )
    q = hkl @ b
    order = np.argsort(np.linalg.norm(q, axis=1))[: int(n_q)]
    return q[order], hkl[order]


def force_structure_factor(dF, masses, positions, q_vectors):
    """
    |F~(q)| = | sum_a m_a^(-1/2) dF_a exp(i q.R_a) |, criterion C3.

    This is the quantity the analytic tail actually targets: it is the
    mass-weighted difference force resolved at wavevector q, which is what the
    projection onto a phonon of wavevector q sees. At q = 0 it reduces to the
    acoustic sum rule. Units follow dF and masses (eV/A/sqrt(amu)).

    Returns:
        (n_q,) array of |F~(q)|.
    """
    dF = np.asarray(dF, dtype=float)
    masses = np.asarray(masses, dtype=float)
    positions = np.asarray(positions, dtype=float)
    q_vectors = np.atleast_2d(np.asarray(q_vectors, dtype=float))

    if masses.shape[0] != dF.shape[0]:
        raise ValueError(
            f"masses has {masses.shape[0]} entries but dF has {dF.shape[0]} rows."
        )

    w = dF / np.sqrt(masses)[:, None]
    phase = np.exp(1j * (positions @ q_vectors.T))  # (N, n_q)
    F_q = np.einsum("aq,ax->qx", phase, w.astype(complex))
    return np.linalg.norm(F_q, axis=1)


# =============================================================================
# Your PL class (mostly unchanged), but uses the improved embedding function
# =============================================================================

class ReadFiles:
    def __init__(self):
        pass

    def ReadStructure(self, path):
        lattice, species, counts, pos_cart = read_poscar(path)
        atoms = dict(zip(species, counts))
        return (pos_cart, atoms)

    def ReadPhononsPhonopy(self, path, freq_cutoff=0.1):
        with open(path, "r") as file:
            lines = [ts.strip() for ts in file]

        atomic_masses = []
        freqs = []
        normal_modes = []

        with open(path, "r") as file:
            for line in file:
                if "mass:" in line:
                    atomic_masses.append(line.split()[1])
        atomic_masses = np.array(atomic_masses, dtype=float)
        total_atoms = len(atomic_masses)

        with open(path, "r") as file:
            line_number = -1
            for line in file:
                line_number += 1
                if "frequency:" in line:
                    freqs.append(float(line.split()[1]))
                    ev_internal = []
                    for i in range(line_number + 3, line_number + 4 * total_atoms + 2, 4):
                        xyz = [lines[i + j].split()[2] for j in range(3)]
                        ev_internal.append(xyz)
                    normal_modes.append(ev_internal)

        freqs = np.array(freqs, dtype=float)
        freqs[freqs < freq_cutoff] = 0
        normal_modes = np.array(
            [[[float(x.strip(",")) for x in sublist] for sublist in outer] for outer in normal_modes],
            dtype=float,
        )
        return atomic_masses, freqs, normal_modes

    def ReadForces(self, path):
        _, forces = read_last_total_force_block(path)
        return forces

    def ReadPosForces(self, path):
        pos, _ = read_last_total_force_block(path)
        return pos


class Photoluminescence(ReadFiles):
    def __init__(self):
        super().__init__()

    def IV(self, iv_low, iv_high, rv_high):
        div = (2 * np.pi) / (2 * rv_high)
        return np.arange(iv_low, iv_high, div)

    def Fourier(self, iv, function):
        div = iv[1] - iv[0]
        rv = 2 * np.pi * np.fft.fftfreq(len(iv), div)
        sort = np.argsort(rv)
        rv = rv[sort]
        DFT = np.fft.fft(function)[sort]
        DFT = div * DFT * np.exp(-1j * rv * iv[0])
        return rv, DFT

    def Trapezoidal(self, integrand, iv, equally_spaced=True):
        div = iv[1] - iv[0]
        return (div / 2) * (np.sum(integrand[1:-1]) + integrand[0] + integrand[-1]) if equally_spaced else np.sum(
            np.array([((iv[i + 1] - iv[i]) / 2) * (integrand[i + 1] + integrand[i]) for i in range(len(iv) - 1)])
        )

    def FreqToEnergy(self, freqs):
        return 4.13566 * freqs  # meV

    def TimeScaling(self, t, reverse=False):
        return t / 658.2119 if not reverse else t * 658.2119

    def Lorentzian(self, x, x0, sigma):
        return ((1 / np.pi) * (sigma * 0.8)) / (((sigma * 0.8) ** 2) + ((x - x0) ** 2))

    def Gaussian(self, x, x0, sigma):
        return (1 / np.sqrt(2 * np.pi * (sigma**2))) * np.exp(-((x - x0) ** 2) / (2 * (sigma**2)))

    def ConfigCoordinates(self, masses, R_es, R_gs, modes):
        masses = np.sqrt(masses)
        R_diff = R_es - R_gs
        mR_diff = np.array([masses[i] * R_diff[i, :] for i in range(len(masses))])
        qk = np.array([np.sum(mR_diff * modes[i, :, :]) for i in range(modes.shape[0])])
        return qk

    def ConfigCoordinatesF(self, masses, F_es, F_gs, modes, Ek, F_diff=None):
        """
        Project the transition force field onto the target's Gamma-point modes.

        F_diff is optional and defaults to F_es - F_gs, which preserves the
        original call signature exactly. Pass F_diff explicitly when the
        difference force field has already been built and corrected on the
        target -- for instance by apply_analytic_tail_to_difference, whose
        monopole tail is a property of the difference alone and cannot be split
        between the two states. F_es and F_gs may then be None.
        """
        masses = np.sqrt(masses)
        if F_diff is None:
            if F_es is None or F_gs is None:
                raise ValueError(
                    "ConfigCoordinatesF needs either F_es and F_gs, or a precomputed F_diff."
                )
            F_diff = F_es - F_gs
        else:
            F_diff = np.asarray(F_diff, dtype=float)
            if F_diff.shape != (len(masses), 3):
                raise ValueError(
                    f"F_diff has shape {F_diff.shape}, expected {(len(masses), 3)} "
                    "(one row per atom, in target POSCAR order)."
                )
        mF_diff = np.array([(1 / masses[i]) * F_diff[i, :] for i in range(len(masses))])
        qk = np.array([np.sum(mF_diff * modes[i, :, :]) for i in range(modes.shape[0])])
        qk = (1 / Ek**2) * qk * 4180.069
        return qk

    def PartialHR(self, freqs, qk):
        return 2 * np.pi * freqs * (qk**2) * 0.166 / (2 * 1.05457)

    def SpectralFunction(self, Sk, Ek, E_meV_positive, sigma=6, Lorentz=False):
        if not Lorentz:
            S_E = np.array([np.dot(Sk, self.Gaussian(i, Ek, sigma)) for i in E_meV_positive])
        else:
            S_E = np.array([np.dot(Sk, self.Lorentzian(i, Ek, sigma)) for i in E_meV_positive])
        return S_E

    def FourierSpectralFunction(self, Sk, Ek, S_E, E_meV_positive):
        t_meV, S_t = self.Fourier(E_meV_positive, S_E)
        S_t_exact = np.array([np.dot(Sk, np.exp(-1j * Ek * i)) for i in t_meV])
        return t_meV, S_t, S_t_exact

    def GeneratingFunction(self, Sk, S_t, t_meV, Ek, T):
        if T == 0.0:
            G_t = np.exp((S_t) - (np.sum(Sk)))
        else:
            Kb = 8.61733326e-2
            nk = 1 / ((np.exp(Ek / (Kb * T))) - 1)
            G_t = np.exp(
                (S_t)
                - (np.sum(Sk))
                + np.sum(nk * Sk * np.exp(1j * Ek * t_meV))
                + np.sum(nk * Sk * np.exp(-1j * Ek * t_meV))
                - 2 * np.sum(nk * Sk)
            )
        return G_t

    def OpticalSpectralFunction(self, G_t, t_meV, zpl, gamma):
        E_meV, A_E = self.Fourier(t_meV, (G_t * np.exp(1j * zpl * t_meV)) * np.exp(-(gamma * np.abs(t_meV))))
        A_E = (1 / len(t_meV)) * A_E
        return E_meV, A_E

    def LuminescenceIntensity(self, E_meV, A_E, zpl):
        A_E = A_E[(E_meV >= (zpl - 500)) & (E_meV <= (zpl + 100))]
        E_meV = E_meV[(E_meV >= (zpl - 500)) & (E_meV <= (zpl + 100))]
        L_E = ((E_meV**3) * A_E) / (self.Trapezoidal(((E_meV**3) * A_E), E_meV))
        return E_meV, L_E

    def InverseParticipationRatio(self, modes):
        p = np.einsum("ijk -> ij", modes**2)
        IPR = 1 / np.einsum("ij -> i", p**2)
        return IPR


# =============================================================================
# Main spectrum calculation with improved force embedding
# =============================================================================

def CalculateSpectrum(
    phonons_source="Phonopy",
    path_phonon_band="band.yaml",
    temperature=0,
    zpl=3400,
    tmax=2000,
    gamma=2,
    # forces tuple formats:
    #   (OUTCAR_es, OUTCAR_gs)                              -> direct
    #   (OUTCAR_es_unit, OUTCAR_gs_unit, POSCAR_super, POSCAR_super) -> embed into super
    forces=None,
    # embedding options:
    embed_tolerance=9e-2,
    embed_pbc=True,
    embed_bijective=True,
    embed_mapping_reference="gs",
    # optional: species sanity check
    unit_poscar_for_species_check=None,
    # optional: write synthetic outcars
    write_synth_es_outcar=None,
    write_synth_gs_outcar=None,
    write_synth_dF_outcar=None,
    # analytic monopole tail for charged (Delta q = +-1) transitions.
    # All default to off, so existing behaviour is unchanged.
    analytic_tail=False,
    tail_defect_index=None,          # index of the C atom in target-POSCAR order
    tail_fit_window=(6.0, 16.0),
    tail_delta_q=None,               # +1 for C_B, -1 for C_N; 0 is rejected (no monopole)
    enforce_sum_rule=False,          # re-impose sum_a dF_a = 0 even without a tail (control)
):
    pl = Photoluminescence()

    if phonons_source == "Phonopy":
        masses, freqs, modes = pl.ReadPhononsPhonopy(path_phonon_band, freq_cutoff=0.1)
        freqs = freqs[: int(freqs.shape[0] / 2)]
        modes = modes[: int(modes.shape[0] / 2), ...]
    else:
        raise ValueError("Only Phonopy band.yaml is wired in this minimal example.")

    Ek = pl.FreqToEnergy(freqs)
    Ek[Ek == 0] = 1e-5

    dF_corrected = None

    if forces is not None:
        if isinstance(forces, tuple) and len(forces) == 4:
            # (OUTCAR_es_unit, OUTCAR_gs_unit, POSCAR_super_es, POSCAR_super_gs)
            outcar_es_unit, outcar_gs_unit, poscar_super_es, poscar_super_gs = forces

            if embed_mapping_reference.lower() == "es":
                reference_outcar = outcar_es_unit
                reference_poscar = poscar_super_es
            elif embed_mapping_reference.lower() == "gs":
                reference_outcar = outcar_gs_unit
                reference_poscar = poscar_super_gs
            else:
                raise ValueError("embed_mapping_reference must be 'es' or 'gs'.")

            target_to_unit, target_positions, mapping_info = build_unit_to_target_force_mapping(
                unit_outcar=reference_outcar,
                target_poscar=reference_poscar,
                tolerance=embed_tolerance,
                pbc=embed_pbc,
                bijective=embed_bijective,
                unit_poscar_for_species_check=unit_poscar_for_species_check,
            )

            F_es = apply_force_embedding_mapping(outcar_es_unit, target_to_unit)
            F_gs = apply_force_embedding_mapping(outcar_gs_unit, target_to_unit)

            if write_synth_es_outcar is not None:
                _, _, _, pos_es = read_poscar(poscar_super_es)
                write_minimal_outcar_total_force_block(
                    write_synth_es_outcar,
                    pos_es,
                    F_es,
                    title="SYNTHETIC OUTCAR ES (force-embedded, shared index map)",
                )
            if write_synth_gs_outcar is not None:
                _, _, _, pos_gs = read_poscar(poscar_super_gs)
                write_minimal_outcar_total_force_block(
                    write_synth_gs_outcar,
                    pos_gs,
                    F_gs,
                    title="SYNTHETIC OUTCAR GS (force-embedded, shared index map)",
                )

            print(f"[Embedding shared map from {embed_mapping_reference.upper()}]", mapping_info)

            # The monopole tail lives on dF = F_es - F_gs only. It is applied here,
            # after the per-state synthetic OUTCARs have been written, so those files
            # keep the raw embedded per-state forces.
            if analytic_tail:
                dF_corrected, tail_info = apply_analytic_tail_to_difference(
                    F_es=F_es,
                    F_gs=F_gs,
                    target_positions=target_positions,
                    target_to_unit=target_to_unit,
                    target_poscar=reference_poscar,
                    delta_q=tail_delta_q,
                    defect_index=tail_defect_index,
                    fit_window=tail_fit_window,
                    enforce_sum_rule=True,
                    verbose=True,
                )
            elif enforce_sum_rule:
                dF_corrected, sum_rule_stats = enforce_force_sum_rule(F_es - F_gs)
                print(
                    "[Sum rule only, no tail] |sum dF| "
                    f"{sum_rule_stats['abs_net_force_before_eVA']:.3e} -> "
                    f"{sum_rule_stats['abs_net_force_after_eVA']:.3e} eV/A"
                )

            if write_synth_dF_outcar is not None:
                write_minimal_outcar_total_force_block(
                    write_synth_dF_outcar,
                    target_positions,
                    dF_corrected if dF_corrected is not None else (F_es - F_gs),
                    title="SYNTHETIC OUTCAR dF = F_es - F_gs (difference force field)",
                )

        elif isinstance(forces, tuple) and len(forces) == 2:
            outcar_es, outcar_gs = forces
            F_es = pl.ReadForces(outcar_es)
            F_gs = pl.ReadForces(outcar_gs)
            if analytic_tail:
                raise ValueError(
                    "analytic_tail=True requires the 4-element `forces` form "
                    "(OUTCAR_es_unit, OUTCAR_gs_unit, POSCAR_super_es, POSCAR_super_gs). "
                    "With a 2-element `forces` there is no embedding, no target_to_unit "
                    "mask, and hence nothing truncated for the tail to continue."
                )
            if enforce_sum_rule:
                dF_corrected, sum_rule_stats = enforce_force_sum_rule(F_es - F_gs)
                print(
                    "[Sum rule only] |sum dF| "
                    f"{sum_rule_stats['abs_net_force_before_eVA']:.3e} -> "
                    f"{sum_rule_stats['abs_net_force_after_eVA']:.3e} eV/A"
                )
        else:
            raise ValueError("Invalid `forces` tuple. Expected 2 or 4 elements.")

        qk = pl.ConfigCoordinatesF(masses, F_es, F_gs, modes, Ek, F_diff=dF_corrected)
    else:
        raise ValueError("This script version expects `forces` to be provided.")

    Sk = pl.PartialHR(freqs, qk)

    Emax = 2.5 * zpl if zpl != 0 else 5000
    tmax_meV = pl.TimeScaling(tmax)
    E_meV_positive = pl.IV(0, Emax, tmax_meV)
    S_E = pl.SpectralFunction(Sk, Ek, E_meV_positive)

    t_meV, S_t, S_t_exact = pl.FourierSpectralFunction(Sk, Ek, S_E, E_meV_positive)
    G_t = pl.GeneratingFunction(Sk, S_t, t_meV, Ek, temperature)

    E_meV, A_E = pl.OpticalSpectralFunction(G_t, t_meV, zpl, gamma)
    E_meV, L_E = pl.LuminescenceIntensity(E_meV, A_E, zpl)

    t_fs = pl.TimeScaling(t_meV, reverse=True)
    IPR = pl.InverseParticipationRatio(modes)

    return (qk, (Ek, Sk), (E_meV_positive, S_E), (t_fs, S_t, S_t_exact), (G_t), (E_meV, A_E), (L_E), IPR)


def Results():
    # ---- EDIT THESE PATHS ----
    forces = (
        "/g/data/im26/ariel/hBN/first_calc/MLFF/hBN/defects/pllines/1-C_B/9_embedding/7x5/ex/0_stage/2_stage/1_cb/7x5/ex/OUTCAR",   # OUTCAR_es_unit
        "/g/data/im26/ariel/hBN/first_calc/MLFF/hBN/defects/pllines/1-C_B/9_embedding/7x5/ex/0_stage/2_stage/1_cb/7x5/gs/OUTCAR",    # OUTCAR_gs_unit
        "../relax/CONTCAR",  # POSCAR_super_es
        "../relax/CONTCAR",  # POSCAR_super_gs (often same)
    )

    (qk, (Ek, Sk), (E_meV_positive, S_E), (t_fs, S_t, S_t_exact), (G_t), (E_meV, A_E), (L_E), IPR) = CalculateSpectrum(
        phonons_source="Phonopy",
        path_phonon_band="../phonons/band.yaml",
        temperature=0,
        zpl=2940,
        tmax=2000,
        gamma=2,
        forces=forces,
        embed_tolerance=9e-2,
        embed_pbc=True,
        embed_bijective=True,
        unit_poscar_for_species_check=None,  # set to "/path/to/unit/POSCAR" to enforce species-order match
        write_synth_es_outcar="OUTCAR_synth_ES",
        write_synth_gs_outcar="OUTCAR_synth_GS",
    )

    plt.figure()
    plt.scatter(Ek, Sk, s=5, marker="s")
    plt.title(f"Total HR factor = {np.sum(Sk)}")
    plt.xlabel("Phonon Energy (meV)")
    plt.ylabel("$S_k$")
    plt.savefig("Sk.eps")
    np.savetxt("Sk.dat", np.column_stack([Ek, Sk]))

    S_E2 = S_E[E_meV_positive <= (max(Ek) + 36)]
    E_meV_positive2 = E_meV_positive[E_meV_positive <= (max(Ek) + 36)]

    plt.figure()
    plt.plot(E_meV_positive2, S_E2)
    plt.xlabel("Phonon Energy (meV)")
    plt.ylabel("S(E)")
    plt.savefig("S.eps")
    np.savetxt("S.dat", np.column_stack([E_meV_positive2, S_E2]))

    mask_t = (t_fs >= 0) & (t_fs <= 550)
    plt.figure()
    plt.plot(t_fs[mask_t], np.real(S_t[mask_t]), label="Real")
    plt.plot(t_fs[mask_t], np.imag(S_t[mask_t]), label="Imag")
    plt.legend()
    plt.xlabel("Time (fs)")
    plt.ylabel("S(t)")
    plt.savefig("St.eps")

    plt.figure()
    plt.plot(t_fs[mask_t], np.real(G_t[mask_t]), label="Real")
    plt.plot(t_fs[mask_t], np.imag(G_t[mask_t]), label="Imag")
    plt.legend()
    plt.xlabel("Time (fs)")
    plt.ylabel("G(t)")
    plt.savefig("Gt.eps")

    plt.figure()
    plt.plot(E_meV, np.abs(L_E))
    plt.xlabel("Photon Energy (meV)")
    plt.ylabel("PL")
    plt.savefig("pl.eps")
    np.savetxt("pl.dat", np.column_stack([E_meV, np.abs(L_E)]))

    plt.figure()
    plt.scatter(Ek, IPR, s=5, marker="s")
    plt.title("Inverse Participation Ratio")
    plt.xlabel("Phonon Energy (meV)")
    plt.ylabel("IPR")
    plt.savefig("ipr.eps")
    np.savetxt("ipr.dat", np.column_stack([Ek, IPR]))


if __name__ == "__main__":
    Results()
