#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exact control for the analytic monopole tail in pl_embedding.py.

The point of the control is that the untruncated answer is known. A 10x8
reference embedded into a 15x9 target is truncated; the 15x9 reference embedded
into the SAME 15x9 target is not, because every target atom is matched. So the
15x9 case is the exact answer that the 10x8 case must reproduce.

Four cases are run, and the middle two are what make the result interpretable:

  1. trunc      10x8 reference, truncated                 (current behaviour)
  2. trunc+SR   10x8 reference, truncated, sum rule re-imposed, NO tail
  3. tail+SR    10x8 reference + analytic monopole tail + sum rule
  4. reference  15x9 reference on the same target         (untruncated answer)

Case 2 isolates the sum rule from the tail. Truncation with bijective mapping
already satisfies sum_a dF_a = 0 to ~1e-5 eV/A, so case 2 is expected to change
almost nothing; if instead it removes a large fraction of the error, then the
tail is solving a smaller problem than the physics argument claims and that must
be known before the result is used.

Alongside the spectral windows the script reports criterion C3,

    F~(q) = | sum_a m_a^(-1/2) dF_a exp(i q.R_a) |

at the smallest few non-zero target q. That is the quantity truncation actually
damages -- F~(0) is left intact by the bijective mapping -- and the 1/E^2 in
ConfigCoordinatesF (S_k ~ proj^2 / E^3) is what turns it into spurious weight
below 5 meV.
"""

import argparse

import numpy as np

from pl_embedding import (
    Photoluminescence,
    _fit_radial_amplitude,
    defect_displacements,
    locate_minority_species_atom,
    species_labels_from_counts,
    apply_analytic_tail_to_difference,
    apply_force_embedding_mapping,
    build_unit_to_target_force_mapping,
    check_phonon_species_order,
    defect_alignment_shift,
    enforce_force_sum_rule,
    force_structure_factor,
    phonon_species_order_message,
    read_poscar,
    smallest_nonzero_q,
    verify_defect_correspondence,
)


# ---- EDIT THESE PATHS ----
OUTCAR_ES_10x8 = "../10x8/ex/OUTCAR"
OUTCAR_GS_10x8 = "../10x8/gs/OUTCAR"
OUTCAR_ES_15x9 = "../15x9/ex/OUTCAR"
OUTCAR_GS_15x9 = "../15x9/gs/OUTCAR"
TARGET_POSCAR = "../relax/CONTCAR"          # the 15x9 target, same for every case
BAND_YAML = "../phonons/band.yaml"          # Gamma-point modes of the 15x9 target

DELTA_Q = 1                                  # +1 for C_B, -1 for C_N
DEFECT_INDEX = None                          # None -> minority species (the carbon)
FIT_WINDOW = (6.0, 16.0)

EMBED_TOLERANCE = 9e-2
EMBED_PBC = True
EMBED_BIJECTIVE = True

SIGMA_MEV = 6.0                              # broadening used by SpectralFunction
E_GRID_MAX_MEV = 260.0
E_GRID_STEP_MEV = 0.05
N_Q_SMALLEST = 5

WINDOWS_MEV = [
    (0.0, 5.0),
    (0.0, 10.0),
    (10.0, 20.0),
    (20.0, 50.0),
    (50.0, 100.0),
    (100.0, 150.0),
    (150.0, 220.0),
]

CASES = ["trunc", "trunc+SR", "tail+SR", "reference"]
REFERENCE_CASE = "reference"


def trapezoid(y, x):
    """Trapezoidal integral, written out so no NumPy version quirks apply."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size < 2:
        return 0.0
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def build_difference_force(outcar_es, outcar_gs, target_poscar, mode, args):
    """
    Embed one reference into the target and return the difference force field.

    mode is one of:
        "raw"      dF = F_es - F_gs, unmatched atoms left at zero (truncation)
        "sumrule"  as "raw", then sum_a dF_a = 0 re-imposed
        "tail"     as "raw", then the unmatched atoms filled with the fitted
                   analytic monopole tail, then the sum rule re-imposed

    Reference cells that substituted a different site of the same host lattice
    are translated onto the target first, so that the untruncated 15x9 case can
    be run against a target built around the 10x8 defect position. A shift below
    the matching tolerance is treated as no shift at all.
    """
    shift, shift_info = None, None
    if args.align_defects:
        candidate, shift_info = defect_alignment_shift(
            unit_outcar=outcar_gs, target_poscar=target_poscar
        )
        if shift_info["shift_norm_A"] > args.tolerance:
            shift = candidate

    target_to_unit, target_positions, mapping_info = build_unit_to_target_force_mapping(
        unit_outcar=outcar_gs,
        target_poscar=target_poscar,
        tolerance=args.tolerance,
        pbc=EMBED_PBC,
        bijective=EMBED_BIJECTIVE,
        unit_position_shift=shift,
    )

    F_es = apply_force_embedding_mapping(outcar_es, target_to_unit)
    F_gs = apply_force_embedding_mapping(outcar_gs, target_to_unit)
    dF = F_es - F_gs

    tail_info = None
    if mode == "sumrule":
        dF, _ = enforce_force_sum_rule(dF)
    elif mode == "tail":
        dF, tail_info = apply_analytic_tail_to_difference(
            F_es=F_es,
            F_gs=F_gs,
            target_positions=target_positions,
            target_to_unit=target_to_unit,
            target_poscar=target_poscar,
            delta_q=args.delta_q,
            defect_index=args.defect_index,
            fit_window=(args.fit_lo, args.fit_hi),
            enforce_sum_rule=True,
            layer_resolved=args.layer_resolved,
            fill_zero_mean=args.fill_zero_mean,
            verbose=True,
        )
    elif mode != "raw":
        raise ValueError(f"Unknown mode '{mode}'.")

    return dF, target_positions, target_to_unit, mapping_info, tail_info, shift_info


def spectrum_from_dF(pl, dF, masses, modes, freqs, Ek, E_grid, sigma):
    """Partial HR factors and the broadened spectral function for one dF."""
    qk = pl.ConfigCoordinatesF(masses, None, None, modes, Ek, F_diff=dF)
    Sk = pl.PartialHR(freqs, qk)
    S_E = pl.SpectralFunction(Sk, Ek, E_grid, sigma=sigma)
    return Sk, S_E


def window_integrals(S_E, E_grid, Sk, Ek):
    """
    Two measures per window: the integral of the broadened S(E), and the raw
    sum of S_k over modes whose energy falls in the window.

    The broadened integral is what the production pipeline produces, but with
    sigma = 6 meV a 0-5 meV window also collects leakage from modes well above
    it. The raw mode sum has no broadening and is therefore unambiguous. If the
    tail works, both must improve.
    """
    integ, raw = [], []
    for lo, hi in WINDOWS_MEV:
        m = (E_grid >= lo) & (E_grid <= hi)
        integ.append(trapezoid(S_E[m], E_grid[m]))
        mk = (Ek >= lo) & (Ek <= hi)
        raw.append(float(np.sum(Sk[mk])))
    return np.array(integ), np.array(raw)


def print_table(title, values, ratios, note=""):
    print("")
    print(title)
    if note:
        print(f"  {note}")
    header = f"  {'window (meV)':<16}"
    for c in CASES:
        header += f"{c:>14}{'ratio':>9}"
    print(header)
    print("  " + "-" * (16 + 23 * len(CASES)))
    for i, (lo, hi) in enumerate(WINDOWS_MEV):
        row = f"  {f'{lo:g} - {hi:g}':<16}"
        for c in CASES:
            row += f"{values[c][i]:>14.6e}"
            row += f"{ratios[c][i]:>9.3f}" if np.isfinite(ratios[c][i]) else f"{'--':>9}"
        print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Exact control for the analytic monopole tail.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--outcar-es-small", default=OUTCAR_ES_10x8)
    parser.add_argument("--outcar-gs-small", default=OUTCAR_GS_10x8)
    parser.add_argument("--outcar-es-large", default=OUTCAR_ES_15x9)
    parser.add_argument("--outcar-gs-large", default=OUTCAR_GS_15x9)
    parser.add_argument("--target-poscar", default=TARGET_POSCAR)
    parser.add_argument("--band-yaml", default=BAND_YAML)
    parser.add_argument("--delta-q", type=int, default=DELTA_Q)
    parser.add_argument("--defect-index", type=int, default=DEFECT_INDEX)
    parser.add_argument("--fit-lo", type=float, default=FIT_WINDOW[0])
    parser.add_argument("--fit-hi", type=float, default=FIT_WINDOW[1])
    parser.add_argument("--tolerance", type=float, default=EMBED_TOLERANCE)
    parser.add_argument("--sigma", type=float, default=SIGMA_MEV)
    parser.add_argument("--n-q", type=int, default=N_Q_SMALLEST)
    parser.add_argument(
        "--layer-resolved",
        action="store_true",
        help="fit and fill one amplitude per species AND layer, instead of one per species",
    )
    parser.add_argument(
        "--fill-zero-mean",
        action="store_true",
        help="remove each group's mean over the filled atoms so the fill contributes no net "
             "force and no mass-weighted q=0 component; truncation already gets q=0 right",
    )
    parser.add_argument(
        "--no-align-defects",
        dest="align_defects",
        action="store_false",
        help="do not translate a reference onto the target's defect site before matching",
    )
    parser.add_argument(
        "--allow-incomplete-reference",
        action="store_true",
        help="continue even if the reference case fails to match every target atom, which "
             "means case 4 is not the untruncated answer",
    )
    parser.add_argument(
        "--include-out-of-plane-q",
        action="store_true",
        help="include reciprocal vectors along the vacuum direction in the C3 table",
    )
    args = parser.parse_args()

    pl = Photoluminescence()

    masses, freqs, modes = pl.ReadPhononsPhonopy(args.band_yaml, freq_cutoff=0.1)
    freqs = freqs[: int(freqs.shape[0] / 2)]
    modes = modes[: int(modes.shape[0] / 2), ...]
    Ek = pl.FreqToEnergy(freqs)
    Ek[Ek == 0] = 1e-5

    lattice, species_names, counts, _ = read_poscar(args.target_poscar)
    n_atoms = int(np.sum(counts))
    defect_index_used = (
        args.defect_index
        if args.defect_index is not None
        else locate_minority_species_atom(species_names, counts)[0]
    )
    species_check = check_phonon_species_order(
        band_yaml=args.band_yaml, target_poscar=args.target_poscar, masses=masses
    )
    if not species_check["ok"]:
        raise ValueError(
            phonon_species_order_message(args.band_yaml, args.target_poscar, species_check)
        )

    print("=" * 74)
    print(" MONOPOLE TAIL VALIDATION")
    print("=" * 74)
    print(f"  target            : {args.target_poscar}  ({n_atoms} atoms, {species_names} {counts})")
    print(f"  modes             : {args.band_yaml}  ({modes.shape[0]} modes)")
    print(f"  phonon/target     : consistent  [{', '.join(species_check['tests_run'])}]"
          + ("" if species_check["symbols_available"] else "; no symbols in band.yaml"))
    print(f"  Delta q           : {args.delta_q:+d}")
    print(f"  fit window        : [{args.fit_lo:g}, {args.fit_hi:g}) A")
    print(f"  broadening sigma  : {args.sigma:g} meV")

    E_grid = np.arange(0.0, E_GRID_MAX_MEV + E_GRID_STEP_MEV, E_GRID_STEP_MEV)

    plan = {
        "trunc": (args.outcar_es_small, args.outcar_gs_small, "raw"),
        "trunc+SR": (args.outcar_es_small, args.outcar_gs_small, "sumrule"),
        "tail+SR": (args.outcar_es_small, args.outcar_gs_small, "tail"),
        "reference": (args.outcar_es_large, args.outcar_gs_large, "raw"),
    }

    dF_all, Sk_all, integ_all, raw_all, stot_all = {}, {}, {}, {}, {}
    mask_all, tail_info_all = {}, {}
    positions_ref = None

    for case in CASES:
        outcar_es, outcar_gs, mode = plan[case]
        print("")
        print("-" * 74)
        print(f" case: {case}   ({mode} embedding of {outcar_gs})")
        print("-" * 74)

        dF, positions, target_to_unit, mapping_info, tail_info, shift_info = build_difference_force(
            outcar_es, outcar_gs, args.target_poscar, mode, args
        )
        positions_ref = positions

        if shift_info is not None:
            applied = mapping_info["unit_position_shift_norm_A"]
            if applied > 0:
                v = mapping_info["unit_position_shift_A"]
                print(
                    f"  defect alignment: reference atom {shift_info['unit_defect_index']} "
                    f"({shift_info['defect_species']}) translated onto target atom "
                    f"{shift_info['target_defect_index']} by "
                    f"({v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f}) A, |shift| {applied:.4f} A"
                )
            else:
                print(
                    f"  defect alignment: already aligned "
                    f"(|shift| {shift_info['shift_norm_A']:.2e} A, below the "
                    f"{args.tolerance:g} A tolerance), no translation applied"
                )

        correspondence = verify_defect_correspondence(
            target_to_unit=target_to_unit, unit_outcar=outcar_gs, target_poscar=args.target_poscar
        )
        if not correspondence["ok"]:
            raise ValueError(
                f"Case '{case}': the target's defect atom "
                f"{correspondence['target_defect_index']} "
                f"({correspondence['defect_species']}) is "
                + (
                    "not matched to any reference atom"
                    if correspondence["defect_unmatched"]
                    else f"matched to reference atom {correspondence['mapped_unit_index']}, "
                         f"not the reference's own defect atom "
                         f"{correspondence['unit_defect_index']}"
                )
                + ".\n"
                "  The reference's force field is centred on a different site from the target's, "
                "so every embedded force belongs to the wrong defect position.\n"
                "  Note that the unmatched-atom count does NOT catch this: two cells that "
                "substituted different sites of one host lattice differ in only two atoms, so "
                "the mapping can look ~99.9% complete while the embedded dF is wholly wrong.\n"
                "  Fix by aligning the defects (this runs by default; --no-align-defects "
                "disables it). If alignment is on and this still fires, the two cells are not "
                "related by a lattice translation and no rigid shift can align them."
            )
        print(
            f"  defect correspondence: target atom {correspondence['target_defect_index']} "
            f"<- reference atom {correspondence['mapped_unit_index']}   [OK]"
        )

        n_unmatched = int(np.sum(target_to_unit < 0))
        print(
            f"  matched {mapping_info['matched_unit_atoms']} / target {mapping_info['n_target']}, "
            f"unmatched target atoms {n_unmatched}, "
            f"max match distance {mapping_info['max_match_distance_A']:.4f} A"
        )
        if case == REFERENCE_CASE and n_unmatched != 0:
            message = (
                f"The reference case left {n_unmatched} of {mapping_info['n_target']} target "
                "atoms unmatched, so it is NOT the untruncated answer and every ratio in this "
                "run would be measured against a truncated baseline.\n"
                "  Likely causes, in order:\n"
                "   - the reference and the target place the defect on different sites and the "
                "translation could not align them (is the shift a host lattice vector?);\n"
                "   - the two cells are not the same structure (different relaxation, cell "
                "shape or atom count), which no rigid translation can fix;\n"
                "   - the matching tolerance is too tight for the relaxation present.\n"
                f"  max match distance was {mapping_info['max_match_distance_A']:.4f} A against "
                f"a tolerance of {args.tolerance:g} A.\n"
                "  Re-run with --allow-incomplete-reference to proceed anyway and see the numbers."
            )
            if args.allow_incomplete_reference:
                print(f"  *** WARNING: {message}")
            else:
                raise ValueError(message)
        print(f"  |sum dF| = {np.linalg.norm(dF.sum(axis=0)):.3e} eV/A")

        Sk, S_E = spectrum_from_dF(pl, dF, masses, modes, freqs, Ek, E_grid, args.sigma)
        integ, raw = window_integrals(S_E, E_grid, Sk, Ek)

        dF_all[case] = dF
        mask_all[case] = target_to_unit < 0
        tail_info_all[case] = tail_info
        Sk_all[case] = Sk
        integ_all[case] = integ
        raw_all[case] = raw
        stot_all[case] = float(np.sum(Sk))

    # ---- fill-region audit --------------------------------------------------
    # The reference case supplies the TRUE dF on exactly the atoms truncation
    # leaves empty, so the analytic fill can be scored against it directly
    # rather than inferred from the spectrum. This is the only place the
    # amplitude can be checked against the thing it is meant to reproduce.
    fill = mask_all["trunc"]
    n_fill = int(np.sum(fill))
    if n_fill and REFERENCE_CASE in dF_all:
        R_vec, r_vec, _ = defect_displacements(lattice, positions_ref, defect_index_used)
        R_hat = np.zeros_like(R_vec)
        nz = r_vec > 0
        R_hat[nz] = R_vec[nz] / r_vec[nz, None]
        labels = species_labels_from_counts(species_names, counts)
        truth = dF_all[REFERENCE_CASE]
        truth_norm = float(np.linalg.norm(truth[fill]))

        print("")
        print(" Fill-region audit: the analytic tail against the reference's true forces")
        print(f"  {n_fill} atoms that truncation leaves empty, r from "
              f"{r_vec[fill].min():.2f} to {r_vec[fill].max():.2f} A")
        print("")
        print(f"  {'case':<16}{'sum|dF| there':>16}{'relative error vs reference':>30}")
        print("  " + "-" * 62)
        for c in CASES:
            err = (
                float(np.linalg.norm(dF_all[c][fill] - truth[fill])) / truth_norm
                if truth_norm > 0 else float("nan")
            )
            print(f"  {c:<16}{float(np.sum(np.abs(dF_all[c][fill]))):>16.6f}{err:>30.4f}")
        print("")
        print("  Truncation scores 1.000 by construction: it puts zero there. Any fill that")
        print("  scores above 1.000 is further from the truth than leaving the region empty.")

        print("")
        print("  Amplitude the reference's own field wants in the fill region, against the")
        print("  amplitude fitted from the reference window and used for the fill:")
        print(f"    {'species':<9}{'A fitted':>13}{'A from truth':>15}{'ratio':>9}"
              f"{'median(truth)':>15}{'R^2 of truth':>14}")
        fitted_amps = (tail_info_all.get("tail+SR") or {}).get("amplitudes_eVA", {})
        truth_rad = np.einsum("ij,ij->i", truth, R_hat)
        for s in dict.fromkeys(species_names):
            sel = fill & (np.array([str(x) for x in labels]) == str(s))
            if int(np.sum(sel)) < 3:
                continue
            t = _fit_radial_amplitude(truth_rad[sel], r_vec[sel])
            a_fit = float(fitted_amps.get(str(s), float("nan")))
            ratio = a_fit / t["amplitude_eVA"] if t["amplitude_eVA"] != 0 else float("nan")
            print(f"    {str(s):<9}{a_fit:>13.4f}{t['amplitude_eVA']:>15.4f}{ratio:>9.2f}"
                  f"{t['amplitude_median_eVA']:>15.4f}{t['r_squared']:>14.4f}")
        print("")
        print("  A ratio far from 1 means the window the amplitude was fitted in does not")
        print("  describe the region it is being extrapolated into. A low R^2 in the last")
        print("  column means no single amplitude describes that region either.")

    ref_integ = integ_all[REFERENCE_CASE]
    ref_raw = raw_all[REFERENCE_CASE]

    ratios_integ = {
        c: np.where(ref_integ != 0, integ_all[c] / np.where(ref_integ != 0, ref_integ, 1.0), np.nan)
        for c in CASES
    }
    ratios_raw = {
        c: np.where(ref_raw != 0, raw_all[c] / np.where(ref_raw != 0, ref_raw, 1.0), np.nan)
        for c in CASES
    }

    print_table(
        " Window-integrated S(E)   [ratio = case / reference]",
        integ_all,
        ratios_integ,
        note=f"broadened with sigma = {args.sigma:g} meV, integrated on a "
             f"{E_GRID_STEP_MEV:g} meV grid",
    )
    print_table(
        " Raw mode sum  sum_k S_k over modes in window   [ratio = case / reference]",
        raw_all,
        ratios_raw,
        note="no broadening, so no leakage between windows",
    )

    print("")
    print(" Total Huang-Rhys factor")
    print(f"  {'case':<16}{'S_tot':>14}{'ratio':>9}{'error':>10}")
    print("  " + "-" * 49)
    for c in CASES:
        ratio = stot_all[c] / stot_all[REFERENCE_CASE]
        print(f"  {c:<16}{stot_all[c]:>14.6f}{ratio:>9.4f}{100.0 * (ratio - 1.0):>9.2f}%")

    # ---- criterion C3 --------------------------------------------------------
    q_vectors, hkl = smallest_nonzero_q(
        lattice, n_q=args.n_q, include_out_of_plane=args.include_out_of_plane_q
    )
    print("")
    print(" Criterion C3: |F~(q)| = |sum_a m_a^(-1/2) dF_a exp(i q.R_a)|   [eV/A/sqrt(amu)]")
    print(
        "  smallest non-zero target q"
        + ("" if args.include_out_of_plane_q else ", in-plane only (the vacuum direction is not")
    )
    if not args.include_out_of_plane_q:
        print("  a physical wavevector for a slab; pass --include-out-of-plane-q to include it)")
    header = f"  {'(h k l)':<12}{'|q| [1/A]':>12}"
    for c in CASES:
        header += f"{c:>14}{'ratio':>9}"
    print(header)
    print("  " + "-" * (24 + 23 * len(CASES)))

    Fq = {c: force_structure_factor(dF_all[c], masses, positions_ref, q_vectors) for c in CASES}
    for i in range(q_vectors.shape[0]):
        row = f"  {str(tuple(int(v) for v in hkl[i])):<12}{np.linalg.norm(q_vectors[i]):>12.5f}"
        for c in CASES:
            row += f"{Fq[c][i]:>14.6e}"
            ref_val = Fq[REFERENCE_CASE][i]
            row += f"{(Fq[c][i] / ref_val):>9.3f}" if ref_val != 0 else f"{'--':>9}"
        print(row)

    q0 = np.zeros((1, 3))
    print("")
    print("  for comparison, q = 0 (the acoustic sum rule):")
    for c in CASES:
        print(f"    {c:<16}{force_structure_factor(dF_all[c], masses, positions_ref, q0)[0]:>14.6e}")

    # ---- success criteria ----------------------------------------------------
    print("")
    print("=" * 74)
    print(" SUCCESS CRITERIA")
    print("=" * 74)

    i05 = WINDOWS_MEV.index((0.0, 5.0))
    i_opt = WINDOWS_MEV.index((150.0, 220.0))

    err_trunc = abs(ratios_integ["trunc"][i05] - 1.0)
    err_sr = abs(ratios_integ["trunc+SR"][i05] - 1.0)
    err_tail = abs(ratios_integ["tail+SR"][i05] - 1.0)

    print(
        f"  S(E < 5 meV) ratio to reference: trunc {ratios_integ['trunc'][i05]:.3f}, "
        f"trunc+SR {ratios_integ['trunc+SR'][i05]:.3f}, tail+SR {ratios_integ['tail+SR'][i05]:.3f}"
    )
    print(f"    [{'PASS' if err_tail < err_trunc else 'FAIL'}] the tail moves S(E < 5 meV) "
          "towards the reference")
    sr_share = (err_trunc - err_sr) / err_trunc if err_trunc > 0 else float("nan")
    print(
        f"    sum rule alone accounts for {100.0 * sr_share:.1f}% of the truncation error. "
        "Expected to be small:"
    )
    print(
        "    truncation with bijective mapping already satisfies the q = 0 sum rule, so a large "
        "value here"
    )
    print("    would mean the tail is solving a smaller problem than the physics argument claims.")

    stot_err = abs(stot_all["tail+SR"] / stot_all[REFERENCE_CASE] - 1.0)
    print(
        f"  S_tot error with tail: {100.0 * stot_err:.3f}%  "
        f"[{'PASS' if stot_err < 0.01 else 'FAIL'}] (must stay within 1%)"
    )

    opt_ref = integ_all[REFERENCE_CASE][i_opt]
    opt_tail = integ_all["tail+SR"][i_opt]
    opt_trunc = integ_all["trunc"][i_opt]
    unchanged = round(opt_tail, 3) == round(opt_trunc, 3)
    print(
        f"  optical window 150-220 meV: trunc {opt_trunc:.6f}, tail+SR {opt_tail:.6f}, "
        f"reference {opt_ref:.6f}"
    )
    print(
        f"    [{'PASS' if unchanged else 'FAIL'}] unchanged by the tail to three decimals"
    )
    print("=" * 74)


if __name__ == "__main__":
    main()
