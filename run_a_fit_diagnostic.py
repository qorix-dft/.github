#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run A: standalone monopole fit diagnostic.

Fits the monopole amplitude of dF = F_ex - F_gs to a reference cell's OWN force
field. There is no target, no band.yaml, no fill and no sum rule: every atom is
its own match, so the fit region is the whole cell and the fill region is empty.
The only thing being measured is whether the far field of dF is described by a
single amplitude A per species with dF_rad = A / r^2.

The question this answers. Fitted over 5-13 A the C_B radial decay exponent is
-2.69 (7x5), -1.97 (10x8), -1.38 (15x9). The largest cell gives the SHALLOWEST
decay, which is the wrong direction for periodic-image contamination: images
steepen a tail, they do not flatten it. The competing explanation is non-local
dielectric screening in a layered 2D system (Rytova-Keldysh), where the
screening weakens toward vacuum at large r and the field falls off more slowly
than A/(eps r^2). Refitting A over consecutive shells discriminates:

    A drifting UP with r    weakening screening, the effect is physical
    A drifting DOWN with r  the reference cell's own images, the window is bad
    A flat                  the single-parameter A/r^2 form is sound

Geometry. All production cells are orthorhombic. Above the isotropic radius
(half the SHORT axis) a radial shell no longer samples all directions equally:
it preferentially samples the long axis, where the cell's own images sit further
away and suppress the field less, so a window reaching past it is biased.

    reference   cell (A)          isotropic sphere   corner r_max
    10x8        25.07 x 34.73     12.53              21.67
    15x9        37.60 x 39.07     18.80              27.32

A1 and A2 stay inside their isotropic spheres. A3 deliberately does not, so the
A3 - A2 difference measures that bias directly.
"""

import argparse

import numpy as np

from pl_embedding import (
    extract_ions_per_type,
    fit_monopole_tail,
    locate_minority_species_atom,
    read_last_total_force_block,
    read_poscar,
    require_api_level,
    species_labels_from_counts,
)


# ---- EDIT THESE PATHS ----
OUTCAR_15x9_EX = "../15x9/ex/OUTCAR"
OUTCAR_15x9_GS = "../15x9/gs/OUTCAR"
OUTCAR_10x8_EX = "../10x8/ex/OUTCAR"
OUTCAR_10x8_GS = "../10x8/gs/OUTCAR"

# The lattice is read from the OUTCAR itself. Set these only if that fails
# (a POSCAR/CONTCAR for the same cell is then used instead).
POSCAR_15x9 = None
POSCAR_10x8 = None

SPECIES_ORDER = ["B", "N", "C"]     # must match the OUTCAR ions-per-type order
DELTA_Q = 1                          # +1 for C_B
EXTRAPOLATION_RADIUS_A = 27.3        # the 15x9 corner radius, for every run

RUNS = [
    {
        "name": "A1",
        "label": "C_B 15x9 self-fit",
        "ex": OUTCAR_15x9_EX,
        "gs": OUTCAR_15x9_GS,
        "poscar": POSCAR_15x9,
        "r_fit": (6.0, 18.0),
        "drift_windows": [(6.0, 10.0), (10.0, 14.0), (14.0, 18.0)],
        "isotropic_radius_A": 18.80,
        "expected_shell_counts": (213, 314, 450),
    },
    {
        "name": "A2",
        "label": "C_B 10x8 self-fit, inside the isotropic sphere",
        "ex": OUTCAR_10x8_EX,
        "gs": OUTCAR_10x8_GS,
        "poscar": POSCAR_10x8,
        "r_fit": (6.0, 12.0),
        "drift_windows": [(6.0, 8.0), (8.0, 10.0), (10.0, 12.0)],
        "isotropic_radius_A": 12.53,
        "expected_shell_counts": (93, 120, 174),
    },
    {
        "name": "A3",
        "label": "C_B 10x8 self-fit, window past the isotropic sphere",
        "ex": OUTCAR_10x8_EX,
        "gs": OUTCAR_10x8_GS,
        "poscar": POSCAR_10x8,
        "r_fit": (6.0, 16.0),
        "drift_windows": [(6.0, 9.0), (9.0, 12.0), (12.0, 16.0)],
        "isotropic_radius_A": 12.53,
        "expected_shell_counts": None,
    },
]

GEOMETRY_TOLERANCE_A = 1e-6


def read_last_lattice_from_outcar(path):
    """
    Lattice vectors from the last 'direct lattice vectors' block of an OUTCAR.

    VASP prints the block once per ionic step, so the last one is the geometry
    the final forces belong to -- the same convention read_last_total_force_block
    already uses for the forces.
    """
    with open(path, "r") as f:
        lines = f.readlines()

    last = None
    for i, line in enumerate(lines):
        if "direct lattice vectors" in line:
            rows = []
            for j in range(i + 1, min(i + 4, len(lines))):
                parts = lines[j].split()
                if len(parts) < 3:
                    break
                try:
                    rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    break
            if len(rows) == 3:
                last = np.array(rows, dtype=float)

    if last is None:
        raise ValueError(
            f"No 'direct lattice vectors' block found in {path}. "
            "Set the corresponding POSCAR path in the EDIT THESE PATHS block instead."
        )
    return last


def load_run(run):
    """Positions, dF, lattice, per-atom species labels and the defect index."""
    pos_ex, F_ex = read_last_total_force_block(run["ex"])
    pos_gs, F_gs = read_last_total_force_block(run["gs"])

    if pos_ex.shape != pos_gs.shape:
        raise ValueError(
            f"{run['name']}: ex has {pos_ex.shape[0]} atoms, gs has {pos_gs.shape[0]}."
        )
    max_dev = float(np.max(np.abs(pos_ex - pos_gs)))
    if max_dev > GEOMETRY_TOLERANCE_A:
        raise ValueError(
            f"{run['name']}: ex and gs geometries differ by up to {max_dev:.3e} A "
            f"(tolerance {GEOMETRY_TOLERANCE_A:.1e}). The embedding method requires both "
            "states evaluated as single points at ONE fixed geometry; a difference here "
            "means dF also contains a displacement, not just the transition force field."
        )

    counts = extract_ions_per_type(run["gs"])
    if len(counts) != len(SPECIES_ORDER):
        raise ValueError(
            f"{run['name']}: OUTCAR has {len(counts)} species types {counts} but "
            f"SPECIES_ORDER is {SPECIES_ORDER}."
        )
    if sum(counts) != pos_ex.shape[0]:
        raise ValueError(f"{run['name']}: ions-per-type {counts} does not sum to "
                         f"{pos_ex.shape[0]} atoms.")

    if run["poscar"]:
        lattice, _, _, _ = read_poscar(run["poscar"])
    else:
        lattice = read_last_lattice_from_outcar(run["gs"])

    labels = species_labels_from_counts(SPECIES_ORDER, counts)
    defect_index, defect_label = locate_minority_species_atom(SPECIES_ORDER, counts)

    return {
        "positions": pos_ex,
        "dF": F_ex - F_gs,
        "lattice": lattice,
        "labels": labels,
        "counts": counts,
        "defect_index": defect_index,
        "defect_label": defect_label,
        "geometry_max_dev_A": max_dev,
    }


def report_run(run, data, fit_info, amplitudes):
    lo, hi = fit_info["r_fit_effective_A"]
    print("")
    print("=" * 78)
    print(f" RUN {run['name']}  --  {run['label']}")
    print("=" * 78)
    print(f"  ex OUTCAR              : {run['ex']}")
    print(f"  gs OUTCAR              : {run['gs']}")
    print(f"  atoms                  : {data['positions'].shape[0]}  "
          f"{dict(zip(SPECIES_ORDER, data['counts']))}")
    print(f"  cell (A)               : "
          + " x ".join(f"{np.linalg.norm(v):.2f}" for v in data["lattice"]))
    print(f"  ex/gs geometry match   : max deviation {data['geometry_max_dev_A']:.2e} A")
    print(f"  |sum dF|               : {np.linalg.norm(data['dF'].sum(axis=0)):.3e} eV/A")
    print(f"  defect atom            : index {data['defect_index']} ({data['defect_label']})")
    print(f"  fit window             : [{lo:.2f}, {hi:.2f}) A, "
          f"{fit_info['n_atoms_in_window']} atoms")
    print(f"  isotropic sphere       : {run['isotropic_radius_A']:.2f} A"
          + ("   <-- window reaches PAST it, bias expected"
             if hi > run["isotropic_radius_A"] else "   (window stays inside)"))
    print(f"  matched region r_max   : {fit_info['matched_r_max_A']:.2f} A")
    if fit_info["skipped_species"]:
        for s, reason in fit_info["skipped_species"]:
            print(f"  skipped species        : {s} ({reason})")

    shells = fit_info["sub_windows_A"]
    totals = []
    for i in range(len(shells)):
        totals.append(sum(
            d["sub_windows"][i]["n_atoms"]
            for d in fit_info["per_species"].values()
            if not d["skipped"] and i < len(d.get("sub_windows", []))
        ))
    line = "  shell totals           : " + ",  ".join(
        f"{a:.0f}-{b:.0f}: {n}" for (a, b), n in zip(shells, totals)
    )
    if run["expected_shell_counts"]:
        line += "   expected " + "/".join(str(x) for x in run["expected_shell_counts"])
    print(line)

    print("")
    print(f"    {'species':<9}{'window [A]':>14}{'N':>7}{'A [eV*A]':>15}{'R^2':>9}{'exponent':>11}")
    for s, d in fit_info["per_species"].items():
        if d["skipped"]:
            continue
        for e in d["sub_windows"]:
            win = f"{e['r_lo_A']:.1f} - {e['r_hi_A']:.1f}"
            if e["sufficient"]:
                print(f"    {s:<9}{win:>14}{e['n_atoms']:>7}{e['amplitude_eVA']:>15.6f}"
                      f"{e['r_squared']:>9.4f}{e['loglog_exponent']:>11.3f}")
            else:
                print(f"    {s:<9}{win:>14}{e['n_atoms']:>7}{'too few atoms':>35}")
        print(f"    {s:<9}{'GLOBAL':>14}{d['n_atoms_in_window']:>7}{d['amplitude_eVA']:>15.6f}"
              f"{d['r_squared']:>9.4f}{d['loglog_exponent']:>11.3f}")
        drift = d.get("drift_percent", float("nan"))
        if np.isfinite(drift):
            tag = "monotone" if d.get("drift_monotone") else "scatter"
            print(f"    {s:<9}{'drift':>14}{'':>7}{drift:>14.1f}%   ({tag}, "
                  f"{d['drift_extrapolated_percent']:+.1f}% extrapolated to "
                  f"{fit_info['extrapolation_radius_A']:.1f} A)")
        print("")

    print(f"  combined log-log exponent : {fit_info['combined_loglog_exponent']:.3f}")
    print(f"  A_B / A_N                 : {fit_info['ratio_A_B_over_A_N']:.3f}   (expect ~ -1)")
    sign_ok = "B" in amplitudes and np.sign(amplitudes["B"]) == np.sign(DELTA_Q)
    print(f"  sign(A_B)                 : {'+' if amplitudes.get('B', 0) > 0 else '-'}"
          f"   expected {'+' if DELTA_Q > 0 else '-'} for Delta q = {DELTA_Q:+d}   "
          f"[{'OK' if sign_ok else 'FAILED'}]")
    print(f"  verdict                   : {fit_info['sub_window_drift_verdict']}")
    for w in fit_info["warnings"]:
        print(f"  warning                   : {w}")


def main():
    parser = argparse.ArgumentParser(
        description="Run A: standalone monopole fit diagnostic (no target, no fill, no sum rule).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--outcar-15x9-ex", default=OUTCAR_15x9_EX)
    parser.add_argument("--outcar-15x9-gs", default=OUTCAR_15x9_GS)
    parser.add_argument("--outcar-10x8-ex", default=OUTCAR_10x8_EX)
    parser.add_argument("--outcar-10x8-gs", default=OUTCAR_10x8_GS)
    parser.add_argument("--poscar-15x9", default=POSCAR_15x9)
    parser.add_argument("--poscar-10x8", default=POSCAR_10x8)
    parser.add_argument("--delta-q", type=int, default=DELTA_Q)
    parser.add_argument("--extrapolation-radius", type=float, default=EXTRAPOLATION_RADIUS_A)
    args = parser.parse_args()

    import pl_embedding
    require_api_level(9, caller="run_a_fit_diagnostic.py")

    override = {
        "A1": (args.outcar_15x9_ex, args.outcar_15x9_gs, args.poscar_15x9),
        "A2": (args.outcar_10x8_ex, args.outcar_10x8_gs, args.poscar_10x8),
        "A3": (args.outcar_10x8_ex, args.outcar_10x8_gs, args.poscar_10x8),
    }
    for run in RUNS:
        run["ex"], run["gs"], run["poscar"] = override[run["name"]]

    print("=" * 78)
    print(" RUN A: STANDALONE MONOPOLE FIT DIAGNOSTIC")
    print(" self-fit of the reference field; no target, no fill, no sum rule")
    print(f"  pl_embedding      : {pl_embedding.__file__}  (API level {pl_embedding.API_LEVEL})")
    print("=" * 78)

    results = {}
    for run in RUNS:
        data = load_run(run)
        n_atoms = data["positions"].shape[0]

        amplitudes, fit_info = fit_monopole_tail(
            dF=data["dF"],
            positions=data["positions"],
            target_to_unit=np.arange(n_atoms),      # self-fit: every atom matched
            defect_index=data["defect_index"],
            species=data["labels"],
            lattice=data["lattice"],
            r_fit=run["r_fit"],
            delta_q=args.delta_q,
            drift_windows=run["drift_windows"],
            extrapolation_radius=args.extrapolation_radius,
            verbose=False,
        )
        report_run(run, data, fit_info, amplitudes)
        results[run["name"]] = {"amplitudes": amplitudes, "fit_info": fit_info}

    # ---- comparisons --------------------------------------------------------
    print("")
    print("=" * 78)
    print(" SUMMARY")
    print("=" * 78)

    def drift_bar(name, s):
        d = results[name]["fit_info"]["per_species"].get(s, {})
        return abs(d.get("drift_percent", float("nan")))

    print("")
    print(" 1. A1 vs A2 -- transferability of the amplitude")
    print("    A is a physical property of the defect, Delta q * Z* / 4 pi eps0 eps, so it")
    print("    must not depend on which cell measured it. Both windows stay inside their")
    print("    own isotropic sphere, so neither is anisotropy biased.")
    print("")
    print(f"    {'species':<9}{'A (15x9)':>14}{'A (10x8)':>14}{'ratio':>10}"
          f"{'discrepancy':>14}{'drift bars':>13}")
    transferable = True
    for s in ("B", "N"):
        a1 = results["A1"]["amplitudes"].get(s)
        a2 = results["A2"]["amplitudes"].get(s)
        if a1 is None or a2 is None or a1 == 0.0:
            continue
        ratio = a2 / a1
        disc = 100.0 * abs(ratio - 1.0)
        bars = drift_bar("A1", s) + drift_bar("A2", s)
        transferable &= disc <= bars
        print(f"    {s:<9}{a1:>14.6f}{a2:>14.6f}{ratio:>10.3f}{disc:>13.1f}%{bars:>12.1f}%")
    flat_a1 = "flat to within" in results["A1"]["fit_info"]["sub_window_drift_verdict"]
    flat_a2 = "flat to within" in results["A2"]["fit_info"]["sub_window_drift_verdict"]

    print("")
    if transferable and (flat_a1 and flat_a2):
        print("    VERDICT: the two cells agree to within their own drift bars. Using the")
        print("    small reference's amplitude to fill a large target is justified.")
    elif transferable:
        print("    VERDICT: the two cells agree to within their drift bars, BUT at least one")
        print("    run is drifting, so those bars are wide and the agreement is weak. A drifting")
        print("    A is not a single number to be transferred: agreement here says the two cells")
        print("    are consistent, not that either has measured a well-defined amplitude. Fix the")
        print("    drift -- narrow the window, or move to a Rytova-Keldysh form -- before relying")
        print("    on transferability.")
    else:
        print("    VERDICT: *** the two cells DISAGREE beyond their drift bars. *** The fit")
        print("    window is contaminated and the transferability premise fails. Do NOT run")
        print("    the embedding on this amplitude until the window is fixed.")

    print("")
    print(" 2. A2 vs A3 -- anisotropy bias from crossing the isotropic sphere")
    print("    Same cell and same data; A3's window reaches to 16 A, past the 10x8 isotropic")
    print("    radius of 12.53 A, where shells preferentially sample the long axis. The")
    print("    difference between them IS the bias.")
    print("")
    print(f"    {'species':<9}{'A (6-12)':>14}{'A (6-16)':>14}{'bias':>12}"
          f"{'exponent 6-12':>16}{'exponent 6-16':>16}")
    for s in ("B", "N"):
        a2 = results["A2"]["amplitudes"].get(s)
        a3 = results["A3"]["amplitudes"].get(s)
        if a2 is None or a3 is None or a2 == 0.0:
            continue
        e2 = results["A2"]["fit_info"]["per_species"][s]["loglog_exponent"]
        e3 = results["A3"]["fit_info"]["per_species"][s]["loglog_exponent"]
        print(f"    {s:<9}{a2:>14.6f}{a3:>14.6f}{100.0 * (a3 / a2 - 1.0):>11.1f}%"
              f"{e2:>16.3f}{e3:>16.3f}")

    print("")
    print(" 3. Sign and sublattice checks")
    print(f"    {'run':<6}{'sign(A_B)':>12}{'A_B/A_N':>11}{'global exponent':>18}{'verdict':>10}")
    for name in ("A1", "A2", "A3"):
        fi = results[name]["fit_info"]
        a_b = results[name]["amplitudes"].get("B", 0.0)
        ok = np.sign(a_b) == np.sign(args.delta_q)
        print(f"    {name:<6}{'+' if a_b > 0 else '-':>12}{fi['ratio_A_B_over_A_N']:>11.3f}"
              f"{fi['combined_loglog_exponent']:>18.3f}{'OK' if ok else 'FAILED':>10}")

    print("")
    print(" 4. Drift directions")
    for name in ("A1", "A2", "A3"):
        print(f"    {name}: {results[name]['fit_info']['sub_window_drift_verdict']}")
    print("")


if __name__ == "__main__":
    main()
