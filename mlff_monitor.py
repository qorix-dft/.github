#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mlff_monitor.py -- look at a VASP machine-learning MD run, locally or over ssh.

This is the read-only half of the workflow. It replaces extract-ave.py, plot.py,
parity.py and the Mathematica notebook that ssh'd into gadi and re-typed the same
ListPlot five times. Nothing here writes into a run directory; md_cycle.sh owns
that side.

    monitor   one dashboard from the .dat files md_cycle.sh writes:
              temperature, energy, pressure, volume, lattice parameters, cell
              angles and the BEEF / ERR / CTIFOR learning curve, each with its
              moving average on top, plus an equilibrated mean +- sigma table.
    parity    per-system parity plots of energy and forces from ML_REG + ML_AB.
    avg       the old extract-ave.py: writes FILE.ave and FILE.cave.

Remote runs are the normal case, so --host/--remote-dir pulls what it needs in a
single ssh call (tar over the wire, into a local cache) instead of one `ssh cat`
per file:

    ./mlff_monitor.py monitor --suffix 300700tri \\
        --host aa2016@gadi.nci.org.au \\
        --remote-dir /g/data/im26/ariel/hBN/first_calc/MLFF/hBN

    ./mlff_monitor.py monitor --suffix 300700tri          # already local
    ./mlff_monitor.py parity  --host ... --remote-dir ...
    ./mlff_monitor.py avg energy.dat300700tri 100 600

Requires numpy and matplotlib. seaborn is used if present, ignored if not.
pandas is no longer needed anywhere.

File format notes, so the column choices below are checkable:

  energy.dat<sfx>    T(K)  E(eV)  F(eV)      from OSZICAR 'T=' lines
  pressure.dat<sfx>  total pressure (kbar)   from OUTCAR
  volume.dat<sfx>    volume (A^3)            from OUTCAR. OUTCAR prints the
                     input geometry's volume too, so this file is normally one
                     row longer than the others; the extra leading row is
                     dropped (--no-trim keeps it).
  cell.dat<sfx>      a11 a22 a33 |a| |b| |c| alpha beta gamma
                     The Mathematica notebook plotted columns 1-3 (the diagonal
                     components). Columns 4-6 (the vector norms) are the actual
                     lattice parameters and are plotted here by default; pass
                     --abc diag for the old behaviour. They differ only by the
                     off-diagonal content of the cell.
  BEEF<sfx>.dat      BEEF  nstep  bee_energy  bee_max_force  bee_ave_force  threshold
  ERR<sfx>.dat       ERR   nstep  rmse_energy rmse_force     rmse_stress
                     Plotted: ERR rmse_force, BEEF bee_max_force, and the
                     CTIFOR threshold -- the three force quantities that decide
                     whether the model is still learning.
"""

import argparse
import io
import os
import re
import shlex
import subprocess
import sys
import tarfile
from collections import Counter, OrderedDict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================================
# style -- a framed, ticks-inward box, the matplotlib equivalent of Frame -> True
# =====================================================================================

PALETTE = {
    "temperature": "#0072B2",   # blue
    "energy":      "#E69F00",   # orange   (Mathematica: Orange)
    "pressure":    "#8B4A2B",   # brown    (Mathematica: Brown)
    "volume":      "#D55E00",   # vermillion (Mathematica: Red)
    "a":           "#0072B2",
    "b":           "#D55E00",
    "c":           "#009E73",
    "err":         "#009E73",
    "beef":        "#000000",
    "ctifor":      "#D55E00",
}

# Okabe-Ito, for the per-system parity panels
PARITY_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#56B4E9",
                  "#CC79A7", "#D55E00", "#000000"]
DIAGONAL_COLOR = "#888888"


def apply_style():
    try:
        import seaborn as sns
        sns.set_theme(style="ticks", context="paper")
    except ImportError:
        pass
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.linewidth": 1.3,
        "axes.labelsize": 13,
        "axes.grid": False,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.frameon": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "figure.dpi": 110,
    })


def frame(ax):
    """Box the axes on all four sides, ticks inward."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.3)
    ax.tick_params(direction="in", top=True, right=True, which="both")
    ax.minorticks_on()


# =====================================================================================
# where the files come from
# =====================================================================================

class FileSource:
    """Local directory, or a remote one mirrored into a cache by one ssh call."""

    def __init__(self, local_dir=".", host=None, remote_dir=None,
                 cache=None, fetch=True):
        self.host = host
        self.remote_dir = remote_dir
        self.fetch = fetch
        if host:
            if not remote_dir:
                sys.exit("ERROR: --host needs --remote-dir.")
            self.dir = cache or os.path.join(".mlff_cache", re.sub(r"\W+", "_", host))
            os.makedirs(self.dir, exist_ok=True)
        else:
            self.dir = local_dir

    def _ssh(self, command, binary=False):
        try:
            proc = subprocess.run(["ssh", "-o", "BatchMode=yes", self.host, command],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as exc:
            sys.exit(f"ERROR: could not run ssh: {exc}")
        if proc.returncode != 0:
            msg = proc.stderr.decode(errors="replace").strip()
            sys.exit(f"ERROR: ssh {self.host} failed ({proc.returncode}): {msg}")
        return proc.stdout if binary else proc.stdout.decode(errors="replace")

    def prefetch(self, names):
        """Copy whichever of `names` exist on the remote host into the cache.

        Two ssh calls total: one to list, one to stream a gzipped tar of the
        files that are actually there. Missing names are simply not fetched --
        a run that has not produced ML_REG yet is not an error here.
        """
        if not self.host or not self.fetch:
            return
        rdir = shlex.quote(self.remote_dir)
        present = set(self._ssh(f"cd {rdir} && ls -1").split())
        wanted = [n for n in dict.fromkeys(names) if n in present]
        if not wanted:
            sys.exit(f"ERROR: none of {sorted(set(names))} exist in "
                     f"{self.host}:{self.remote_dir}")
        print(f"fetching {len(wanted)} file(s) from {self.host}:{self.remote_dir}")
        quoted = " ".join(shlex.quote(n) for n in wanted)
        blob = self._ssh(f"cd {rdir} && tar czf - -- {quoted}", binary=True)
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            for member in tar.getmembers():
                # only plain files, only into the cache directory
                if not member.isfile() or os.path.sep in member.name or ".." in member.name:
                    continue
                data = tar.extractfile(member).read()
                with open(os.path.join(self.dir, member.name), "wb") as fh:
                    fh.write(data)
        for name in wanted:
            print(f"  {name}  ({os.path.getsize(os.path.join(self.dir, name))/1024:.0f} kB)")

    def path(self, *candidates):
        """First existing candidate, or None."""
        for name in candidates:
            full = os.path.join(self.dir, name)
            if os.path.exists(full) and os.path.getsize(full) > 0:
                return full
        return None


# =====================================================================================
# readers
# =====================================================================================

def read_table(path):
    """Whitespace table -> (n, ncols) float array.

    Comments and blank lines are skipped, as are rows VASP mangled into '*****'
    by a too-narrow format field. Rows whose width differs from the majority
    (a half-written last line in a running job) are dropped too.
    """
    rows, bad = [], 0
    with open(path) as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if not line:
                continue
            try:
                rows.append([float(x) for x in line.split()])
            except ValueError:
                bad += 1
    if not rows:
        sys.exit(f"ERROR: no numeric rows in '{path}'.")
    width = Counter(len(r) for r in rows).most_common(1)[0][0]
    ragged = sum(1 for r in rows if len(r) != width)
    rows = [r for r in rows if len(r) == width]
    if bad or ragged:
        print(f"  note: {os.path.basename(path)}: skipped {bad} unreadable and "
              f"{ragged} ragged row(s)")
    return np.array(rows, dtype=float)


def read_tagged(path, tag):
    """Lines beginning with `tag` -> float array of the fields after the tag."""
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if parts and parts[0] == tag:
                try:
                    rows.append([float(x) for x in parts[1:]])
                except ValueError:
                    continue
    if not rows:
        return np.empty((0, 0))
    width = Counter(len(r) for r in rows).most_common(1)[0][0]
    return np.array([r for r in rows if len(r) == width], dtype=float)


# =====================================================================================
# averages
# =====================================================================================

def moving_average(y, nwin):
    """Trailing mean over `nwin` points. Returns (x_index, values), x aligned to
    the last point of each window -- the same alignment pandas' rolling().mean()
    used in extract-ave.py."""
    y = np.asarray(y, dtype=float)
    if nwin < 2 or y.size < nwin:
        return np.arange(y.size), y
    kernel = np.ones(nwin) / nwin
    sma = np.convolve(y, kernel, mode="valid")
    return np.arange(nwin - 1, y.size), sma


def cumulative_average(y, neq):
    """Running mean of everything after the first `neq` points."""
    y = np.asarray(y, dtype=float)[neq:]
    if y.size == 0:
        return np.empty(0), np.empty(0)
    return np.arange(neq, neq + y.size), np.cumsum(y) / np.arange(1, y.size + 1)


def stats(y, neq):
    """(n, mean, std, drift) over the production part. `drift` compares the
    means of the first and last quarter -- a large drift means not equilibrated."""
    y = np.asarray(y, dtype=float)[neq:]
    if y.size == 0:
        return 0, np.nan, np.nan, np.nan
    q = max(1, y.size // 4)
    return y.size, float(y.mean()), float(y.std(ddof=1) if y.size > 1 else 0.0), \
        float(y[-q:].mean() - y[:q].mean())


# =====================================================================================
# monitor
# =====================================================================================

def draw_series(ax, y, color, label=None, nwin=100, raw_alpha=0.35, x0=1, ls="-"):
    """Raw trace faint, moving average solid on top."""
    x = np.arange(x0, x0 + len(y))
    ax.plot(x, y, color=color, lw=0.9, alpha=raw_alpha, zorder=2)
    mx, my = moving_average(y, nwin)
    ax.plot(mx + x0, my, color=color, lw=1.9, ls=ls, label=label, zorder=3)


def panel_single(ax, y, color, ylabel, nwin, neq, unit=""):
    draw_series(ax, y, color, nwin=nwin)
    n, mean, std, _ = stats(y, neq)
    if n:
        ax.axhline(mean, color=color, lw=1.0, ls="--", alpha=0.8, zorder=4)
        ax.text(0.975, 0.05, f"{mean:.4g} $\\pm$ {std:.2g} {unit}".strip(),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#cccccc", alpha=0.85))
    ax.set_xlabel("MD step")
    ax.set_ylabel(ylabel)
    frame(ax)


def panel_multi(ax, series, ylabel, nwin, ylim=None, legend_loc="best"):
    # a/b/c and alpha/beta/gamma routinely lie on top of each other (90 deg, or a
    # cell that is metrically hexagonal), so the dashes matter as much as colour
    styles = ["-", "--", ":"]
    for (y, color, label), ls in zip(series, styles):
        draw_series(ax, y, color, label=label, nwin=nwin, raw_alpha=0.25, ls=ls)
    ax.set_xlabel("MD step")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc=legend_loc, ncol=len(series), handlelength=2.0, columnspacing=1.0)
    frame(ax)


def panel_learning(ax, beef, err, beef_start, err_start):
    """ERR rmse_force, BEEF max force error and the CTIFOR threshold."""
    if err.size:
        ex, ey = err[err_start:, 0], err[err_start:, 2]
        order = np.argsort(ex)
        ax.plot(ex[order], ey[order], color=PALETTE["err"], lw=2.0,
                marker="o", ms=3.0, label="ERR (RMSE force)", zorder=5)
    if beef.size:
        ax.plot(beef[beef_start:, 0], beef[beef_start:, 2], color=PALETTE["beef"],
                lw=1.7, label="BEEF (max force err)", zorder=3)
        ax.plot(beef[beef_start:, 0], beef[beef_start:, 4], color=PALETTE["ctifor"],
                lw=1.7, label="CTIFOR (threshold)", zorder=2)
    ax.set_xlabel("MD step")
    ax.set_ylabel("Force error / threshold (eV/$\\AA$)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", handlelength=1.4)
    frame(ax)


def cmd_monitor(args):
    sfx = args.suffix
    src = FileSource(args.dir, args.host, args.remote_dir, args.cache, not args.no_fetch)
    names = [f"energy.dat{sfx}", f"pressure.dat{sfx}", f"volume.dat{sfx}",
             f"cell.dat{sfx}", f"BEEF{sfx}.dat", f"ERR{sfx}.dat",
             f"ML_LOGFILE{sfx}", "ML_LOGFILE"]
    src.prefetch(names)

    p_energy = src.path(f"energy.dat{sfx}")
    p_press = src.path(f"pressure.dat{sfx}")
    p_vol = src.path(f"volume.dat{sfx}")
    p_cell = src.path(f"cell.dat{sfx}")
    p_beef = src.path(f"BEEF{sfx}.dat", f"ML_LOGFILE{sfx}", "ML_LOGFILE")
    p_err = src.path(f"ERR{sfx}.dat", f"ML_LOGFILE{sfx}", "ML_LOGFILE")
    if not any([p_energy, p_press, p_vol, p_cell, p_beef]):
        sys.exit(f"ERROR: found nothing for suffix '{sfx}' in {src.dir}.")

    nwin, neq = args.window, args.neq
    print(f"\nreading suffix '{sfx}' from {src.dir}")

    energy = read_table(p_energy) if p_energy else None
    nsteps = len(energy) if energy is not None else None

    def trim(arr, what):
        """OUTCAR reports the input geometry as well; drop that leading row."""
        if arr is None or nsteps is None or args.no_trim:
            return arr
        if len(arr) == nsteps + 1:
            print(f"  note: {what} has {len(arr)} rows for {nsteps} MD steps -- "
                  f"dropping the leading (input geometry) row")
            return arr[1:]
        if len(arr) != nsteps:
            print(f"  note: {what} has {len(arr)} rows but energy has {nsteps}; "
                  f"plotted on its own step axis")
        return arr

    press = trim(read_table(p_press), "pressure") if p_press else None
    vol = trim(read_table(p_vol), "volume") if p_vol else None
    cell = trim(read_table(p_cell), "cell") if p_cell else None
    beef = read_tagged(p_beef, "BEEF") if p_beef else np.empty((0, 0))
    err = read_tagged(p_err, "ERR") if p_err else np.empty((0, 0))

    # --- panels, in the order they are drawn -------------------------------------
    panels, summary = [], []

    if energy is not None:
        T, E = energy[:, 0], energy[:, 1]
        panels.append(("Temperature",
                       lambda ax: panel_single(ax, T, PALETTE["temperature"],
                                               "Temperature (K)", nwin, neq, "K")))
        panels.append(("Energy",
                       lambda ax: panel_single(ax, E, PALETTE["energy"],
                                               "Energy $E$ (eV)", nwin, neq, "eV")))
        summary += [("temperature (K)", stats(T, neq)), ("energy E (eV)", stats(E, neq))]
        if energy.shape[1] > 2:
            summary.append(("free energy F (eV)", stats(energy[:, 2], neq)))

    if press is not None:
        P = press[:, 0]
        panels.append(("Pressure",
                       lambda ax: panel_single(ax, P, PALETTE["pressure"],
                                               "Pressure (kbar)", nwin, neq, "kbar")))
        summary.append(("pressure (kbar)", stats(P, neq)))

    if vol is not None:
        V = vol[:, 0]
        panels.append(("Volume",
                       lambda ax: panel_single(ax, V, PALETTE["volume"],
                                               "Volume ($\\AA^3$)", nwin, neq, "A^3")))
        summary.append(("volume (A^3)", stats(V, neq)))

    if cell is not None and cell.shape[1] >= 9:
        cols = (0, 1, 2) if args.abc == "diag" else (3, 4, 5)
        a, b, c = (cell[:, i] for i in cols)
        al, be, ga = cell[:, 6], cell[:, 7], cell[:, 8]
        panels.append(("Lattice parameters",
                       lambda ax: panel_multi(ax, [(a, PALETTE["a"], "$a$"),
                                                   (b, PALETTE["b"], "$b$"),
                                                   (c, PALETTE["c"], "$c$")],
                                              "Lattice parameters ($\\AA$)", nwin,
                                              args.abc_ylim)))
        panels.append(("Cell angles",
                       lambda ax: panel_multi(ax, [(al, PALETTE["a"], r"$\alpha$"),
                                                   (be, PALETTE["b"], r"$\beta$"),
                                                   (ga, PALETTE["c"], r"$\gamma$")],
                                              "Angles (deg)", nwin, args.ang_ylim)))
        label = "a11 a22 a33" if args.abc == "diag" else "|a| |b| |c|"
        summary += [(f"{label.split()[0]} (A)", stats(a, neq)),
                    (f"{label.split()[1]} (A)", stats(b, neq)),
                    (f"{label.split()[2]} (A)", stats(c, neq)),
                    ("alpha (deg)", stats(al, neq)),
                    ("beta (deg)", stats(be, neq)),
                    ("gamma (deg)", stats(ga, neq))]

    if beef.size or err.size:
        panels.append(("Learning curve",
                       lambda ax: panel_learning(ax, beef, err,
                                                 args.beef_start, args.err_start)))

    # --- render -------------------------------------------------------------------
    apply_style()
    out = args.out or f"monitor_{sfx}.png"
    if args.separate:
        stem = os.path.splitext(out)[0]
        for title, draw in panels:
            fig, ax = plt.subplots(figsize=(5.8, 4.1))
            draw(ax)
            fig.tight_layout()
            name = f"{stem}_{re.sub(r'[^A-Za-z0-9]+', '_', title).lower()}.png"
            fig.savefig(name, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"  wrote {name}")
    else:
        ncols = 2
        nrows = (len(panels) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, 3.5 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax, (title, draw) in zip(axes, panels):
            draw(ax)
            ax.set_title(title, fontsize=12, pad=6)
        for ax in axes[len(panels):]:
            ax.set_visible(False)
        fig.suptitle(f"{sfx}" + (f"   ({args.host}:{args.remote_dir})" if args.host else ""),
                     fontsize=13, y=1.0)
        fig.tight_layout()
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"\nwrote {out}")

    # --- the numbers ---------------------------------------------------------------
    print(f"\nproduction averages (skipping the first {neq} steps, "
          f"moving-average window {nwin}):")
    print(f"  {'quantity':<20} {'n':>7} {'mean':>14} {'sigma':>12} {'drift':>12}")
    for name, (n, mean, std, drift) in summary:
        if not n:
            print(f"  {name:<20} {'--':>7}   (fewer than {neq} steps)")
            continue
        print(f"  {name:<20} {n:>7} {mean:>14.5g} {std:>12.4g} {drift:>12.4g}")
    print("\n  drift = mean(last quarter) - mean(first quarter) of the production part;")
    print("  if it is not small compared with sigma, raise --neq.")


# =====================================================================================
# parity (from parity.py, unchanged in substance)
# =====================================================================================

def parse_ml_ab(path):
    """[(system_name, n_atoms), ...] in file order, keyed off the ML_AB labels."""
    names, natoms = [], []
    with open(path) as fh:
        lines = fh.readlines()
    i, n = 0, len(lines)
    while i < n:
        label = lines[i].strip()
        # the value always sits two lines below its header:
        #   <label> / ---------- / <value>
        if label == "System name":
            names.append(lines[i + 2].strip())
            i += 3
            continue
        # exact match: "The number of atom types" shares a prefix with this one
        if label == "The number of atoms":
            natoms.append(int(lines[i + 2].split()[0]))
            i += 3
            continue
        i += 1
    if len(names) != len(natoms):
        sys.exit(f"ERROR: ML_AB parse mismatch: {len(names)} 'System name' entries "
                 f"but {len(natoms)} 'The number of atoms' entries.")
    if not names:
        sys.exit(f"ERROR: no configurations found in '{path}'. Is it an ML_AB file?")
    return list(zip(names, natoms))


def parse_ml_reg(path):
    """(energies, forces, stress), each (M, 2): col 0 ab-initio, col 1 fitted."""
    sections = {"energy": [], "force": [], "stress": []}
    current = None
    with open(path) as fh:
        for line in fh:
            low = line.strip().lower()
            if low.startswith("total energies"):
                current = "energy"; continue
            if low.startswith("forces"):
                current = "force"; continue
            if low.startswith("stress"):
                current = "stress"; continue
            if current is None:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                sections[current].append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    energies = np.array(sections["energy"], dtype=float)
    if energies.size == 0:
        sys.exit(f"ERROR: no 'Total energies' data found in '{path}'.")
    return (energies,
            np.array(sections["force"], dtype=float),
            np.array(sections["stress"], dtype=float))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def parity_panel(ref, ml, color, system, kind_title, axis_label, rmse_value,
                 rmse_unit, rmse_fmt, marker_size, alpha, out_file, dpi):
    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    both = np.concatenate([ref, ml])
    lo, hi = both.min(), both.max()
    pad = 0.03 * (hi - lo) if hi > lo else 1.0
    lo, hi = lo - pad, hi + pad

    ax.plot([lo, hi], [lo, hi], "--", color=DIAGONAL_COLOR, lw=1.2, zorder=1)
    ax.scatter(ref, ml, s=marker_size, alpha=alpha, color=color,
               edgecolors="black", linewidths=0.3,
               rasterized=(ref.size > 5000), zorder=2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Reference / DFT  {axis_label}")
    ax.set_ylabel(f"ML predicted  {axis_label}")
    ax.set_title(f"{system}  --  {kind_title}")
    ax.grid(True, color="#dddddd", lw=0.6, zorder=0)
    ax.text(0.03, 0.97, f"RMSE {rmse_fmt.format(rmse_value)} {rmse_unit}\nn = {ref.size}",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", alpha=0.9))
    frame(ax)
    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_file}")


def cmd_parity(args):
    src = FileSource(args.dir, args.host, args.remote_dir, args.cache, not args.no_fetch)
    src.prefetch([args.ml_reg, args.ml_ab])
    p_ab, p_reg = src.path(args.ml_ab), src.path(args.ml_reg)
    if not p_ab or not p_reg:
        sys.exit(f"ERROR: need both {args.ml_ab} and {args.ml_reg} in {src.dir}.")

    print(f"reading {p_ab}")
    configs = parse_ml_ab(p_ab)
    names = [c[0] for c in configs]
    atoms = np.array([c[1] for c in configs], dtype=int)
    order = list(OrderedDict.fromkeys(names))
    colors = {n: PARITY_PALETTE[k % len(PARITY_PALETTE)] for k, n in enumerate(order)}
    print(f"  {len(configs)} configurations, {len(order)} distinct systems")

    print(f"reading {p_reg}")
    energies, forces, stress = parse_ml_reg(p_reg)
    print(f"  energy rows={len(energies)} force rows={len(forces)} stress rows={len(stress)}")

    if len(energies) != len(configs):
        sys.exit(f"ERROR: ML_REG has {len(energies)} energy entries but ML_AB has "
                 f"{len(configs)} configurations. They must come from the same run.")
    expected = int(3 * atoms.sum())
    if len(forces) != expected:
        sys.exit(f"ERROR: ML_REG has {len(forces)} force rows but ML_AB implies "
                 f"3*sum(N) = {expected}. They must match.")

    e_ref, e_ml = energies[:, 0].copy(), energies[:, 1].copy()
    if args.per_atom:
        e_ref, e_ml = e_ref / atoms, e_ml / atoms
        e_axis, e_unit, e_scale, e_fmt = "energy (eV/atom)", "meV/atom", 1000.0, "{:.2f}"
    else:
        e_axis, e_unit, e_scale, e_fmt = "total energy (eV)", "eV", 1.0, "{:.4f}"

    # forces: the flat list is sliced into consecutive 3N blocks, one per config
    starts = np.concatenate(([0], np.cumsum(3 * atoms)))
    f_ref = {n: [] for n in order}
    f_ml = {n: [] for n in order}
    for i, name in enumerate(names):
        s, e = starts[i], starts[i + 1]
        f_ref[name].append(forces[s:e, 0])
        f_ml[name].append(forces[s:e, 1])

    apply_style()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"plotting into '{args.out_dir}/' ({2 * len(order)} images):")
    summary = []
    for name in order:
        mask = np.array([nm == name for nm in names])
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
        er, em = e_ref[mask], e_ml[mask]
        e_rmse = rmse(er, em) * e_scale
        parity_panel(er, em, colors[name], name, "Energy parity (ML vs. DFT)",
                     e_axis, e_rmse, e_unit, e_fmt, 32, 0.85,
                     os.path.join(args.out_dir, f"parity_energy_{safe}.png"), args.dpi)
        fr = np.concatenate(f_ref[name]); fm = np.concatenate(f_ml[name])
        f_rmse = rmse(fr, fm)
        parity_panel(fr, fm, colors[name], name, "Force-component parity (ML vs. DFT)",
                     "force (eV/$\\AA$)", f_rmse, "eV/A", "{:.4f}", 6, 0.30,
                     os.path.join(args.out_dir, f"parity_forces_{safe}.png"), args.dpi)
        summary.append((name, er.size, e_rmse, fr.size, f_rmse))

    print("\nper-system RMSE:")
    print(f"  {'system':<16} {'n_struct':>8} {'E-RMSE':>12}   {'n_force':>9} {'F-RMSE (eV/A)':>14}")
    for name, ne, er_, nf, fr_ in summary:
        print(f"  {name:<16} {ne:>8} {e_fmt.format(er_):>8} {e_unit:<4}   "
              f"{nf:>9} {fr_:>14.4f}")


# =====================================================================================
# avg -- drop-in replacement for extract-ave.py
# =====================================================================================

def cmd_avg(args):
    """Writes FILE.ave (moving average) and FILE.cave (cumulative average), same
    layout as extract-ave.py: first column is the original row index."""
    data = read_table(args.file)
    n, ncols = data.shape

    def write(path, index, block):
        with open(path, "w") as fh:
            for i, row in zip(index, block):
                fh.write(" ".join([str(int(i))] + [repr(float(v)) for v in row]) + "\n")
        print(f"wrote {path} ({len(index)} rows)")

    if n >= args.window:
        idx = np.arange(args.window - 1, n)
        sma = np.column_stack([moving_average(data[:, j], args.window)[1]
                               for j in range(ncols)])
        write(args.file + ".ave", idx, sma)
    else:
        print(f"skipping {args.file}.ave: only {n} rows, window is {args.window}")

    if n > args.neq:
        tail = data[args.neq:]
        cave = np.cumsum(tail, axis=0) / np.arange(1, len(tail) + 1)[:, None]
        write(args.file + ".cave", np.arange(args.neq, n), cave)
    else:
        print(f"skipping {args.file}.cave: only {n} rows, neq is {args.neq}")


# =====================================================================================
# CLI
# =====================================================================================

def pair(text):
    lo, hi = text.split(",")
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    remote = argparse.ArgumentParser(add_help=False)
    remote.add_argument("--dir", default=".", help="local run directory (default: .)")
    remote.add_argument("--host", help="ssh target, e.g. aa2016@gadi.nci.org.au")
    remote.add_argument("--remote-dir", help="directory on --host")
    remote.add_argument("--cache", help="where fetched files land "
                                        "(default: .mlff_cache/<host>)")
    remote.add_argument("--no-fetch", action="store_true",
                        help="use whatever is already in the cache, do not ssh")
    remote.add_argument("--dpi", type=int, default=300)

    m = sub.add_parser("monitor", parents=[remote],
                       help="dashboard of T, E, P, V, cell and the learning curve")
    m.add_argument("--suffix", required=True, help="e.g. 300700tri")
    m.add_argument("--window", type=int, default=100, help="moving-average window")
    m.add_argument("--neq", type=int, default=600,
                   help="equilibration steps to skip in the averages")
    m.add_argument("--abc", choices=["norm", "diag"], default="norm",
                   help="lattice parameters from the vector norms (default) or "
                        "from a11/a22/a33 as the Mathematica notebook did")
    m.add_argument("--abc-ylim", type=pair, metavar="LO,HI")
    m.add_argument("--ang-ylim", type=pair, metavar="LO,HI")
    m.add_argument("--beef-start", type=int, default=0,
                   help="drop this many leading BEEF rows (the first few are huge)")
    m.add_argument("--err-start", type=int, default=0)
    m.add_argument("--no-trim", action="store_true",
                   help="keep the leading OUTCAR row in volume/pressure")
    m.add_argument("--separate", action="store_true", help="one file per panel")
    m.add_argument("--out", help="output png (default: monitor_<suffix>.png)")
    m.set_defaults(func=cmd_monitor)

    p = sub.add_parser("parity", parents=[remote],
                       help="per-system energy and force parity plots")
    p.add_argument("--ml-reg", default="ML_REG")
    p.add_argument("--ml-ab", default="ML_AB")
    p.add_argument("--out-dir", default="parity_plots")
    p.add_argument("--total-energy", dest="per_atom", action="store_false",
                   help="plot total energies instead of eV/atom")
    p.set_defaults(func=cmd_parity, per_atom=True)

    a = sub.add_parser("avg", help="moving and cumulative averages of a data file")
    a.add_argument("file")
    a.add_argument("window", type=int, help="moving-average window")
    a.add_argument("neq", type=int, help="equilibration steps to skip")
    a.set_defaults(func=cmd_avg)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
