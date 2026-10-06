#!/bin/bash
# -*- sh -*-
#
# md_cycle.sh -- close out one VASP machine-learning MD segment and stage the next.
#
# Replaces doextract.sh. The split of labour is deliberate:
#
#   md_cycle.sh       MUTATES the run directory. Runs on the cluster, in the job
#                     directory, between segments. Pure bash/awk: no python, no
#                     "module load", nothing to install.
#   mlff_monitor.py   READS the run directory (locally or over ssh) and makes all
#                     the plots and averages. Never touches the run.
#
# What it does, in order:
#   1. refuses to run if anything is missing or already archived under $SUFFIX
#   2. extracts energy / pressure / volume / cell / ML errors into .dat$SUFFIX
#   3. copies the BEEF and ERR lines out of ML_LOGFILE for the learning curve
#   4. updates TEBEG, TEEND and ML_CTIFOR in INCAR
#   5. archives XDATCAR, ML_REG, ML_LOGFILE, OSZICAR, ML_ABN under $SUFFIX
#   6. builds the next POSCAR out of CONTCAR, so the next segment continues the
#      trajectory: final positions AND final velocities, not a cold restart
#
# Usage:  ./md_cycle.sh SUFFIX TEBEG TEEND
#   e.g.  ./md_cycle.sh 300700tri 300 700
#
# Environment overrides:
#   FAC=1          divide lattice vectors by this (e.g. the supercell multiple)
#   CELL_MODE=diag keep only a11,a22,a33 from CONTCAR's cell and take the
#                  off-diagonals from the previous POSCAR, so the cell stays
#                  "sort of orthorhombic" (the doextract.sh behaviour).
#                  CELL_MODE=full takes CONTCAR's cell as it stands, which is
#                  what a strict continuation wants.
#   DROP_PC=1      cut the predictor-corrector block off the new POSCAR. Needed
#                  ONLY when the next segment changes ensemble (NVT -> NpT, say):
#                  those coordinates belong to the old ensemble and VASP crashes
#                  on them. Same ensemble: leave it alone.
#   DRY_RUN=1      do the extraction, skip every mv/gzip/sed
#
# Differences from doextract.sh worth knowing about:
#   * The POSCAR lattice patch no longer uses `sed s/$old/$new/g`, which replaced
#     that number string EVERYWHERE in POSCAR -- including in atomic coordinates
#     that happened to share the digits. Lines 3-5 are now rewritten in place.
#   * Running it twice with the same SUFFIX used to half-clobber the archive.
#     It now refuses.
#   * The XDATCAR parser keys off the "configuration=" markers instead of
#     counting lines modulo natoms, so it handles both NPT (header repeated every
#     step) and NVT (header written once), and any number of species.
#   * The moving/cumulative averages moved to `mlff_monitor.py avg`.
#   * The new POSCAR is built FROM CONTCAR rather than by patching the old
#     POSCAR's lattice, so it carries the final positions and the final
#     velocities. doextract.sh handed the next segment the old coordinates with
#     no velocities at all, i.e. a 0 K start, and every segment then spent its
#     first steps climbing back to temperature.
#
# On the velocity block (vasp.at/wiki/POSCAR, vasp.at/wiki/CONTCAR): after the
# ion positions CONTCAR writes an empty line, then one velocity per ion, then a
# further block of predictor-corrector coordinates. The empty line is not
# padding -- blank, like a leading C/c/K/k, selects "Cartesian" mode, so those
# velocities are in A/fs and a change of cell does not rescale them. This script
# therefore copies CONTCAR through byte for byte and rewrites nothing but lines
# 3-5.

set -euo pipefail

usage() {
    echo "usage: $(basename "$0") SUFFIX TEBEG TEEND" >&2
    echo "  e.g. $(basename "$0") 300700tri 300 700" >&2
    exit 1
}

die() { echo "md_cycle: $*" >&2; exit 1; }

# poscar_from_contcar OLD_POSCAR CHECK_ONLY
# Writes the next POSCAR to stdout: CONTCAR verbatim, except that its lattice
# lines are rewritten per CELL_MODE, taking the off-diagonals from OLD_POSCAR.
# Everything below the ion positions -- the empty line that marks the velocities
# Cartesian, the velocities themselves, the predictor-corrector block -- is
# copied through untouched, which is what makes the next segment a continuation
# rather than a 0 K restart. With CHECK_ONLY=1 it validates and prints nothing.
poscar_from_contcar() {
    awk -v mode="$CELL_MODE" -v droppc="$DROP_PC" -v check="$2" '
function allint(s,   i, n, p) {
    n = split(s, p, /[ \t]+/)
    for (i = 1; i <= n; i++) if (p[i] != "" && p[i] !~ /^[0-9]+$/) return 0
    return 1
}
function numeric(s,   i, n, p, k) {
    n = split(s, p, /[ \t]+/)
    for (i = 1; i <= n; i++) if (p[i] != "" && p[i] ~ /^[-+.0-9]/) k++
    return k
}
function bail(msg) { print "md_cycle: CONTCAR: " msg > "/dev/stderr"; exit 1 }

NR == FNR { if (FNR >= 3 && FNR <= 5) for (i = 1; i <= 3; i++) old[FNR,i] = $i; next }
{ c[FNR] = $0; last = FNR }

END {
    # counts line: line 6 (VASP4) or line 7 (VASP5, species names on 6)
    for (k = 6; k <= 7 && !cl; k++) if (c[k] != "" && allint(c[k])) cl = k
    if (!cl) bail("no ion-count line on line 6 or 7")
    n = split(c[cl], p, /[ \t]+/)
    for (i = 1; i <= n; i++) nat += p[i]
    if (nat < 1) bail("ion count is " nat)

    ml = cl + 1
    if (c[ml] ~ /^[ \t]*[Ss]/) ml++                       # "Selective dynamics"
    if (c[ml] !~ /^[ \t]*[DdCcKk]/) bail("no Direct/Cartesian line after the ion counts")

    pos = ml + 1                                          # first position line
    sep = pos + nat                                       # blank -> Cartesian velocities
    vel = sep + 1                                         # first velocity line
    for (k = pos; k < pos + nat; k++)
        if (numeric(c[k]) < 3) bail("position line " k " is short -- truncated file?")

    nvel = 0; vmax = 0
    if (last >= sep && (c[sep] ~ /^[ \t]*$/ || c[sep] ~ /^[ \t]*[DdCcKk]/)) {
        for (k = vel; k < vel + nat && k <= last; k++) {
            if (numeric(c[k]) < 3) break
            nvel++
            m = split(c[k], q, /[ \t]+/)
            for (i = 1; i <= m; i++) {
                if (q[i] == "") continue
                a = q[i] + 0; if (a < 0) a = -a
                if (a > vmax) vmax = a
            }
        }
    }
    if (nvel && nvel != nat) bail("velocity block has " nvel " of " nat " ions -- truncated file?")

    pc = (nvel ? vel + nat : 0)                           # predictor-corrector starts here
    if (check) exit 0                                     # pre-flight: errors only
    if (nvel == 0)
        print "md_cycle: WARNING: CONTCAR carries no velocities; the next segment" \
              " starts cold" > "/dev/stderr"
    else if (vmax == 0)
        print "md_cycle: WARNING: CONTCAR velocities are all zero; the next" \
              " segment starts cold" > "/dev/stderr"
    else
        printf "velocities: %d ions, max |v| = %.5f A/fs (Cartesian)%s\n",
               nvel, vmax, (pc && pc <= last ? ", predictor-corrector block follows" : "") \
               > "/dev/stderr"

    for (k = 1; k <= last; k++) {
        if (droppc == 1 && pc && k >= pc) break           # cut before the PC block
        if (k >= 3 && k <= 5) {
            r = k - 2
            nv = split(c[k], v, /[ \t]+/)
            j = 0
            for (i = 1; i <= nv; i++) if (v[i] != "" && ++j <= 3) w[j] = v[i]
            if (j < 3) bail("lattice line " k " does not have three components")
            for (i = 1; i <= 3; i++)
                printf "  %20.14f", (mode == "full" || i == r) ? w[i] : old[k,i]
            printf "\n"
        } else print c[k]
    }
}
    ' "$1" "$DIR/CONTCAR"
}

[ $# -eq 3 ] || usage
sfx=$1; tbeg=$2; tend=$3

case $tbeg$tend in
    *[!0-9.]*) die "TEBEG and TEEND must be numbers (got '$tbeg' '$tend')" ;;
esac

FAC=${FAC:-1}
CELL_MODE=${CELL_MODE:-diag}
DROP_PC=${DROP_PC:-0}
DRY_RUN=${DRY_RUN:-0}
DIR=${DIR:-.}

echo "suffix              : $sfx"
echo "next TEBEG -> TEEND : $tbeg -> $tend K"
echo "lattice mode        : $CELL_MODE (fac=$FAC)"
if [ "$DROP_PC" = 1 ]; then echo "predictor-corrector : dropped from the new POSCAR"; fi
if [ "$DRY_RUN" = 1 ]; then echo "DRY RUN             : nothing will be moved or edited"; fi

# ---------------------------------------------------------------------------
# 0. refuse to run on an incomplete or already-archived directory
# ---------------------------------------------------------------------------
for f in OSZICAR OUTCAR XDATCAR ML_LOGFILE INCAR POSCAR CONTCAR; do
    [ -s "$DIR/$f" ] || die "$f is missing or empty -- did the run finish?"
done
for f in "XDATCAR$sfx" "XDATCAR$sfx.gz" "OSZICAR$sfx" "ML_LOGFILE$sfx"; do
    [ -e "$DIR/$f" ] && die "$f already exists: suffix '$sfx' has been used. Pick another."
done
grep -q 'T=' "$DIR/OSZICAR" || die "no 'T=' lines in OSZICAR -- not an MD run?"

# CONTCAR becomes the next POSCAR, so check it is whole before anything is moved
if [ "$CELL_MODE" = diag ]; then
    # its off-diagonals come from the old POSCAR, so the two files have to agree
    # on the scale factor or splicing their lattice lines rescales the cell
    sc=$(awk 'NR==2 {print $1+0}' "$DIR/CONTCAR")
    sp=$(awk 'NR==2 {print $1+0}' "$DIR/POSCAR")
    awk -v a="$sc" -v b="$sp" 'BEGIN { exit !(a == b) }' \
        || die "CONTCAR scale ($sc) != POSCAR scale ($sp); refusing to mix lattices"
fi
poscar_from_contcar "$DIR/POSCAR" 1 > /dev/null || exit 1

# ---------------------------------------------------------------------------
# 1. energy, pressure, volume
# ---------------------------------------------------------------------------
energy=energy.dat$sfx; pressure=pressure.dat$sfx; volume=volume.dat$sfx
cell=cell.dat$sfx;     error=error-ml.dat$sfx
rm -f "$energy" "$pressure" "$volume" "$cell" "$error"

# OSZICAR MD line:  N T= <temp> E= <etot> F= <free energy> E0= ... EK= ...
echo '# T(K)    E(eV)    F(eV)' > "$energy"
grep 'T=' "$DIR/OSZICAR" | awk '{print $3, $5, $7}' >> "$energy"

echo '# Total pressure (kbar)' > "$pressure"
grep 'total pressure' "$DIR/OUTCAR" | awk '{print $4}' >> "$pressure" || true

echo '# Total volume (A^3)' > "$volume"
grep 'volume of cell' "$DIR/OUTCAR" | awk '{print $5}' >> "$volume" || true

# ---------------------------------------------------------------------------
# 2. cell: lattice vectors -> a11 a22 a33 |a| |b| |c| alpha beta gamma
# ---------------------------------------------------------------------------
# One row per "configuration=" marker. The header block preceding a marker is
# the cell of that configuration (NPT); if there is only one header it is reused
# for every configuration (NVT).
echo '# a11    a22    a33    a    b    c    alpha  beta  gamma' > "$cell"
awk -v fac="$FAC" '
function acos(x) { if (x > 1) x = 1; if (x < -1) x = -1; return atan2(sqrt(1 - x*x), x) }
function allint(s,   i, n, p) {
    n = split(s, p, /[ \t]+/)
    for (i = 1; i <= n; i++) if (p[i] != "" && p[i] !~ /^[0-9]+$/) return 0
    return 1
}
function emit(   a11, a12, a13, a21, a22, a23, a31, a32, a33, a, b, c, al, be, ga) {
    a11 = s*v[1,1]/fac; a12 = s*v[1,2]/fac; a13 = s*v[1,3]/fac
    a21 = s*v[2,1]/fac; a22 = s*v[2,2]/fac; a23 = s*v[2,3]/fac
    a31 = s*v[3,1]/fac; a32 = s*v[3,2]/fac; a33 = s*v[3,3]/fac
    a = sqrt(a11*a11 + a12*a12 + a13*a13)
    b = sqrt(a21*a21 + a22*a22 + a23*a23)
    c = sqrt(a31*a31 + a32*a32 + a33*a33)
    ga = acos((a11*a21 + a12*a22 + a13*a23)/(a*b))*deg
    be = acos((a11*a31 + a12*a32 + a13*a33)/(a*c))*deg
    al = acos((a21*a31 + a22*a32 + a23*a33)/(b*c))*deg
    printf "%14.8f %14.8f %14.8f %14.8f %14.8f %14.8f %10.4f %10.4f %10.4f\n",
           a11, a22, a33, a, b, c, al, be, ga
}
BEGIN { deg = 45.0/atan2(1, 1); s = 1; hdr = 0; skip = 0; nat = 0 }
/configuration=/ {
    if (nat == 0) { print "md_cycle: cannot read the XDATCAR header" > "/dev/stderr"; exit 1 }
    emit(); hdr = 0; skip = nat; next
}
skip > 0 { skip--; next }
{
    hdr++
    if      (hdr == 2)                 { s = $1 + 0 }
    else if (hdr >= 3 && hdr <= 5)     { v[hdr-2,1] = $1; v[hdr-2,2] = $2; v[hdr-2,3] = $3 }
    else if (hdr == 6 || hdr == 7)     { if (allint($0)) { nat = 0; for (i = 1; i <= NF; i++) nat += $i } }
}
' "$DIR/XDATCAR" >> "$cell"

ne=$(grep -vc '^#' < "$energy" || true)
np=$(grep -vc '^#' < "$pressure" || true)
nv=$(grep -vc '^#' < "$volume" || true)
nc=$(grep -vc '^#' < "$cell" || true)
printf 'extracted: %s steps energy, %s pressure, %s volume, %s cell\n' "$ne" "$np" "$nv" "$nc"
[ "$nc" -gt 0 ] || die "no configurations found in XDATCAR"
# OUTCAR prints "volume of cell" once for the input geometry as well, so nv is
# normally ne+1. mlff_monitor.py trims that leading entry when it plots.

# ---------------------------------------------------------------------------
# 3. ML errors, basis-set growth, learning curve
# ---------------------------------------------------------------------------
{
    printf '# Final errors: %s\n'  "$(grep ERR      "$DIR/ML_LOGFILE" | tail -1 | awk '{print $3, $4, $5}')"
    printf '# Final no. conf. and basis sets: %s\n' \
                                   "$(grep SPRSC    "$DIR/ML_LOGFILE" | tail -1 | awk '{print $4, $5, $7, $8, $10}')"
    printf '# No. of radial and angular descriptors per element: %s\n' \
                                   "$(grep 'NDESC ' "$DIR/ML_LOGFILE" | tail -1 | awk '{print $3, $5}')"
    echo '#'
    echo '#'
} > "$error"
paste <(grep 'ERR'   "$DIR/ML_LOGFILE" || true) \
      <(grep 'STDAB' "$DIR/ML_LOGFILE" || true) >> "$error"
printf '\n\n' >> "$error"
grep 'SPRSC' "$DIR/ML_LOGFILE" >> "$error" || true
printf '\n\n' >> "$error"
grep 'BEEF'  "$DIR/ML_LOGFILE" >> "$error" || true

grep 'BEEF' "$DIR/ML_LOGFILE" > "BEEF$sfx.dat" || true
grep 'ERR'  "$DIR/ML_LOGFILE" > "ERR$sfx.dat"  || true
echo "extracted: $(wc -l < "BEEF$sfx.dat") BEEF, $(wc -l < "ERR$sfx.dat") ERR lines"

if [ "$DRY_RUN" = 1 ]; then
    echo "dry run: stopping before any mv/gzip/sed."
    exit 0
fi

# ---------------------------------------------------------------------------
# 4. INCAR for the next segment
# ---------------------------------------------------------------------------
set_incar() {  # set_incar TAG VALUE -- replace the tag's line, or append it
    local tag=$1 val=$2
    if grep -qE "^[[:space:]]*$tag[[:space:]]*=" "$DIR/INCAR"; then
        sed -i "s|^[[:space:]]*$tag[[:space:]]*=.*|$tag = $val|" "$DIR/INCAR"
    else
        printf '%s = %s\n' "$tag" "$val" >> "$DIR/INCAR"
    fi
    printf '  INCAR: %-10s = %s\n' "$tag" "$val"
}
set_incar TEBEG "$tbeg"
set_incar TEEND "$tend"

cti=$(grep 'BEEF' "$DIR/ML_LOGFILE" | tail -1 | awk '{print $6}')
if [ -n "$cti" ]; then
    set_incar ML_CTIFOR "$cti"
else
    echo "  INCAR: no BEEF line found, ML_CTIFOR left as it was" >&2
fi

# ---------------------------------------------------------------------------
# 5. archive
# ---------------------------------------------------------------------------
mv "$DIR/XDATCAR" "XDATCAR$sfx" && gzip -f "XDATCAR$sfx"
for f in ML_REG ML_LOGFILE OSZICAR; do
    [ -e "$DIR/$f" ] && mv "$DIR/$f" "$f$sfx"
done
if [ -e "$DIR/ML_ABN" ]; then
    cp "$DIR/ML_ABN" "$DIR/ML_AB"          # ML_ABN becomes the next run's input
    mv "$DIR/ML_ABN" "ML_ABN$sfx"
fi
echo "archived under suffix $sfx"

# ---------------------------------------------------------------------------
# 6. next POSCAR = CONTCAR: final positions, final velocities, chosen cell
# ---------------------------------------------------------------------------
# CONTCAR is taken whole and only its lattice lines are rewritten, so the
# velocity block -- and the empty line in front of it that marks the velocities
# Cartesian -- reaches the next segment exactly as VASP wrote it.
cp "$DIR/POSCAR" "POSCAR$sfx"
if ! poscar_from_contcar "POSCAR$sfx" 0 > "$DIR/POSCAR.next"; then
    rm -f "$DIR/POSCAR.next"
    die "could not build the next POSCAR; the current one is untouched"
fi
mv "$DIR/POSCAR.next" "$DIR/POSCAR"

echo "POSCAR rebuilt from CONTCAR (previous one kept as POSCAR$sfx):"
sed -n '3,5p' "$DIR/POSCAR"
echo "ready for the next segment."
