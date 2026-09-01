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
#   6. copies the relaxed lattice from CONTCAR into POSCAR
#
# Usage:  ./md_cycle.sh SUFFIX TEBEG TEEND
#   e.g.  ./md_cycle.sh 300700tri 300 700
#
# Environment overrides:
#   FAC=1          divide lattice vectors by this (e.g. the supercell multiple)
#   CELL_MODE=diag copy only a11,a22,a33 from CONTCAR into POSCAR (keeps the cell
#                  "sort of orthorhombic", the doextract.sh behaviour).
#                  CELL_MODE=full copies all nine components instead.
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

set -euo pipefail

usage() {
    echo "usage: $(basename "$0") SUFFIX TEBEG TEEND" >&2
    echo "  e.g. $(basename "$0") 300700tri 300 700" >&2
    exit 1
}

die() { echo "md_cycle: $*" >&2; exit 1; }

[ $# -eq 3 ] || usage
sfx=$1; tbeg=$2; tend=$3

case $tbeg$tend in
    *[!0-9.]*) die "TEBEG and TEEND must be numbers (got '$tbeg' '$tend')" ;;
esac

FAC=${FAC:-1}
CELL_MODE=${CELL_MODE:-diag}
DRY_RUN=${DRY_RUN:-0}
DIR=${DIR:-.}

echo "suffix              : $sfx"
echo "next TEBEG -> TEEND : $tbeg -> $tend K"
echo "lattice mode        : $CELL_MODE (fac=$FAC)"
[ "$DRY_RUN" = 1 ] && echo "DRY RUN             : nothing will be moved or edited"

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
# 6. carry the relaxed lattice from CONTCAR into POSCAR
# ---------------------------------------------------------------------------
# CONTCAR and POSCAR must share a scale factor, otherwise mixing their lattice
# lines silently rescales the cell.
sc=$(awk 'NR==2 {print $1+0}' "$DIR/CONTCAR")
sp=$(awk 'NR==2 {print $1+0}' "$DIR/POSCAR")
awk -v a="$sc" -v b="$sp" 'BEGIN { exit !(a == b) }' \
    || die "CONTCAR scale ($sc) != POSCAR scale ($sp); refusing to mix lattices"

cp "$DIR/POSCAR" "POSCAR$sfx"
awk -v mode="$CELL_MODE" '
NR == FNR { if (FNR >= 3 && FNR <= 5) for (i = 1; i <= 3; i++) n[FNR,i] = $i; next }
FNR >= 3 && FNR <= 5 {
    r = FNR - 2
    for (i = 1; i <= 3; i++) {
        if (mode == "full" || i == r) v[i] = n[FNR,i]; else v[i] = $i
    }
    printf "  %20.14f %20.14f %20.14f\n", v[1], v[2], v[3]
    next
}
{ print }
' "$DIR/CONTCAR" "POSCAR$sfx" > "$DIR/POSCAR"

echo "POSCAR lattice updated from CONTCAR (previous POSCAR kept as POSCAR$sfx):"
sed -n '3,5p' "$DIR/POSCAR"
echo "ready for the next segment."
