#!/usr/bin/env bash
# =============================================================================
#  run_vasp_ml_ai.sh - two-stage VASP driver (ML force field -> ab initio)
#
#  Everything site-specific lives in the four USER SECTIONS below: paste your
#  own submission-script header, your INCAR, your KPOINTS, and the command line
#  that launches VASP. The script is scheduler-agnostic because it never writes
#  batch directives itself - it copies your header verbatim into the job script
#  and submits it with the command you name.
#
#  What it does:
#    Stage 1 (ml) : runs VASP with the ML force field (INCAR as given, ML_ tags
#                   active), restarting from CONTCAR until the run converges,
#                   up to MAX_ML_CYCLES times.
#    Stage 2 (ai) : runs VASP fully ab initio - same INCAR with every ML_ line
#                   commented out - starting from the CONTCAR stage 1 produced,
#                   up to MAX_AI_CYCLES times.
#
#  Usage:
#      ./run_vasp_ml_ai.sh                # write job script(s) and submit
#      ./run_vasp_ml_ai.sh write          # write job script(s), do not submit
#      ./run_vasp_ml_ai.sh run            # run both stages here and now
#      ./run_vasp_ml_ai.sh run ml         # run only the ML stage
#      ./run_vasp_ml_ai.sh run ai         # run only the ab initio stage
#
#  Inputs expected in the run directory: POSCAR, POTCAR, ML_FF.
#  INCAR and KPOINTS are written by the script; do not hand-edit them.
# =============================================================================

set -uo pipefail

# #############################################################################
#  USER SECTION 1 - your submission script header
# #############################################################################
#  Paste your own header between the two markers, exactly as your machine wants
#  it: shebang, batch directives, module loads, exports, ulimit - everything up
#  to (but not including) the line that actually launches VASP.
#
#  Two optional placeholders are substituted when the job script is written:
#      @JOBNAME@   -> JOB_NAME (with a -ml / -ai suffix when jobs are split)
#      @WALLTIME@  -> WALLTIME_ML or WALLTIME_AI (see the CONFIG section)
#  Use them if you want the per-stage walltimes to take effect; otherwise your
#  header is copied through untouched.
# #############################################################################
read -r -d '' SUBMIT_HEADER << 'END_SUBMIT_HEADER'
#!/bin/bash -l
#SBATCH --job-name=@JOBNAME@
#SBATCH --account=pawsey1141
#SBATCH --partition=work
#SBATCH --ntasks=64              # total no. of cpus requested
#SBATCH --ntasks-per-node=64     # no. of cpus/node
##SBATCH --exclusive             # uncomment if using a full node (ntasks=128)
#SBATCH --cpus-per-task=1
#SBATCH --mem=115G               # memory limit per node (here using half a node)
#SBATCH --time=@WALLTIME@

# VASP.6.4.3
module load hdf5/1.14.5-parallel-api-v112 netlib-scalapack/2.2.0 fftw/3.3.10
export VASP=/software/projects/pawsey1141/cverdi/vasp.6.4.3-vtst/bin/vasp_std
# VASP.6.5.1 (comment above and uncomment below)
#module load hdf5/1.14.5-parallel-api-v112 netlib-scalapack/2.2.0 fftw/3.3.10
#export VASP=/software/projects/pawsey1141/cverdi/vasp.6.5.1/bin/vasp_std

# OpenMP settings
export OMP_NUM_THREADS=1         # number of threads
# Settings for MPI jobs
export MPICH_OFI_STARTUP_CONNECT=1
export MPICH_OFI_VERBOSE=1
# Temporary workaround for avoiding Slingshot issues on shared nodes:
export FI_CXI_DEFAULT_VNI=$(od -vAn -N4 -tu < /dev/urandom | tr -d ' ')
# VASP uses a large amount of stack memory in addition to heap memory.
# This does not unlock more physical memory.
ulimit -s unlimited
END_SUBMIT_HEADER

# #############################################################################
#  USER SECTION 2 - INCAR
# #############################################################################
#  Write it with the ML tags ACTIVE. Stage 2 reuses this same INCAR with every
#  line matching ML_LINE_REGEX (default: any tag starting with ML_) commented
#  out, so you only maintain one INCAR.
# #############################################################################
read -r -d '' INCAR_TEMPLATE << 'END_INCAR'
SYSTEM = test

# ab initio

PREC   = Accurate
EDIFFG = -0.01
NELMIN = 2             # no. of e. self-consistency steps
IVDW = 12
LCHARG = .FALSE.
LWAVE = .FALSE.
NUPDOWN = 0
ISPIN = 2
ISMEAR = 0             # Gaussian smearing
SIGMA = 0.05

LREAL = Auto           # Projection operators in real space

ENCUT = 500            # Plane wave cutoff

IBRION = 2             # MD (treat ionic degrees of freedom)
NSW    = 300           # no of ionic steps
ISYM = 0               # disable when relaxing a defect
POTIM  = 0.5           # MD time step in fs
ISIF = 2               # update positions, cell shape and volume
LATTICE_CONSTRAINTS = .TRUE. .TRUE. .FALSE.

# machine-learned force field (commented out automatically for stage 2)
ML_LMLFF = T
ML_MODE = run
END_INCAR

# Optional: a completely separate INCAR for the ab initio stage. Leave empty to
# use INCAR_TEMPLATE with the ML_ lines commented out (the normal case).
INCAR_AI_TEMPLATE=""

# #############################################################################
#  USER SECTION 3 - KPOINTS
# #############################################################################
#  Leave empty (KPOINTS_TEMPLATE="") to use a KPOINTS file you provide yourself.
# #############################################################################
read -r -d '' KPOINTS_TEMPLATE << 'END_KPOINTS'
hBN     # System label
 0
Gamma
 1 1 1
 0  0  0
END_KPOINTS

# #############################################################################
#  USER SECTION 4 - how VASP is launched
# #############################################################################
#  Paste your own launch line, WITHOUT redirection (stdout goes to vasp.out).
#  Single quotes matter: the variables are expanded inside the job, not now, so
#  scheduler variables such as $SLURM_NTASKS work as they do in your script.
#  Examples:
#      'srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS -c $OMP_NUM_THREADS -m block:block:block "$VASP"'
#      'mpirun -np $PBS_NP "$VASP"'
#      'mpiexec -n 64 "$VASP"'
#      '"$VASP"'                      # serial / already-parallel wrapper
# #############################################################################
VASP_CMD='srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS -c $OMP_NUM_THREADS -m block:block:block "$VASP"'

# #############################################################################
#  CONFIG
# #############################################################################

# ---- submission ------------------------------------------------------------
SUBMIT_CMD="${SUBMIT_CMD:-sbatch}"          # qsub | bsub | sbatch | ...
SUBMIT_VIA_STDIN="${SUBMIT_VIA_STDIN:-no}"  # yes for LSF ("bsub < script")
JOB_NAME="${JOB_NAME:-vasp}"

# ---- walltimes (substituted into @WALLTIME@ in your header) ----------------
# MD / ML force-field cycles are normally much cheaper than the ab initio ones.
WALLTIME_ML="${WALLTIME_ML:-04:00:00}"      # stage 1
WALLTIME_AI="${WALLTIME_AI:-24:00:00}"      # stage 2

# ---- job layout ------------------------------------------------------------
#   single : one job runs both stages; @WALLTIME@ becomes WALLTIME_ML+WALLTIME_AI
#   chain  : one job per stage, each with its own walltime. The ML job submits
#            the ab initio job itself once it converges, so no scheduler
#            dependency syntax is needed (requires that the submit command works
#            from a compute node - true on most, but not all, machines).
JOB_LAYOUT="${JOB_LAYOUT:-single}"

# ---- workflow --------------------------------------------------------------
MAX_ML_CYCLES="${MAX_ML_CYCLES:-4}"         # restart cap, ML stage
MAX_AI_CYCLES="${MAX_AI_CYCLES:-2}"         # restart cap, ab initio stage

# Convergence is detected by this line in vasp.out.
CONV_REGEX="${CONV_REGEX:-reached required accuracy - stopping structural energy minimi[sz]ation}"
# For a pure MD run (IBRION = 0) VASP never prints that line; set the matching
# flag to "no" and normal termination (DONE_REGEX) is accepted instead.
REQUIRE_CONV_ML="${REQUIRE_CONV_ML:-yes}"
REQUIRE_CONV_AI="${REQUIRE_CONV_AI:-yes}"
DONE_REGEX="${DONE_REGEX:-General timing and accounting}"

# INCAR lines commented out for the ab initio stage.
ML_LINE_REGEX="${ML_LINE_REGEX:-^[[:space:]]*ML_}"

# Files that must be present before a stage starts.
REQUIRED_ML="${REQUIRED_ML:-POSCAR POTCAR ML_FF}"
REQUIRED_AI="${REQUIRED_AI:-POSCAR POTCAR}"

# When ML_MODE is train/refit, promote the new force field between cycles.
PROMOTE_ML_FF="${PROMOTE_ML_FF:-no}"

# Files archived after every cycle, into stage_ml/ and stage_ai/.
ARCHIVE_FILES="${ARCHIVE_FILES:-INCAR vasp.out OUTCAR OSZICAR CONTCAR XDATCAR REPORT ML_LOGFILE ML_ABN ML_FFN}"

# #############################################################################
#  END OF USER-EDITABLE PART
# #############################################################################

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

log  () { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
die  () { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
#  walltime arithmetic (HH:MM:SS)
# ---------------------------------------------------------------------------
wt_seconds () {
    local t="$1" h=0 m=0 s=0
    case "$t" in
        *:*:*) IFS=: read -r h m s <<< "$t" ;;
        *:*)   IFS=: read -r m s   <<< "$t" ;;
        *)     s="$t" ;;
    esac
    printf '%s' $(( 10#$h * 3600 + 10#$m * 60 + 10#$s ))
}
wt_sum () {
    local s=$(( $(wt_seconds "$1") + $(wt_seconds "$2") ))
    printf '%02d:%02d:%02d' $(( s/3600 )) $(( (s%3600)/60 )) $(( s%60 ))
}

# ---------------------------------------------------------------------------
#  input files
# ---------------------------------------------------------------------------
write_incar () {                    # $1 = ml | ai
    if [[ "$1" == ml ]]; then
        printf '%s\n' "$INCAR_TEMPLATE" > INCAR
    elif [[ -n "$INCAR_AI_TEMPLATE" ]]; then
        printf '%s\n' "$INCAR_AI_TEMPLATE" > INCAR
    else
        # same INCAR, every ML_ line commented out
        printf '%s\n' "$INCAR_TEMPLATE" |
            awk -v re="$ML_LINE_REGEX" \
                '{ if ($0 ~ re && $0 !~ /^[[:space:]]*#/) print "#" $0; else print }' > INCAR
    fi
}

write_kpoints () {
    if [[ -n "$KPOINTS_TEMPLATE" ]]; then
        printf '%s\n' "$KPOINTS_TEMPLATE" > KPOINTS
    elif [[ ! -s KPOINTS ]]; then
        die "KPOINTS_TEMPLATE is empty and no KPOINTS file is present in $(pwd)."
    fi
}

check_inputs () {                   # $1 = space-separated file list
    local f
    for f in $1; do
        [[ -s "$f" ]] || die "required input '$f' not found or empty in $(pwd)."
    done
}

# ---------------------------------------------------------------------------
#  running VASP
# ---------------------------------------------------------------------------
run_vasp () {
    # one VASP invocation; vasp.out is overwritten each call
    log "launching: $VASP_CMD"
    eval "$VASP_CMD" > vasp.out 2>&1
}

converged  () { [[ -f vasp.out ]] && grep -qE -- "$CONV_REGEX" vasp.out; }
terminated () { [[ -f vasp.out ]] && grep -qE -- "$DONE_REGEX" vasp.out; }

seed_from_contcar () {
    [[ -s CONTCAR ]] || die "CONTCAR missing or empty; cannot continue."
    cp POSCAR "POSCAR.bak.$(date +%Y%m%d-%H%M%S)"
    cp CONTCAR POSCAR
}

archive_cycle () {                  # $1 = stage, $2 = cycle
    local dir="stage_$1" f
    mkdir -p "$dir"
    for f in $ARCHIVE_FILES; do
        [[ -s "$f" ]] && cp -f "$f" "$dir/${f}.cycle$2"
    done
    return 0
}

promote_ml_ff () {
    [[ "$PROMOTE_ML_FF" == yes ]] || return 0
    [[ -s ML_FFN ]] && cp ML_FFN ML_FF
    [[ -s ML_ABN ]] && cp ML_ABN ML_AB
    return 0
}

run_stage () {                      # $1 = ml | ai
    local stage="$1" label max_cycles need_conv required cyc rc

    case "$stage" in
        ml) label="ML force field"; max_cycles="$MAX_ML_CYCLES"
            need_conv="$REQUIRE_CONV_ML"; required="$REQUIRED_ML" ;;
        ai) label="ab initio";      max_cycles="$MAX_AI_CYCLES"
            need_conv="$REQUIRE_CONV_AI"; required="$REQUIRED_AI" ;;
        *)  die "unknown stage '$stage' (expected ml or ai)." ;;
    esac

    # A finished stage is skipped, so a requeued or re-run job resumes.
    if [[ -f ".stage_${stage}.done" ]]; then
        log "=== stage ${stage} (${label}) already complete - skipping ==="
        return 0
    fi

    log "=== stage ${stage}: ${label} ==="
    check_inputs "$required"
    write_kpoints
    write_incar "$stage"

    for (( cyc=1; cyc<=max_cycles; cyc++ )); do
        log "--- ${stage} cycle ${cyc} of ${max_cycles} ---"
        rc=0
        run_vasp || rc=$?
        archive_cycle "$stage" "$cyc"
        [[ "$stage" == ml ]] && promote_ml_ff
        [[ "$rc" -ne 0 ]] && log "WARNING: VASP exited with status ${rc}."

        if [[ "$need_conv" == yes ]]; then
            if converged; then
                log "${label} converged on cycle ${cyc}."
                seed_from_contcar          # hand the geometry to the next stage
                touch ".stage_${stage}.done"
                return 0
            fi
            log "${label} not converged yet; restarting from CONTCAR."
        else
            if terminated && [[ "$rc" -eq 0 ]]; then
                log "${label} finished normally on cycle ${cyc}."
                seed_from_contcar
                touch ".stage_${stage}.done"
                return 0
            fi
            log "${label} did not terminate normally; restarting from CONTCAR."
        fi
        seed_from_contcar
    done

    die "stage ${stage} (${label}) did not finish within ${max_cycles} cycles."
}

# ---------------------------------------------------------------------------
#  run mode - executed inside the job (or interactively)
# ---------------------------------------------------------------------------
do_run () {                         # $1 = ml | ai | all
    local what="${1:-all}"

    # Re-apply the header: batch directives are comments, so this just replays
    # the module loads, exports and ulimit. Makes "run" work standalone too.
    eval "$SUBMIT_HEADER"

    log "run mode '${what}' in $(pwd)"
    [[ -n "${VASP:-}" ]] || die "\$VASP is not set - export it in USER SECTION 1."

    case "$what" in
        ml)  run_stage ml ;;
        ai)  run_stage ai ;;
        all) run_stage ml; run_stage ai ;;
        *)   die "unknown stage '$what' (expected ml, ai or all)." ;;
    esac

    # In chain layout the ML job submits the ab initio job itself.
    if [[ "$what" == ml && "$JOB_LAYOUT" == chain ]]; then
        log "submitting the ab initio stage"
        submit_stage ai
    fi

    log "=== finished '${what}' successfully ==="
}

# ---------------------------------------------------------------------------
#  job-script writing and submission
# ---------------------------------------------------------------------------
job_script_name () {                # $1 = ml | ai | all
    printf 'job_%s_%s.sub' "$JOB_NAME" "$1"
}

write_job_script () {               # $1 = ml | ai | all  -> prints the filename
    local what="$1" file hdr name walltime
    file="$(job_script_name "$what")"

    case "$what" in
        ml)  name="${JOB_NAME}-ml"; walltime="$WALLTIME_ML" ;;
        ai)  name="${JOB_NAME}-ai"; walltime="$WALLTIME_AI" ;;
        all) name="$JOB_NAME";      walltime="$(wt_sum "$WALLTIME_ML" "$WALLTIME_AI")" ;;
    esac

    hdr="$SUBMIT_HEADER"
    hdr="${hdr//@JOBNAME@/$name}"
    hdr="${hdr//@WALLTIME@/$walltime}"

    {
        printf '%s\n\n' "$hdr"
        printf '# ---------------------------------------------------------------\n'
        printf '#  Written by run_vasp_ml_ai.sh - edit that script, not this one.\n'
        printf '#  stage(s): %s   walltime: %s\n' "$what" "$walltime"
        printf '# ---------------------------------------------------------------\n'
        printf 'cd %q || exit 1\n' "$PWD"
        printf 'exec bash %q run %s\n' "$SELF" "$what"
    } > "$file"
    chmod +x "$file"
    printf '%s' "$file"
}

submit_stage () {                   # $1 = ml | ai | all
    local file; file="$(write_job_script "$1")"
    log "submitting $file with: $SUBMIT_CMD"
    if [[ "$SUBMIT_VIA_STDIN" == yes ]]; then
        $SUBMIT_CMD < "$file"
    else
        $SUBMIT_CMD "$file"
    fi
}

do_submit () {                      # $1 = write | submit
    local action="$1" stages file

    case "$JOB_LAYOUT" in
        single) stages="all" ;;
        chain)  stages="ml" ;;       # the ML job submits the ab initio job
        *)      die "JOB_LAYOUT must be 'single' or 'chain'." ;;
    esac

    if [[ "$JOB_LAYOUT" == chain && "$SUBMIT_HEADER" != *@WALLTIME@* ]]; then
        log "NOTE: your header has no @WALLTIME@ placeholder, so WALLTIME_ML"
        log "      and WALLTIME_AI are not applied - both jobs use the walltime"
        log "      hardcoded in the header."
    fi

    if [[ "$action" == write ]]; then
        for stage in $stages; do
            file="$(write_job_script "$stage")"
            log "wrote $file"
        done
        [[ "$JOB_LAYOUT" == chain ]] && log "(the ab initio job script is written when the ML job finishes)"
    else
        for stage in $stages; do submit_stage "$stage"; done
    fi
}

# ---------------------------------------------------------------------------
#  entry point
# ---------------------------------------------------------------------------
case "${1:-submit}" in
    submit) do_submit submit ;;
    write)  do_submit write ;;
    run)    do_run "${2:-all}" ;;
    -h|--help|help)
        sed -n '2,32p' "$SELF" ;;
    *)      die "unknown command '$1' (expected submit, write or run)." ;;
esac
