#!/usr/bin/env bash
# =============================================================================
#  make_vasp_submit.sh - generate scheduler-agnostic VASP submission scripts
#
#  This script does not run VASP. It *writes* the submission script(s) that do,
#  emitting the correct batch directives for the chosen scheduler (Slurm, PBS/
#  Torque, LSF, SGE, or none/plain shell) from a single configuration block.
#
#  The generated scripts implement the two-stage workflow:
#
#    Stage 1 (ml) : machine-learned force field run (ML_LMLFF = T), typically
#                   an MD or a cheap ML-driven relaxation. Loops, restarting
#                   from CONTCAR, until it converges (or until the cycle cap).
#    Stage 2 (ai) : pure ab initio run with the ML tags commented out, seeded
#                   from the geometry produced by stage 1.
#
#  Because the two stages have very different cost, each gets its own walltime:
#  WALLTIME_ML and WALLTIME_AI (see the CONFIGURATION block below).
#
#  Usage:
#      ./make_vasp_submit.sh              # write scripts into ./ (OUTDIR)
#      SCHEDULER=pbs ./make_vasp_submit.sh
#      WALLTIME_ML=04:00:00 WALLTIME_AI=24:00:00 ./make_vasp_submit.sh
#
#  Every configuration variable below can also be overridden from the
#  environment, so the hardcoded values act as defaults, not as a straitjacket.
# =============================================================================

set -euo pipefail

# =============================================================================
#  CONFIGURATION - edit these
# =============================================================================

# ---- scheduler -------------------------------------------------------------
# slurm | pbs | lsf | sge | none
SCHEDULER="${SCHEDULER:-slurm}"

# Split the workflow into two chained jobs (one per stage, each with its own
# walltime) or run both stages inside a single job whose walltime is the sum.
SPLIT_JOBS="${SPLIT_JOBS:-yes}"

# ---- walltimes (HH:MM:SS) --------------------------------------------------
# MD / ML force-field work is normally much cheaper than the ab initio stage.
WALLTIME_ML="${WALLTIME_ML:-04:00:00}"      # stage 1: ML force field / MD
WALLTIME_AI="${WALLTIME_AI:-24:00:00}"      # stage 2: ab initio

# ---- job / accounting ------------------------------------------------------
JOB_NAME="${JOB_NAME:-vasp}"
ACCOUNT="${ACCOUNT-pawsey1141}"            # empty string -> directive omitted
QUEUE="${QUEUE-work}"                      # partition (Slurm) / queue (others)

# ---- resources -------------------------------------------------------------
NODES="${NODES:-1}"
NTASKS_PER_NODE="${NTASKS_PER_NODE:-64}"    # MPI ranks per node
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"         # OpenMP threads per rank
MEM_PER_NODE="${MEM_PER_NODE-115G}"        # empty string -> directive omitted
EXCLUSIVE="${EXCLUSIVE:-no}"                # yes -> request whole nodes

# ---- code / environment ----------------------------------------------------
VASP_BIN="${VASP_BIN:-/software/projects/pawsey1141/cverdi/vasp.6.4.3-vtst/bin/vasp_std}"

# Modules loaded inside the job (space separated, in load order). May be empty.
MODULES="${MODULES-hdf5/1.14.5-parallel-api-v112 netlib-scalapack/2.2.0 fftw/3.3.10}"

# Extra "KEY=VALUE" exports for the job environment, one per array element.
ENV_EXPORTS=(
    "MPICH_OFI_STARTUP_CONNECT=1"
    "MPICH_OFI_VERBOSE=1"
)

# HPE Slingshot shared-node workaround (random FI_CXI_DEFAULT_VNI). Cray only.
SLINGSHOT_VNI_WORKAROUND="${SLINGSHOT_VNI_WORKAROUND:-yes}"

# VASP needs a large stack; "ulimit -s unlimited" inside the job.
UNLIMIT_STACK="${UNLIMIT_STACK:-yes}"

# MPI launcher: auto | srun | mpirun | mpiexec | none
#   auto -> srun under Slurm, mpirun everywhere else, none when SCHEDULER=none
LAUNCHER="${LAUNCHER:-auto}"
# Extra launcher flags appended verbatim (e.g. Slurm rank placement).
LAUNCHER_EXTRA="${LAUNCHER_EXTRA--m block:block:block}"

# ---- workflow control ------------------------------------------------------
MAX_ML_CYCLES="${MAX_ML_CYCLES:-4}"         # restart cap, stage 1
MAX_AI_CYCLES="${MAX_AI_CYCLES:-2}"         # restart cap, stage 2

# Require the ionic-relaxation convergence line before a stage is called done.
# Set the ML one to "no" for a plain MD run (IBRION = 0), where VASP simply
# runs NSW steps and never prints that line; normal termination is then enough.
REQUIRE_CONV_ML="${REQUIRE_CONV_ML:-yes}"
REQUIRE_CONV_AI="${REQUIRE_CONV_AI:-yes}"

# Regexes matched against vasp.out.
CONV_REGEX="${CONV_REGEX:-reached required accuracy - stopping structural energy minimi[sz]ation}"
DONE_REGEX="${DONE_REGEX:-General timing and accounting}"

# ML mode for stage 1: run (use existing ML_FF) | train | refit
ML_MODE="${ML_MODE:-run}"
# Input files that must exist before the job starts. ML_FF is added
# automatically when ML_MODE=run.
REQUIRED_INPUTS="${REQUIRED_INPUTS:-POSCAR POTCAR}"

# ---- INCAR -----------------------------------------------------------------
SYSTEM_NAME="${SYSTEM_NAME:-test}"

# Tags shared by both stages. Comment lines and blank lines are passed through.
INCAR_COMMON=(
    "PREC   = Accurate"
    "EDIFFG = -0.01"
    "NELMIN = 2              # min. no. of electronic self-consistency steps"
    "IVDW   = 12"
    "LCHARG = .FALSE."
    "LWAVE  = .FALSE."
    "NUPDOWN = 0"
    "ISPIN  = 2"
    "ISMEAR = 0              # Gaussian smearing"
    "SIGMA  = 0.05"
    "LREAL  = Auto           # projection operators in real space"
    "ENCUT  = 500            # plane-wave cutoff (eV)"
    "IBRION = 2"
    "NSW    = 300            # no. of ionic steps"
    "ISYM   = 0              # disable when relaxing a defect"
    "POTIM  = 0.5            # ionic step / MD time step (fs)"
    "ISIF   = 2              # relax positions only"
    "LATTICE_CONSTRAINTS = .TRUE. .TRUE. .FALSE."
)

# Stage-specific tags. A key given here replaces the same key in INCAR_COMMON,
# so the two stages can differ (cheaper cutoff, more steps for MD, ...).
INCAR_ML_EXTRA=(
    "NSW    = 300"
)
INCAR_AI_EXTRA=(
    "NSW    = 300"
)

# ML tags: written as-is in stage 1, commented out in stage 2.
INCAR_ML_TAGS=(
    "ML_LMLFF = T"
    "ML_MODE  = ${ML_MODE}"
)

# ---- KPOINTS ---------------------------------------------------------------
# Set WRITE_KPOINTS=no to keep an existing hand-written KPOINTS file.
WRITE_KPOINTS="${WRITE_KPOINTS:-yes}"
KPOINTS_LABEL="${KPOINTS_LABEL:-${SYSTEM_NAME}}"
KPOINTS_MODE="${KPOINTS_MODE:-Gamma}"       # Gamma | Monkhorst
KPOINTS_MESH="${KPOINTS_MESH:-1 1 1}"
KPOINTS_SHIFT="${KPOINTS_SHIFT:-0 0 0}"

# ---- output ----------------------------------------------------------------
OUTDIR="${OUTDIR:-.}"

# =============================================================================
#  END OF CONFIGURATION - implementation below
# =============================================================================

die () { printf 'make_vasp_submit.sh: %s\n' "$*" >&2; exit 1; }

NTASKS=$(( NODES * NTASKS_PER_NODE ))

case "$SCHEDULER" in
    slurm|pbs|lsf|sge|none) ;;
    *) die "unknown SCHEDULER '$SCHEDULER' (slurm|pbs|lsf|sge|none)" ;;
esac
case "$SPLIT_JOBS" in yes|no) ;; *) die "SPLIT_JOBS must be yes or no" ;; esac

# ---------------------------------------------------------------------------
#  walltime helpers
# ---------------------------------------------------------------------------
wt_seconds () {                     # HH:MM:SS or MM:SS or SS -> seconds
    local t="$1" h=0 m=0 s=0
    case "$t" in
        *:*:*) IFS=: read -r h m s <<< "$t" ;;
        *:*)   IFS=: read -r m s   <<< "$t" ;;
        *)     s="$t" ;;
    esac
    printf '%s' $(( 10#$h * 3600 + 10#$m * 60 + 10#$s ))
}
wt_hms   () { local s="$1"; printf '%02d:%02d:%02d' $(( s/3600 )) $(( (s%3600)/60 )) $(( s%60 )); }
wt_hhmm  () { local s="$1"; printf '%d:%02d'        $(( s/3600 )) $(( (s%3600)/60 )); }   # LSF
wt_sum   () { wt_hms $(( $(wt_seconds "$1") + $(wt_seconds "$2") )); }

# ---------------------------------------------------------------------------
#  INCAR assembly
# ---------------------------------------------------------------------------
tag_key () {                        # "ENCUT  = 500   # comment" -> "ENCUT"
    local k="${1%%=*}"
    k="${k#"${k%%[![:space:]]*}"}"
    k="${k%"${k##*[![:space:]]}"}"
    printf '%s' "$k"
}

merge_incar () {                    # $1 = base array name, $2 = override name
    local base_name="$1" ovr_name="$2"
    eval "local base=(\"\${${base_name}[@]}\")"
    eval "local ovr=(\"\${${ovr_name}[@]}\")"
    local line key o okey used="" out=()

    for line in ${base+"${base[@]}"}; do
        case "$line" in
            ''|'#'*) out+=("$line"); continue ;;
        esac
        key="$(tag_key "$line")"
        for o in ${ovr+"${ovr[@]}"}; do
            okey="$(tag_key "$o")"
            if [[ "$okey" == "$key" ]]; then line="$o"; used="$used $okey"; break; fi
        done
        out+=("$line")
    done
    for o in ${ovr+"${ovr[@]}"}; do            # overrides that added new keys
        okey="$(tag_key "$o")"
        [[ " $used " == *" $okey "* ]] || out+=("$o")
    done
    printf '%s\n' ${out+"${out[@]}"}
}

incar_for_stage () {                # $1 = ml | ai  -> full INCAR text
    local stage="$1" body tag
    if [[ "$stage" == "ml" ]]; then
        body="$(merge_incar INCAR_COMMON INCAR_ML_EXTRA)"
    else
        body="$(merge_incar INCAR_COMMON INCAR_AI_EXTRA)"
    fi
    printf 'SYSTEM = %s\n\n' "$SYSTEM_NAME"
    printf '%s\n\n' "$body"
    if [[ "$stage" == "ml" ]]; then
        printf '# machine-learned force field\n'
        for tag in ${INCAR_ML_TAGS+"${INCAR_ML_TAGS[@]}"}; do printf '%s\n' "$tag"; done
    else
        printf '# machine-learned force field (disabled: pure ab initio stage)\n'
        for tag in ${INCAR_ML_TAGS+"${INCAR_ML_TAGS[@]}"}; do printf '#%s\n' "$tag"; done
    fi
}

kpoints_text () {
    printf '%s\n' "$KPOINTS_LABEL" \
                  " 0" \
                  "$KPOINTS_MODE" \
                  " $KPOINTS_MESH" \
                  " $KPOINTS_SHIFT"
}

# ---------------------------------------------------------------------------
#  scheduler directives
# ---------------------------------------------------------------------------
emit_directives () {                # $1 = job name, $2 = walltime HH:MM:SS
    local name="$1" wt="$2" secs; secs="$(wt_seconds "$2")"
    case "$SCHEDULER" in
    slurm)
        printf '#SBATCH --job-name=%s\n' "$name"
        [[ -n "$ACCOUNT" ]] && printf '#SBATCH --account=%s\n' "$ACCOUNT"
        [[ -n "$QUEUE"   ]] && printf '#SBATCH --partition=%s\n' "$QUEUE"
        printf '#SBATCH --nodes=%s\n'            "$NODES"
        printf '#SBATCH --ntasks=%s\n'           "$NTASKS"
        printf '#SBATCH --ntasks-per-node=%s\n'  "$NTASKS_PER_NODE"
        printf '#SBATCH --cpus-per-task=%s\n'    "$CPUS_PER_TASK"
        [[ -n "$MEM_PER_NODE" ]] && printf '#SBATCH --mem=%s\n' "$MEM_PER_NODE"
        [[ "$EXCLUSIVE" == yes ]] && printf '#SBATCH --exclusive\n'
        printf '#SBATCH --time=%s\n'             "$wt"
        printf '#SBATCH --output=%s-%%j.out\n'   "$name"
        printf '#SBATCH --error=%s-%%j.err\n'    "$name"
        ;;
    pbs)
        printf '#PBS -N %s\n' "$name"
        [[ -n "$ACCOUNT" ]] && printf '#PBS -A %s\n' "$ACCOUNT"
        [[ -n "$QUEUE"   ]] && printf '#PBS -q %s\n' "$QUEUE"
        printf '#PBS -l select=%s:ncpus=%s:mpiprocs=%s:ompthreads=%s%s\n' \
               "$NODES" "$(( NTASKS_PER_NODE * CPUS_PER_TASK ))" \
               "$NTASKS_PER_NODE" "$CPUS_PER_TASK" \
               "${MEM_PER_NODE:+:mem=$MEM_PER_NODE}"
        printf '#PBS -l walltime=%s\n' "$wt"
        printf '#PBS -j oe\n'
        ;;
    lsf)
        printf '#BSUB -J %s\n' "$name"
        [[ -n "$ACCOUNT" ]] && printf '#BSUB -P %s\n' "$ACCOUNT"
        [[ -n "$QUEUE"   ]] && printf '#BSUB -q %s\n' "$QUEUE"
        printf '#BSUB -n %s\n' "$NTASKS"
        printf '#BSUB -R "span[ptile=%s]"\n' "$NTASKS_PER_NODE"
        [[ -n "$MEM_PER_NODE" ]] && printf '#BSUB -R "rusage[mem=%s]"\n' "$MEM_PER_NODE"
        [[ "$EXCLUSIVE" == yes ]] && printf '#BSUB -x\n'
        printf '#BSUB -W %s\n' "$(wt_hhmm "$secs")"
        printf '#BSUB -o %s-%%J.out\n' "$name"
        printf '#BSUB -e %s-%%J.err\n' "$name"
        ;;
    sge)
        printf '#$ -N %s\n' "$name"
        printf '#$ -cwd\n'
        printf '#$ -j y\n'
        [[ -n "$ACCOUNT" ]] && printf '#$ -A %s\n' "$ACCOUNT"
        [[ -n "$QUEUE"   ]] && printf '#$ -q %s\n' "$QUEUE"
        printf '#$ -pe %s %s\n' "${SGE_PE:-mpi}" "$NTASKS"
        printf '#$ -l h_rt=%s\n' "$wt"
        ;;
    none)
        printf '# no scheduler: run this script directly (walltime %s is advisory)\n' "$wt"
        ;;
    esac
}

launcher_cmd () {                   # array literal for the generated script
    local l="$LAUNCHER"
    if [[ "$l" == auto ]]; then
        if   [[ "$SCHEDULER" == none  ]]; then l=mpirun
        elif [[ "$SCHEDULER" == slurm ]]; then l=srun
        else l=mpirun; fi
    fi
    case "$l" in
        none)   printf '' ;;
        srun)   printf 'srun -N %s -n %s -c %s %s' "$NODES" "$NTASKS" "$CPUS_PER_TASK" "$LAUNCHER_EXTRA" ;;
        mpirun|mpiexec)
                printf '%s -n %s %s' "$l" "$NTASKS" "${LAUNCHER_EXTRA/-m block:block:block/}" ;;
        *)      printf '%s' "$l" ;;
    esac
}

submit_cmd () {                     # how the generated scripts get submitted
    case "$SCHEDULER" in
        slurm) printf 'sbatch' ;;
        pbs|sge) printf 'qsub' ;;
        lsf)   printf 'bsub' ;;
        none)  printf 'bash' ;;
    esac
}

# ---------------------------------------------------------------------------
#  script writer
# ---------------------------------------------------------------------------
write_stage_script () {             # $1 = file, $2 = job name, $3 = walltime,
                                    # $4 = stage list ("ml", "ai", "ml ai")
    local file="$1" name="$2" wt="$3" stages="$4"
    local required="$REQUIRED_INPUTS"
    [[ "$stages" == *ml* && "$ML_MODE" == run ]] && required="$required ML_FF"

    {
        printf '#!/bin/bash -l\n'
        emit_directives "$name" "$wt"
        printf '\n'
        cat << EOF
# ---------------------------------------------------------------------------
#  Generated by make_vasp_submit.sh - do not edit by hand; edit the generator.
#  scheduler : ${SCHEDULER}
#  stages    : ${stages}
#  walltime  : ${wt}
#  resources : ${NODES} node(s) x ${NTASKS_PER_NODE} rank(s) x ${CPUS_PER_TASK} thread(s) = ${NTASKS} rank(s)
# ---------------------------------------------------------------------------

set -u

STAGES="${stages}"
VASP="${VASP_BIN}"
MODULES="${MODULES}"
LAUNCH=($(launcher_cmd))
OMP_THREADS="${CPUS_PER_TASK}"
MAX_ML_CYCLES=${MAX_ML_CYCLES}
MAX_AI_CYCLES=${MAX_AI_CYCLES}
REQUIRE_CONV_ML="${REQUIRE_CONV_ML}"
REQUIRE_CONV_AI="${REQUIRE_CONV_AI}"
ML_MODE="${ML_MODE}"
WRITE_KPOINTS="${WRITE_KPOINTS}"
SLINGSHOT_VNI_WORKAROUND="${SLINGSHOT_VNI_WORKAROUND}"
UNLIMIT_STACK="${UNLIMIT_STACK}"
REQUIRED_INPUTS="${required}"
CONV_REGEX='${CONV_REGEX}'
DONE_REGEX='${DONE_REGEX}'
ENV_EXPORTS=(
EOF
        local e
        for e in ${ENV_EXPORTS+"${ENV_EXPORTS[@]}"}; do printf '    %q\n' "$e"; done
        printf ')\n\n'

        printf "read -r -d '' INCAR_ML << 'CLAUDE_INCAR_ML_EOF' || true\n"
        incar_for_stage ml
        printf 'CLAUDE_INCAR_ML_EOF\n\n'

        printf "read -r -d '' INCAR_AI << 'CLAUDE_INCAR_AI_EOF' || true\n"
        incar_for_stage ai
        printf 'CLAUDE_INCAR_AI_EOF\n\n'

        printf "read -r -d '' KPOINTS_TEXT << 'CLAUDE_KPOINTS_EOF' || true\n"
        kpoints_text
        printf 'CLAUDE_KPOINTS_EOF\n\n'

        cat << 'CLAUDE_BODY_EOF'
# ---------------------------------------------------------------------------
#  job environment
# ---------------------------------------------------------------------------
# Start in the directory the job was submitted from, whatever the scheduler.
for d in "${SLURM_SUBMIT_DIR:-}" "${PBS_O_WORKDIR:-}" "${LS_SUBCWD:-}" "${SGE_O_WORKDIR:-}"; do
    [[ -n "$d" && -d "$d" ]] && cd "$d" && break
done
echo "Working directory: $(pwd)"

if [[ -n "$MODULES" ]] && command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    module load $MODULES
fi

export OMP_NUM_THREADS="$OMP_THREADS"
for kv in ${ENV_EXPORTS+"${ENV_EXPORTS[@]}"}; do export "${kv?}"; done

# Temporary workaround for Slingshot issues on shared Cray nodes.
if [[ "$SLINGSHOT_VNI_WORKAROUND" == yes && -r /dev/urandom ]]; then
    export FI_CXI_DEFAULT_VNI=$(od -vAn -N4 -tu < /dev/urandom | tr -d ' ')
fi

# VASP uses a lot of stack in addition to heap; this unlocks no extra RAM.
[[ "$UNLIMIT_STACK" == yes ]] && ulimit -s unlimited 2>/dev/null || true

# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------
run_vasp () {
    # single VASP invocation; overwrites vasp.out each call
    if [[ ${#LAUNCH[@]} -gt 0 ]]; then
        "${LAUNCH[@]}" "$VASP" > vasp.out 2>&1
    else
        "$VASP" > vasp.out 2>&1
    fi
}

converged () {  [[ -f vasp.out ]] && grep -qE -- "$CONV_REGEX" vasp.out; }
terminated () { [[ -f vasp.out ]] && grep -qE -- "$DONE_REGEX" vasp.out; }

seed_from_contcar () {
    # carry the relaxed / evolved geometry forward into the next cycle
    if [[ -s CONTCAR ]]; then
        cp POSCAR "POSCAR.bak.$(date +%Y%m%d-%H%M%S)"
        cp CONTCAR POSCAR
    else
        echo "ERROR: CONTCAR missing or empty; cannot continue." >&2
        exit 1
    fi
}

promote_ml_ff () {
    # when training/refitting, the new force field becomes the next input
    if [[ "$ML_MODE" != run ]]; then
        [[ -s ML_FFN ]] && cp ML_FFN ML_FF
        [[ -s ML_ABN ]] && cp ML_ABN ML_AB
    fi
    return 0
}

archive_cycle () {                  # $1 = stage, $2 = cycle
    local dir="stage_$1"
    mkdir -p "$dir"
    local f
    for f in vasp.out OUTCAR OSZICAR CONTCAR XDATCAR REPORT ML_LOGFILE ML_ABN ML_FFN INCAR; do
        [[ -s "$f" ]] && cp -f "$f" "$dir/${f}.cycle$2"
    done
    return 0
}

write_incar () {                    # $1 = ml | ai
    if [[ "$1" == ml ]]; then printf '%s\n' "$INCAR_ML" > INCAR
    else                      printf '%s\n' "$INCAR_AI" > INCAR
    fi
}

# ---------------------------------------------------------------------------
#  input sanity checks
# ---------------------------------------------------------------------------
for f in $REQUIRED_INPUTS; do
    if [[ ! -s "$f" ]]; then
        echo "ERROR: required input '$f' not found or empty in $(pwd)." >&2
        exit 1
    fi
done

if [[ "$WRITE_KPOINTS" == yes ]]; then
    printf '%s\n' "$KPOINTS_TEXT" > KPOINTS
elif [[ ! -s KPOINTS ]]; then
    echo "ERROR: WRITE_KPOINTS=no but no KPOINTS file is present." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
#  stage driver
# ---------------------------------------------------------------------------
for stage in $STAGES; do
    case "$stage" in
        ml) label="ML force field / MD"; max_cycles="$MAX_ML_CYCLES"; need_conv="$REQUIRE_CONV_ML" ;;
        ai) label="ab initio";           max_cycles="$MAX_AI_CYCLES"; need_conv="$REQUIRE_CONV_AI" ;;
        *)  echo "ERROR: unknown stage '$stage'." >&2; exit 1 ;;
    esac

    # A stage already finished by an earlier run of this script is skipped, so
    # a requeued job resumes instead of starting over.
    if [[ -f ".stage_${stage}.done" ]]; then
        echo "=== STAGE ${stage} (${label}) already complete - skipping ==="
        continue
    fi

    echo "=== STAGE ${stage}: ${label} relaxation ==="
    write_incar "$stage"

    stage_done=0
    for (( cyc=1; cyc<=max_cycles; cyc++ )); do
        echo "--- ${stage} cycle ${cyc} of ${max_cycles} ($(date)) ---"
        rc=0
        run_vasp || rc=$?
        archive_cycle "$stage" "$cyc"
        [[ "$stage" == ml ]] && promote_ml_ff

        if [[ "$rc" -ne 0 ]]; then
            echo "WARNING: VASP exited with status ${rc} on ${stage} cycle ${cyc}." >&2
        fi

        if [[ "$need_conv" == yes ]]; then
            if converged; then
                echo "${label} converged on cycle ${cyc}."
                seed_from_contcar
                stage_done=1
                break
            fi
            echo "${label} not yet converged; restarting from CONTCAR."
            seed_from_contcar
        else
            # e.g. plain MD: no convergence line is ever printed, so normal
            # termination of the run is what we check for.
            if terminated && [[ "$rc" -eq 0 ]]; then
                echo "${label} finished normally on cycle ${cyc}."
                seed_from_contcar
                stage_done=1
                break
            fi
            echo "${label} did not terminate normally; restarting from CONTCAR."
            seed_from_contcar
        fi
    done

    if [[ "$stage_done" -ne 1 ]]; then
        echo "ERROR: stage ${stage} (${label}) did not finish within ${max_cycles} cycles." >&2
        exit 2
    fi
    touch ".stage_${stage}.done"
done

echo "=== DONE: stages '${STAGES}' completed successfully ($(date)). ==="
CLAUDE_BODY_EOF
    } > "$file"
    chmod +x "$file"
}

write_submit_driver () {            # $1 = file, $2... = generated scripts
    local file="$1"; shift
    local scripts=("$@")
    {
        printf '#!/usr/bin/env bash\n'
        cat << EOF
# Submit the generated VASP job script(s) for SCHEDULER=${SCHEDULER}.
# Stage 2 only starts if stage 1 finishes successfully.
set -euo pipefail
cd "\$(dirname "\$0")"

EOF
        if [[ ${#scripts[@]} -eq 1 ]]; then
            case "$SCHEDULER" in
                lsf)  printf 'bsub < %q\n' "${scripts[0]}" ;;
                none) printf 'bash %q\n'   "${scripts[0]}" ;;
                *)    printf '%s %q\n' "$(submit_cmd)" "${scripts[0]}" ;;
            esac
        else
            case "$SCHEDULER" in
            slurm)
                cat << EOF
jid1=\$(sbatch --parsable ${scripts[0]})
echo "stage 1 (ML/MD) submitted as \$jid1"
jid2=\$(sbatch --parsable --dependency=afterok:\$jid1 ${scripts[1]})
echo "stage 2 (ab initio) submitted as \$jid2 (after \$jid1)"
EOF
                ;;
            pbs)
                cat << EOF
jid1=\$(qsub ${scripts[0]})
echo "stage 1 (ML/MD) submitted as \$jid1"
jid2=\$(qsub -W depend=afterok:\$jid1 ${scripts[1]})
echo "stage 2 (ab initio) submitted as \$jid2 (after \$jid1)"
EOF
                ;;
            lsf)
                cat << EOF
out=\$(bsub < ${scripts[0]})
echo "\$out"
jid1=\$(sed -E 's/.*<([0-9]+)>.*/\1/' <<< "\$out")
bsub -w "done(\$jid1)" < ${scripts[1]}
EOF
                ;;
            sge)
                cat << EOF
out=\$(qsub -terse ${scripts[0]})
jid1=\${out%%.*}
echo "stage 1 (ML/MD) submitted as \$jid1"
qsub -hold_jid "\$jid1" ${scripts[1]}
EOF
                ;;
            none)
                cat << EOF
bash ${scripts[0]}
bash ${scripts[1]}
EOF
                ;;
            esac
        fi
    } > "$file"
    chmod +x "$file"
}

# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------
mkdir -p "$OUTDIR"

case "$SCHEDULER" in
    slurm) ext="slurm" ;; pbs) ext="pbs" ;; lsf) ext="lsf" ;;
    sge)   ext="sge"   ;; none) ext="sh"  ;;
esac

generated=()
if [[ "$SPLIT_JOBS" == yes ]]; then
    s1="submit_ml.${ext}"; s2="submit_ai.${ext}"
    write_stage_script "$OUTDIR/$s1" "${JOB_NAME}-ml" "$WALLTIME_ML" "ml"
    write_stage_script "$OUTDIR/$s2" "${JOB_NAME}-ai" "$WALLTIME_AI" "ai"
    generated=("$s1" "$s2")
else
    s1="submit_${JOB_NAME}.${ext}"
    write_stage_script "$OUTDIR/$s1" "$JOB_NAME" "$(wt_sum "$WALLTIME_ML" "$WALLTIME_AI")" "ml ai"
    generated=("$s1")
fi
write_submit_driver "$OUTDIR/submit_all.sh" ${generated+"${generated[@]}"}

cat << EOF
Generated in ${OUTDIR}:
$(printf '  %s\n' ${generated+"${generated[@]}"} submit_all.sh)

  scheduler        : ${SCHEDULER}
  job layout       : $( [[ "$SPLIT_JOBS" == yes ]] && echo "two chained jobs" || echo "single job, both stages" )
  walltime (ML/MD) : ${WALLTIME_ML}
  walltime (ab in.): ${WALLTIME_AI}
  resources        : ${NODES} node(s) x ${NTASKS_PER_NODE} rank(s) x ${CPUS_PER_TASK} thread(s) = ${NTASKS} rank(s)
  binary           : ${VASP_BIN}

Submit with:  ./submit_all.sh
EOF
