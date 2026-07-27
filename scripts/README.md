# `make_vasp_submit.sh` — scheduler-agnostic VASP submission scripts

A generator, not a runner. You configure the job once at the top of
`make_vasp_submit.sh`; it writes the batch script(s) with the right directives
for whichever scheduler the machine happens to use, plus a `submit_all.sh`
driver that submits them with the correct dependency syntax.

Supported schedulers: `slurm`, `pbs` (PBSPro/Torque), `lsf`, `sge`, and `none`
(plain shell, e.g. a workstation or an interactive allocation).

## The workflow it generates

Same two-stage strategy in every case:

| Stage | INCAR | Loop |
|-------|-------|------|
| 1 — `ml` | ML tags active (`ML_LMLFF = T`, `ML_MODE = run`) | restart from `CONTCAR` until converged, up to `MAX_ML_CYCLES` |
| 2 — `ai` | ML tags commented out | seeded from the stage-1 geometry, up to `MAX_AI_CYCLES` |

Convergence is detected in `vasp.out` via
`reached required accuracy - stopping structural energy minimisation`.
For a plain MD run (`IBRION = 0`) that line never appears, so set
`REQUIRE_CONV_ML=no` and normal termination (`General timing and accounting`)
is used instead.

Each cycle's `vasp.out`, `OUTCAR`, `OSZICAR`, `CONTCAR`, `XDATCAR`, `INCAR`,
and ML files are archived under `stage_ml/` and `stage_ai/`. A completed stage
drops a `.stage_<name>.done` marker, so a requeued job resumes rather than
redoing finished work.

## Walltimes

The two stages get separate hardcoded walltimes, since ML/MD work is normally
much cheaper than ab initio:

```bash
WALLTIME_ML="${WALLTIME_ML:-04:00:00}"      # stage 1
WALLTIME_AI="${WALLTIME_AI:-24:00:00}"      # stage 2
```

With `SPLIT_JOBS=yes` (default) each stage becomes its own job carrying its own
walltime, chained so stage 2 only starts if stage 1 succeeds. With
`SPLIT_JOBS=no` both stages run in one job whose walltime is the sum.

## Usage

```bash
./make_vasp_submit.sh                  # write scripts into ./
./submit_all.sh                        # submit them

# any config variable can also be overridden from the environment
SCHEDULER=pbs QUEUE=normal ./make_vasp_submit.sh
WALLTIME_ML=02:00:00 WALLTIME_AI=48:00:00 NODES=2 ./make_vasp_submit.sh
SCHEDULER=none LAUNCHER=mpirun OUTDIR=runs/defect_a ./make_vasp_submit.sh
```

Required inputs in the run directory: `POSCAR`, `POTCAR`, and `ML_FF` (the last
only when `ML_MODE=run`). `INCAR` and `KPOINTS` are written by the job itself.

## Configuration reference

| Variable | Meaning |
|---|---|
| `SCHEDULER` | `slurm` \| `pbs` \| `lsf` \| `sge` \| `none` |
| `SPLIT_JOBS` | `yes` = two chained jobs; `no` = one job, walltimes summed |
| `WALLTIME_ML` / `WALLTIME_AI` | per-stage walltime, `HH:MM:SS` (converted per scheduler) |
| `JOB_NAME`, `ACCOUNT`, `QUEUE` | naming and accounting; empty `ACCOUNT`/`QUEUE` omits the directive |
| `NODES`, `NTASKS_PER_NODE`, `CPUS_PER_TASK` | ranks = `NODES × NTASKS_PER_NODE`; `CPUS_PER_TASK` sets `OMP_NUM_THREADS` |
| `MEM_PER_NODE`, `EXCLUSIVE` | memory request (empty = omit), whole-node request |
| `VASP_BIN`, `MODULES`, `ENV_EXPORTS` | binary, modules to load, extra environment |
| `LAUNCHER`, `LAUNCHER_EXTRA` | `auto` picks `srun` under Slurm, `mpirun` otherwise; `none` runs serially |
| `SLINGSHOT_VNI_WORKAROUND`, `UNLIMIT_STACK` | Cray shared-node `FI_CXI_DEFAULT_VNI` fix; `ulimit -s unlimited` |
| `MAX_ML_CYCLES`, `MAX_AI_CYCLES` | restart caps per stage |
| `REQUIRE_CONV_ML`, `REQUIRE_CONV_AI` | require the convergence line, or accept normal termination |
| `CONV_REGEX`, `DONE_REGEX` | patterns matched against `vasp.out` |
| `ML_MODE` | `run` \| `train` \| `refit`; non-`run` promotes `ML_FFN`→`ML_FF` between cycles |
| `SYSTEM_NAME`, `INCAR_COMMON`, `INCAR_ML_EXTRA`, `INCAR_AI_EXTRA`, `INCAR_ML_TAGS` | INCAR contents; the per-stage arrays override matching keys in `INCAR_COMMON` |
| `WRITE_KPOINTS`, `KPOINTS_MODE`, `KPOINTS_MESH`, `KPOINTS_SHIFT` | KPOINTS generation (`WRITE_KPOINTS=no` keeps an existing file) |
| `OUTDIR` | where the generated scripts are written |

The `INCAR_*` and `ENV_EXPORTS` settings are bash arrays, so edit those in the
file rather than passing them through the environment.

## Porting to a new machine

Usually three lines: `SCHEDULER`, `VASP_BIN`, `MODULES` — plus `ACCOUNT`/`QUEUE`
and the resource block. Everything scheduler-specific (directives, the MPI
launcher, walltime format, job-dependency syntax, the submit-directory `cd`)
is derived from `SCHEDULER`.
