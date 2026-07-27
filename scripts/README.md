# `run_vasp_ml_ai.sh` — two-stage VASP driver (ML force field → ab initio)

One script that runs VASP. Everything machine-specific lives in four heredoc
sections at the top, which you fill in by pasting your own files:

| Section | You paste | Marker |
|---|---|---|
| 1 | your submission-script header (directives, modules, exports, `ulimit`) | `END_SUBMIT_HEADER` |
| 2 | your INCAR, with the ML tags active | `END_INCAR` |
| 3 | your KPOINTS | `END_KPOINTS` |
| 4 | the line that launches VASP (`srun …`, `mpirun …`) | `VASP_CMD=` |

It is scheduler-agnostic because it never writes batch directives itself: your
header is copied verbatim into the generated job script and submitted with the
command you name in `SUBMIT_CMD` (`sbatch`, `qsub`, `bsub`, …).

## What it does

1. **Stage `ml`** — writes INCAR and KPOINTS, runs VASP with the ML force field
   active, and if the run has not converged copies `CONTCAR` → `POSCAR` and runs
   again, up to `MAX_ML_CYCLES` times.
2. **Stage `ai`** — writes the *same* INCAR with every `ML_` line commented out,
   starting from the `CONTCAR` stage 1 produced, up to `MAX_AI_CYCLES` times.

Convergence is the usual VASP line in `vasp.out`:
`reached required accuracy - stopping structural energy minimisation`.

Each cycle's `INCAR`, `vasp.out`, `OUTCAR`, `OSZICAR`, `CONTCAR`, `XDATCAR` and
ML files are archived into `stage_ml/` and `stage_ai/` as `<file>.cycleN`, and
each `POSCAR` is backed up before being overwritten. A finished stage drops a
`.stage_<name>.done` marker, so a requeued or re-run job resumes instead of
repeating work already done.

## Usage

```bash
./run_vasp_ml_ai.sh              # write the job script and submit it
./run_vasp_ml_ai.sh write        # write the job script, don't submit
./run_vasp_ml_ai.sh run          # run both stages here and now
./run_vasp_ml_ai.sh run ml       # run only the ML stage
./run_vasp_ml_ai.sh run ai       # run only the ab initio stage
```

Inputs needed in the run directory: `POSCAR`, `POTCAR`, `ML_FF`. `INCAR` and
`KPOINTS` are written by the script — edit the heredocs, not those files.

`run` replays your header first (batch directives are just comments), so it
works unchanged inside an interactive allocation or on a workstation.

## Walltimes and job layout

The two stages have separate hardcoded walltimes, since ML/MD cycles are
normally much cheaper than ab initio ones:

```bash
WALLTIME_ML="${WALLTIME_ML:-04:00:00}"      # stage 1
WALLTIME_AI="${WALLTIME_AI:-24:00:00}"      # stage 2
```

They are substituted into the `@WALLTIME@` placeholder in your header (and
`@JOBNAME@` for the job name). How they are used depends on `JOB_LAYOUT`:

- `single` (default) — one job runs both stages; `@WALLTIME@` becomes the sum
  (`28:00:00` with the defaults above).
- `chain` — one job per stage, each with its own walltime. The ML job submits
  the ab initio job itself once it converges, so no scheduler dependency syntax
  is involved. Requires that your submit command works from a compute node,
  which most but not all sites allow.

If your header has no `@WALLTIME@` placeholder, it is copied through untouched
and whatever walltime you hardcoded in it applies to both stages; `chain` mode
says so when you submit.

## Other settings

| Variable | Meaning |
|---|---|
| `SUBMIT_CMD`, `SUBMIT_VIA_STDIN` | how to submit (`SUBMIT_VIA_STDIN=yes` gives `bsub < script` for LSF) |
| `JOB_NAME` | base job name, `-ml`/`-ai` suffixed in chain layout |
| `MAX_ML_CYCLES`, `MAX_AI_CYCLES` | restart caps per stage |
| `REQUIRE_CONV_ML`, `REQUIRE_CONV_AI` | set to `no` for a pure MD run (`IBRION = 0`), where the convergence line is never printed and normal termination is accepted instead |
| `CONV_REGEX`, `DONE_REGEX` | patterns matched against `vasp.out` |
| `ML_LINE_REGEX` | which INCAR lines get commented out for stage 2 (default: anything starting with `ML_`) |
| `INCAR_AI_TEMPLATE` | optional separate INCAR for stage 2 instead of auto-commenting |
| `REQUIRED_ML`, `REQUIRED_AI` | files checked before each stage starts |
| `PROMOTE_ML_FF` | `yes` copies `ML_FFN`→`ML_FF` between cycles (for `ML_MODE = train`/`refit`) |
| `ARCHIVE_FILES` | files copied into `stage_*/` after every cycle |

All of them can also be set from the environment for a one-off run, e.g.
`WALLTIME_AI=48:00:00 JOB_LAYOUT=chain ./run_vasp_ml_ai.sh`.

## Porting to another machine

Replace section 1 with that machine's header, section 4 with its launch line,
and set `SUBMIT_CMD`. Nothing else is scheduler-aware: the job script `cd`s to
an absolute path rather than relying on `$SLURM_SUBMIT_DIR`/`$PBS_O_WORKDIR`,
and chaining is done by self-submission rather than dependency flags.

## Testing it without a queue

Point `SUBMIT_CMD` at `bash` to execute the generated job script immediately in
the current shell — useful for checking the workflow end to end (optionally with
a stub `$VASP`) before burning an allocation:

```bash
SUBMIT_CMD=bash ./run_vasp_ml_ai.sh
```
