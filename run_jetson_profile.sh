#!/usr/bin/env bash
# Jetson TX2 profiling run for the MPE sparse-search comparison.
#
# Puts the board in a fixed, reproducible state, then runs
# jetson_mpe_profile.py pinned to a single Cortex-A57 core.
#
#   ./run_jetson_profile.sh                      # defaults below
#   ./run_jetson_profile.sh --repeats 40         # any flag is passed through
#   CORE=4 ./run_jetson_profile.sh               # pick a different A57 core
#
# Why each step matters:
#   nvpmodel -m 0 + jetson_clocks   locks clocks, so DVFS and thermal ramp do
#                                   not make the first repeat differ from the last
#   OPENBLAS_CORETYPE=ARMV8         without it NumPy SIGILLs on the TX2's cores
#   taskset onto an A57             the TX2 is heterogeneous: cores 1-2 are
#                                   Denver 2 (wide, speculative, atypical),
#                                   cores 0/3/4/5 are Cortex-A57. A57 is the
#                                   honest stand-in for a generic edge part, and
#                                   pinning stops the scheduler from migrating
#                                   between two different microarchitectures
#   --threads 1                     an edge codec gets one core; it also stops
#                                   OpenBLAS multithreading OMP's lstsq, which
#                                   would flatter OMP specifically
set -euo pipefail

# ---------------------------------------------------------------- settings --
CORE="${CORE:-3}"                       # an A57 core (0, 3, 4 or 5)
AUDIO="${AUDIO:-input/input.wav}"       # quote globs: AUDIO='speech/*.wav'
K_VALUES="${K_VALUES:-1,2,5,10,20,35}"  # K=35 is where GCLST's scaling win shows
ALGOS="${ALGOS:-omp,lst,gclst}"
REPEATS="${REPEATS:-30}"
ITERS="${ITERS:-200}"
OUT_PREFIX="${OUT_PREFIX:-jetson_mpe_profiling}"
PY="${PY:-python3}"

LOG="${OUT_PREFIX}_run.log"

# ------------------------------------------------------------ device setup --
echo "=== Locking power mode and clocks ==="
if command -v nvpmodel >/dev/null 2>&1; then
    sudo nvpmodel -m 0 || echo "  nvpmodel failed (continuing)"
    sudo jetson_clocks   || echo "  jetson_clocks failed (continuing)"
    nvpmodel -q || true
else
    echo "  nvpmodel not found -- not a Jetson? continuing unlocked"
fi

# NumPy SIGILLs on the TX2's ARM cores without this.
export OPENBLAS_CORETYPE=ARMV8

echo
echo "=== Environment ==="
echo "  core pinned   : $CORE"
echo "  algos         : $ALGOS"
echo "  K values      : $K_VALUES"
echo "  repeats       : $REPEATS   iters/run: $ITERS"
echo "  audio         : $AUDIO"
uname -srm
$PY -c "import numpy; print('  numpy', numpy.__version__)"

# Record the thermal starting point; a run that begins hot reads differently
# from one that begins cold, and that belongs in the log.
if ls /sys/devices/virtual/thermal/thermal_zone*/temp >/dev/null 2>&1; then
    echo -n "  temps (C)     : "
    for z in /sys/devices/virtual/thermal/thermal_zone*/temp; do
        awk '{printf "%.1f ", $1/1000}' "$z"
    done
    echo
fi

# ------------------------------------------------------------------- run ----
echo
echo "=== Profiling (logging to $LOG) ==="
# shellcheck disable=SC2086
taskset -c "$CORE" $PY jetson_mpe_profile.py \
    --audio $AUDIO \
    --algos "$ALGOS" \
    --k-values "$K_VALUES" \
    --repeats "$REPEATS" \
    --iters "$ITERS" \
    --threads 1 \
    --out-prefix "$OUT_PREFIX" \
    "$@" 2>&1 | tee "$LOG"

echo
echo "=== Done ==="
echo "  ${OUT_PREFIX}.csv           one row per (algo, K, run)"
echo "  ${OUT_PREFIX}_summary.csv   across-run mean / 95% CI / P99 per (algo, K)"
echo "  ${LOG}                      full console log"
