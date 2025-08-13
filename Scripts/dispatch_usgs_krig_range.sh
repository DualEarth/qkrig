#!/usr/bin/env bash
set -euo pipefail

# ---- user knobs ----
CONFIG="${1:-configs/usgsgaugekrig.yaml}"
START_DATE="${2:-2020-01-01}"
END_DATE="${3:-2020-12-31}"
PLOT_CONFIG_OVERRIDE="${4:-}"   # optional; leave empty to use config's plot_config
MAX_PROCS="${MAX_PROCS:-16}"    # concurrency
PYTHON_BIN="${PYTHON_BIN:-python}"  # python interpreter (use your conda env)

# Avoid oversubscription when running many processes
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# headless plotting
export MPLBACKEND=Agg

# Build the date list using env vars to avoid argv issues with heredoc
DATE_LIST="$(
  SDATE="$START_DATE" EDATE="$END_DATE" \
  "$PYTHON_BIN" - <<'PY'
from datetime import datetime, timedelta
import os, sys
start = os.environ["SDATE"]
end   = os.environ["EDATE"]
sd = datetime.strptime(start, "%Y-%m-%d").date()
ed = datetime.strptime(end,   "%Y-%m-%d").date()
if ed < sd:
    sys.exit("END_DATE must be >= START_DATE")
d = sd
while d <= ed:
    print(d.isoformat())
    d += timedelta(days=1)
PY
)"

# Function to run one day
run_one_day() {
  local d="$1"
  local y="${d:0:4}"
  local m="${d:5:2}"
  local day="${d:8:2}"

  if [[ -n "$PLOT_CONFIG_OVERRIDE" ]]; then
    "$PYTHON_BIN" Scripts/run_usgs_krig_day.py \
      --config "$CONFIG" --year "$y" --month "$m" --day "$day" \
      --plot-config "$PLOT_CONFIG_OVERRIDE"
  else
    "$PYTHON_BIN" Scripts/run_usgs_krig_day.py \
      --config "$CONFIG" --year "$y" --month "$m" --day "$day"
  fi
}

export -f run_one_day
export CONFIG PLOT_CONFIG_OVERRIDE PYTHON_BIN

# Prefer GNU parallel if available
if command -v parallel >/dev/null 2>&1; then
  echo "$DATE_LIST" | parallel -j "$MAX_PROCS" --halt now,fail=1 run_one_day {}
else
  # Fallback: xargs -P
  echo "$DATE_LIST" | xargs -n1 -P "$MAX_PROCS" -I{} bash -c 'run_one_day "$@"' _ {}
fi
