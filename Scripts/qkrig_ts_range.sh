#!/bin/bash
set -e

START_DATE="2011-06-01"
END_DATE="2019-09-30"
OUT_DIR="/mnt/disk1/qkrig/conus/"
JOBS=4  # number of parallel jobs

mkdir -p "$OUT_DIR"

# Generate list of dates
DATES=$(seq 0 $(( ($(date -d "$END_DATE" +%s) - $(date -d "$START_DATE" +%s)) / 86400 )) | \
        xargs -I{} date -I -d "$START_DATE + {} days")

# Run extraction in parallel using xargs
echo "$DATES" | xargs -n 1 -P $JOBS -I{} python3 qkrig_ts_daily.py {} "$OUT_DIR"

echo "✅ All water-year catchment CSVs updated in $OUT_DIR"

