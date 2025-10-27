#!/bin/bash
set -e

START_DATE="2020-01-01"
END_DATE="2025-08-08"
CSV_DIR="/mnt/disk1/qkrig/conus/"
JOBS=16  # number of parallel jobs

mkdir -p "$CSV_DIR"

# Generate list of dates
DATES=$(seq 0 $(( ($(date -d "$END_DATE" +%s) - $(date -d "$START_DATE" +%s)) / 86400 )) | \
        xargs -I{} date -I -d "$START_DATE + {} days")

# Run extraction in parallel using xargs
echo "$DATES" | xargs -n 1 -P $JOBS -I{} python3 qkrig_ts_daily.py {} "$CSV_DIR"

echo "All daily CSVs saved in $CSV_DIR"

# Optional: combine all CSVs into a single file
FINAL_CSV="/mnt/disk1/qkrig/conus/TS_full.csv"
echo "Merging daily CSVs into $FINAL_CSV..."
python3 - <<EOF
import pandas as pd
import glob

csv_files = sorted(glob.glob("$CSV_DIR/ts_*.csv"))
dfs = [pd.read_csv(f) for f in csv_files]
df_all = pd.concat(dfs, ignore_index=True)
df_all.to_csv("$FINAL_CSV", index=False)
print(f"Saved combined CSV to $FINAL_CSV")
EOF

echo "Done!"
