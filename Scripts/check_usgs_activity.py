#!/usr/bin/env python3
"""
Check USGS gauge activity on October 1st for a given year.

- Reads usgs_site_info.csv with a column 'site_no' (string or numeric).
- For each site, attempts to fetch daily discharge (00060) on Oct 1.
- If the 'as-is' site id fails, tries site_id.zfill(8) as a fallback.
- Classifies sites as ACTIVE (data available & parseable) or INACTIVE.
- Outputs:
    ./usgs_retrieval_logs/active_sites_Oct01_<year>.txt
    ./usgs_retrieval_logs/inactive_sites_Oct01_<year>.txt
    ./usgs_retrieval_logs/activity_summary_Oct01_<year>.csv

Usage:
    python check_usgs_activity.py --year 2024 --csv usgs_site_info.csv --concurrency 32
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

import dataretrieval.nwis as nwis


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Check USGS gauge activity on Oct 1 for a given year.")
    ap.add_argument("--year", type=int, required=True, help="Year to check (Oct 1 of this year).")
    ap.add_argument("--csv", type=str, default="usgs_site_info.csv",
                    help="Path to USGS site info CSV (must contain 'site_no').")
    ap.add_argument("--concurrency", type=int, default=16, help="Thread pool size.")
    ap.add_argument("--max-retries", type=int, default=3, help="Retries per site on request failures.")
    ap.add_argument("--retry-backoff-seconds", type=float, default=0.75,
                    help="Base backoff (with jitter) between retries.")
    return ap.parse_args()


def safe_site_id(raw: str) -> str:
    """Normalize a site ID to string without whitespace."""
    return str(raw).strip()


def try_fetch(site_id: str, date_str: str) -> Tuple[bool, str]:
    """
    Attempt to fetch daily mean discharge for a single date.

    Returns (ok, reason) where:
      ok=True  => df had a valid 00060 Mean value
      ok=False => reason describes the failure
    """
    try:
        df = nwis.get_record(
            sites=site_id,
            service="dv",
            start=date_str,
            end=date_str,
            parameterCd="00060",
        )
        if df is None or df.empty:
            return (False, "nwis_empty")

        cols = [c for c in df.columns if ("00060" in c and "Mean" in c)]
        if not cols:
            return (False, "missing_mean_col")

        try:
            val = float(df.iloc[0][cols[0]])
        except Exception:
            return (False, "bad_value")

        if not np.isfinite(val):
            return (False, "nonfinite")

        # We consider any finite daily mean as "active" for the date
        return (True, "ok")

    except Exception as e:
        return (False, "exception")


def check_one_site(
    site_id_raw: str,
    date_str: str,
    max_retries: int,
    retry_backoff: float,
) -> Tuple[str, bool, str, str]:
    """
    Check one site for activity on date_str.

    Strategy:
      1) Try site ID as-is.
      2) If not ok and the site_no looks numeric and len < 8, try zfill(8).
      Retries around the network call with backoff+jitter.

    Returns (final_site_id_used, is_active, reason, original_site_id)
    """
    original = safe_site_id(site_id_raw)
    candidates = [original]

    # zfill(8) fallback when it could help (numeric and not already >=8 chars)
    if original.isdigit() and len(original) < 8:
        candidates.append(original.zfill(8))

    # De-duplicate in case original already equals zfilled
    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    for cand in candidates:
        attempt = 0
        while True:
            ok, reason = try_fetch(cand, date_str)
            if ok:
                return (cand, True, "ok", original)
            attempt += 1
            if attempt > max_retries:
                break
            # backoff with jitter
            time.sleep(retry_backoff * (1 + random.random()) * attempt)

    # If none of the candidates worked, report last reason from the last candidate
    # (for clarity, we return reason for the final attempt of the final candidate)
    return (candidates[-1], False, reason, original)


def main():
    args = parse_args()
    year = args.year
    csv_path = args.csv
    concurrency = args.concurrency
    max_retries = args.max_retries
    retry_backoff = args.retry_backoff_seconds

    date_str = f"{year}-10-01"

    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path, comment="#", dtype={"site_no": str})
    if "site_no" not in df.columns:
        print("ERROR: CSV must contain a 'site_no' column.", file=sys.stderr)
        sys.exit(1)

    # Prepare output dir
    logs_dir = os.path.join(os.getcwd(), "usgs_retrieval_logs")
    os.makedirs(logs_dir, exist_ok=True)

    sites: List[str] = [safe_site_id(s) for s in df["site_no"].astype(str).tolist() if str(s).strip()]
    print(f"Checking {len(sites)} sites for activity on {date_str} with concurrency={concurrency} ...")

    results: List[Tuple[str, bool, str, str]] = []  # (final_id_used, is_active, reason, original_id)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {
            ex.submit(check_one_site, sid, date_str, max_retries, retry_backoff): sid
            for sid in sites
        }
        for fut in as_completed(futs):
            final_id, is_active, reason, original = fut.result()
            results.append((final_id, is_active, reason, original))

    # Split active/inactive
    active = sorted([orig for _, ok, _, orig in results if ok])
    inactive = sorted([orig for _, ok, _, orig in results if not ok])

    # Write text lists
    active_txt = os.path.join(logs_dir, f"active_sites_Oct01_{year}.txt")
    inactive_txt = os.path.join(logs_dir, f"inactive_sites_Oct01_{year}.txt")
    with open(active_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(active) + ("\n" if active else ""))
    with open(inactive_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(inactive) + ("\n" if inactive else ""))

    # Write a summary CSV with columns: original_site_id, final_site_id_used, status, reason
    summary_csv = os.path.join(logs_dir, f"activity_summary_Oct01_{year}.csv")
    rows = []
    for final_id, ok, reason, original in results:
        rows.append({
            "original_site_id": original,
            "final_site_id_used": final_id,
            "status": "ACTIVE" if ok else "INACTIVE",
            "reason": reason,
        })
    pd.DataFrame(rows).to_csv(summary_csv, index=False)

    print(f"Done.\n  Active: {len(active)}\n  Inactive: {len(inactive)}")
    print(f"Wrote:\n  {active_txt}\n  {inactive_txt}\n  {summary_csv}")


if __name__ == "__main__":
    main()
