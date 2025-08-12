#!/usr/bin/env python3
"""
Bulk USGS daily streamflow retrieval (2020 -> present) with per-day outputs.

- Fetches full daily time-series for each site (per-site requests, not per-day)
- Converts cfs -> mm/day using drainage area from metadata
- Writes per-day KV caches and human-readable logs to ./usgs_retrieval_logs/
- Processes year-by-year to bound memory
- Parallel per-site with retries and jittered backoff

Usage:
    python Scripts/usgs_bulk_fetch.py --config configs/usgsgaugekrig.yaml \
        --start 2020-01-01 --end today --concurrency 16 --max-retries 3

Notes:
- Requires your project to be installed (pip install -e .) so internal imports work.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml

from loaders.usgs_loader import USGSLoader  # reuses your filtering logic
import dataretrieval.nwis as nwis


# -------------------------- Utilities --------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bulk USGS DV fetcher (per-site, 2020->present) writing daily KV caches.")
    p.add_argument("--config", required=True, help="Path to YAML config (e.g., configs/usgsgaugekrig.yaml)")
    p.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD) or '2020-01-01'")
    p.add_argument("--end", default="today", help="End date (YYYY-MM-DD) or 'today'")
    p.add_argument("--concurrency", type=int, default=None, help="Override concurrency from config")
    p.add_argument("--max-retries", type=int, default=None, help="Override max_retries from config")
    p.add_argument("--retry-backoff-seconds", type=float, default=None, help="Override retry backoff from config")
    p.add_argument("--logs-dir", default="usgs_retrieval_logs", help="Directory for KV/log outputs")
    p.add_argument("--year-chunks", action="store_true",
                   help="Process one calendar year at a time (recommended to bound memory).")
    return p.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def daterange_year_chunks(start_dt: date, end_dt: date) -> List[Tuple[date, date]]:
    """Split [start_dt, end_dt] into calendar-year chunks."""
    chunks: List[Tuple[date, date]] = []
    y = start_dt.year
    while True:
        y_start = date(y, 1, 1) if y > start_dt.year else start_dt
        y_end = date(y, 12, 31)
        if y_end > end_dt:
            y_end = end_dt
        chunks.append((y_start, y_end))
        if y_end >= end_dt:
            break
        y += 1
    return chunks


# -------------------------- KV / log writers --------------------------

def kv_path_for_date(logs_dir: str, date_str: str) -> str:
    return os.path.join(logs_dir, f"{date_str}.kv.txt")

def log_path_for_date(logs_dir: str, date_str: str) -> str:
    return os.path.join(logs_dir, f"{date_str}.txt")

def save_kv_cache(
    logs_dir: str,
    date_str: str,
    successes: List[Tuple[float, float, float, str]],
    failures: List[Tuple[str, str]],
) -> None:
    """
    Write the KV cache file for a single day:
      OK lines:   <site_id>=OK,<lon>,<lat>,<mm_day>
      FAIL lines: <site_id>=FAIL,<reason>
    """
    path = kv_path_for_date(logs_dir, date_str)
    lines = []
    lines.append(f"# KV cache for USGS retrieval on {date_str}")
    lines.append(f"# Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for lon, lat, mm_day, sid in sorted(successes, key=lambda r: r[3]):
        lines.append(f"{sid}=OK,{lon:.8f},{lat:.8f},{mm_day:.8f}")
    for sid, reason in sorted(failures, key=lambda r: r[0]):
        reason_clean = str(reason).replace(",", ";")
        lines.append(f"{sid}=FAIL,{reason_clean}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_log(
    logs_dir: str,
    date_str: str,
    attempted_sites: List[str],
    successes: List[Tuple[float, float, float, str]],
    failures: List[Tuple[str, str]],
) -> None:
    """Human-readable log per day."""
    path = log_path_for_date(logs_dir, date_str)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(attempted_sites)
    ok = len(successes)
    bad = len(failures)

    lines = []
    lines.append("USGS Retrieval Log")
    lines.append(f"Timestamp: {ts}")
    lines.append(f"Date queried: {date_str}")
    lines.append(f"Total sites attempted: {total}")
    lines.append(f"Successful: {ok}")
    lines.append(f"Unsuccessful: {bad}")
    lines.append("")
    lines.append("=== Successful retrievals ===")
    if successes:
        for lon, lat, mm_day, sid in sorted(successes, key=lambda r: r[3]):
            lines.append(f"OK  site={sid}  lon={lon:.6f}  lat={lat:.6f}  streamflow_mm_day={mm_day:.6f}")
    else:
        lines.append("(none)")

    lines.append("")
    lines.append("=== Unsuccessful retrievals ===")
    if failures:
        for sid, reason in sorted(failures, key=lambda r: r[0]):
            lines.append(f"FAIL  site={sid}  reason={reason}")
    else:
        lines.append("(none)")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# -------------------------- Fetch per-site (full period) --------------------------

def fetch_site_timeseries(
    site_id: str,
    start_str: str,
    end_str: str,
    max_retries: int,
    retry_backoff: float,
) -> Optional[pd.DataFrame]:
    """
    Fetch a full DV time series (parameterCd 00060) for one site in [start_str, end_str].
    Returns a DataFrame indexed by date with ONE column that includes '00060' and 'Mean',
    or None if unavailable after retries.
    """
    attempt = 0
    while True:
        try:
            df = nwis.get_record(
                sites=site_id,
                service="dv",
                start=start_str,
                end=end_str,
                parameterCd="00060",
            )
            if df is None or df.empty:
                return None

            cols = [c for c in df.columns if ("00060" in c and "Mean" in c)]
            if not cols:
                return None

            # Reduce to a single-series DF: daily mean discharge (cfs)
            ddf = df[[cols[0]]].copy()
            ddf = ddf.rename(columns={cols[0]: "cfs"})
            # Ensure datetime index
            if not isinstance(ddf.index, pd.DatetimeIndex):
                ddf.index = pd.to_datetime(ddf.index)
            ddf = ddf.sort_index()
            return ddf

        except Exception:
            attempt += 1
            if attempt > max_retries:
                return None
            sleep_s = retry_backoff * (1 + random.random()) * attempt
            time.sleep(sleep_s)


# -------------------------- Main --------------------------

def main():
    args = parse_args()
    ensure_dir(args.logs_dir)

    # Resolve date bounds
    if args.end.lower() == "today":
        end_dt = date.today()
    else:
        end_dt = datetime.strptime(args.end, "%Y-%m-%d").date()
    start_dt = datetime.strptime(args.start, "%Y-%m-%d").date()
    if end_dt < start_dt:
        raise ValueError("end date must be >= start date")

    # Load config and USGSLoader (to reuse your metadata filtering)
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    loader = USGSLoader(args.config)

    # Allow CLI overrides for concurrency/retries
    if args.concurrency is not None:
        loader.concurrency = args.concurrency
    if args.max_retries is not None:
        loader.max_retries = args.max_retries
    if args.retry_backoff_seconds is not None:
        loader.retry_backoff = args.retry_backoff_seconds

    sites = list(loader.gauge_metadata.index.values)
    if not sites:
        print("No sites to fetch (after filtering). Exiting.")
        return

    # Precompute per-site static metadata for conversion
    meta = loader.gauge_metadata  # index: gauge_id
    site_meta: Dict[str, Tuple[float, float, float]] = {}  # sid -> (lon, lat, area_km2)
    for sid, row in meta.iterrows():
        try:
            site_meta[sid] = (float(row["gauge_lon"]), float(row["gauge_lat"]), float(row["area_km2"]))
        except Exception:
            continue

    # Process year-by-year to bound memory (recommended)
    year_spans = daterange_year_chunks(start_dt, end_dt) if args.year_chunks else [(start_dt, end_dt)]

    for y_start, y_end in year_spans:
        start_str = y_start.strftime("%Y-%m-%d")
        end_str = y_end.strftime("%Y-%m-%d")
        print(f"\n=== Processing span {start_str} -> {end_str} for {len(sites)} sites ===")

        # Parallel per-site fetches for this span
        per_site_series: Dict[str, Optional[pd.DataFrame]] = {}
        with ThreadPoolExecutor(max_workers=loader.concurrency) as ex:
            futures = {ex.submit(fetch_site_timeseries, sid, start_str, end_str, loader.max_retries, loader.retry_backoff): sid
                       for sid in sites}
            for fut in as_completed(futures):
                sid = futures[fut]
                per_site_series[sid] = fut.result()

        # Aggregate to per-day outputs: build dicts date_str -> successes/failures lists
        # We generate the full list of dates in the span to ensure a KV file per day (even if empty).
        all_days = pd.date_range(start=start_str, end=end_str, freq="D")
        successes_by_day: Dict[str, List[Tuple[float, float, float, str]]] = {d.strftime("%Y-%m-%d"): [] for d in all_days}
        failures_by_day: Dict[str, List[Tuple[str, str]]] = {d.strftime("%Y-%m-%d"): [] for d in all_days}

        for sid, ddf in per_site_series.items():
            lon, lat, area_km2 = site_meta.get(sid, (np.nan, np.nan, np.nan))
            if not (np.isfinite(area_km2) and area_km2 > 0):
                # Mark all days in span as failures for this site (invalid area)
                for d in all_days:
                    failures_by_day[d.strftime("%Y-%m-%d")].append((sid, "invalid_area"))
                continue

            if ddf is None or ddf.empty:
                # No data returned for this span; mark failures
                for d in all_days:
                    failures_by_day[d.strftime("%Y-%m-%d")].append((sid, "nwis_empty"))
                continue

            # Align series to full day list; missing entries remain NaN
            series = ddf["cfs"].reindex(all_days)
            # Convert cfs -> mm/day
            area_m2 = area_km2 * 1e6
            mm_day_series = (series.astype(float) * 0.0283168 * 86400.0 / area_m2) * 1000.0

            for d, v in mm_day_series.items():
                ds = d.strftime("%Y-%m-%d")
                if pd.notna(v) and np.isfinite(v):
                    # Skip obviously bad magnitudes if you want (optional)
                    if v < -69 or v > 69:
                        failures_by_day[ds].append((sid, "Large_magnitude_flow"))
                    else:
                        successes_by_day[ds].append((lon, lat, float(v), sid))
                else:
                    failures_by_day[ds].append((sid, "missing"))

        # Write all days in this span
        attempted_sites = sites  # same set each day
        for d in all_days:
            ds = d.strftime("%Y-%m-%d")
            save_kv_cache(args.logs_dir, ds, successes_by_day[ds], failures_by_day[ds])
            write_log(args.logs_dir, ds, attempted_sites, successes_by_day[ds], failures_by_day[ds])

        print(f"✓ Finished span {start_str} -> {end_str}. Wrote {len(all_days)} KV + log files to {args.logs_dir}.")

    print("\nAll done.")
    print(f"Outputs are in: {os.path.abspath(args.logs_dir)}")


if __name__ == "__main__":
    # Ensure project root on sys.path if not installed; but recommended: pip install -e .
    sys.exit(main())
