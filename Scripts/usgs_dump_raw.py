#!/usr/bin/env python3
"""
Simple USGS DV downloader: fetch raw daily discharge (00060) for each site
and save the raw response to a CSV file per site.

- Input: text/CSV file of site IDs (one per line, or a CSV with a site column).
- Output: one CSV per site in the output directory.
- Window: fixed 2020-01-01 to 2025-01-01 (edit constants below if needed).
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

import pandas as pd
import numpy as np

# dataretrieval is the USGS helper package
import dataretrieval.nwis as nwis

# ---------- User-editable constants ----------
SITE_LIST_PATH = "/home/exouser/qkrig/data/usgs_active_sites_Oct01_2024.txt"
OUT_DIR = "/home/exouser/qkrig/usgs_raw_daily_retrievals"
START_DATE = "2020-01-01"
END_DATE = "2025-08-08"
# --------------------------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_site_ids(path: str) -> List[str]:
    """
    Load site IDs from a text or CSV file, one ID per line (or a CSV column).
    Zero-pad to 8 digits. Removes blanks and duplicates.
    """
    # Try as a simple text file (one site ID per line)
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if lines:
            site_ids = lines
        else:
            site_ids = []
    except Exception:
        site_ids = []

    # If the text approach yielded nothing, try reading as CSV
    if not site_ids:
        try:
            df = pd.read_csv(path)
            # Heuristics: try common column names, else take first column
            for col in ["site_no", "site", "gauge_id", "id", "site_id"]:
                if col in df.columns:
                    site_ids = df[col].astype(str).tolist()
                    break
            if not site_ids and df.shape[1] >= 1:
                site_ids = df.iloc[:, 0].astype(str).tolist()
        except Exception:
            pass

    # Final cleanup: strip, keep digits, zero-pad to 8
    cleaned = []
    for s in site_ids:
        s = str(s).strip()
        if not s:
            continue
        # Some files may include commas or spaces; keep numeric part
        s_digits = "".join(ch for ch in s if ch.isdigit())
        if not s_digits:
            continue
        cleaned.append(s_digits.zfill(8))

    # Deduplicate while preserving order
    seen = set()
    out = []
    for sid in cleaned:
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def pick_dv_column(df: pd.DataFrame) -> Optional[str]:
    """
    Accept either '...Mean' or '...00003' (USGS statCd mean).
    Fallback: if exactly one '00060' column exists, take it.
    """
    if df is None or df.empty:
        return None
    cols = [str(c) for c in df.columns]
    for c in cols:
        if "00060" in c and "Mean" in c:
            return c
    for c in cols:
        if "00060" in c and "00003" in c:
            return c
    cand = [c for c in cols if "00060" in c]
    if len(cand) == 1:
        return cand[0]
    return None


def get_site_dv_raw(site_id: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """
    Try the canonical DV endpoint for daily mean first; fallback to generic record.
    Returns the raw DataFrame (as returned by dataretrieval), or None if empty.
    """
    # 1) Preferred: get_dv with statCd=00003 (mean), include inactive sites historically
    try:
        dv = nwis.get_dv(
            sites=site_id,
            startDT=start,
            endDT=end,
            parameterCd="00060",
            statCd="00003",
            siteStatus="all",
        )
        if dv is not None and not dv.empty and pick_dv_column(dv) is not None:
            return dv
    except Exception:
        pass

    # 2) Fallback: get_record (service="dv")
    try:
        df = nwis.get_record(
            sites=site_id,
            service="dv",
            start=start,
            end=end,
            parameterCd="00060",
        )
        if df is not None and not df.empty and pick_dv_column(df) is not None:
            return df
    except Exception:
        pass

    # 3) As a last resort, try unfiltered emptiness (some sites might have non-mean stats)
    try:
        dv_any = nwis.get_dv(
            sites=site_id,
            startDT=start,
            endDT=end,
            parameterCd="00060",
            siteStatus="all",
        )
        if dv_any is not None and not dv_any.empty:
            return dv_any
    except Exception:
        pass

    return None


def main() -> int:
    ensure_dir(OUT_DIR)

    sites = load_site_ids(SITE_LIST_PATH)
    if not sites:
        print(f"[error] No site IDs found in {SITE_LIST_PATH}")
        return 2

    print(f"Found {len(sites)} sites. Downloading raw DV (00060) {START_DATE} → {END_DATE}")
    ok = 0
    empty = 0
    for i, sid in enumerate(sites, 1):
        # Try both padded and raw (some sites already 8-digit)
        candidates = [sid]
        if sid != sid.zfill(8):
            candidates.append(sid.zfill(8))

        df_raw = None
        for c in candidates:
            df_raw = get_site_dv_raw(c, START_DATE, END_DATE)
            if df_raw is not None and not df_raw.empty:
                break

        out_path = os.path.join(OUT_DIR, f"{sid}.csv")
        if df_raw is None or df_raw.empty:
            empty += 1
            print(f"[{i}/{len(sites)}] {sid}: NO DATA")
            # Still write an empty placeholder CSV with a note?
            # If you don't want empty files, comment the next two lines:
            # pd.DataFrame().to_csv(out_path, index=False)
            continue

        # Ensure a DatetimeIndex for consistent saves
        if not isinstance(df_raw.index, pd.DatetimeIndex):
            try:
                df_raw.index = pd.to_datetime(df_raw.index)
            except Exception:
                # If index can't be parsed, leave as-is; it's "raw"
                pass

        # Save exactly as returned (raw). Include index for dates.
        df_raw.to_csv(out_path)
        ok += 1
        print(f"[{i}/{len(sites)}] {sid}: saved → {out_path}")

    print(f"\nDone. Saved {ok} site files; {empty} empty.")
    print(f"Output dir: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

