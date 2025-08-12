# usgs_loader.py

from __future__ import annotations

import os
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd

from core.base_loader import BaseLoader
import dataretrieval.nwis as nwis


class USGSLoader(BaseLoader):
    """
    Parallel USGS daily-discharge loader.

    Reads site metadata from a CSV (same schema you used before), optionally filters
    by a provided site list, geographic bounding box, and minimum drainage area, and
    then fetches daily discharge for a target date in parallel via NWIS.

    Config keys used:
    
    data:
      metadata_file: path to USGS site info CSV
      site_list_file: optional text file with one site id per line
    settings:
      date_format: "%Y-%m-%d" (unused here but retained for parity)
      add_random_sites: 0 (optional; appends a random sample from metadata)
      concurrency: 16                 # number of threads
      max_retries: 3                  # per-site retry attempts
      retry_backoff_seconds: 0.75     # base backoff (with jitter)
      min_area_km2: 0                 # filter small basins
      bbox: [-125, 24, -66, 50]       # [min_lon, min_lat, max_lon, max_lat]
    """

    def __init__(self, config_path: str):
        super().__init__(config_path)
        dcfg = self.config.get("data", {})
        scfg = self.config.get("settings", {})

        self.metadata_file: str = dcfg["metadata_file"]
        self.site_list_file: Optional[str] = dcfg.get("site_list_file")

        # Parallelism & robustness knobs
        self.concurrency: int = int(scfg.get("concurrency", 16))
        self.max_retries: int = int(scfg.get("max_retries", 3))
        self.retry_backoff: float = float(scfg.get("retry_backoff_seconds", 0.75))

        # Optional filters
        self.min_area_km2: float = float(scfg.get("min_area_km2", 0.0))
        self.bbox: Optional[List[float]] = scfg.get("bbox")  # [min_lon, min_lat, max_lon, max_lat]

        # Logging directory (in main/current working directory)
        self.logs_dir = os.path.join(os.getcwd(), "usgs_retrieval_logs")
        os.makedirs(self.logs_dir, exist_ok=True)

        # Gauge metadata loaded by BaseLoader->_load_gauge_metadata
        # (set as self.gauge_metadata)

    # ---------------------------------------------------------------------
    # BaseLoader requirements
    # ---------------------------------------------------------------------
    def _load_gauge_metadata(self) -> pd.DataFrame:
        dcfg = self.config.get("data", {})
        scfg = self.config.get("settings", {})

        df = pd.read_csv(dcfg["metadata_file"], comment="#", dtype={"site_no": str})
        df = df.rename(
            columns={
                "site_no": "gauge_id",
                "dec_lat_va": "gauge_lat",
                "dec_long_va": "gauge_lon",
                "drain_area_va": "area_km2",
            }
        )
        df = df[["gauge_id", "gauge_lat", "gauge_lon", "area_km2"]].dropna()

        # Filter by optional site list
        site_list_file = dcfg.get("site_list_file")
        if site_list_file and os.path.exists(site_list_file):
            with open(site_list_file, "r") as f:
                wanted = {line.strip() for line in f if line.strip()}
            df = df[df["gauge_id"].isin(wanted)]

        # Optional add_random_sites
        add_random = int(scfg.get("add_random_sites", 0))
        if add_random > 0:
            all_sites = pd.read_csv(dcfg["metadata_file"], comment="#", dtype={"site_no": str})
            all_sites = all_sites.rename(
                columns={
                    "site_no": "gauge_id",
                    "dec_lat_va": "gauge_lat",
                    "dec_long_va": "gauge_lon",
                    "drain_area_va": "area_km2",
                }
            )[["gauge_id", "gauge_lat", "gauge_lon", "area_km2"]].dropna()
            current = set(df["gauge_id"]) 
            candidates = all_sites[~all_sites["gauge_id"].isin(current)]
            sample_n = min(add_random, len(candidates))
            if sample_n > 0:
                df = pd.concat([df, candidates.sample(n=sample_n, random_state=42)], ignore_index=True)

        # Optional area filter
        min_area = float(scfg.get("min_area_km2", 0.0))
        if min_area > 0:
            df = df[df["area_km2"] >= min_area]

        # Optional geographic bounding box filter
        bbox = scfg.get("bbox")
        if bbox and len(bbox) == 4:
            min_lon, min_lat, max_lon, max_lat = map(float, bbox)
            df = df[
                (df["gauge_lon"] >= min_lon)
                & (df["gauge_lon"] <= max_lon)
                & (df["gauge_lat"] >= min_lat)
                & (df["gauge_lat"] <= max_lat)
            ]

        return df.set_index("gauge_id")

    # ---------------------------------------------------------------------
    # Helpers: file paths
    # ---------------------------------------------------------------------
    def _log_path_for_date(self, date_str: str) -> str:
        # Human-readable log, e.g., usgs_retrieval_logs/2025-08-11.txt
        return os.path.join(self.logs_dir, f"{date_str}.txt")

    def _kv_path_for_date(self, date_str: str) -> str:
        # Key-value cache, e.g., usgs_retrieval_logs/2025-08-11.kv.txt
        return os.path.join(self.logs_dir, f"{date_str}.kv.txt")

    # ---------------------------------------------------------------------
    # Helpers: KV cache (simple key=value lines)
    #   OK line:   <site_id>=OK,<lon>,<lat>,<mm_day>
    #   FAIL line: <site_id>=FAIL,<reason>
    #   Lines starting with '#' are comments and ignored.
    # ---------------------------------------------------------------------
    def _save_kv_cache(
        self,
        date_str: str,
        successes: List[Tuple[float, float, float, str]],
        failures: List[Tuple[str, str]],
    ) -> None:
        path = self._kv_path_for_date(date_str)
        lines = []
        lines.append(f"# KV cache for USGS retrieval on {date_str}")
        lines.append(f"# Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        # successes: (lon, lat, mm_day, site_id)
        for lon, lat, mm_day, sid in sorted(successes, key=lambda r: r[3]):
            lines.append(f"{sid}=OK,{lon:.8f},{lat:.8f},{mm_day:.8f}")
        # failures: (site_id, reason)
        for sid, reason in sorted(failures, key=lambda r: r[0]):
            # Ensure reason has no commas that would break simple parsing
            reason_clean = str(reason).replace(",", ";")
            lines.append(f"{sid}=FAIL,{reason_clean}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _load_kv_cache(
        self,
        date_str: str
    ) -> Optional[Tuple[List[Tuple[float, float, float, str]], List[Tuple[str, str]]]]:
        path = self._kv_path_for_date(date_str)
        if not os.path.exists(path):
            return None
        successes: List[Tuple[float, float, float, str]] = []
        failures: List[Tuple[str, str]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                key, val = s.split("=", 1)
                parts = val.split(",")
                if not parts:
                    continue
                status = parts[0].strip().upper()
                if status == "OK" and len(parts) == 4:
                    try:
                        lon = float(parts[1]); lat = float(parts[2]); mm = float(parts[3])
                        successes.append((lon, lat, mm, key))
                    except Exception:
                        failures.append((key, "kv_parse_error"))
                elif status == "FAIL" and len(parts) >= 2:
                    reason = ",".join(parts[1:]).strip()
                    failures.append((key, reason))
                else:
                    failures.append((key, "kv_bad_line"))
        return successes, failures

    # ---------------------------------------------------------------------
    # Helpers: human-readable logging (unchanged)
    # ---------------------------------------------------------------------
    def _write_log(
        self,
        date_str: str,
        attempted_sites: List[str],
        successes: List[Tuple[float, float, float, str]],
        failures: List[Tuple[str, str]],
    ) -> None:
        """
        Write a human-readable log file enumerating successes and failures.
        failures: list of (site_id, reason)
        successes: list of (lon, lat, mm_day, site_id)
        """
        path = self._log_path_for_date(date_str)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total = len(attempted_sites)
        ok = len(successes)
        bad = len(failures)

        lines = []
        lines.append(f"USGS Retrieval Log")
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

        # Write file (overwrite for this date run)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def get_streamflow(self, year: int, month: int, day: int) -> List[Tuple[float, float, float, str]]:
        """
        Fetch (lon, lat, streamflow_mm_day, gauge_id) for all gauges on a date in parallel.
        Returns a list of tuples, with negatives removed.
        Caching:
          - If ./usgs_retrieval_logs/<YYYY-MM-DD>.kv.txt exists, load from it and skip NWIS calls.
        Also writes a detailed retrieval log to ./usgs_retrieval_logs/<YYYY-MM-DD>.txt
        and a KV cache to ./usgs_retrieval_logs/<YYYY-MM-DD>.kv.txt when retrieval is performed.
        """
        target = pd.Timestamp(year=year, month=month, day=day)
        date_str = target.strftime("%Y-%m-%d")

        # If KV cache exists, load and return immediately
        cached = self._load_kv_cache(date_str)
        if cached is not None:
            successes, failures = cached
            if successes:
                arr = np.array(successes, dtype=[("lon", float), ("lat", float), ("streamflow", float), ("gauge_id", "U15")])
                vals = arr["streamflow"]
                print(f"[cache] Summary for {date_str}")
                print(f"  - Observations: {len(vals)}")
                print(f"  - Min: {np.min(vals):.2f}, Max: {np.max(vals):.2f}, Mean: {np.mean(vals):.2f}")
                return arr.tolist()
            else:
                print(f"[cache] No data for {date_str}.")
                return []

        # Fast-exit if no metadata, but still emit a log and KV cache
        if self.gauge_metadata.empty:
            # No gauges => write empty files for traceability
            self._write_log(date_str, attempted_sites=[], successes=[], failures=[])
            self._save_kv_cache(date_str, successes=[], failures=[])
            print(f"No gauges to query for {date_str}.")
            return []

        # Build tasks
        sites = list(self.gauge_metadata.index.values)

        results: List[Tuple[float, float, float, str]] = []
        failures: List[Tuple[str, str]] = []  # (site_id, reason)

        def fetch_one(site_id: str) -> Tuple[Optional[Tuple[float, float, float, str]], str, str]:
            """
            Fetch one site's daily mean discharge and convert to mm/day, with retries.
            Returns (record_or_None, status_reason, site_id).
            """
            try:
                meta = self.gauge_metadata.loc[site_id]
            except KeyError:
                return (None, "missing_metadata", site_id)

            lon, lat, area_km2 = float(meta["gauge_lon"]), float(meta["gauge_lat"]), float(meta["area_km2"])
            if not np.isfinite(area_km2) or area_km2 <= 0:
                return (None, "invalid_area", site_id)

            attempt = 0
            while True:
                if attempt > 0:
                    site_id = site_id.zfill(8)
                try:
                    df = nwis.get_record(
                        sites=site_id,
                        service="dv",
                        start=date_str,
                        end=date_str,
                        parameterCd="00060",  # discharge
                    )
                    if df is None or df.empty:
                        return (None, "nwis_empty", site_id)

                    # find column with 00060 and Mean (daily value)
                    cols = [c for c in df.columns if ("00060" in c and "Mean" in c)]
                    if not cols:
                        return (None, "missing_mean_col", site_id)

                    try:
                        cfs = float(df.iloc[0][cols[0]])
                    except Exception:
                        return (None, "bad_value", site_id)

                    if not np.isfinite(cfs):
                        return (None, "nonfinite_cfs", site_id)

                    # cfs -> mm/day
                    area_m2 = area_km2 * 1e6
                    mm_day = (cfs * 0.0283168 * 86400.0 / area_m2) * 1000.0

                    # filter large negatives here; return None to drop
                    if mm_day < -99:
                        return (None, "Large_negative_flow", site_id)

                    return ((lon, lat, mm_day, site_id), "ok", site_id)

                except Exception as e:
                    attempt += 1
                    if attempt > self.max_retries:
                        # Avoid dumping huge traces to the log; keep reason compact
                        return (None, "exception_after_retries", site_id)
                    # backoff with jitter
                    sleep_s = self.retry_backoff * (1 + random.random()) * attempt
                    time.sleep(sleep_s)

        # Parallel execution with bounded workers
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            futures = {ex.submit(fetch_one, sid): sid for sid in sites}
            for fut in as_completed(futures):
                rec, status, sid = fut.result()
                if rec is not None and status == "ok":
                    results.append(rec)
                else:
                    failures.append((sid, status))

        # Write retrieval log (success + failure, including all attempted)
        self._write_log(
            date_str=date_str,
            attempted_sites=sites,
            successes=results,
            failures=failures,
        )

        # Write KV cache for future fast reads
        self._save_kv_cache(date_str, successes=results, failures=failures)

        if not results:
            print(f"No data for {date_str}.")
            return []

        # Summary & negative screening already handled; convert to structured array then list
        arr = np.array(results, dtype=[("lon", float), ("lat", float), ("streamflow", float), ("gauge_id", "U15")])
        vals = arr["streamflow"]
        print(f"Summary for {date_str}")
        print(f"  - Observations: {len(vals)}")
        print(f"  - Min: {np.min(vals):.2f}, Max: {np.max(vals):.2f}, Mean: {np.mean(vals):.2f}")

        return arr.tolist()