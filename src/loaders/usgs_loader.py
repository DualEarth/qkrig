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
    Parallel USGS 15-minute instantaneous value (IV) loader with post-load bbox filtering.

    Reads site metadata from a CSV, optionally filters by a provided site list,
    geographic bounding box, and minimum drainage area, and then fetches 15-minute
    instantaneous discharge for a target date/time in parallel via NWIS.

    NEW: Fetches 15-minute IV data instead of daily values. Returns streamflow in mm/15min.
    
    Usage:
      - get_streamflow(year, month, day) -> Returns dict with all 96 15-min timestamps
      - get_streamflow(year, month, day, hour, minute) -> Returns list for that timestamp

    Config keys used:

    data:
      metadata_file: path to USGS site info CSV
      site_list_file: optional text file with one site id per line
      data_cache_directory: output dir for KV/logs
    settings:
      date_format: "%Y-%m-%d"
      add_random_sites: 0
      concurrency: 16
      max_retries: 3
      retry_backoff_seconds: 0.75
      min_area_km2: 0
      bbox: [-125, 24, -66, 50]       # [min_lon, min_lat, max_lon, max_lat]
      bbox_pad_deg: 0.0               # small padding (deg) applied at filtering time
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
        self.bbox_pad_deg: float = float(scfg.get("bbox_pad_deg", 0.0))

        # Logging directory
        self.logs_dir = dcfg.get("data_cache_directory")
        os.makedirs(self.logs_dir, exist_ok=True)

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

        # Optional geographic bounding box filter (metadata-level)
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
        return os.path.join(self.logs_dir, f"{date_str}.txt")

    def _kv_path_for_date(self, date_str: str) -> str:
        return os.path.join(self.logs_dir, f"{date_str}.kv.txt")

    # ---------------------------------------------------------------------
    # Helpers: KV cache (simple key=value lines)
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
        for lon, lat, mm_day, sid in sorted(successes, key=lambda r: r[3]):
            lines.append(f"{sid}=OK,{lon:.8f},{lat:.8f},{mm_day:.8f}")
        for sid, reason in sorted(failures, key=lambda r: r[0]):
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
    # Helpers: human-readable logging
    # ---------------------------------------------------------------------
    def _write_log(
        self,
        date_str: str,
        attempted_sites: List[str],
        successes: List[Tuple[float, float, float, str]],
        failures: List[Tuple[str, str]],
    ) -> None:
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

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # ---------------------------------------------------------------------
    # NEW: post-load bbox filter
    # ---------------------------------------------------------------------
    def _filter_by_bbox(
        self,
        records: List[Tuple[float, float, float, str]]
    ) -> List[Tuple[float, float, float, str]]:
        """
        Apply bbox (with optional padding) to (lon, lat, mm, site_id) records.
        Does not modify cache files; only filters what we return.
        """
        if not records or not self.bbox or len(self.bbox) != 4:
            return records

        min_lon, min_lat, max_lon, max_lat = map(float, self.bbox)
        pad = float(self.bbox_pad_deg)
        min_lon -= pad; min_lat -= pad; max_lon += pad; max_lat += pad

        filtered = [
            (lon, lat, mm, sid)
            for (lon, lat, mm, sid) in records
            if (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat)
        ]
        dropped = len(records) - len(filtered)
        if dropped > 0:
            print(f"[bbox] Dropped {dropped} record(s) outside {min_lon:.3f},{min_lat:.3f},{max_lon:.3f},{max_lat:.3f}")
        return filtered
    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def get_streamflow(
        self, 
        year: int, 
        month: int, 
        day: int,
        hour: Optional[int] = None,
        minute: Optional[int] = None
    ):
        """
        Fetch 15-minute instantaneous streamflow data for all gauges on a date.
        
        Args:
            year, month, day: Target date
            hour, minute: Optional - if provided, returns data for that specific
                         15-minute window only. If not provided, returns all
                         96 timestamps for the day.
        
        Returns:
            If hour/minute specified: List[Tuple[float, float, float, str]]
                List of (lon, lat, streamflow_mm_15min, gauge_id) tuples
            If hour/minute NOT specified: Dict[str, List[Tuple[float, float, float, str]]]
                Dict keyed by timestamp string (YYYY-MM-DD_HH-MM) with lists of tuples
        
        Caching:
            Uses timestamped cache files (YYYY-MM-DD_HH-MM.kv.txt).
            If fetching all timestamps, checks/creates cache for each.
        
        NOTE: Always applies bbox filtering to results before returning.
        """
        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        
        # Determine if we're fetching specific time or all timestamps
        fetch_single = hour is not None and minute is not None
        
        if fetch_single:
            # Single timestamp mode - return list
            return self._get_streamflow_single(year, month, day, hour, minute)
        else:
            # All timestamps mode - return dict with all 96 15-min intervals
            return self._get_streamflow_all(year, month, day)

    def _get_streamflow_single(
        self, 
        year: int, 
        month: int, 
        day: int,
        hour: int,
        minute: int
    ) -> List[Tuple[float, float, float, str]]:
        """Fetch data for a single 15-minute timestamp."""
        # NWIS IV data comes in UTC, so target must be tz-aware
        target = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute, tz='UTC')
        ts_str = target.strftime("%Y-%m-%d_%H-%M")
        date_str = f"{year:04d}-{month:02d}-{day:02d}"

        # Check cache
        cached = self._load_kv_cache(ts_str)
        if cached is not None:
            successes, failures = cached
            if successes:
                successes = self._filter_by_bbox(successes)
                if not successes:
                    print(f"[cache+bbox] No data within bbox for {ts_str}.")
                    return []
                arr = np.array(successes, dtype=[("lon", float), ("lat", float), ("streamflow", float), ("gauge_id", "U15")])
                vals = arr["streamflow"]
                print(f"[cache] Summary for {ts_str} (after bbox filter)")
                print(f"  - Observations: {len(vals)}")
                print(f"  - Min: {np.min(vals):.4f}, Max: {np.max(vals):.4f}, Mean: {np.mean(vals):.4f}")
                return arr.tolist()
            else:
                print(f"[cache] No data for {ts_str}.")
                return []

        if self.gauge_metadata.empty:
            self._write_log(ts_str, attempted_sites=[], successes=[], failures=[])
            self._save_kv_cache(ts_str, successes=[], failures=[])
            print(f"No gauges to query for {ts_str}.")
            return []

        sites = list(self.gauge_metadata.index.values)
        results: List[Tuple[float, float, float, str]] = []
        failures: List[Tuple[str, str]] = []

        print(f"[USGS IV] Fetching data for {ts_str}")
        print(f"[USGS IV] Sites to query: {len(sites)}")

        def fetch_one_iv(site_id: str) -> Tuple[Optional[Tuple[float, float, float, str]], str, str]:
            """Fetch one site's IV discharge for the target timestamp."""
            try:
                meta = self.gauge_metadata.loc[site_id]
            except KeyError:
                return (None, "missing_metadata", site_id)

            lon, lat, area_km2 = float(meta["gauge_lon"]), float(meta["gauge_lat"]), float(meta["area_km2"])
            if not np.isfinite(area_km2) or area_km2 <= 0:
                return (None, "invalid_area", site_id)

            attempt = 0
            sid = site_id.zfill(8)  # Always zero-pad to 8 digits for NWIS
            while True:
                try:
                    # get_iv returns (DataFrame, metadata) tuple
                    result = nwis.get_iv(
                        sites=sid,
                        startDT=date_str,
                        endDT=date_str,
                        parameterCd="00060",
                    )
                    # Unpack tuple - first element is the DataFrame
                    if isinstance(result, tuple):
                        df = result[0]
                    else:
                        df = result
                    
                    if df is None or df.empty:
                        return (None, "nwis_empty", site_id)

                    # Find 00060 column (IV data has no "Mean" suffix)
                    cols = [c for c in df.columns if "00060" in c]
                    if not cols:
                        return (None, "missing_00060_col", site_id)

                    # Find matching timestamp
                    df.index = pd.to_datetime(df.index)
                    if target in df.index:
                        row = df.loc[target]
                    else:
                        # Find nearest within 7.5-minute window
                        mask = abs(df.index - target) <= pd.Timedelta(minutes=7)
                        if not mask.any():
                            return (None, "no_matching_time", site_id)
                        row = df[mask].iloc[0]

                    try:
                        cfs = float(row[cols[0]])
                    except Exception:
                        return (None, "bad_value", site_id)

                    if not np.isfinite(cfs):
                        return (None, "nonfinite_cfs", site_id)

                    # cfs -> mm/15min (900 seconds)
                    area_m2 = area_km2 * 1e6
                    mm_15min = (cfs * 0.0283168 * 900.0 / area_m2) * 1000.0

                    # Filter large magnitudes (scaled for 15-min)
                    if mm_15min < -1 or mm_15min > 5:
                        return (None, "large_magnitude_flow", site_id)

                    return ((lon, lat, mm_15min, site_id), "ok", site_id)

                except Exception as e:
                    attempt += 1
                    if attempt > self.max_retries:
                        return (None, f"exception:{str(e)[:50]}", site_id)
                    sleep_s = self.retry_backoff * (1 + random.random()) * attempt
                    time.sleep(sleep_s)

        # Parallel execution
        completed = 0
        total_sites = len(sites)
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            futures = {ex.submit(fetch_one_iv, sid): sid for sid in sites}
            for fut in as_completed(futures):
                rec, status, sid = fut.result()
                completed += 1
                if rec is not None and status == "ok":
                    results.append(rec)
                else:
                    failures.append((sid, status))
                if completed % 100 == 0 or completed == total_sites:
                    print(f"[USGS IV] Progress: {completed}/{total_sites}")

        self._write_log(ts_str, attempted_sites=sites, successes=results, failures=failures)
        self._save_kv_cache(ts_str, successes=results, failures=failures)

        if not results:
            print(f"No data for {ts_str}.")
            return []

        results = self._filter_by_bbox(results)
        if not results:
            print(f"[bbox] No data within bbox for {ts_str}.")
            return []

        arr = np.array(results, dtype=[("lon", float), ("lat", float), ("streamflow", float), ("gauge_id", "U15")])
        vals = arr["streamflow"]
        print(f"Summary for {ts_str} (after bbox filter)")
        print(f"  - Observations: {len(vals)}")
        print(f"  - Min: {np.min(vals):.4f}, Max: {np.max(vals):.4f}, Mean: {np.mean(vals):.4f}")

        return arr.tolist()

    def _get_streamflow_all(
        self, 
        year: int, 
        month: int, 
        day: int
    ) -> dict:
        """
        Fetch data for all 96 15-minute timestamps in a day.
        Returns dict keyed by timestamp string.
        """
        results_dict = {}
        
        # Generate all 96 15-minute timestamps for the day
        timestamps = []
        for h in range(24):
            for m in [0, 15, 30, 45]:
                timestamps.append((h, m))
        
        print(f"[USGS IV] Fetching all 96 timestamps for {year:04d}-{month:02d}-{day:02d}")
        
        for h, m in timestamps:
            ts_str = f"{year:04d}-{month:02d}-{day:02d}_{h:02d}-{m:02d}"
            data = self._get_streamflow_single(year, month, day, h, m)
            if data:
                results_dict[ts_str] = data
        
        print(f"[USGS IV] Completed: {len(results_dict)}/96 timestamps have data")
        return results_dict

