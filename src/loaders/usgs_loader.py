from __future__ import annotations

import os
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict
from datetime import datetime

import numpy as np
import pandas as pd

from core.base_loader import BaseLoader
import dataretrieval.nwis as nwis


class USGSLoader(BaseLoader):
    """
    Parallel USGS discharge loader with support for daily and 15-min IV data,
    plus post-load bounding box filtering.

    Modes:
      - Daily (default): get_streamflow(year, month, day)
      - Hourly IV:       get_streamflow(year, month, day, hour=H)
      - Single IV:       get_streamflow(year, month, day, hour=H, minute=M)

    Config keys:
      data:
        metadata_file, site_list_file, data_cache_directory
      settings:
        concurrency, max_retries, retry_backoff_seconds,
        min_area_km2, bbox, bbox_pad_deg
    """

    def __init__(self, config_path: str):
        super().__init__(config_path)
        dcfg = self.config.get("data", {})
        scfg = self.config.get("settings", {})

        self.metadata_file: str = dcfg["metadata_file"]
        self.site_list_file: Optional[str] = dcfg.get("site_list_file")
        self.concurrency: int = int(scfg.get("concurrency", 16))
        self.max_retries: int = int(scfg.get("max_retries", 3))
        self.retry_backoff: float = float(scfg.get("retry_backoff_seconds", 0.75))
        self.min_area_km2: float = float(scfg.get("min_area_km2", 0.0))
        self.bbox: Optional[List[float]] = scfg.get("bbox")
        self.bbox_pad_deg: float = float(scfg.get("bbox_pad_deg", 0.0))
        self.logs_dir = dcfg.get("data_cache_directory")
        os.makedirs(self.logs_dir, exist_ok=True)

    def _load_gauge_metadata(self) -> pd.DataFrame:
        dcfg = self.config.get("data", {})
        scfg = self.config.get("settings", {})

        df = pd.read_csv(
            dcfg["metadata_file"], comment="#", dtype={"site_no": str},
            on_bad_lines="skip", engine="python",
        )
        df = df.rename(columns={
            "site_no": "gauge_id",
            "dec_lat_va": "gauge_lat",
            "dec_long_va": "gauge_lon",
            "drain_area_va": "area_km2",
        })

        required = ["gauge_id", "gauge_lat", "gauge_lon", "area_km2"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[USGSLoader] Missing columns: {missing}")
            return pd.DataFrame(columns=required).set_index("gauge_id")

        df = df[required].dropna()
        df["gauge_id"] = df["gauge_id"].str.zfill(8)

        site_list_file = dcfg.get("site_list_file")
        if site_list_file and os.path.exists(site_list_file):
            with open(site_list_file, "r") as f:
                wanted = {line.strip() for line in f if line.strip()}
            df = df[df["gauge_id"].isin(wanted)]

        add_random = int(scfg.get("add_random_sites", 0))
        if add_random > 0:
            all_sites = pd.read_csv(
                dcfg["metadata_file"], comment="#", dtype={"site_no": str},
                on_bad_lines="skip", engine="python",
            )
            all_sites = all_sites.rename(columns={
                "site_no": "gauge_id",
                "dec_lat_va": "gauge_lat",
                "dec_long_va": "gauge_lon",
                "drain_area_va": "area_km2",
            })[required].dropna()
            current = set(df["gauge_id"])
            candidates = all_sites[~all_sites["gauge_id"].isin(current)]
            sample_n = min(add_random, len(candidates))
            if sample_n > 0:
                df = pd.concat([df, candidates.sample(n=sample_n, random_state=42)], ignore_index=True)

        min_area = float(scfg.get("min_area_km2", 0.0))
        if min_area > 0:
            df = df[df["area_km2"] >= min_area]

        bbox = scfg.get("bbox")
        if bbox and len(bbox) == 4:
            min_lon, min_lat, max_lon, max_lat = map(float, bbox)
            df = df[
                (df["gauge_lon"] >= min_lon) & (df["gauge_lon"] <= max_lon)
                & (df["gauge_lat"] >= min_lat) & (df["gauge_lat"] <= max_lat)
            ]

        return df.set_index("gauge_id")

    def _log_path_for_date(self, date_str: str) -> str:
        return os.path.join(self.logs_dir, f"{date_str}.txt")

    def _kv_path_for_date(self, date_str: str) -> str:
        return os.path.join(self.logs_dir, f"{date_str}.kv.txt")

    def _save_kv_cache(self, date_str, successes, failures):
        path = self._kv_path_for_date(date_str)
        lines = [
            f"# KV cache for USGS retrieval on {date_str}",
            f"# Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        for lon, lat, mm, sid in sorted(successes, key=lambda r: r[3]):
            lines.append(f"{sid}=OK,{lon:.8f},{lat:.8f},{mm:.8f}")
        for sid, reason in sorted(failures, key=lambda r: r[0]):
            lines.append(f"{sid}=FAIL,{str(reason).replace(',', ';')}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _load_kv_cache(self, date_str):
        path = self._kv_path_for_date(date_str)
        if not os.path.exists(path):
            return None
        successes, failures = [], []
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
                        lon, lat, mm = float(parts[1]), float(parts[2]), float(parts[3])
                        successes.append((lon, lat, mm, key))
                    except Exception:
                        failures.append((key, "kv_parse_error"))
                elif status == "FAIL" and len(parts) >= 2:
                    failures.append((key, ",".join(parts[1:]).strip()))
                else:
                    failures.append((key, "kv_bad_line"))
        return successes, failures

    def _write_log(self, date_str, attempted_sites, successes, failures):
        path = self._log_path_for_date(date_str)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "USGS Retrieval Log",
            f"Timestamp: {ts}",
            f"Date queried: {date_str}",
            f"Total sites attempted: {len(attempted_sites)}",
            f"Successful: {len(successes)}",
            f"Unsuccessful: {len(failures)}",
            "",
            "=== Successful retrievals ===",
        ]
        if successes:
            for lon, lat, mm, sid in sorted(successes, key=lambda r: r[3]):
                lines.append(f"OK  site={sid}  lon={lon:.6f}  lat={lat:.6f}  streamflow={mm:.6f}")
        else:
            lines.append("(none)")
        lines += ["", "=== Unsuccessful retrievals ==="]
        if failures:
            for sid, reason in sorted(failures, key=lambda r: r[0]):
                lines.append(f"FAIL  site={sid}  reason={reason}")
        else:
            lines.append("(none)")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _filter_by_bbox(self, records):
        """Apply bbox filter to (lon, lat, mm, site_id) records."""
        if not records or not self.bbox or len(self.bbox) != 4:
            return records
        min_lon, min_lat, max_lon, max_lat = map(float, self.bbox)
        pad = float(self.bbox_pad_deg)
        min_lon -= pad; min_lat -= pad; max_lon += pad; max_lat += pad
        filtered = [
            (lon, lat, mm, sid) for (lon, lat, mm, sid) in records
            if (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat)
        ]
        dropped = len(records) - len(filtered)
        if dropped > 0:
            print(f"[bbox] Dropped {dropped} records outside bbox")
        return filtered

    def _return_cached(self, cache_key):
        """Load from KV cache if available. Returns filtered list or None."""
        cached = self._load_kv_cache(cache_key)
        if cached is None:
            return None
        successes, _ = cached
        if not successes:
            print(f"[cache] No data for {cache_key}.")
            return []
        successes = self._filter_by_bbox(successes)
        if not successes:
            print(f"[cache] No data within bbox for {cache_key}.")
            return []
        arr = np.array(successes, dtype=[
            ("lon", float), ("lat", float), ("streamflow", float), ("gauge_id", "U15")
        ])
        vals = arr["streamflow"]
        print(f"[cache] {cache_key}: {len(vals)} obs, "
              f"min={np.min(vals):.4f}, max={np.max(vals):.4f}, mean={np.mean(vals):.4f}")
        return arr.tolist()

    def _finalize_results(self, cache_key, sites, results, failures):
        """Write log/cache, apply bbox, return final list."""
        self._write_log(cache_key, attempted_sites=sites, successes=results, failures=failures)
        self._save_kv_cache(cache_key, successes=results, failures=failures)
        if not results:
            print(f"No data for {cache_key}.")
            return []
        results = self._filter_by_bbox(results)
        if not results:
            print(f"[bbox] No data within bbox for {cache_key}.")
            return []
        arr = np.array(results, dtype=[
            ("lon", float), ("lat", float), ("streamflow", float), ("gauge_id", "U15")
        ])
        vals = arr["streamflow"]
        print(f"{cache_key}: {len(vals)} obs, "
              f"min={np.min(vals):.4f}, max={np.max(vals):.4f}, mean={np.mean(vals):.4f}")
        return arr.tolist()

    # --- Public API ---

    def get_streamflow(self, year, month, day, hour=None, minute=None):
        """
        Fetch discharge data for all gauges.

        Modes:
          - Daily:     get_streamflow(year, month, day)      -> List
          - Hourly IV: get_streamflow(year, month, day, H)   -> Dict of 4 Lists
          - Single IV: get_streamflow(year, month, day, H, M) -> List

        For bulk IV data, use the offline pipeline:
          usgs_raw_iv.py -> usgs_raw_to_iv_kv.py -> .kv.txt cache
        """
        if hour is not None and minute is not None:
            return self._get_streamflow_iv(year, month, day, hour, minute)
        elif hour is not None:
            return self._get_streamflow_iv_hour(year, month, day, hour)
        else:
            return self._get_streamflow_daily(year, month, day)

    # --- Daily mode ---

    def _get_streamflow_daily(self, year, month, day):
        """Fetch daily mean discharge (mm/day) via nwis.get_record(service='dv')."""
        date_str = f"{year:04d}-{month:02d}-{day:02d}"

        from_cache = self._return_cached(date_str)
        if from_cache is not None:
            return from_cache

        if self.gauge_metadata.empty:
            self._write_log(date_str, [], [], [])
            self._save_kv_cache(date_str, [], [])
            print(f"No gauges to query for {date_str}.")
            return []

        sites = list(self.gauge_metadata.index.values)
        results, failures = [], []
        print(f"[DV] Fetching {date_str} ({len(sites)} sites)")

        def fetch_one(site_id):
            try:
                meta = self.gauge_metadata.loc[site_id]
            except KeyError:
                return (None, "missing_metadata", site_id)
            lon, lat, area_km2 = float(meta["gauge_lon"]), float(meta["gauge_lat"]), float(meta["area_km2"])
            if not np.isfinite(area_km2) or area_km2 <= 0:
                return (None, "invalid_area", site_id)
            attempt = 0
            sid = site_id.zfill(8)
            while True:
                try:
                    df = nwis.get_record(sites=sid, service="dv",
                                         start=date_str, end=date_str, parameterCd="00060")
                    if df is None or df.empty:
                        return (None, "nwis_empty", site_id)
                    cols = [c for c in df.columns if "00060" in c and "Mean" in c]
                    if not cols:
                        return (None, "missing_mean_col", site_id)
                    cfs = float(df.iloc[0][cols[0]])
                    if not np.isfinite(cfs):
                        return (None, "nonfinite_cfs", site_id)
                    area_m2 = area_km2 * 1e6
                    mm_day = (cfs * 0.0283168 * 86400.0 / area_m2) * 1000.0
                    if mm_day < -500 or mm_day > 500:
                        return (None, "large_magnitude_flow", site_id)
                    return ((lon, lat, mm_day, site_id), "ok", site_id)
                except Exception as e:
                    attempt += 1
                    if attempt > self.max_retries:
                        return (None, f"exception:{str(e)[:50]}", site_id)
                    time.sleep(self.retry_backoff * (1 + random.random()) * attempt)

        completed, total = 0, len(sites)
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            futures = {ex.submit(fetch_one, sid): sid for sid in sites}
            for fut in as_completed(futures):
                rec, status, sid = fut.result()
                completed += 1
                if rec is not None and status == "ok":
                    results.append(rec)
                else:
                    failures.append((sid, status))
                if completed % 100 == 0 or completed == total:
                    print(f"[DV] {completed}/{total}")

        return self._finalize_results(date_str, sites, results, failures)

    # --- IV mode: single timestamp ---

    def _get_streamflow_iv(self, year, month, day, hour, minute):
        """Fetch IV discharge (mm/15min) for a single timestamp via nwis.get_iv()."""
        target = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute, tz='UTC')
        ts_str = target.strftime("%Y-%m-%d_%H-%M")
        date_str = f"{year:04d}-{month:02d}-{day:02d}"

        from_cache = self._return_cached(ts_str)
        if from_cache is not None:
            return from_cache

        if self.gauge_metadata.empty:
            self._write_log(ts_str, [], [], [])
            self._save_kv_cache(ts_str, [], [])
            print(f"No gauges to query for {ts_str}.")
            return []

        sites = list(self.gauge_metadata.index.values)
        results, failures = [], []
        print(f"[IV] Fetching {ts_str} ({len(sites)} sites)")

        def fetch_one(site_id):
            try:
                meta = self.gauge_metadata.loc[site_id]
            except KeyError:
                return (None, "missing_metadata", site_id)
            lon, lat, area_km2 = float(meta["gauge_lon"]), float(meta["gauge_lat"]), float(meta["area_km2"])
            if not np.isfinite(area_km2) or area_km2 <= 0:
                return (None, "invalid_area", site_id)
            attempt = 0
            sid = site_id.zfill(8)
            while True:
                try:
                    result = nwis.get_iv(sites=sid, startDT=date_str, endDT=date_str, parameterCd="00060")
                    df = result[0] if isinstance(result, tuple) else result
                    if df is None or df.empty:
                        return (None, "nwis_empty", site_id)
                    cols = [c for c in df.columns if "00060" in c]
                    if not cols:
                        return (None, "missing_00060_col", site_id)
                    df.index = pd.to_datetime(df.index)
                    if target in df.index:
                        row = df.loc[target]
                    else:
                        mask = abs(df.index - target) <= pd.Timedelta(minutes=7)
                        if not mask.any():
                            return (None, "no_matching_time", site_id)
                        row = df[mask].iloc[0]
                    cfs = float(row[cols[0]])
                    if not np.isfinite(cfs):
                        return (None, "nonfinite_cfs", site_id)
                    area_m2 = area_km2 * 1e6
                    mm_15min = (cfs * 0.0283168 * 900.0 / area_m2) * 1000.0
                    if mm_15min < -10 or mm_15min > 100:
                        return (None, "large_magnitude_flow", site_id)
                    return ((lon, lat, mm_15min, site_id), "ok", site_id)
                except Exception as e:
                    attempt += 1
                    if attempt > self.max_retries:
                        return (None, f"exception:{str(e)[:50]}", site_id)
                    time.sleep(self.retry_backoff * (1 + random.random()) * attempt)

        completed, total = 0, len(sites)
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            futures = {ex.submit(fetch_one, sid): sid for sid in sites}
            for fut in as_completed(futures):
                rec, status, sid = fut.result()
                completed += 1
                if rec is not None and status == "ok":
                    results.append(rec)
                else:
                    failures.append((sid, status))
                if completed % 100 == 0 or completed == total:
                    print(f"[IV] {completed}/{total}")

        return self._finalize_results(ts_str, sites, results, failures)

    # --- IV mode: hourly (4 timestamps) ---

    def _get_streamflow_iv_hour(self, year, month, day, hour):
        """
        Fetch IV discharge for all 4 timestamps in a given hour.
        Returns dict keyed by 'YYYY-MM-DD_HH-MM'.
        One API call per site covers all 4 timestamps.
        """
        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        targets = [
            pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=m, tz='UTC')
            for m in (0, 15, 30, 45)
        ]
        ts_strs = [t.strftime("%Y-%m-%d_%H-%M") for t in targets]

        results_dict: Dict[str, List] = {}
        need_fetch = []
        for i, ts_str in enumerate(ts_strs):
            from_cache = self._return_cached(ts_str)
            if from_cache is not None:
                results_dict[ts_str] = from_cache
            else:
                need_fetch.append(i)

        if not need_fetch:
            return results_dict

        fetch_targets = [targets[i] for i in need_fetch]
        fetch_ts_strs = [ts_strs[i] for i in need_fetch]

        if self.gauge_metadata.empty:
            for ts_str in fetch_ts_strs:
                self._write_log(ts_str, [], [], [])
                self._save_kv_cache(ts_str, [], [])
                results_dict[ts_str] = []
            print(f"No gauges to query for hour {hour:02d}.")
            return results_dict

        sites = list(self.gauge_metadata.index.values)
        ts_results = {ts: [] for ts in fetch_ts_strs}
        ts_failures = {ts: [] for ts in fetch_ts_strs}
        print(f"[IV] Fetching hour {hour:02d} ({len(sites)} sites, {len(fetch_ts_strs)} timestamps)")

        def fetch_one(site_id):
            try:
                meta = self.gauge_metadata.loc[site_id]
            except KeyError:
                return [(ts, None, "missing_metadata") for ts in fetch_ts_strs]
            lon, lat, area_km2 = float(meta["gauge_lon"]), float(meta["gauge_lat"]), float(meta["area_km2"])
            if not np.isfinite(area_km2) or area_km2 <= 0:
                return [(ts, None, "invalid_area") for ts in fetch_ts_strs]
            attempt = 0
            sid = site_id.zfill(8)
            while True:
                try:
                    result = nwis.get_iv(sites=sid, startDT=date_str, endDT=date_str, parameterCd="00060")
                    df = result[0] if isinstance(result, tuple) else result
                    if df is None or df.empty:
                        return [(ts, None, "nwis_empty") for ts in fetch_ts_strs]
                    cols = [c for c in df.columns if "00060" in c]
                    if not cols:
                        return [(ts, None, "missing_00060_col") for ts in fetch_ts_strs]
                    df.index = pd.to_datetime(df.index)
                    area_m2 = area_km2 * 1e6
                    out = []
                    for tgt, ts_str in zip(fetch_targets, fetch_ts_strs):
                        if tgt in df.index:
                            row = df.loc[tgt]
                        else:
                            mask = abs(df.index - tgt) <= pd.Timedelta(minutes=7)
                            if not mask.any():
                                out.append((ts_str, None, "no_matching_time"))
                                continue
                            row = df[mask].iloc[0]
                        try:
                            cfs = float(row[cols[0]])
                        except Exception:
                            out.append((ts_str, None, "bad_value"))
                            continue
                        if not np.isfinite(cfs):
                            out.append((ts_str, None, "nonfinite_cfs"))
                            continue
                        mm_15min = (cfs * 0.0283168 * 900.0 / area_m2) * 1000.0
                        if mm_15min < -10 or mm_15min > 100:
                            out.append((ts_str, None, "large_magnitude_flow"))
                            continue
                        out.append((ts_str, (lon, lat, mm_15min, site_id), "ok"))
                    return out
                except Exception as e:
                    attempt += 1
                    if attempt > self.max_retries:
                        return [(ts, None, f"exception:{str(e)[:50]}") for ts in fetch_ts_strs]
                    time.sleep(self.retry_backoff * (1 + random.random()) * attempt)

        completed, total = 0, len(sites)
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            futures = {ex.submit(fetch_one, sid): sid for sid in sites}
            for fut in as_completed(futures):
                for ts_str, rec, status in fut.result():
                    if rec is not None and status == "ok":
                        ts_results[ts_str].append(rec)
                    else:
                        sid_from = rec[3] if rec else futures[fut]
                        ts_failures[ts_str].append((str(sid_from), status))
                completed += 1
                if completed % 100 == 0 or completed == total:
                    print(f"[IV] {completed}/{total}")

        for ts_str in fetch_ts_strs:
            results_dict[ts_str] = self._finalize_results(
                ts_str, sites, ts_results[ts_str], ts_failures[ts_str]
            )

        ok_count = sum(1 for v in results_dict.values() if v)
        print(f"[IV] Hour {hour:02d}: {ok_count}/{len(ts_strs)} timestamps have data")
        return results_dict
