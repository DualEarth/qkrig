# usgs_loader.py

from __future__ import annotations

import os
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional

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
    # Public API
    # ---------------------------------------------------------------------
    def get_streamflow(self, year: int, month: int, day: int) -> List[Tuple[float, float, float, str]]:
        """
        Fetch (lon, lat, streamflow_mm_day, gauge_id) for all gauges on a date in parallel.
        Returns a list of tuples, with negatives removed.
        """
        target = pd.Timestamp(year=year, month=month, day=day)
        date_str = target.strftime("%Y-%m-%d")

        # Fast-exit
        if self.gauge_metadata.empty:
            print(f"No gauges to query for {date_str}.")
            return []

        # Build tasks
        sites = list(self.gauge_metadata.index.values)

        results: List[Tuple[float, float, float, str]] = []

        def fetch_one(site_id: str) -> Optional[Tuple[float, float, float, str]]:
            """Fetch one site's daily mean discharge and convert to mm/day, with retries."""
            meta = self.gauge_metadata.loc[site_id]
            lon, lat, area_km2 = float(meta["gauge_lon"]), float(meta["gauge_lat"]), float(meta["area_km2"])
            if not np.isfinite(area_km2) or area_km2 <= 0:
                return None

            attempt = 0
            while True:
                try:
                    df = nwis.get_record(
                        sites=site_id,
                        service="dv",
                        start=date_str,
                        end=date_str,
                        parameterCd="00060",  # discharge
                    )
                    if df is None or df.empty:
                        return None

                    # find column with 00060 and Mean (daily value)
                    cols = [c for c in df.columns if ("00060" in c and "Mean" in c)]
                    if not cols:
                        return None

                    cfs = float(df.iloc[0][cols[0]])
                    if not np.isfinite(cfs):
                        return None

                    # cfs -> mm/day
                    area_m2 = area_km2 * 1e6
                    mm_day = (cfs * 0.0283168 * 86400.0 / area_m2) * 1000.0

                    # filter negatives here; return None to drop
                    if mm_day < 0:
                        return None

                    return (lon, lat, mm_day, site_id)

                except Exception:
                    attempt += 1
                    if attempt > self.max_retries:
                        return None
                    # backoff with jitter
                    sleep_s = self.retry_backoff * (1 + random.random()) * attempt
                    time.sleep(sleep_s)

        # Parallel execution with bounded workers
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            futures = {ex.submit(fetch_one, sid): sid for sid in sites}
            for fut in as_completed(futures):
                rec = fut.result()
                if rec is not None:
                    results.append(rec)

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


