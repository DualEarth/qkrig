#!/usr/bin/env python3
"""
Extract TS for a single date and save into water-year-wise catchment CSVs.
Usage: python qkrig_ts_daily.py YYYY-MM-DD /output_dir
"""

import sys, os, datetime as dt
import numpy as np
import pandas as pd
import geopandas as gpd

# --- USER CONFIG ---
GPKG_PATH    = "~/.ngiab/hydrofabric/v2.2/conus_nextgen.gpkg"
EXPORT_DIR   = "/mnt/disk1/usgskrig/exports/gridsize/200/conus/range/100km/all_guages/"
LAYER        = "divides"
ID_FIELD     = "divide_id"
GRID_LON_KEY = "grid_lon"
GRID_LAT_KEY = "grid_lat"
GRID_VAL_KEY = "z_interp"
GRID_VAR_KEY = "kriging_variance"   # variance key confirmed

# --- Load GDF ---
gdf = gpd.read_file(GPKG_PATH, layer=LAYER)
if gdf.crs.is_geographic:
    gdf_proj = gdf.to_crs("EPSG:5070")
else:
    gdf_proj = gdf
cent_proj = gdf_proj.geometry.centroid
gdf["centroid"] = gpd.GeoSeries(cent_proj, crs=gdf_proj.crs).to_crs(4326)

# --- Helper functions ---
def npz_path_for_date(d: dt.date) -> str:
    return os.path.join(EXPORT_DIR, f"interp_{d.isoformat()}.npz")

def nearest_grid_value(lons, lats, vals, pt_lon, pt_lat):
    ix = np.argmin(np.abs(lons - pt_lon))
    iy = np.argmin(np.abs(lats - pt_lat))
    return float(vals[iy, ix])

def grid_sample_both(npz_path, centroids):
    with np.load(npz_path, allow_pickle=True) as z:
        L, A = z[GRID_LON_KEY], z[GRID_LAT_KEY]
        V, VV = z[GRID_VAL_KEY], z.get(GRID_VAR_KEY, None)
    if VV is None:
        return centroids.apply(lambda pt: (nearest_grid_value(L, A, V, pt.x, pt.y), np.nan))
    return centroids.apply(lambda pt: (
        nearest_grid_value(L, A, V, pt.x, pt.y),
        nearest_grid_value(L, A, VV, pt.x, pt.y)
    ))

def water_year(d: dt.date) -> int:
    """Return the water year for a given date (Oct 1 - Sep 30)."""
    return d.year + 1 if d.month >= 10 else d.year

# --- MAIN ---
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python qkrig_ts_daily.py YYYY-MM-DD /output_dir")
        sys.exit(1)

    date_str = sys.argv[1]
    out_dir = sys.argv[2]
    d = dt.date.fromisoformat(date_str)
    npz_file = npz_path_for_date(d)

    if not os.path.exists(npz_file):
        print(f"No NPZ file for {d}, skipping")
        sys.exit(0)

    # --- Sample daily values (both mean + variance) ---
    vals = grid_sample_both(npz_file, gdf["centroid"])
    ser_val = vals.apply(lambda x: x[0])
    ser_var = vals.apply(lambda x: x[1])

    wy = water_year(d)
    wy_dir = os.path.join(out_dir, f"WY{wy}")
    os.makedirs(wy_dir, exist_ok=True)

    # --- Append to per-catchment CSVs ---
    for cat_id, val, var_val in zip(gdf[ID_FIELD].values, ser_val.values, ser_var.values):
        cat_file = os.path.join(wy_dir, f"{cat_id}.csv")
        df_day = pd.DataFrame({"date": [d], "qkrig": [val], "variance": [var_val]})
        if os.path.exists(cat_file):
            df_day.to_csv(cat_file, mode='a', header=False, index=False)
        else:
            df_day.to_csv(cat_file, mode='w', header=True, index=False)

    print(f"Appended daily values + variance for {date_str} to WY{wy} catchment files")
