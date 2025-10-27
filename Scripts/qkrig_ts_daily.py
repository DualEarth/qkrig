#!/usr/bin/env python3
"""
Extract TS for a single date and save as CSV.
Usage: python qkrig_ts_daily.py YYYY-MM-DD /output_dir
"""

import sys, os, datetime as dt
import numpy as np
import pandas as pd
import geopandas as gpd

# --- USER CONFIG ---
GPKG_PATH   = "~/.ngiab/hydrofabric/v2.2/conus_nextgen.gpkg"
EXPORT_DIR  = "/mnt/disk1/usgskrig/exports/gridsize/200/conus/range/100km/"
LAYER       = "divides"
ID_FIELD    = "divide_id"
GRID_LON_KEY = "grid_lon"
GRID_LAT_KEY = "grid_lat"
GRID_VAL_KEY = "z_interp"

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

def grid_sample(npz_path: str, centroids: pd.Series, lon_key: str, lat_key: str, val_key: str) -> pd.Series:
    with np.load(npz_path, allow_pickle=True) as z:
        L = z[lon_key]; A = z[lat_key]; V = z[val_key]
    return centroids.apply(lambda pt: nearest_grid_value(L, A, V, pt.x, pt.y))

# --- MAIN ---
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_ts_csv.py YYYY-MM-DD /output_dir")
        sys.exit(1)

    date_str = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    d = dt.date.fromisoformat(date_str)
    npz_file = npz_path_for_date(d)
    if not os.path.exists(npz_file):
        print(f"No NPZ file for {d}, skipping")
        sys.exit(0)

    ser = grid_sample(npz_file, gdf["centroid"], GRID_LON_KEY, GRID_LAT_KEY, GRID_VAL_KEY)

    df_day = pd.DataFrame({
        "date": pd.to_datetime(d),
        "divide_id": gdf[ID_FIELD].values,
        "value": ser.values
    })

    csv_path = os.path.join(out_dir, f"ts_{date_str}.csv")
    df_day.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
