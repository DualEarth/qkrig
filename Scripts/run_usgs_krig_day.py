#!/usr/bin/env python3
import argparse
import os
import sys
import yaml
import numpy as np

# If installed with `pip install -e .`, these imports just work:
from loaders.usgs_loader import USGSLoader
from interpolation.usgs_krig import USGSKrig

def parse_args():
    p = argparse.ArgumentParser(description="Run USGS kriging for a single day and save plots.")
    p.add_argument("--config", required=True, help="Path to usgsgaugekrig.yaml")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--day", type=int, required=True)
    p.add_argument("--plot-config", default=None,
                   help="Optional override: path to plot_config.yaml (otherwise uses config['plot_config']).")
    return p.parse_args()

def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    plot_cfg_path = args.plot_config or cfg.get("plot_config")
    if plot_cfg_path is None:
        print("WARNING: No plot_config path provided; using plot defaults.")

    # 1) Load streamflow for the day (will bbox-filter on return)
    loader = USGSLoader(args.config)
    data = loader.get_streamflow(args.year, args.month, args.day)
    if not data:
        print(f"[{args.year:04d}-{args.month:02d}-{args.day:02d}] No data returned.")
        return 0

    # 2) Krige
    krig = USGSKrig(data, args.config, args.year, args.month, args.day)
    # Let the visualizer know where plot YAML is:
    krig.plot_config_path = plot_cfg_path

    # Compute
    krig.compute_semivariogram()
    krig.compute_kriging()
    krig.plot_variogram()
    krig.map_krig_interpolation()
    interp_path, vario_path = krig.export_all()
    print("data exported to")
    print(interp_path)
    print(vario_path)

    return 0

if __name__ == "__main__":
    sys.exit(main())

