# base_krig.py

import numpy as np
import yaml
from pyproj import Geod
from pykrige.ok import OrdinaryKriging

class BaseKrig:
    def __init__(self, data, config_path, year, month, day):
        if len(data) == 0:
            raise ValueError("Input data is empty.")

        self.data = np.array(data, dtype=float)
        self.lons = self.data[:, 0]
        self.lats = self.data[:, 1]
        self.values = self.data[:, 2]
        self.year = year
        self.month = month
        self.day = day
        self.geod = Geod(ellps="WGS84")

        # Load kriging config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        kcfg = self.config.get("kriging", {})

        # Validate config entries
        required_keys = ["grid_size", "variogram_model", "variogram_bins"]
        for key in required_keys:
            if key not in kcfg:
                raise KeyError(f"Missing '{key}' in kriging config.")

        self.grid_size = kcfg["grid_size"]
        self.variogram_model = kcfg["variogram_model"]
        self.variogram_bins = kcfg["variogram_bins"]

        # Create interpolation grid
        lon_min, lon_max = np.min(self.lons), np.max(self.lons)
        lat_min, lat_max = np.min(self.lats), np.max(self.lats)
        self.grid_lon = np.linspace(lon_min, lon_max, self.grid_size)
        self.grid_lat = np.linspace(lat_min, lat_max, self.grid_size)
        self.grid_lon_mesh, self.grid_lat_mesh = np.meshgrid(self.grid_lon, self.grid_lat)

        # Placeholders for kriging results
        self.z_interp = None
        self.kriging_variance = None

    def compute_kriging(self):
        kcfg = self.config.get("kriging", {})
        variogram_params = None

        if kcfg.get("range"):
            variogram_params = {
                "sill": kcfg.get("sill", None),
                "range": kcfg["range"] / 111,  # degrees
                "nugget": kcfg.get("nugget", 0.0)
            }

        ok = OrdinaryKriging(
            self.lons, self.lats, self.values,
            variogram_model=self.variogram_model,
            exact_values=kcfg.get("exact_values", True),
            nlags=kcfg.get("nlags", 12),
            weight=kcfg.get("weight", True),
            variogram_parameters=variogram_params
        )

        self.z_interp, self.kriging_variance = ok.execute("grid", self.grid_lon, self.grid_lat)

    def plot_variogram(self):
        raise NotImplementedError("Use visualization module to plot variogram.")

    def map_krig_interpolation(self):
        raise NotImplementedError("Use visualization module to plot interpolation.")

    def map_krig_error_variance(self):
        raise NotImplementedError("Use visualization module to plot error variance.")