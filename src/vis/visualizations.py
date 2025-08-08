# visualizations.py

import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
from cartopy.io import shapereader as shpreader
from shapely.ops import unary_union
from shapely.geometry import box
from shapely import vectorized

class PlotConfig:
    def __init__(self, path=None):
        self.cfg = self._load_yaml_or_default(path)

    def _load_yaml_or_default(self, path):
        default = {
            "variogram": {
                "figure_size": [8, 5],
                "color": "blue",
                "label": "Empirical Variogram",
                "xlabel": "Distance (km)",
                "ylabel": "Semi-variance",
                "title_prefix": "Empirical Variogram",
                "legend": True,
            },
            "kriging_interpolation": {
                "figure_size": [8, 6],
                "cmap": "coolwarm",
                "levels": 15,
                "colorbar_label": "Interpolated Streamflow (mm/day)",
                "scatter": {
                    "cmap": "coolwarm",
                    "s": 8,
                    "edgecolors": "none",
                    "label": "Observed Data",
                },
                "xlabel": "Longitude",
                "ylabel": "Latitude",
                "title_prefix": "Kriging Interpolation",
                "legend": True,
            },
            "kriging_error": {
                "figure_size": [8, 6],
                "cmap": "viridis",
                "levels": 15,
                "colorbar_label": "Kriging Error Variance",
                "xlabel": "Longitude",
                "ylabel": "Latitude",
                "title": "Kriging Error Variance Map",
            },
        }

        if path and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    cfg = yaml.safe_load(f) or {}
                for k, v in default.items():
                    if k in cfg and isinstance(cfg[k], dict) and isinstance(v, dict):
                        merged = v.copy()
                        merged.update(cfg[k])
                        default[k] = merged
                    elif k in cfg:
                        default[k] = cfg[k]
            except Exception:
                pass  # keep defaults
        return default

    def __getitem__(self, item):
        return self.cfg.get(item, {})


def _get_land_mask(krig):
    """
    Returns boolean mask [lat, lon] where True==land, False==water.
    Caches on krig.land_mask. Only accepts masks that match (ny, nx).
    """
    ny, nx = krig.grid_lat.size, krig.grid_lon.size

    # If a mask is cached, ensure it matches the *current* grid
    if getattr(krig, "land_mask", None) is not None:
        m = krig.land_mask
        if isinstance(m, np.ndarray) and m.shape == (ny, nx):
            return m
        # stale or mismatched — drop it
        krig.land_mask = None

    # 1) Only accept external raster if shape matches exactly
    mask_path = krig.config.get("data", {}).get("land_mask")
    if mask_path and os.path.exists(mask_path):
        try:
            arr = np.load(mask_path)
            if arr.shape == (ny, nx):
                krig.land_mask = arr.astype(bool)
                return krig.land_mask
        except Exception:
            pass  # ignore and fall through

    # 2) Build on-the-fly from Natural Earth on the krig grid
    try:

        # Normalize longitudes to [-180, 180] to match Natural Earth
        glon = krig.grid_lon.astype(float).copy()
        glon = ((glon + 180.0) % 360.0) - 180.0
        glat = krig.grid_lat.astype(float)

        xx, yy = np.meshgrid(glon, glat)  # (ny, nx)

        shpfilename = shpreader.natural_earth(
            resolution="50m", category="physical", name="land"
        )
        geoms = list(shpreader.Reader(shpfilename).geometries())
        if not geoms:
            krig.land_mask = None
            return None

        # Crop geometries to our grid bbox for speed
        lon_min, lon_max = float(glon.min()), float(glon.max())
        lat_min, lat_max = float(glat.min()), float(glat.max())
        bbox = box(lon_min, lat_min, lon_max, lat_max)
        geoms = [g for g in geoms if g.intersects(bbox)]
        if not geoms:
            krig.land_mask = None
            return None

        land_union = unary_union(geoms).buffer(0)

        # Prefer covers (includes boundaries); fall back to contains|touches
        if hasattr(vectorized, "covers"):
            mask = vectorized.covers(land_union, xx, yy)
        else:
            mask = vectorized.contains(land_union, xx, yy) | vectorized.touches(land_union, xx, yy)

        mask = np.asarray(mask, dtype=bool)

        # Guarantee mask shape == (ny, nx)
        if mask.shape != (ny, nx):
            # Transpose if needed (some backends flip axes)
            if mask.T.shape == (ny, nx):
                mask = mask.T
            else:
                # As a last resort, bail out rather than returning a mismatched array
                krig.land_mask = None
                return None

        krig.land_mask = mask
        return krig.land_mask

    except Exception:
        krig.land_mask = None
        return None


class VariogramPlotter:
    def __init__(self, krig_obj):
        self.krig = krig_obj
        self.plot_cfg = PlotConfig(getattr(self.krig, "plot_config_path", None))
        self.config = self.plot_cfg["variogram"]

    def plot(self):
        distances, differences = [], []
        num_points = len(self.krig.lons)

        for i in range(num_points):
            for j in range(i + 1, num_points):
                _, _, d = self.krig.geod.inv(
                    self.krig.lons[i], self.krig.lats[i],
                    self.krig.lons[j], self.krig.lats[j]
                )
                distances.append(d / 1000.0)  # km
                differences.append((self.krig.values[i] - self.krig.values[j]) ** 2)

        distances = np.array(distances)
        differences = np.array(differences)

        if distances.size == 0:
            raise ValueError("Not enough points to compute a variogram (need at least 2).")

        bins = self.config.get("bins", getattr(self.krig, "variogram_bins", 25))
        bin_edges = np.linspace(0, np.max(distances), bins + 1)
        bin_indices = np.digitize(distances, bin_edges) - 1

        semi_variance = []
        for i in range(bins):
            mask = (bin_indices == i)
            semi_variance.append(differences[mask].mean() if np.any(mask) else np.nan)
        semi_variance = np.array(semi_variance)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        plt.figure(figsize=self.config.get("figure_size", [8, 5]))
        plt.scatter(
            bin_centers, semi_variance,
            c=self.config.get("color", "blue"),
            label=self.config.get("label", "Empirical Variogram"),
        )
        plt.xlabel(self.config.get("xlabel", "Distance (km)"))
        plt.ylabel(self.config.get("ylabel", "Semi-variance"))
        title_prefix = self.config.get("title_prefix", "Empirical Variogram")
        date_str = f"{self.krig.year}-{self.krig.month:02d}-{self.krig.day:02d}"
        plt.title(f"{title_prefix} - {date_str}")
        if self.config.get("legend", True):
            plt.legend()
        plt.show()


class KrigingMapPlotter:
    def __init__(self, krig_obj):
        self.krig = krig_obj
        self.plot_cfg = PlotConfig(getattr(self.krig, "plot_config_path", None))
        self.config_interp = self.plot_cfg["kriging_interpolation"]
        self.config_error = self.plot_cfg["kriging_error"]

    def plot_interpolation(self):
        if self.krig.z_interp is None:
            raise RuntimeError("compute_kriging() must be run before plotting interpolation.")

        z = np.clip(self.krig.z_interp, np.min(self.krig.values), np.max(self.krig.values))
        mask = _get_land_mask(self.krig)
        if mask is not None:
            z = np.ma.masked_where(~mask, z)

        plt.figure(figsize=self.config_interp.get("figure_size", [8, 6]))
        plt.contourf(
            self.krig.grid_lon, self.krig.grid_lat, z,
            levels=self.config_interp.get("levels", 15),
            cmap=self.config_interp.get("cmap", "coolwarm"),
        )
        plt.colorbar(label=self.config_interp.get("colorbar_label", "Interpolated Streamflow (mm/day)"))

        scatter_cfg = self.config_interp.get("scatter", {})
        plt.scatter(
            self.krig.lons, self.krig.lats, c=self.krig.values,
            s=scatter_cfg.get("s", 8),
            cmap=scatter_cfg.get("cmap", "coolwarm"),
            edgecolors=scatter_cfg.get("edgecolors", "none"),
            label=scatter_cfg.get("label", "Observed Data"),
        )

        plt.xlabel(self.config_interp.get("xlabel", "Longitude"))
        plt.ylabel(self.config_interp.get("ylabel", "Latitude"))
        title_prefix = self.config_interp.get("title_prefix", "Kriging Interpolation")
        plt.title(f"{title_prefix} ({self.krig.variogram_model} model)")
        if self.config_interp.get("legend", True):
            plt.legend()
        plt.show()

    def plot_error_variance(self):
        if self.krig.kriging_variance is None:
            raise RuntimeError("compute_kriging() must be run before plotting error variance.")

        var = self.krig.kriging_variance

        # Apply land mask (True==land, False==water)
        land_mask = _get_land_mask(self.krig)
        if land_mask is not None and land_mask.shape == var.shape:
            var = np.ma.masked_where(~land_mask.astype(bool), var)

        plt.figure(figsize=self.config_error.get("figure_size", [8, 6]))
        plt.contourf(
            self.krig.grid_lon, self.krig.grid_lat, var,
            levels=self.config_error.get("levels", 15),
            cmap=self.config_error.get("cmap", "viridis"),
        )
        plt.colorbar(label=self.config_error.get("colorbar_label", "Kriging Error Variance"))
        plt.xlabel(self.config_error.get("xlabel", "Longitude"))
        plt.ylabel(self.config_error.get("ylabel", "Latitude"))
        plt.title(self.config_error.get("title", "Kriging Error Variance Map"))
        plt.show()