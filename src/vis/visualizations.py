# visualizations.py

import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
from cartopy.io import shapereader as shpreader
from shapely.ops import unary_union
from shapely.geometry import box
from shapely import vectorized
from matplotlib.colors import LogNorm, PowerNorm, BoundaryNorm

class PlotConfig:
    def __init__(self, path=None):
        self.cfg = self._load_yaml_or_default(path)

    def _load_yaml_or_default(self, path):
        default = {
            "save_plots": False,
            "show_plots": True,
            "plots_directory": "./plots",
            "variogram": {
                "figure_size": [8, 5],
                "color": "blue",
                "label": "Empirical Variogram",
                "xlabel": "Distance (km)",
                "ylabel": "Semi-variance",
                "title_prefix": "Empirical Variogram",
                "legend": True,
                "min_value": None,   # y-axis lower bound
                "max_value": None,   # y-axis upper bound
                "ylog": False,       # log scale on y-axis
            },
            "kriging_interpolation": {
                "figure_size": [8, 6],
                "cmap": "coolwarm",
                "levels": 15,
                "colorbar_label": "Interpolated Streamflow (mm/day)",
                "max_value": None,
                "min_value": None,
                "log_scale": False,
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



def _get_conus_mask(krig):
    """
    Boolean mask (ny, nx) True==inside CONUS, False==outside.
    """
    ny, nx = krig.grid_lat.size, krig.grid_lon.size
    glon = krig.grid_lon.astype(float).copy()
    glon = ((glon + 180.0) % 360.0) - 180.0
    glat = krig.grid_lat.astype(float)
    xx, yy = np.meshgrid(glon, glat)

    # Load US polygon and clip to CONUS bounds
    shpfilename = shpreader.natural_earth(
        resolution="50m", category="cultural", name="admin_0_countries"
    )
    geoms = [rec.geometry for rec in shpreader.Reader(shpfilename).records()
             if rec.attributes.get("NAME") == "United States of America"]
    if not geoms:
        return None

    usa_union = unary_union(geoms)

    # Hard CONUS bbox: approx [-125, -66.5] lon, [24.5, 49.5] lat
    conus_bbox = box(-125.0, 24.5, -66.5, 49.5)
    conus_geom = usa_union.intersection(conus_bbox)

    mask = vectorized.contains(conus_geom, xx, yy) | vectorized.touches(conus_geom, xx, yy)
    return np.asarray(mask, dtype=bool)

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

        if not self.krig.semivariogram_ready():
            raise RuntimeError(
                "Semivariogram not computed. Call `krig.compute_semivariogram(...)` before plotting."
            )

        bin_centers, semi_variance = self.krig._semivar_cache

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

        ymax_cfg = self.config.get("max_value", None)

        if self.config.get("ylog", False):
            # Avoid non-positive values on log scale
            current_ymin, current_ymax = plt.ylim()
            eps = 1e-12
            # If min <= 0, bump slightly positive
            if current_ymin <= 0:
                current_ymin = eps
            plt.ylim(bottom=current_ymin, top=ymax_cfg)
            plt.yscale("log")

        if self.config.get("legend", True):
            plt.legend(loc="lower left")
        save_plots = self.plot_cfg.cfg.get("save_plots", False)
        show_plots = self.plot_cfg.cfg.get("show_plots", True)
        plots_dir = self.plot_cfg.cfg.get("plots_directory", "./plots")

        if save_plots:
            os.makedirs(plots_dir, exist_ok=True)
            fname = f"variogram_{self.krig.year:04d}-{self.krig.month:02d}-{self.krig.day:02d}.png"
            plt.savefig(os.path.join(plots_dir, fname), dpi=300, bbox_inches="tight")

        if show_plots:
            plt.show()
        else:
            plt.close()


class KrigingMapPlotter:
    def __init__(self, krig_obj):
        self.krig = krig_obj
        self.plot_cfg = PlotConfig(getattr(self.krig, "plot_config_path", None))
        self.config_interp = self.plot_cfg["kriging_interpolation"]
        self.config_error = self.plot_cfg["kriging_error"]

    def plot_interpolation(self):
        if self.krig.z_interp is None:
            raise RuntimeError("compute_kriging() must be run before plotting interpolation.")

        cfg = self.config_interp

        # Observed bounds
        data_min = float(np.nanmin(self.krig.values))
        data_max = float(np.nanmax(self.krig.values))

        # Config bounds
        vmin = cfg.get("min_value", cfg.get("vmin", None))
        vmax = cfg.get("max_value", cfg.get("vmax", None))
        vmin = data_min if vmin is None else float(vmin)
        vmax = data_max if vmax is None else float(vmax)
        if vmin > vmax:
            vmin, vmax = vmax, vmin

        # Clip surface
        z = np.clip(self.krig.z_interp, vmin, vmax)

        # Land mask (True==land)
        mask = _get_land_mask(self.krig)
        if mask is not None:
            z = np.ma.masked_where(~mask, z)

        # CONUS mask
        conus_mask = _get_conus_mask(self.krig)
        if conus_mask is not None:
            z = np.ma.masked_where(~conus_mask, z)

        # Pick normalization
        # Back-compat: if log_scale is set, default to "log" unless user overrides "norm"
        norm_name = cfg.get("norm", "log" if cfg.get("log_scale", False) else "linear").lower()
        cmap = cfg.get("cmap", "viridis")

        norm = None
        eps = 1e-12

        if norm_name == "log":
            vmin_eff = max(vmin, eps)
            z = np.ma.masked_where(z <= 0, z)
            norm = LogNorm(vmin=vmin_eff, vmax=vmax)
        elif norm_name == "power":
            gamma = float(cfg.get("power_gamma", 0.5))  # <1 stretches low end
            # PowerNorm works with non-negative values, allow zeros
            vmin_eff = max(vmin, 0.0)
            norm = PowerNorm(gamma=gamma, vmin=vmin_eff, vmax=vmax)
        else:
            norm = None  # linear; vmin/vmax passed to plotting call

        # --- Render mode ---
        render_mode = cfg.get("render_mode", "pcolormesh").lower()

        plt.figure(figsize=cfg.get("figure_size", [8, 6]))

        if render_mode == "pcolormesh":
            # Continuous shading -> no “few colors” problem
            pc = plt.pcolormesh(
                self.krig.grid_lon, self.krig.grid_lat, z,
                shading="auto",
                cmap=cmap,
                norm=norm,
                vmin=None if norm is not None else vmin,
                vmax=None if norm is not None else vmax,
            )
            cbar = plt.colorbar(pc, label=cfg.get("colorbar_label", "Interpolated Streamflow (mm/day)"))

        else:  # "contourf"
            levels_cfg = cfg.get("levels", 15)
            levels = levels_cfg

            # If log norm + int levels -> build log-spaced boundaries
            if isinstance(levels_cfg, int) and norm_name == "log":
                base = float(cfg.get("log_scale_base", 10.0))
                start = np.log(max(vmin, eps)) / np.log(base)
                stop  = np.log(vmax) / np.log(base)
                if stop <= start:
                    stop = start + 1.0
                levels = np.logspace(start, stop, int(levels_cfg), base=base)
                # Boundary norm to map bins to full colormap
                norm = BoundaryNorm(levels, ncolors=plt.get_cmap(cmap).N, clip=True)

            cs = plt.contourf(
                self.krig.grid_lon, self.krig.grid_lat, z,
                levels=levels,
                cmap=cmap,
                norm=norm,
                vmin=None if norm is not None else vmin,
                vmax=None if norm is not None else vmax,
                extend="both",
            )
            cbar = plt.colorbar(cs, label=cfg.get("colorbar_label", "Interpolated Streamflow (mm/day)"))

        # Observations (use same norm)
        scatter_cfg = cfg.get("scatter", {})
        plt.scatter(
            self.krig.lons, self.krig.lats, c=self.krig.values,
            s=scatter_cfg.get("s", 8),
            cmap=scatter_cfg.get("cmap", cmap),
            edgecolors=scatter_cfg.get("edgecolors", "none"),
            label=scatter_cfg.get("label", "Observed Data"),
            norm=norm,
            vmin=None if norm is not None else vmin,
            vmax=None if norm is not None else vmax,
        )

        plt.xlabel(cfg.get("xlabel", "Longitude"))
        plt.ylabel(cfg.get("ylabel", "Latitude"))
        title_prefix = cfg.get("title_prefix", "Kriging Interpolation")
        plt.title(f"{title_prefix} ({self.krig.variogram_model} model)")
        if cfg.get("legend", True):
            plt.legend(loc="upper right")

        # Save/show
        save_plots = self.plot_cfg.cfg.get("save_plots", False)
        show_plots = self.plot_cfg.cfg.get("show_plots", True)
        plots_dir = self.plot_cfg.cfg.get("plots_directory", "./plots")
        if save_plots:
            os.makedirs(plots_dir, exist_ok=True)
            fname = f"kriging_interp_{self.krig.year:04d}-{self.krig.month:02d}-{self.krig.day:02d}.png"
            plt.savefig(os.path.join(plots_dir, fname), dpi=300, bbox_inches="tight")
        if show_plots:
            plt.show()
        else:
            plt.close()


    def plot_error_variance(self):
        if self.krig.kriging_variance is None:
            raise RuntimeError("compute_kriging() must be run before plotting error variance.")

        var = self.krig.kriging_variance

        # Apply land mask (True==land, False==water)
        land_mask = _get_land_mask(self.krig)
        if land_mask is not None and land_mask.shape == var.shape:
            var = np.ma.masked_where(~land_mask.astype(bool), var)

        # CONUS mask
        conus_mask = _get_conus_mask(self.krig)
        if conus_mask is not None:
            var = np.ma.masked_where(~conus_mask, var)

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
        save_plots = self.plot_cfg.cfg.get("save_plots", False)
        show_plots = self.plot_cfg.cfg.get("show_plots", True)
        plots_dir = self.plot_cfg.cfg.get("plots_directory", "./plots")

        if save_plots:
            os.makedirs(plots_dir, exist_ok=True)
            fname = f"kriging_error_{self.krig.year:04d}-{self.krig.month:02d}-{self.krig.day:02d}.png"
            plt.savefig(os.path.join(plots_dir, fname), dpi=300, bbox_inches="tight")

        if show_plots:
            plt.show()
        else:
            plt.close()