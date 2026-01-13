# base_krig.py

import os
import numpy as np
import yaml
from pyproj import Geod
from pykrige.ok import OrdinaryKriging
from typing import Optional, Tuple, Dict
from scipy.optimize import curve_fit

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
            self.config = yaml.safe_load(f) or {}

        kcfg = self.config.get("kriging", {}) or {}
        self.plot_config_path = self.config.get("plot_config", None)

        land_mask_path = self.config.get("data", {}).get("land_mask")
        if land_mask_path and os.path.exists(land_mask_path):
            self.land_mask = np.load(land_mask_path)
        else:
            self.land_mask = None

        # Validate config entries
        required_keys = ["grid_size", "variogram_model", "variogram_bins"]
        for key in required_keys:
            if key not in kcfg:
                raise KeyError(f"Missing '{key}' in kriging config.")

        self.grid_size = int(kcfg["grid_size"])
        self.variogram_model = kcfg["variogram_model"]
        self.variogram_bins = int(kcfg["variogram_bins"])

        # Create interpolation grid
        lon_min, lon_max = np.min(self.lons), np.max(self.lons)
        lat_min, lat_max = np.min(self.lats), np.max(self.lats)
        self.grid_lon = np.linspace(lon_min, lon_max, self.grid_size)
        self.grid_lat = np.linspace(lat_min, lat_max, self.grid_size)
        self.grid_lon_mesh, self.grid_lat_mesh = np.meshgrid(self.grid_lon, self.grid_lat)

        # Placeholders for kriging results
        self.z_interp: Optional[np.ndarray] = None
        self.kriging_variance: Optional[np.ndarray] = None

        # Semivariogram cache
        self._semivar_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (bin_centers_km, semi_variance)
        self._semivar_bins_used: Optional[int] = None

    # ---------------------------------------------------------------------
    # Core computations
    # ---------------------------------------------------------------------
    def _spherical_model(self, h, sill, rng, nugget):
        """Spherical variogram model."""
        h = np.asarray(h, dtype=float)
        gamma = np.where(
            h <= rng,
            nugget + sill * (1.5 * (h / rng) - 0.5 * (h / rng) ** 3),
            nugget + sill,
        )
        return gamma
    
    def fit_variogram_from_empirical(self, bins: Optional[int] = None):
        """
        Fit a spherical variogram model to the empirical semivariogram.
        Returns sill, range_km, nugget.
        """
        if not self.semivariogram_ready(bins):
            self.compute_semivariogram(bins=bins)

        h_km, gamma = self._semivar_cache

        mask = np.isfinite(gamma)
        h_km = h_km[mask]
        gamma = gamma[mask]

        if len(h_km) < 3:
            raise RuntimeError("Not enough variogram points to fit model.")

        # --- Initial guesses (robust defaults)
        nugget0 = self.config["kriging"].get("nugget", 0.0)
        sill0 = np.nanmax(gamma)
        range0 = self.config["kriging"].get("range", np.nanmax(h_km))

        p0 = [sill0, range0, nugget0]

        bounds = (
            (0.0, 1e-6, 0.0),      # lower
            (np.inf, np.inf, np.inf),  # upper
        )

        popt, _ = curve_fit(
            self._spherical_model,
            h_km,
            gamma,
            p0=p0,
            bounds=bounds,
            maxfev=10_000,
        )

        sill, range_km, nugget = popt
        return float(sill), float(range_km), float(nugget)

    



    



    def compute_kriging(self):
        kcfg = self.config.get("kriging", {}) or {}
        # --- Fit variogram from empirical data (PER TIMESTEP)
        sill, range_km, nugget = self.fit_variogram_from_empirical(
            bins=kcfg.get("variogram_bins")
        )

        variogram_params = {
            "sill": sill,
            "range": range_km / 111.0,  # km → degrees
            "nugget": nugget,
        }

        ok = OrdinaryKriging(
            self.lons, self.lats, self.values,
            variogram_model=self.variogram_model,
            exact_values=kcfg.get("exact_values", True),
            nlags=kcfg.get("nlags", 12),
            weight=kcfg.get("weight", True),
            variogram_parameters=variogram_params,
        )

        self.z_interp, self.kriging_variance = ok.execute("grid", self.grid_lon, self.grid_lat)

    def compute_semivariogram(self, bins: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute and CACHE the empirical semivariogram on geodesic distances.
        Must be called before plotting or exporting the variogram.

        Returns:
            bin_centers_km: (B,) array
            semi_variance:  (B,) array (NaN where no pairs fall in bin)
        """
        if bins is None:
            bins = int(self.variogram_bins)

        n = len(self.lons)
        if n < 2:
            raise ValueError("Need at least two points to compute a semivariogram.")

        dists_km = []
        sqdiff = []
        for i in range(n):
            for j in range(i + 1, n):
                _, _, d_m = self.geod.inv(self.lons[i], self.lats[i], self.lons[j], self.lats[j])
                dists_km.append(d_m / 1000.0)
                sqdiff.append((self.values[i] - self.values[j]) ** 2)

        dists_km = np.asarray(dists_km, dtype=float)
        sqdiff = np.asarray(sqdiff, dtype=float)

        if dists_km.size == 0:
            raise ValueError("Not enough unique pairs to compute a semivariogram.")

        bin_edges = np.linspace(0.0, float(np.nanmax(dists_km)), bins + 1)
        bin_idx = np.digitize(dists_km, bin_edges) - 1

        semi_variance = np.full(bins, np.nan, dtype=float)
        for b in range(bins):
            mask = (bin_idx == b)
            if np.any(mask):
                semi_variance[b] = float(np.nanmean(sqdiff[mask]))

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Cache & record bins used
        self._semivar_cache = (bin_centers, semi_variance)
        self._semivar_bins_used = int(bins)
        return bin_centers, semi_variance

    # ---------------------------------------------------------------------
    # Readiness helpers
    # ---------------------------------------------------------------------
    def semivariogram_ready(self, bins: Optional[int] = None) -> bool:
        """Return True if a semivariogram is cached (and bins match if provided)."""
        if self._semivar_cache is None:
            return False
        if bins is None:
            return True
        return (self._semivar_bins_used == int(bins))

    # ---------------------------------------------------------------------
    # Export helpers
    # ---------------------------------------------------------------------
    def _date_str(self) -> str:
        return f"{self.year:04d}-{self.month:02d}-{self.day:02d}"

    def _resolve_exports(self) -> str:
        exp_cfg: Dict = (self.config.get("exports") or {})
        export_dir = exp_cfg.get("directory", "./exports")
        os.makedirs(export_dir, exist_ok=True)
        return export_dir

    def export_all(self, bins: Optional[int] = None) -> Tuple[str, str]:
        """
        Export both interpolation (.npz) and semivariogram (.csv) to the configured directory.
        Requires that compute_kriging() and compute_semivariogram() were called beforehand.
        """
        if self.z_interp is None or self.kriging_variance is None:
            raise RuntimeError("compute_kriging() must be run before export_all().")
        if not self.semivariogram_ready(bins):
            raise RuntimeError("compute_semivariogram() must be run (with matching bins) before export_all().")

        export_dir = self._resolve_exports()
        d = self._date_str()

        interp_path = os.path.join(export_dir, f"interp_{d}.npz")
        vario_path  = os.path.join(export_dir, f"variogram_{d}.csv")

        self.export_interpolation(interp_path)
        self.export_variogram(vario_path, bins=bins)

        return interp_path, vario_path

    # ---------------------------------------------------------------------
    # Exports
    # ---------------------------------------------------------------------
    def export_interpolation(self, out_path: str):
        """
        Export interpolation grids (z and variance) plus axes to NPZ.
        Requires compute_kriging() beforehand.
        """
        if self.z_interp is None or self.kriging_variance is None:
            raise RuntimeError("compute_kriging() must be run before exporting interpolation.")

        meta = {
            "date": self._date_str(),
            "variogram_model": self.variogram_model,
            "grid_size": int(self.grid_size),
        }

        np.savez_compressed(
            out_path,
            grid_lon=self.grid_lon,
            grid_lat=self.grid_lat,
            z_interp=self.z_interp,
            kriging_variance=self.kriging_variance,
            **{f"meta_{k}": v for k, v in meta.items()},
        )

    def export_variogram(self, out_path: str, bins: Optional[int] = None):
        """
        Export the empirical semivariogram to CSV.
        Requires compute_semivariogram() beforehand (with same bins if provided).
        """
        if not self.semivariogram_ready(bins):
            raise RuntimeError(
                "compute_semivariogram() must be called before exporting the variogram "
                "(and bins must match if specified)."
            )

        bin_centers, semi_variance = self._semivar_cache
        rows = np.column_stack([bin_centers, semi_variance])
        header = "distance_km,semi_variance"
        np.savetxt(out_path, rows, delimiter=",", header=header, comments="")

    # ---------------------------------------------------------------------
    # Plot delegation
    # ---------------------------------------------------------------------
    def plot_variogram(self):
        raise NotImplementedError("Use visualization module to plot variogram.")

    def map_krig_interpolation(self):
        raise NotImplementedError("Use visualization module to plot interpolation.")

    def map_krig_error_variance(self):
        raise NotImplementedError("Use visualization module to plot error variance.")
    
    def plot_interpolation_with_variogram():
        raise NotImplementedError("Use visualization module to plot combo.")