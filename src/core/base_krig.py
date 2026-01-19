# base_krig.py

import os
import numpy as np
import yaml
from pyproj import Geod
from pykrige.ok import OrdinaryKriging
from typing import Optional, Tuple, Dict

class BaseKrig:
    def __init__(self, data, config_path, year, month, day):
        if len(data) == 0:
            raise ValueError("Input data is empty.")

        self.data = np.array(data, dtype=float)
        self.lons = self.data[:, 0]
        self.lats = self.data[:, 1]
        self.values = self.data[:, 2]
        self.eps = 1e-3
        self.values_log = np.log(self.values + self.eps)
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
    
    def fit_daily_variogram(self):
        """
        Fit empirical variogram for this day in LOG-SPACE.

        FIXED:
        - nugget = 1
        - range = 100 km

        ESTIMATED (daily):
        - sill (robust log-space variance)

        Returns sill, nugget, range in DEGREES (PyKrige-ready).
        """
        num_points = len(self.lons)
        if num_points < 5:
            raise ValueError("Too few points for variogram fitting")

        # ------------------------------
        # --- safe log-transform with less aggressive floor ---
        eps = 0.2  # small flows treated as 0.2, keeps sill reasonable
        values_log = np.log(np.maximum(self.values, eps))


        # Fallback if any NaNs remain
        if np.any(np.isnan(values_log)) or len(values_log) < 2:
            print("[Variogram] invalid values detected, using config defaults")
            return {
                "nugget": 1.0,
                "sill": 1.0,
                "range": 100.0 / 111.0
            }

        distances = []
        semivariances = []

        # --- Pairwise distances and semivariances ---
        for i in range(num_points):
            for j in range(i + 1, num_points):
                _, _, d_m = self.geod.inv(
                    self.lons[i], self.lats[i],
                    self.lons[j], self.lats[j]
                )
                distances.append(d_m / 1000.0)  # km
                semivariances.append(
                    0.5 * (values_log[i] - values_log[j]) ** 2
                )

        distances = np.asarray(distances)
        semivariances = np.asarray(semivariances)

        # --- Bin empirical variogram ---
        n_bins = self.variogram_bins
        max_dist = np.percentile(distances, 90)
        bins = np.linspace(0, max_dist, n_bins + 1)
        bin_ids = np.digitize(distances, bins)

        gamma = []
        bin_centers = []

        for k in range(1, len(bins)):
            mask = bin_ids == k
            if np.any(mask):
                gamma.append(np.mean(semivariances[mask]))
                bin_centers.append(0.5 * (bins[k] + bins[k - 1]))

        gamma = np.asarray(gamma)
        bin_centers = np.asarray(bin_centers)

        # ------------------------------------------------------------------
        # FIXED PARAMETERS
        # ------------------------------------------------------------------
        nugget = 1.0                 # fixed nugget
        range_km = 100.0             # fixed range (km)
        range_deg = range_km / 111.0 # convert to degrees

        # --- sill from gamma ---
        if len(gamma) > 0:
            sill = float(np.nanmax(gamma))
 
        else:
            sill = nugget + 1e-6

        # ensure sill >= nugget
        sill = max(sill, nugget + 1e-6)


        print(f"[Variogram] log-std={np.std(values_log):.3f}, sill={sill:.3f}")

        return {
            "nugget": nugget,
            "sill": sill,
            "range": range_deg
        }


    # ---------------------------------------------------------------------
    # Core computations
    # ---------------------------------------------------------------------
    def compute_kriging(self):
        kcfg = self.config.get("kriging", {}) or {}
        use_daily = kcfg.get("fit_daily_variogram", False)

        variogram_params = None

        if use_daily:
            try:
                vg = self.fit_daily_variogram()

                if vg["range"] <= 0 or not np.isfinite(vg["sill"]):
                    raise ValueError("Invalid daily variogram")

                variogram_params = {
                    "sill": vg["sill"],
                    "range": vg["range"],
                    "nugget": vg["nugget"],
                }

                print(
                    f"[{self.year}-{self.month:02d}-{self.day:02d}] "
                    f"Daily variogram | "
                    f"sill={vg['sill']:.3f}, "
                    f"range={vg['range']*111:.1f} km, "
                    f"nugget={vg['nugget']:.3f}"
                )

            except Exception as e:
                print(f"⚠️ Daily variogram failed, using config values: {e}")

        if variogram_params is None and kcfg.get("range"):
            variogram_params = {
                "sill": kcfg.get("sill", None),
                "range": float(kcfg["range"]) / 111.0,
                "nugget": kcfg.get("nugget", 0.0),
            }

        ok = OrdinaryKriging(
            self.lons,
            self.lats,
            self.values,
            variogram_model=self.variogram_model,
            exact_values=kcfg.get("exact_values", True),
            nlags=kcfg.get("nlags", 12),
            weight=kcfg.get("weight", True),
            variogram_parameters=variogram_params,
        )

        self.z_interp, self.kriging_variance = ok.execute(
            "grid", self.grid_lon, self.grid_lat
        )



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