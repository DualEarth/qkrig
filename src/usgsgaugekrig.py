import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Geod
import scipy.spatial.distance as dist
from pykrige.ok import OrdinaryKriging
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import random
import dataretrieval.nwis as nwis

class USGSLoader:

    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.metadata_file = self.config["data"]["metadata_file"]
        self.site_list_file = self.config["data"].get("site_list_file", None)
        print(self.site_list_file)
        self.date_format = self.config["settings"]["date_format"]
        self.add_random_sites = self.config["settings"].get("add_random_sites", 0)
        self.gauge_metadata = self._load_gauge_metadata()

    def _load_config(self, config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _load_gauge_metadata(self):
        df = pd.read_csv(self.metadata_file, comment="#", dtype={"site_no": str})
        df = df.rename(columns={
            "site_no": "gauge_id",
            "dec_lat_va": "gauge_lat",
            "dec_long_va": "gauge_lon",
            "drain_area_va": "area_km2"
        })

        df = df[["gauge_id", "gauge_lat", "gauge_lon", "area_km2"]]
        df.dropna(inplace=True)

        if self.site_list_file and os.path.exists(self.site_list_file):
            with open(self.site_list_file, "r") as f:
                site_ids = [line.strip() for line in f if line.strip()]

            df = df[df["gauge_id"].isin(site_ids)]

        # Add random sites
        if self.add_random_sites > 0:
            all_metadata = pd.read_csv(self.metadata_file, comment="#", dtype={"site_no": str})
            all_metadata = all_metadata.rename(columns={
                "site_no": "gauge_id",
                "dec_lat_va": "gauge_lat",
                "dec_long_va": "gauge_lon",
                "drain_area_va": "area_km2"
            })
            all_metadata = all_metadata[["gauge_id", "gauge_lat", "gauge_lon", "area_km2"]].dropna()
            current_sites = set(df["gauge_id"])
            candidate_sites = all_metadata[~all_metadata["gauge_id"].isin(current_sites)]
            sample_size = min(self.add_random_sites, len(candidate_sites))
            random_sites = candidate_sites.sample(n=sample_size, random_state=42)
            df = pd.concat([df, random_sites])

        df.set_index("gauge_id", inplace=True)

        return df

    def get_streamflow(self, year, month, day):
        """Get (lon, lat, streamflow mm/day) for each gauge on a given date."""
        results = []
        date = pd.Timestamp(year=year, month=month, day=day)
        date_str = date.strftime("%Y-%m-%d")

        for gauge_id, row in self.gauge_metadata.iterrows():
            try:
                # Use dataretrieval to get daily discharge (parameterCd 00060 = discharge)
                df = nwis.get_record(
                    sites=gauge_id,
                    service="dv",
                    start=date_str,
                    end=date_str,
                    parameterCd="00060"
                )

                if not df.empty:
                    # Look for discharge value, column name ends with "00060_Mean"
                    discharge_col = [col for col in df.columns if "00060" in col and "Mean" in col]
                    if discharge_col:
                        streamflow_cfs = df.iloc[0][discharge_col[0]]

                        # Convert cfs → mm/day
                        area_km2 = row["area_km2"]
                        area_m2 = area_km2 * 1e6
                        streamflow_mm = (streamflow_cfs * 0.0283168 * 86400 / area_m2) * 1000

                        results.append((
                            row["gauge_lon"],
                            row["gauge_lat"],
                            streamflow_mm,
                            gauge_id
                        ))

            except Exception as e:
                print(f"Failed for gauge {gauge_id}: {e}")
                continue

        if results:
            dtype = [("lon", float), ("lat", float), ("streamflow", float), ("gauge_id", "U15")]
            results = np.array(results, dtype=dtype)
            streamflows = results["streamflow"]

            print(f"\nSummary for {date_str}")
            print(f"  - Observations: {len(streamflows)}")
            print(f"  - Min: {np.min(streamflows):.2f}, Max: {np.max(streamflows):.2f}, Mean: {np.mean(streamflows):.2f}")

            neg_vals = results[streamflows < 0]
            if len(neg_vals):
                print(f"  - ⚠️ Negative values found ({len(neg_vals)}):")
                for row in neg_vals:
                    print(f"    {row['gauge_id']} @ ({row['lat']:.4f}, {row['lon']:.4f}) = {row['streamflow']:.2f}")

            results = results[streamflows >= 0]

        else:
            print(f"No data for {date_str}")

        return results.tolist()

class USGSKrig:
    def __init__(self, data, config_path, year, month, day):
        """
        Initialize the kriging analysis with parameters from the configuration file.

        :param data: List of (lon, lat, streamflow) tuples.
        :param config_path: Path to the YAML configuration file.
        :param year: Year of the data.
        :param month: Month of the data.
        :param day: Day of the data.
        """
        self.data = np.array(data, dtype=float)
        self.lons = self.data[:, 0]
        self.lats = self.data[:, 1]
        self.values = self.data[:, 2]
        self.year = year
        self.month = month
        self.day = day
        self.geod = Geod(ellps="WGS84")

        # Load configuration settings
        self.config = self._load_config(config_path)
        self.grid_size = self.config["kriging"]["grid_size"]
        self.variogram_model = self.config["kriging"]["variogram_model"]
        self.variogram_bins = self.config["kriging"]["variogram_bins"]

        # Grid setup for interpolation
        lon_min, lon_max = np.min(self.lons), np.max(self.lons)
        lat_min, lat_max = np.min(self.lats), np.max(self.lats)
        self.grid_lon = np.linspace(lon_min, lon_max, self.grid_size)
        self.grid_lat = np.linspace(lat_min, lat_max, self.grid_size)
        self.grid_lon_mesh, self.grid_lat_mesh = np.meshgrid(self.grid_lon, self.grid_lat)

        # Initialize variables to store kriging results
        self.z_interp = None
        self.kriging_variance = None

    def _load_config(self, config_path):
        """Loads the configuration from the YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def compute_kriging(self):
        """
        Computes ordinary kriging interpolation and error variance.
        Stores the results in self.z_interp and self.kriging_variance.
        """
        print(f"Computing Kriging using {self.variogram_model} variogram model...")

        # Read sensitivity parameters from config
        sill = self.config["kriging"].get("sill", None)
        nugget = self.config["kriging"].get("nugget", 0.0)
        exact_values = self.config["kriging"].get("exact_values", True)
        nlags = self.config["kriging"].get("nlags", 12)
        weight = self.config["kriging"].get("weight", True)
        variogram_range_km = self.config["kriging"].get("range", None)

        # Convert range from km to degrees
        if variogram_range_km:
            variogram_range_deg = variogram_range_km / 111  # Approximate conversion (1° ≈ 111 km)
        else:
            variogram_range_deg = None  # Auto-fit if not provided

        # Define variogram parameters
        variogram_parameters = None
        if variogram_range_deg:
            variogram_parameters = {"sill": sill, "range": variogram_range_deg, "nugget": nugget}

        min_val, max_val = np.min(self.values), np.max(self.values)

        # Ordinary Kriging with adjusted range
        ok = OrdinaryKriging(
            self.lons, self.lats, self.values,
            variogram_model=self.variogram_model,
            exact_values=exact_values,
            nlags=nlags,
            weight=weight,
            variogram_parameters=variogram_parameters
        )

        self.z_interp, self.kriging_variance = ok.execute("grid", self.grid_lon, self.grid_lat)

    def plot_variogram(self):
        """
        Computes and plots the empirical variogram with distances in kilometers.
        The number of bins is set from the config file.
        """
        # Compute pairwise distances in meters using geodesic distance
        num_points = len(self.lons)
        distances = []
        differences = []

        for i in range(num_points):
            for j in range(i + 1, num_points):
                _, _, distance_m = self.geod.inv(self.lons[i], self.lats[i], self.lons[j], self.lats[j])
                distances.append(distance_m / 1000)  # Convert meters to kilometers
                differences.append((self.values[i] - self.values[j]) ** 2)

        distances = np.array(distances)
        differences = np.array(differences)

        # Bin distances and compute semi-variance
        bin_edges = np.linspace(0, np.max(distances), self.variogram_bins + 1)
        bin_indices = np.digitize(distances, bin_edges) - 1
        semi_variance = [differences[bin_indices == i].mean() for i in range(self.variogram_bins)]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Format the date
        date_str = f"{self.year}-{self.month:02d}-{self.day:02d}"

        # Plot variogram
        plt.figure(figsize=(8, 5))
        plt.scatter(bin_centers, semi_variance, c="blue", label="Empirical Variogram")
        plt.xlabel("Distance (km)")
        plt.ylabel("Semi-variance")
        plt.title(f"Empirical Variogram ({self.variogram_model} model) - {date_str}")
        plt.legend()
        plt.show()

    def map_krig_interpolation(self):
        """
        Plots the kriging interpolated values while ensuring the interpolated values 
        are bounded by the observed data range.
        """
        if self.z_interp is None:
            raise RuntimeError("You must run compute_kriging() first!")

        # Get min/max bounds from observed values
        min_val, max_val = np.min(self.values), np.max(self.values)

        # Clip the interpolated values to be within the observed range
        self.z_interp = np.clip(self.z_interp, min_val, max_val)

        # Plot bounded kriging interpolation
        plt.figure(figsize=(8, 6))
        plt.contourf(self.grid_lon, self.grid_lat, self.z_interp, cmap="coolwarm", levels=15)
        plt.colorbar(label="Interpolated Streamflow")
        plt.scatter(self.lons, self.lats, c=self.values, edgecolors=None, label="Observed Data", cmap="coolwarm", s=1)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"Kriging Interpolation (Bounded, {self.variogram_model} model)")
        plt.legend()
        plt.show()

    def map_krig_error_variance(self):
        """
        Plots the kriging error variance.
        """
        if self.kriging_variance is None:
            raise RuntimeError("You must run compute_kriging() first!")

        plt.figure(figsize=(8, 6))
        plt.contourf(self.grid_lon, self.grid_lat, self.kriging_variance, cmap="viridis", levels=15)
        plt.colorbar(label="Kriging Error Variance")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Kriging Error Variance Map")
        plt.show()