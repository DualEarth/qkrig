# ./src/simulationkrig.py

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

class SimulationLoader:
    def __init__(self, config_path):
        """
        Initialize the loader for simulation data from a given CSV file.
        """
        self.config = self._load_config(config_path)
        self.metadata_file = self.config["data"]["metadata_file"]
        self.data_dir = self.config["data"]["data_dir"]
        self.model_file = os.path.join(self.data_dir, self.config["data"]["model_file"])
        self.date_format = self.config["settings"]["date_format"]
        self.model_file = os.path.join(self.data_dir, self.model_file)
        self.gauge_metadata = self._load_gauge_metadata()
        self.sim_df = pd.read_csv(self.model_file, index_col="date", parse_dates=True)

    def _load_config(self, config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _load_gauge_metadata(self):
        df = pd.read_csv(self.metadata_file, sep=";", dtype={"gauge_id": str})
        df.set_index("gauge_id", inplace=True)
        return df

    def get_streamflow(self, year, month, day):
        """
        Retrieves (longitude, latitude, streamflow in mm/day, gauge_id) for all gauges on a specified date.
        """
        results = []
        target_date = pd.Timestamp(year=year, month=month, day=day)

        if target_date not in self.sim_df.index:
            print(f"⚠️ No data for {target_date.date()} in {os.path.basename(self.model_file)}.")
            return []

        row = self.sim_df.loc[target_date]

        for gauge_id, flow in row.items():
            if pd.notna(flow) and gauge_id in self.gauge_metadata.index:
                meta = self.gauge_metadata.loc[gauge_id]
                lon = meta["gauge_lon"]
                lat = meta["gauge_lat"]
                results.append((lon, lat, flow, gauge_id))

        if results:
            results = np.array(results, dtype=[("lon", float), ("lat", float), ("streamflow", float), ("gauge_id", "U10")])
            streamflows = results["streamflow"]

            print(f"Summary Statistics for {target_date.date()}:")
            print(f"  - Number of simulations: {len(streamflows)}")
            print(f"  - Min: {np.min(streamflows):.2f} mm/day")
            print(f"  - Max: {np.max(streamflows):.2f} mm/day")
            print(f"  - Mean: {np.mean(streamflows):.2f} mm/day")
            print(f"  - Std Dev: {np.std(streamflows):.2f} mm/day")

            # Check for negative values
            negative_mask = streamflows < 0
            if np.any(negative_mask):
                print(f"  - WARNING: {np.sum(negative_mask)} negative values found:")
                for row in results[negative_mask]:
                    print(f"    - {row['gauge_id']}: ({row['lat']:.4f}, {row['lon']:.4f}) → {row['streamflow']:.2f}")

            # Remove negative values
            results = results[~negative_mask]

        else:
            print(f"⚠️ No valid streamflow data on {target_date.date()}.")

        return results.tolist()

class SimulationKrig:
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