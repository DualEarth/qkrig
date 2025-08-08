# ./src/camelskrig.py

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

class CamelsLoader:

    def __init__(self, config_path):
        """
        Initialize the loader using a configuration file.
        
        :param config_path: Path to the YAML configuration file.
        """
        self.config = self._load_config(config_path)
        self.metadata_file = self.config["data"]["metadata_file"]
        self.data_dir = self.config["data"]["data_dir"]
        self.date_format = self.config["settings"]["date_format"]
        self.land_mask = self.config["data"]["land_mask"]
        self.gauge_metadata = self._load_gauge_metadata()

    def _load_config(self, config_path):
        """Loads the configuration from the YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _load_gauge_metadata(self):
        """Loads the gauge metadata from the configured file."""
        df = pd.read_csv(self.metadata_file, sep=";", dtype={"gauge_id": str})
        df.set_index("gauge_id", inplace=True)  # Index by gauge ID for easy lookup
        return df

    def _find_gauge_file(self, gauge_id):
        """Finds the file containing the streamflow data for the given gauge_id."""
        for subdir in os.listdir(self.data_dir):
            subdir_path = os.path.join(self.data_dir, subdir)
            if os.path.isdir(subdir_path):
                file_path = os.path.join(subdir_path, f"{gauge_id}_streamflow_qc.txt")
                if os.path.exists(file_path):
                    return file_path
        return None  # File not found

    def get_streamflow(self, year, month, day):
        """
        Retrieves (longitude, latitude, normalized streamflow) for all gauges on a specified date.
        
        :param year: Year (int)
        :param month: Month (int)
        :param day: Day (int)
        :return: List of tuples [(lon, lat, streamflow in mm/day), ...]
        """
        results = []
        target_date = f"{year} {month:02d} {day:02d}"  # Match format in streamflow file

        for gauge_id, row in self.gauge_metadata.iterrows():
            file_path = self._find_gauge_file(gauge_id)
            if not file_path:
                continue  # Skip if no file found

            with open(file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    if " ".join(parts[1:4]) == target_date:
                        lon, lat = row["gauge_lon"], row["gauge_lat"]
                        streamflow_cfs = float(parts[4])

                        # Get basin area (use area_geospa_fabric instead of area_gages2)
                        area_m2 = row["area_geospa_fabric"] * 1e6  # Convert km² to m²
                        if area_m2 > 0:  # Avoid division by zero
                            # Use correct conversion formula
                            streamflow_mm = (streamflow_cfs * 0.0283168 * 86400 / area_m2) * 1000
                            results.append((lon, lat, streamflow_mm, gauge_id))

                        break  # Move to the next gauge once data is found

        # Convert results to a structured NumPy array
        if results:
            dtype = [("lon", float), ("lat", float), ("streamflow", float), ("gauge_id", "U10")]
            results = np.array(results, dtype=dtype)

            streamflows = results["streamflow"]  # Extract streamflow values

            print(f"Summary Statistics for {year}-{month:02d}-{day:02d}:")
            print(f"  - Number of observations: {len(streamflows)}")
            print(f"  - Min streamflow: {np.min(streamflows):.2f} mm/day")
            print(f"  - Max streamflow: {np.max(streamflows):.2f} mm/day")
            print(f"  - Mean streamflow: {np.mean(streamflows):.2f} mm/day")
            print(f"  - Std deviation: {np.std(streamflows):.2f} mm/day")
            
            # Identify and print negative values
            negative_indices = streamflows < 0
            num_negatives = np.sum(negative_indices)

            if num_negatives > 0:
                print(f"  - WARNING: {num_negatives} negative values detected!")
                print("  - Locations of negative values (gauge_id, lon, lat, streamflow):")
                for row in results[negative_indices]:
                    print(f"    - Gauge {row['gauge_id']}: ({row['lon']:.4f}, {row['lat']:.4f}) -> {row['streamflow']:.2f} mm/day")

            # Remove negative values
            results = results[~negative_indices]

        else:
            print(f"No streamflow data found for {year}-{month:02d}-{day:02d}.")

        return results.tolist()  # Convert back to a list of tuples

class CamelsKrig:
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

        land_mask_path = "."
        land_mask = np.load(land_mask_path)  # shape: (H, W)

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