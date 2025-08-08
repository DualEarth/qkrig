# visualizations.py

import matplotlib.pyplot as plt
import numpy as np
import yaml
import os

class PlotConfig:
    def __init__(self, config_path="/Users/jmframe/qkrig/configs/plot_config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

    def __getitem__(self, item):
        return self.cfg.get(item, {})


class VariogramPlotter:
    def __init__(self, krig_obj):
        self.krig = krig_obj
        self.config = PlotConfig()["variogram"]

    def plot(self):
        distances, differences = [], []
        num_points = len(self.krig.lons)

        for i in range(num_points):
            for j in range(i + 1, num_points):
                _, _, d = self.krig.geod.inv(
                    self.krig.lons[i], self.krig.lats[i],
                    self.krig.lons[j], self.krig.lats[j]
                )
                distances.append(d / 1000)
                differences.append((self.krig.values[i] - self.krig.values[j]) ** 2)

        distances = np.array(distances)
        differences = np.array(differences)

        bins = self.config.get("bins", self.krig.variogram_bins)
        bin_edges = np.linspace(0, np.max(distances), bins + 1)
        bin_indices = np.digitize(distances, bin_edges) - 1
        semi_variance = [differences[bin_indices == i].mean() for i in range(bins)]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        plt.figure(figsize=self.config.get("figure_size", [8, 5]))
        plt.scatter(bin_centers, semi_variance,
                    c=self.config.get("color", "blue"),
                    label=self.config.get("label", "Empirical Variogram"))
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
        self.config_interp = PlotConfig()["kriging_interpolation"]
        self.config_error = PlotConfig()["kriging_error"]

    def plot_interpolation(self):
        if self.krig.z_interp is None:
            raise RuntimeError("compute_kriging() must be run before plotting interpolation.")

        z = np.clip(self.krig.z_interp, np.min(self.krig.values), np.max(self.krig.values))

        plt.figure(figsize=self.config_interp.get("figure_size", [8, 6]))
        plt.contourf(self.krig.grid_lon, self.krig.grid_lat, z,
                     levels=self.config_interp.get("levels", 15),
                     cmap=self.config_interp.get("cmap", "coolwarm"))
        plt.colorbar(label=self.config_interp.get("colorbar_label", "Interpolated Streamflow"))

        scatter_cfg = self.config_interp.get("scatter", {})
        plt.scatter(self.krig.lons, self.krig.lats, c=self.krig.values,
                    s=scatter_cfg.get("s", 1),
                    cmap=scatter_cfg.get("cmap", "coolwarm"),
                    edgecolors=scatter_cfg.get("edgecolors", "none"),
                    label=scatter_cfg.get("label", "Observed Data"))

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

        plt.figure(figsize=self.config_error.get("figure_size", [8, 6]))
        plt.contourf(self.krig.grid_lon, self.krig.grid_lat, self.krig.kriging_variance,
                     levels=self.config_error.get("levels", 15),
                     cmap=self.config_error.get("cmap", "viridis"))
        plt.colorbar(label=self.config_error.get("colorbar_label", "Kriging Error Variance"))
        plt.xlabel(self.config_error.get("xlabel", "Longitude"))
        plt.ylabel(self.config_error.get("ylabel", "Latitude"))
        plt.title(self.config_error.get("title", "Kriging Error Variance Map"))
        plt.show()
