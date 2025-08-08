# camels_krig.py

import numpy as np
from core.base_krig import BaseKrig
from vis.visualizations import VariogramPlotter, KrigingMapPlotter

class CamelsKrig(BaseKrig):
    def __init__(self, data, config_path, year, month, day):
        super().__init__(data, config_path, year, month, day)

        # Initialize plotting utilities using visualization module
        self.variogram_plotter = VariogramPlotter(self)
        self.krig_map_plotter = KrigingMapPlotter(self)

    def plot_variogram(self):
        self.variogram_plotter.plot()

    def map_krig_interpolation(self):
        self.krig_map_plotter.plot_interpolation()

    def map_krig_error_variance(self):
        self.krig_map_plotter.plot_error_variance()
