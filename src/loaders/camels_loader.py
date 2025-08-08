# camels_loader.py

import os
import pandas as pd
import numpy as np
from datetime import datetime
from core.base_loader import BaseLoader

class CamelsLoader(BaseLoader):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.data_dir = self.config["data"]["data_dir"]

    def _load_gauge_metadata(self):
        df = pd.read_csv(self.config["data"]["metadata_file"], sep=";", dtype={"gauge_id": str})
        df.set_index("gauge_id", inplace=True)
        return df

    def _find_gauge_file(self, gauge_id):
        for subdir in os.listdir(self.data_dir):
            subdir_path = os.path.join(self.data_dir, subdir)
            if os.path.isdir(subdir_path):
                file_path = os.path.join(subdir_path, f"{gauge_id}_streamflow_qc.txt")
                if os.path.exists(file_path):
                    return file_path
        return None

    def get_streamflow(self, year, month, day):
        results = []
        target_date = f"{year} {month:02d} {day:02d}"

        for gauge_id, row in self.gauge_metadata.iterrows():
            file_path = self._find_gauge_file(gauge_id)
            if not file_path:
                continue

            with open(file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    if " ".join(parts[1:4]) == target_date:
                        streamflow_cfs = float(parts[4])
                        area_m2 = row["area_geospa_fabric"] * 1e6
                        if area_m2 > 0:
                            streamflow_mm = (streamflow_cfs * 0.0283168 * 86400 / area_m2) * 1000
                            results.append((row["gauge_lon"], row["gauge_lat"], streamflow_mm, gauge_id))
                        break

        if results:
            dtype = [("lon", float), ("lat", float), ("streamflow", float), ("gauge_id", "U10")]
            results = np.array(results, dtype=dtype)
            streamflows = results["streamflow"]

            print(f"Summary Statistics for {year}-{month:02d}-{day:02d}:")
            print(f"  - Observations: {len(streamflows)}")
            print(f"  - Min: {np.min(streamflows):.2f} mm/day")
            print(f"  - Max: {np.max(streamflows):.2f} mm/day")
            print(f"  - Mean: {np.mean(streamflows):.2f} mm/day")
            print(f"  - Std: {np.std(streamflows):.2f} mm/day")

            negatives = streamflows < 0
            if np.any(negatives):
                print(f"  - ⚠️ {np.sum(negatives)} negative values detected")
                for row in results[negatives]:
                    print(f"    - {row['gauge_id']}: ({row['lat']:.4f}, {row['lon']:.4f}) = {row['streamflow']:.2f}")
                results = results[~negatives]
        else:
            print(f"No streamflow data found for {year}-{month:02d}-{day:02d}.")

        return results.tolist()
