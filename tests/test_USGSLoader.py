import sys
from pprint import pprint
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "src")))
from usgsgaugekrig import USGSLoader

def main():
    # Path to your config YAML file
    config_path = "../configs/usgsgaugekrig.yaml"

    # Date to test
    year, month, day = 2022, 7, 15

    # Instantiate the loader
    print("Initializing USGSLoader...")
    loader = USGSLoader(config_path)

    # Fetch streamflow data
    print(f"\nFetching streamflow for {year}-{month:02d}-{day:02d}...")
    results = loader.get_streamflow(year, month, day)

    # Show output
    if results:
        print(f"\n Retrieved {len(results)} records.")
        print("\nFirst 5 results (lon, lat, streamflow_mm/day, gauge_id):")
        pprint(results[:5])
    else:
        print("\n No valid streamflow data returned.")

if __name__ == "__main__":
    main()
