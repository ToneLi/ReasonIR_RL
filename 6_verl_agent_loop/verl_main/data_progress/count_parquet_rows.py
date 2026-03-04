import pandas as pd
import os
import glob

def count_rows(path):
    """Count rows in a parquet file or all parquet files in a directory."""
    if os.path.isfile(path):
        df = pd.read_parquet(path)
        print(f"{path}: {len(df)} rows")
    elif os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.parquet")))
        if not files:
            print(f"No parquet files found in {path}")
            return
        for f in files:
            try:
                df = pd.read_parquet(f)
                print(f"{os.path.basename(f)}: {len(df)} rows")
            except Exception as e:
                print(f"{os.path.basename(f)}: ERROR - {e}")
    else:
        print(f"Path not found: {path}")

if __name__ == "__main__":
    import sys
    base="/data/experiment_data_gamma/mingchen/7_verl_agent_loop/ReasonIR_RL/6_verl_agent_loop/verl_main/data_progress/bright/part_48_49_50_rounds_dev.parquet"
    count_rows(base)
  