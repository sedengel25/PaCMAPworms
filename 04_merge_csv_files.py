#!/usr/bin/env python3

import pandas as pd
import glob
import sys
from pathlib import Path

def main():
    if len(sys.argv) != 2:
        print("Usage: merge_results.py <RUN_ID>")
        sys.exit(1)

    run_id = sys.argv[1]
    base_path = Path(f"runs/{run_id}/results")
    files = sorted(base_path.glob("res_*.csv"))

    if not files:
        print(f"Keine Dateien gefunden in {base_path}.")
        return

    print(f"✔ {len(files)} Dateien gefunden. Starte Merge...")

    # Erste Datei inkl. Header laden
    df = pd.read_csv(files[0])

    # Restliche Dateien ohne Header anhängen
    for file in files[1:]:
        temp_df = pd.read_csv(file)
        df = pd.concat([df, temp_df], ignore_index=True)

    # Ergebnis speichern
    out_path = f"results_{run_id}.csv"
    df.to_csv(out_path, index=False)
    print(f"✔ {len(files)} Dateien zusammengeführt in '{out_path}'.")

if __name__ == "__main__":
    main()

