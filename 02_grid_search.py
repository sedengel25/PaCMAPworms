#!/usr/bin/env python3

import os
import glob
import pandas as pd

# 1. Alle Files mit "highD_orig" im data-Verzeichnis finden
embedded_files = glob.glob("data/*highD_orig*")

# 2. Parameterräume definieren
dimred_methods      = ["tSNE", "UMAP", "TriMap", "PaCMAP"]
# Für tSNE nur 2 und 3, sonst 2–10
target_dims_all     = list(range(2, 11))   # 2 bis 10
target_dims_tsne    = [2, 3]
min_cluster_sizes   = [10, 50, 100, 200, 500, 1000]
pacmap_n_neighbors  = [1, 2, 5, 10, 20, 40]

# 3. Grid generieren
configs = []
for file in embedded_files:
    for method in dimred_methods:
        # je nach Methode die target_dims wählen
        if method == "tSNE":
            target_dims = target_dims_tsne
        else:
            target_dims = target_dims_all

        for target_dim in target_dims:
            for mcs in min_cluster_sizes:
                # drei min_samples-Werte: 75%, 100%, 125% von mcs
                samples_list = [round(mcs * 0.75), mcs, round(mcs * 1.25)]
                if method == "PaCMAP":
                    for nn in pacmap_n_neighbors:
                        for ms in samples_list:
                            configs.append({
                                "file": file,
                                "dimred_method":    method,
                                "target_dim":       target_dim,
                                "min_cluster_size": mcs,
                                "min_samples":      ms,
                                "n_neighbors":      nn
                            })
                else:
                    for ms in samples_list:
                        configs.append({
                            "file": file,
                            "dimred_method":    method,
                            "target_dim":       target_dim,
                            "min_cluster_size": mcs,
                            "min_samples":      ms,
                            "n_neighbors":      0
                        })

# 4. DataFrame und CSV schreiben
df = pd.DataFrame(configs)
out_csv = "grid_search.csv"
df.to_csv(out_csv, index=False)
print(f"Wrote configuration grid to {out_csv} ({len(df)} rows).")
