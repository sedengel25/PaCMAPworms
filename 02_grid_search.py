#!/usr/bin/env python3

import os
import glob
import pandas as pd

RUN_PATH = "runs/run_20250701_1454"   
RUN_ID = os.path.basename(RUN_PATH)

embedded_files = glob.glob(os.path.join(RUN_PATH, "data/highmapped_xd", "*"))



dimred_methods      = ["tSNE", "UMAP", "TriMap", "PaCMAP"]
target_dims_all     = list(range(2, 11))   # 2 bis 10
target_dims_tsne    = [2, 3]
min_cluster_sizes   = [10, 50, 100, 200, 500, 1000]
pacmap_n_neighbors  = [1, 2, 5, 10, 20, 40]


configs = []
for file in embedded_files:
    file = os.path.basename(file)
    for method in dimred_methods:
        if method == "tSNE":
            target_dims = target_dims_tsne
        else:
            target_dims = target_dims_all

        for target_dim in target_dims:
            for mcs in min_cluster_sizes:
                samples_list = [round(mcs * 0.5), round(mcs * 0.75), mcs]
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
out_csv = os.path.join(RUN_PATH, f"grid_search_{RUN_ID}.csv")
df.to_csv(out_csv, index=False)
print(f"✅ Wrote configuration grid to {out_csv} ({len(df)} rows).")
