import os
import glob
import re
import argparse
import pandas as pd

# Argumente parsen
parser = argparse.ArgumentParser(description="Generate grid_search CSV for a specific run folder.")
parser.add_argument("--run_id", type=str, default="20250701_1549",
                    help="ID des Runs, z.B. 20250701_1454")
args = parser.parse_args()

RUN_ID = args.run_id
RUN_PATH = os.path.join("runs", RUN_ID)

# Highmapped XD Dateien finden
embedded_files = glob.glob(os.path.join(RUN_PATH, "data/input/highmapped_xd", "*"))
embedded_files = [os.path.basename(f) for f in embedded_files]

# Parameterraum
dimred_methods      = ["tSNE", "UMAP", "TriMap", "PaCMAP"]
min_cluster_sizes   = [30]
pacmap_n_neighbors  = [10]
nn = 10
configs = []

for file in embedded_files:
    # XD extrahieren
    m = re.search(r'_(\d+)d\.txt$', file)
    if not m:
        print(f"Warning: could not extract XD from {file}, skipping.")
        continue
    xd = int(m.group(1))

    # dynamische target_dims: [2, 25%, 50%, 75%, 100%] von xd
    raw = [2,
           int(round(0.25 * xd)),
           int(round(0.5  * xd)),
           int(round(0.75 * xd)),
           xd]
    target_dims = sorted(set([d for d in raw if d >= 2]))
    target_dim = 2
    print(target_dims)
    ms = 30
    target_repeats = 5
    for method in dimred_methods:
        # tSNE nur für 2D und 3D
        #if method == "tSNE":
        #    method_dims = [d for d in target_dims if d in (2, 3)]
        #else:
        #    method_dims = target_dims
        #method_dims = target_dims
        #for target_dim in method_dims:
            for mcs in min_cluster_sizes:
                for rep in range(1, target_repeats + 1):
                  #samples = [round(mcs * 0.5), round(mcs * 0.75), mcs]
                  if method == "PaCMAP":
                      #for nn in pacmap_n_neighbors:
                          #for ms in samples:
                    configs.append({
                        "file": file,
                        "dimred_method": method,
                        "input_dim": xd,
                        "target_dim": target_dim,
                        "min_cluster_size": mcs,
                        "min_samples": ms,
                        "n_neighbors": nn,
                        "rep": rep
                    })
                  else:
                      #for ms in samples:
                    configs.append({
                        "file": file,
                        "dimred_method": method,
                        "input_dim": xd,
                        "target_dim": target_dim,
                        "min_cluster_size": mcs,
                        "min_samples": ms,
                        "n_neighbors": nn,
                        "rep": rep
                    })
# Als CSV speichern
df = pd.DataFrame(configs)
out_csv = os.path.join(RUN_PATH, f"grid_search_{RUN_ID}.csv")
print(len(df))
df.to_csv(out_csv, index=False)
