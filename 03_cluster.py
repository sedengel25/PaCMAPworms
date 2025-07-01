#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
from sklearn.manifold import TSNE
import umap
import trimap
import pacmap
import hdbscan
from sklearn.metrics import adjusted_rand_score
from hdbscan import validity
import argparse
import sys
import re

# 🔹 Fester, testfreundlicher Default für lokale Läufe
RUN_PATH_DEFAULT = "runs/run_20250701_1454"

def remove_last_postfix_before_extension(filename):
    """
    Entfernt den letzten _postfix vor der Dateiendung (.txt),
    gibt base_clean OHNE .txt und ext (".txt") zurück.
    """
    base, ext = os.path.splitext(filename)
    parts = base.split('_')
    if len(parts) > 1:
        base_clean = '_'.join(parts[:-1])
    else:
        base_clean = base
    return base_clean, ext

def process_config(row, run_path):
    filename_rel = row['file']  # enthält z.B. "04_0nm_03run_10d.txt"
    base_clean, ext = remove_last_postfix_before_extension(filename_rel)
    dir_path = os.path.join(run_path, "data", "highmapped_xd")
    filename = os.path.join(dir_path, filename_rel)

    print(filename_rel)
    print(filename)
    print(base_clean)
    print(ext)

    # Noise Mult extrahieren
    m = re.search(r'_(\d+)nm', filename_rel)
    nm = int(m.group(1)) if m else -1

    # Daten laden
    X = np.loadtxt(filename).astype(np.float64)
    n_points = X.shape[0]

    # Labels laden
    labels_file = os.path.join(run_path, "data", "true_labels", f"{base_clean}_labels{ext}")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"❌ Labels file not found: {labels_file}")
    y_true = np.loadtxt(labels_file).astype(int)

    # HDBSCAN auf highD
    mcs = int(row['min_cluster_size'])
    ms = int(row['min_samples'])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms)
    labels_pred_mapped = clusterer.fit_predict(X)

    ari_mapped = adjusted_rand_score(y_true, labels_pred_mapped)
    dbcv_mapped = validity.validity_index(X, labels_pred_mapped)
    n_noise_m = int((labels_pred_mapped == -1).sum())
    n_clusters_m = len(set(labels_pred_mapped)) - (1 if -1 in labels_pred_mapped else 0)

    pred_mapped_file = os.path.join(dir_path, f"{base_clean}_pred_labels{ext}")
    np.savetxt(pred_mapped_file, labels_pred_mapped, fmt='%d')

    # DimRed
    method = row['dimred_method']
    target_dim = int(row['target_dim'])
    nn = int(row['n_neighbors'])

    if method == 'tSNE':
        dr = TSNE(n_components=target_dim, init='random', random_state=42,
                  method='exact' if target_dim > 3 else 'barnes_hut')
    elif method == 'UMAP':
        dr = umap.UMAP(n_components=target_dim, random_state=42)
    elif method == 'TriMap':
        dr = trimap.TRIMAP(n_dims=target_dim)
    elif method == 'PaCMAP':
        dr = pacmap.PaCMAP(n_components=target_dim, n_neighbors=nn, random_state=42)
    else:
        raise ValueError(f"Unknown dimred method: {method}")

    X_red = dr.fit_transform(X).astype(np.float64)
    embedded_file = os.path.join(dir_path, f"{base_clean}_{target_dim}d_emb{ext}")
    np.savetxt(embedded_file, X_red, fmt='%.6f')

    # HDBSCAN auf Embedding
    clusterer_red = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms)
    labels_pred_red = clusterer_red.fit_predict(X_red)

    ari_red = adjusted_rand_score(y_true, labels_pred_red)
    dbcv_red_m = validity.validity_index(X, labels_pred_red)
    dbcv_red_e = validity.validity_index(X_red, labels_pred_red)
    n_noise_r = int((labels_pred_red == -1).sum())
    n_clusters_r = len(set(labels_pred_red)) - (1 if -1 in labels_pred_red else 0)

    pred_red_file = os.path.join(dir_path, f"{base_clean}_{target_dim}d_emb_pred_labels{ext}")
    np.savetxt(pred_red_file, labels_pred_red, fmt='%d')

    # Ergebnisse zurückgeben
    return {
        **row.to_dict(),
        'noise_mult': nm,
        'n_points': n_points,
        'n_clusters_orig': n_clusters_m,
        'n_noise_orig': n_noise_m,
        'ARI_orig': ari_mapped,
        'DBCV_orig': dbcv_mapped,
        'n_clusters_embedded': n_clusters_r,
        'n_noise_embedded': n_noise_r,
        'ARI_embedded': ari_red,
        'DBCV_embedded_m': dbcv_red_m,
        'DBCV_embedded_e': dbcv_red_e,
    }

def main(run_path, out_prefix, idx):
    run_id = os.path.basename(run_path)
    grid_csv = os.path.join(run_path, f"grid_search_{run_id}.csv")

    df = pd.read_csv(grid_csv)
    total = len(df)
    if idx < 0 or idx >= total:
        raise IndexError(f"Index {idx} außerhalb der Grid-Länge {total}")
    width = len(str(total - 1))
    row = df.iloc[idx]

    res = process_config(row, run_path)
    print(res)
    out_df = pd.DataFrame([res])

    results_dir = os.path.join(run_path, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, f"{out_prefix}_{idx:0{width}d}.csv")
    out_df.to_csv(out_file, index=False)
    print(f"✅ Wrote partial results to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DR+HDBSCAN on one grid row within a run folder.")
    parser.add_argument('--run_path', type=str, default=RUN_PATH_DEFAULT,
                        help="Pfad zum Run-Ordner, z.B. runs/run_20250701_1454")
    parser.add_argument('--out', type=str, default='res',
                        help="Prefix für Ergebnisdateien")
    parser.add_argument('--index', type=int, default=1,
                        help="Zeile der Grid CSV (falls None wird SLURM_ARRAY_TASK_ID verwendet)")
    args = parser.parse_args()

    idx = args.index if args.index is not None else int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    main(args.run_path, args.out, idx)
    sys.exit(0)
