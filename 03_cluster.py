#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
from sklearn.manifold import TSNE
import umap
import trimap      # liefert TRIMAP
import pacmap
import hdbscan
from sklearn.metrics import adjusted_rand_score
from hdbscan import validity
import argparse
import sys
import glob, re

def process_config(row):
    # ─── 1) Daten laden ─────────────────────────────────────────────────────────
    filename = row['file']
    m = re.search(r'_(\d+)nm', filename)
    nm = int(m.group(1))
    X = np.loadtxt(filename)
    X = X.astype(np.float64)
    n_points = X.shape[0]
    labels_file = row['file'].replace('_mapped_', '_labels_')
    y_true = np.loadtxt(labels_file).astype(int)

    # ─── 2) HDBSCAN auf die gemappten (2D/10D/…) Daten ──────────────────────────
    mcs = int(row['min_cluster_size'])
    ms  = int(row['min_samples'])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms)
    
    labels_pred_mapped = clusterer.fit_predict(X)
    ari_mapped  = adjusted_rand_score(y_true, labels_pred_mapped)
    dbcv_mapped = validity.validity_index(X, labels_pred_mapped)
    n_noise_m   = int((labels_pred_mapped == -1).sum())
    n_clusters_m = len(set(labels_pred_mapped)) - (1 if -1 in labels_pred_mapped else 0)

    # speichere die gemappten Prädiktionen
    pred_mapped_file = row['file'].replace('_mapped_', '_labels_pred_mapped_')
    np.savetxt(pred_mapped_file, labels_pred_mapped, fmt='%d')

    # ─── 3) DimRed auswählen und anwenden ───────────────────────────────────────
    method     = row['dimred_method']
    target_dim = int(row['target_dim'])
    nn = int(row['n_neighbors'])
    if method == 'tSNE':
        dr = TSNE(n_components=target_dim, init='random', random_state=42)
        X_red = dr.fit_transform(X)
    elif method == 'UMAP':
        dr = umap.UMAP(n_components=target_dim, random_state=42)
        X_red = dr.fit_transform(X)
    elif method == 'TriMap':
        dr = trimap.TRIMAP(n_dims=target_dim)
        X_red = dr.fit_transform(X)
    elif method == 'PaCMAP':
        dr = pacmap.PaCMAP(n_components=target_dim, n_neighbors=nn, random_state=42)
        X_red = dr.fit_transform(X)
    else:
        raise ValueError(f"Unknown dimred method: {method}")

    X_red = X_red.astype(np.float64)
    # speichere das embedding mit target_dim im Dateinamen
    embedded_file = row['file'].replace('_mapped_', f'_embedded_{target_dim}d_')
    np.savetxt(embedded_file, X_red, fmt='%.6f')

    # ─── 4) HDBSCAN auf das Embedded ────────────────────────────────────────────
    clusterer_red = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms)
    labels_pred_red = clusterer_red.fit_predict(X_red)
    ari_red   = adjusted_rand_score(y_true, labels_pred_red)
    dbcv_red_m  = validity.validity_index(X, labels_pred_red)
    dbcv_red_e  = validity.validity_index(X_red, labels_pred_red)
    n_noise_r  = int((labels_pred_red == -1).sum())
    n_clusters_r = len(set(labels_pred_red)) - (1 if -1 in labels_pred_red else 0)

    # speichere die embedded Prädiktionen
    pred_red_file = row['file'].replace('_mapped_', '_labels_pred_embedded_')
    np.savetxt(pred_red_file, labels_pred_red, fmt='%d')

    # ─── 5) Ergebnisse in Dict packen ──────────────────────────────────────────
    return {
        **row.to_dict(),
        'file': filename,
        'noise_mult': nm,
        'DR': method,
        'target_dim':target_dim,
        'min_cluster_size':mcs,
        'min_samples':ms,
        'n_neighbors': nn,
        'n_points': n_points,
        'n_clusters_orig':  n_clusters_m,
        'n_noise_orig':     n_noise_m,
        'ARI_orig':         ari_mapped,
        'DBCV_orig':        dbcv_mapped,
        'n_clusters_embedded': n_clusters_r,
        'n_noise_embedded':    n_noise_r,
        'ARI_embedded':        ari_red,
        'DBCV_embedded_m':       dbcv_red_m,
        'DBCV_embedded_e':       dbcv_red_e,
    }

def main(grid_csv, out_prefix, idx):
    # 1. Grid einlesen
    df = pd.read_csv(grid_csv)
    total = len(df)
    if idx < 0 or idx >= total:
        raise IndexError(f"Index {idx} außerhalb der Grid-Länge {total}")
    # Bestimme Breite für Null-Padding basierend auf Anzahl der Konfigurationen
    width = len(str(total - 1))
    row = df.iloc[idx]

    # 2. Eine Konfiguration verarbeiten
    res = process_config(row)
    print(res)
    out_df = pd.DataFrame([res])

    # 3. Job-spezifische CSV schreiben
    out_file = f"{out_prefix}_{idx:0{width}d}.csv"
    out_df.to_csv(out_file, index=False)
    print(f"Wrote partial results to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one DR+HDBSCAN job per grid row")
    parser.add_argument('--grid',   type=str, default='grid_search.csv',
                        help="Input CSV mit allen Konfigurationen")
    parser.add_argument('--out',    type=str, default='result',
                        help="Prefix für die Ausgabe-CSV (erhält _00001.csv etc.)")
    parser.add_argument('--index',  type=int, default=1000,
                        help="Welche Zeile aus dem Grid zu verarbeiten ist")
    args = parser.parse_args()

    # 4. Index wählen: CLI-Argument oder SLURM_ARRAY_TASK_ID
    if args.index is None:
        args.index = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))  # Fallback 0
    main(args.grid, args.out, args.index)
    sys.exit(0)