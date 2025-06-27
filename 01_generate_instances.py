#!/usr/bin/env python3

import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import sys
import os
def gen_one_worm(c, var_range, numsteps, steepness, nump, num_noisep, stepl, c_trail, tree, cluster_id, noise_mult, n_collision):
    dims = c.shape[0]
    new_cs = []
    Xnew = []
    labels_new = []

    # initial random direction
    num_rdirs = 3
    rdir = np.zeros((num_rdirs, dims))
    for j in range(num_rdirs):
      x = np.random.rand(dims) - 0.5
      x /= np.linalg.norm(x)
      rdir[j, :] = x
    v = rdir[0, :]

    # Compute null space of v to get orthogonal vectors
    orthos = null_space(v.reshape(1, -1))  # Shape: (dims, dims-1)
    
    # Random unit vector orthogonal to v
    v3 = orthos @ np.random.rand(dims - 1)
    v3 /= np.linalg.norm(v3)
    
    for i_step in range(1, numsteps + 1):
        p = i_step / numsteps

        # update null space each step (v changes)
        orthos = null_space(v.reshape(1, -1))
        v3 = orthos @ np.random.rand(dims - 1)
        v3 /= np.linalg.norm(v3)
    
        v2 = v3
    
        v = v + v2 * (steepness / numsteps)
        v /= np.linalg.norm(v)
        
        # interpolate variance
        varr = var_range[0] * (1 - p) + var_range[1] * p
        c = c + v * stepl
    
        # collision check
        if tree is not None:
            dist, _ = tree.query(c.reshape(1, -1), k=1)
            if dist[0][0] < 50:
                n_collision += 1
                break

        new_cs.append(c.copy())

        # 1️⃣ Dichte Punkte: label = cluster_id
        dp = np.random.normal(loc=c, scale=np.sqrt(varr), size=(nump, dims))
        Xnew.append(dp)
        labels_new.append(np.full(nump, cluster_id, dtype=int))

        # 2️⃣ Noise-Punkte: label = 0
        rp = np.random.normal(loc=c, scale=np.sqrt(varr * noise_mult), size=(num_noisep, dims))
        Xnew.append(rp)
        labels_new.append(np.zeros(num_noisep, dtype=int))

    # only accept if enough steps were taken
    if len(new_cs) > numsteps / 4:
        return np.vstack(Xnew), np.hstack(labels_new), new_cs, n_collision
    else:
        return np.empty((0, dims)), np.empty((0,), dtype=int), [], n_collision



def gen_2d_worms(ds_id=1, max_clusters=10, nump=10, num_noisep=4, stepl=2, dims = 2, start_var = 1, end_var = 120, noise_mult = 8):
    X_blocks = []
    label_blocks = []
    c_trail = []
    cluster_id = 1
    n_collision = 0

    while n_collision < max_clusters:
        steepness = 1 + np.random.rand() * 4
        numsteps = np.random.randint(180, 301)
        var_range = np.array([start_var, end_var]) * 20
        c0 = np.random.rand(dims) * 2000 - 1000
        
        tree = None
        if c_trail:
            tree = KDTree(np.vstack(c_trail))


        Xnew, labels_new, new_cs, n_collision = gen_one_worm(
            c0, var_range, numsteps, steepness,
            nump, num_noisep, stepl,
            c_trail, tree, cluster_id, noise_mult, n_collision
        )

        print(cluster_id, n_collision)
        if Xnew.size > 0:
            X_blocks.append(Xnew)
            label_blocks.append(labels_new)
            c_trail.extend(new_cs)
            cluster_id += 1

    n_clusters = cluster_id - 1
    if X_blocks and n_clusters > 0:
        # combine all clusters
        X = np.vstack(X_blocks)
        labels = np.hstack(label_blocks)

        # shift so minimum is zero
        X -= X.min(axis=0)
        return X, labels, n_clusters


    else:
        print("No clusters generated.")



def normalize1(data):
    return (data - data.mean()) / (data.std() * 0.7) #2 * (data - data.min()) / (data.max() - data.min()) - 1

def embed_2d_data(h, hidden_dim=100, output_dim=3, n_hidden_layers=12, seed=44):
    rng = np.random.default_rng(seed)

    W_in = rng.standard_normal((hidden_dim, 2))
    hidden_weights = [rng.standard_normal((hidden_dim, hidden_dim))
                      for _ in range(n_hidden_layers - 1)]
    U = rng.standard_normal((output_dim, hidden_dim))

    X = normalize1(h.dot(W_in.T))
    X = np.tanh(X)

    for i, W_h in enumerate(hidden_weights):
        X = normalize1(X.dot(W_h.T))
        X = np.tanh(X)

    X = normalize1(X.dot(U.T))
    X = np.tanh(X)

    return X



def generate_worm_datasets_batch(
        noise_mult_list = [0, 5, 10, 20, 50],
        output_dims_list = [3, 5, 10],
        datasets_per_combo = 5,
        base_ds_id = 1,
        max_clusters = 10,
        nump = 10,
        num_noisep = 4,
        stepl = 2,
        dims = 2,
        start_var = 1,
        end_var = 120
    ):
    
    base_ds_id = 1
    os.makedirs("data", exist_ok=True)
    os.makedirs("img", exist_ok=True)

    for noise_mult in noise_mult_list:
        effective_noisep = 0 if noise_mult == 0 else num_noisep
        for output_dim in output_dims_list:
            for i in range(datasets_per_combo):
                ds_id = f"{base_ds_id:02d}id_{noise_mult}nm_{output_dim}d_{i}"
                base_ds_id += 1
                print(f"\n=== Generating ds_id {ds_id} (noise_mult={noise_mult}, output_dim={output_dim}) ===")
                
                # 1️⃣ Generate worm dataset
                X, labels, n_clusters = gen_2d_worms(
                    ds_id=ds_id,
                    max_clusters=max_clusters,
                    nump=nump,
                    num_noisep=effective_noisep,
                    stepl=stepl,
                    dims=dims,
                    start_var=start_var,
                    end_var=end_var,
                    noise_mult=noise_mult
                )
                
                # 2️⃣ Normalize
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)



                # 3️⃣ Save original normalized data
                original_fname = f"data/worms_2D_orig_{ds_id}.txt"
                print(X_scaled.shape)
                np.savetxt(original_fname, X_scaled, fmt="%.6f")
                print(f"Saved normalized original data: {original_fname}")

                plt.figure(figsize=(6,6))
                # labels[:,0] enthält die cluster_id
                plt.scatter(X_scaled[:, 0], X_scaled[:, 1],
                            c=labels, cmap="tab10", s=5, alpha=0.05)
                plt.title(f"Original 2D worms (ds_id={ds_id})")
                plt.xlabel("Feature 1")
                plt.ylabel("Feature 2")
                plt.colorbar(label="Cluster ID")
                plot2d_name = f"img/worms_2D_plot_{ds_id}.png"
                plt.tight_layout()
                plt.savefig(plot2d_name)
                plt.close()
                print(f"Saved 2D scatter plot: {plot2d_name}")


                # 4️⃣ Embed to higher dimension
                X_mapped = embed_2d_data(X_scaled, output_dim=output_dim)
                print(X_mapped.shape)
                embed_fname = f"data/worms_highD_orig_{ds_id}.txt"
                np.savetxt(embed_fname, X_mapped, fmt="%.6f")
                print(f"Saved mapped data: {embed_fname}")

                if output_dim == 3:
                    X_mapped_3d = X_mapped
                    embed_3d_fname = f"data/worms_3D_vis{ds_id}.txt"
                    np.savetxt(embed_3d_fname, X_mapped_3d, fmt="%.6f")
                    print(f"Saved 3D mapped data for visualization: {embed_3d_fname}")
                else:
                    X_mapped_3d = embed_2d_data(X_scaled, output_dim=3)
                    embed_3d_fname = f"data/worms_3D_vis{ds_id}.txt"
                    np.savetxt(embed_3d_fname, X_mapped_3d, fmt="%.6f")
                    print(f"Saved 3D mapped data for visualization: {embed_3d_fname}")

                fig = go.Figure(data=[
                    go.Scatter3d(
                        x=X_mapped_3d[:, 0],
                        y=X_mapped_3d[:, 1],
                        z=X_mapped_3d[:, 2],
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=labels,
                            colorscale='Viridis',
                            opacity=0.8,
                            showscale=True,
                            colorbar=dict(title='Cluster ID')
                        )
                    )
                ])
                fig.update_layout(
                    title=f"3D Embedding (ds_id={ds_id})",
                    scene=dict(
                        xaxis_title='Dim 1',
                        yaxis_title='Dim 2',
                        zaxis_title='Dim 3'
                    ),
                    margin=dict(l=0, r=0, b=0, t=30)
                )

                html_fname = f"img/worms_3D_plot_{ds_id}.html"
                fig.write_html(html_fname)
                print(f"Saved interactive 3D plot: {html_fname}")


                # 6️⃣ Save labels
                label_fname = f"data/worms_labels_{ds_id}.txt"
                print(labels.shape)
                np.savetxt(label_fname, labels.astype(int), fmt="%d")
                print(f"Saved labels: {label_fname}")
                #sys.exit("break")
                # 7️⃣ Compute and save pairwise distance matrix
                # distmat = squareform(pdist(X_mapped))
                # distmat_fname = f"worms_distmat_{output_dim}_{ds_id}.npy"
                # np.save(distmat_fname, distmat)
                # print(f"Saved distance matrix: {distmat_fname}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch generate worm datasets with multiple noise and embedding dimensions.")
    parser.add_argument("--datasets_per_combo", type=int, default=5, help="Datasets per noise_mult-output_dim combination.")
    parser.add_argument("--max_clusters", type=int, default=5, help="Maximum clusters per dataset.")
    parser.add_argument("--dims", type=int, default=2, help="Data generation dimensionality before embedding.")
    parser.add_argument("--stepl", type=float, default=1.5, help="Step length of worm movement.")
    parser.add_argument("--start_var", type=float, default=0.5, help="Start variance.")
    parser.add_argument("--end_var", type=float, default=150, help="End variance.")
    parser.add_argument("--nump", type=int, default=2, help="Dense points per step.")
    parser.add_argument("--num_noisep", type=int, default=2, help="Noise points per step.")

    args = parser.parse_args()

    noise_mult_list = [0, 5, 10, 20, 50]
    output_dims_list = [10]


    generate_worm_datasets_batch(
        noise_mult_list=noise_mult_list,
        output_dims_list=output_dims_list,
        datasets_per_combo=args.datasets_per_combo,
        max_clusters=args.max_clusters,
        dims=args.dims,
        stepl=args.stepl,
        start_var=args.start_var,
        end_var=args.end_var,
        nump=args.nump,
        num_noisep=args.num_noisep
    )


