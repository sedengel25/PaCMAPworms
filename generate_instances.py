#!/usr/bin/env python3

import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial.distance import pdist
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import argparse


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

        # generate dense points
        dp = np.random.normal(loc=c, scale=np.sqrt(varr), size=(nump, dims))
        Xnew.append(dp)
        labels_new.append(np.column_stack([
            np.full(nump, cluster_id),
            np.ones(nump, dtype=int)
        ]))

        # generate noise points
        rp = np.random.normal(loc=c, scale=np.sqrt(varr * noise_mult), size=(num_noisep, dims))
        Xnew.append(rp)
        labels_new.append(np.column_stack([
            np.full(num_noisep, cluster_id),
            np.full(num_noisep, 2, dtype=int)
        ]))

    # only accept if enough steps were taken
    if len(new_cs) > numsteps / 4:
        return np.vstack(Xnew), np.vstack(labels_new), new_cs, n_collision
    else:
        return np.empty((0, dims)), np.empty((0, 2), dtype=int), [], n_collision


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
        labels = np.vstack(label_blocks)

        # shift so minimum is zero
        X -= X.min(axis=0)

        # write to files with noise, dimension and cluster count in filename
        fname = f"worms_noise{num_noisep}_d{dims}_cl{n_clusters}_{ds_id}.txt"
        labname = f"worms_noise{num_noisep}_d{dims}_cl{n_clusters}_{ds_id}-labels.txt"
        np.savetxt(fname, X, fmt="%.6f")
        np.savetxt(labname, labels, fmt="%d %d")
        print(f"Dataset file generated: {fname}")
        print(f"Ground truth file generated: {labname}")

        # histogram of pairwise distances
        numsamples = min(5000, X.shape[0])
        idxs = np.random.permutation(X.shape[0])[:numsamples]
        d = pdist(X[idxs])
        plt.figure()
        plt.hist(d, bins=200)
        plt.xlabel("Pairwise distance")
        plt.ylabel("Frequency")
        plt.title("Distance histogram")
        plt.savefig(f"worms_{ds_id}.png")
        print(f"Histogram saved as worms_{ds_id}.png")
    else:
        print("No clusters generated.")


# Direct call without argparse for PyCharm
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic worm dataset.")

    parser.add_argument("--ds_id", type=int, default=1, help="Dataset ID suffix for filenames.")
    parser.add_argument("--max_clusters", type=int, default=20, help="Maximum number of clusters.")
    parser.add_argument("--nump", type=int, default=10, help="Number of dense points per step.")
    parser.add_argument("--num_noisep", type=int, default=4, help="Number of noise points per step.")
    parser.add_argument("--noise_mult", type=int, default=8, help="Noise variance multiplicator.")
    parser.add_argument("--stepl", type=float, default=2.0, help="Step length of the worm movement.")
    parser.add_argument("--dims", type=int, default=3, help="Number of dimensions.")
    parser.add_argument("--start_var", type=float, default=1.0, help="Start of variance range.")
    parser.add_argument("--end_var", type=float, default=120.0, help="End of variance range.")

    args = parser.parse_args()

    # Call the function with parsed arguments
    gen_2d_worms(
        ds_id=args.ds_id,
        max_clusters=args.max_clusters,
        nump=args.nump,
        num_noisep=args.num_noisep,
        stepl=args.stepl,
        dims=args.dims,
        start_var=args.start_var,
        end_var=args.end_var
    )

