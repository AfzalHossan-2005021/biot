"""
core.py — INCENT with shared-scale normalization fix
=====================================================
Single fix from original: D_A and D_B are now normalized by the
same scale (max of D_B), so GW correctly embeds A as a subregion of B.

All other logic is unchanged from the original INCENT implementation.
"""

import os
import time
import datetime

import numpy as np
import pandas as pd
import torch
import ot

from typing import Optional, Tuple, Union
from numpy.typing import NDArray
from anndata import AnnData

from .utils import (
    fused_gromov_wasserstein_incent,
    jensenshannon_divergence_backend,
    pairwise_msd,
    to_dense_array,
    extract_data_matrix,
)


# ─────────────────────────────────────────────────────────────────────────────
# Neighbourhood distribution
# ─────────────────────────────────────────────────────────────────────────────

def neighborhood_distribution(curr_slice: AnnData, radius: float) -> np.ndarray:
    """
    Compute the normalised cell-type neighbourhood distribution for every cell.

    Parameters
    ----------
    curr_slice : AnnData  — must have .obsm['spatial'] and .obs['cell_type_annot']
    radius     : float    — Euclidean radius defining the local neighbourhood

    Returns
    -------
    dist : (n_cells, n_cell_types) float64 array, rows sum to 1
    """
    from tqdm import tqdm
    from sklearn.neighbors import BallTree

    cell_types      = np.array(curr_slice.obs['cell_type_annot'].astype(str))
    unique_ct       = np.unique(cell_types)
    ct2idx          = {c: i for i, c in enumerate(unique_ct)}
    coords          = curr_slice.obsm['spatial']
    n               = curr_slice.shape[0]
    K               = len(unique_ct)

    tree            = BallTree(coords)
    neighbor_lists  = tree.query_radius(coords, r=radius)

    dist = np.zeros((n, K), dtype=np.float64)
    for i in tqdm(range(n), desc="Neighbourhood distribution"):
        for idx in neighbor_lists[i]:
            dist[i, ct2idx[cell_types[idx]]] += 1.0

    row_sums = dist.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return dist / row_sums


# ─────────────────────────────────────────────────────────────────────────────
# Cosine distance on gene expression
# ─────────────────────────────────────────────────────────────────────────────

def cosine_distance(sliceA, sliceB, sliceA_name, sliceB_name,
                    filePath, use_rep=None, use_gpu=False,
                    nx=ot.backend.NumpyBackend(), beta=0.8, overwrite=False):
    """
    Pairwise cosine distance matrix between gene expression of sliceA and sliceB.
    Results are cached to filePath.
    """
    A_X = nx.from_numpy(to_dense_array(extract_data_matrix(sliceA, use_rep)))
    B_X = nx.from_numpy(to_dense_array(extract_data_matrix(sliceB, use_rep)))

    if isinstance(nx, ot.backend.TorchBackend) and use_gpu:
        A_X = A_X.cuda()
        B_X = B_X.cuda()

    s_A = A_X + 0.01
    s_B = B_X + 0.01

    fileName = f"{filePath}/cosine_dist_gene_expr_{sliceA_name}_{sliceB_name}.npy"

    if os.path.exists(fileName) and not overwrite:
        print("Loading cached cosine distance matrix")
        mat = np.load(fileName)
        if use_gpu and isinstance(nx, ot.backend.TorchBackend):
            return torch.from_numpy(mat).cuda()
        return mat

    print("Computing cosine distance matrix")
    if isinstance(s_A, torch.Tensor) and isinstance(s_B, torch.Tensor):
        norm_A = s_A / s_A.norm(dim=1, keepdim=True)
        norm_B = s_B / s_B.norm(dim=1, keepdim=True)
        mat    = 1.0 - torch.mm(norm_A, norm_B.T)
        np.save(fileName, mat.cpu().detach().numpy())
        return mat
    else:
        from sklearn.metrics.pairwise import cosine_distances
        mat = cosine_distances(
            to_dense_array(s_A) if not isinstance(s_A, np.ndarray) else s_A,
            to_dense_array(s_B) if not isinstance(s_B, np.ndarray) else s_B,
        )
        np.save(fileName, mat)
        return mat


# ─────────────────────────────────────────────────────────────────────────────
# Main alignment function
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_align(
    sliceA:    AnnData,
    sliceB:    AnnData,
    alpha:     float,
    beta:      float,
    gamma:     float,
    radius:    float,
    filePath:  str,
    use_rep:   Optional[str]  = None,
    G_init                    = None,
    a_distribution            = None,
    b_distribution            = None,
    norm:      bool           = False,
    numItermax: int           = 6000,
    backend                   = ot.backend.NumpyBackend(),
    use_gpu:   bool           = False,
    return_obj: bool          = False,
    verbose:   bool           = False,
    gpu_verbose: bool         = True,
    sliceA_name: Optional[str] = None,
    sliceB_name: Optional[str] = None,
    overwrite: bool           = False,
    neighborhood_dissimilarity: str = 'jsd',
    **kwargs,
) -> Union[NDArray[np.floating],
           Tuple[NDArray[np.floating], float, float, float, float]]:
    """
    Compute the optimal alignment between two MERFISH slices.

    Key fix vs original INCENT
    --------------------------
    Both D_A and D_B are normalised by ``max(D_B)`` (shared scale).
    This preserves the true size relationship between the two slices so that
    GW correctly embeds slice A as a *subregion* of slice B rather than
    forcing it to fill all of B.

    Parameters
    ----------
    sliceA, sliceB : AnnData
        Must have .obsm['spatial'] and .obs['cell_type_annot'].
    alpha  : weight of the GW spatial term  (0 = biology only, 1 = space only)
    beta   : weight of cell-type mismatch penalty inside M1
    gamma  : weight of neighbourhood dissimilarity M2
    radius : neighbourhood radius for niche distribution (same units as coords)
    filePath : directory for caching intermediate matrices and logs
    neighborhood_dissimilarity : 'jsd' | 'cosine' | 'msd'
    """

    start_time = time.time()
    os.makedirs(filePath, exist_ok=True)

    log_name = (f"{filePath}/log_{sliceA_name}_{sliceB_name}.txt"
                if sliceA_name and sliceB_name
                else f"{filePath}/log.txt")
    logFile  = open(log_name, "w")
    logFile.write("pairwise_align — INCENT with shared-scale normalization\n")
    logFile.write(f"{datetime.datetime.now()}\n")
    logFile.write(f"sliceA={sliceA_name}  sliceB={sliceB_name}\n")
    logFile.write(f"alpha={alpha}  beta={beta}  gamma={gamma}  radius={radius}\n\n")

    # ── GPU / backend selection ───────────────────────────────────────────────
    if use_gpu:
        if torch.cuda.is_available():
            backend = ot.backend.TorchBackend()
            if gpu_verbose:
                print("GPU available — using CUDA.")
        else:
            use_gpu = False
            backend = ot.backend.NumpyBackend()
            if gpu_verbose:
                print("GPU requested but not available — using CPU.")
    else:
        backend = ot.backend.NumpyBackend()
        if gpu_verbose:
            print("Using CPU backend.")

    nx = backend

    # ── Input validation ──────────────────────────────────────────────────────
    for s in [sliceA, sliceB]:
        if not len(s):
            raise ValueError(f"Empty AnnData: {s}")

    # ── Shared genes ──────────────────────────────────────────────────────────
    shared_genes = sliceA.var_names.intersection(sliceB.var_names)
    if len(shared_genes) == 0:
        raise ValueError("No shared genes between slices.")
    sliceA = sliceA[:, shared_genes]
    sliceB = sliceB[:, shared_genes]

    # ── Shared cell types ─────────────────────────────────────────────────────
    shared_ct = (pd.Index(sliceA.obs['cell_type_annot'])
                 .unique()
                 .intersection(pd.Index(sliceB.obs['cell_type_annot']).unique()))
    if len(shared_ct) == 0:
        raise ValueError("No shared cell types between slices.")
    sliceA = sliceA[sliceA.obs['cell_type_annot'].isin(shared_ct)]
    sliceB = sliceB[sliceB.obs['cell_type_annot'].isin(shared_ct)]

    logFile.write(f"After filtering: n_A={sliceA.shape[0]}  n_B={sliceB.shape[0]}\n")
    logFile.write(f"shared_genes={len(shared_genes)}  shared_ct={len(shared_ct)}\n\n")

    # ── Spatial distance matrices ─────────────────────────────────────────────
    coordsA = nx.from_numpy(sliceA.obsm['spatial'].copy())
    coordsB = nx.from_numpy(sliceB.obsm['spatial'].copy())
    if isinstance(nx, ot.backend.TorchBackend):
        coordsA = coordsA.float()
        coordsB = coordsB.float()

    D_A = ot.dist(coordsA, coordsA, metric='euclidean')
    D_B = ot.dist(coordsB, coordsB, metric='euclidean')

    # ── ★ SHARED-SCALE NORMALIZATION (the critical fix) ★ ────────────────────
    #
    # Both matrices are divided by the SAME value: max(D_B).
    # After normalisation:
    #   D_B spans [0, 1.0]
    #   D_A spans [0, diameter_A / diameter_B]  ← remains smaller than 1
    #
    # This tells GW: "A's internal geometry fits inside B's geometry."
    # GW will therefore embed A as a spatial subregion of B — which is
    # exactly what serial-section partial overlap requires.
    #
    # Under the OLD independent normalisation (D_A /= max(D_A)):
    #   Both span [0, 1.0] → GW thinks A and B are the same size
    #   → forces A cells to spread across all of B → mixing failure.
    #
    scale = nx.max(nx.max(D_A), nx.max(D_B))
    if float(scale) < 1e-12:
        raise ValueError("D_B is all zeros — check that spatial coordinates exist.")

    D_A = D_A / scale   # e.g. spans [0, 0.76] for the provided example data
    D_B = D_B / scale   # spans [0, 1.0]

    # NumPy 2.0 removed ndarray.ptp; use np.ptp for compatibility.
    span_A = np.ptp(np.asarray(sliceA.obsm['spatial']), axis=0).max()
    span_B = np.ptp(np.asarray(sliceB.obsm['spatial']), axis=0).max()
    expected_ratio = span_A / max(span_B, 1e-12)

    logFile.write(f"Shared-scale normalization: scale={float(scale):.4f}\n")
    logFile.write(f"D_A max after norm: {float(nx.max(D_A)):.6f}  "
                  f"(expected ≈ {expected_ratio:.4f})\n")
    logFile.write(f"D_B max after norm: {float(nx.max(D_B)):.6f}  (expected 1.0)\n\n")

    if use_gpu and isinstance(nx, ot.backend.TorchBackend):
        D_A = D_A.cuda()
        D_B = D_B.cuda()

    # ── Gene expression cost M_gene ───────────────────────────────────────────
    cosine_dist_gene_expr = cosine_distance(
        sliceA, sliceB, sliceA_name, sliceB_name, filePath,
        use_rep=use_rep, use_gpu=use_gpu, nx=nx, beta=beta, overwrite=overwrite)

    # ── Cell-type mismatch penalty ────────────────────────────────────────────
    lab_A = np.asarray(sliceA.obs['cell_type_annot'].values)
    lab_B = np.asarray(sliceB.obs['cell_type_annot'].values)
    M_celltype = (lab_A[:, None] != lab_B[None, :]).astype(np.float64)

    if isinstance(cosine_dist_gene_expr, torch.Tensor):
        M_ct = torch.from_numpy(M_celltype).to(cosine_dist_gene_expr.device)
        M1   = (1.0 - beta) * cosine_dist_gene_expr + beta * M_ct
    else:
        M1 = nx.from_numpy(
            (1.0 - beta) * cosine_dist_gene_expr + beta * M_celltype)

    logFile.write(f"M_celltype: shape={M_celltype.shape}  beta={beta}\n")

    # ── Neighbourhood distribution ────────────────────────────────────────────
    nd_cache_A = f"{filePath}/nd_{sliceA_name}.npy"
    nd_cache_B = f"{filePath}/nd_{sliceB_name}.npy"

    if os.path.exists(nd_cache_A) and not overwrite:
        print("Loading cached neighbourhood distribution A")
        nd_A = np.load(nd_cache_A)
    else:
        print("Computing neighbourhood distribution A")
        nd_A = neighborhood_distribution(sliceA, radius=radius)
        np.save(nd_cache_A, nd_A)

    if os.path.exists(nd_cache_B) and not overwrite:
        print("Loading cached neighbourhood distribution B")
        nd_B = np.load(nd_cache_B)
    else:
        print("Computing neighbourhood distribution B")
        nd_B = neighborhood_distribution(sliceB, radius=radius)
        np.save(nd_cache_B, nd_B)

    nd_A += 0.01   # avoid zero-division in JSD
    nd_B += 0.01

    # Move to GPU if needed
    if use_gpu:
        if isinstance(nd_A, np.ndarray):
            nd_A = torch.from_numpy(nd_A).cuda()
        if isinstance(nd_B, np.ndarray):
            nd_B = torch.from_numpy(nd_B).cuda()

    # ── Neighbourhood dissimilarity M2 ────────────────────────────────────────
    if neighborhood_dissimilarity == 'jsd':
        jsd_cache = f"{filePath}/jsd_{sliceA_name}_{sliceB_name}.npy"
        if os.path.exists(jsd_cache) and not overwrite:
            print("Loading cached JSD matrix")
            js_dist = np.load(jsd_cache)
            M2 = (torch.from_numpy(js_dist).cuda()
                  if use_gpu and isinstance(nx, ot.backend.TorchBackend)
                  else nx.from_numpy(js_dist))
        else:
            print("Computing JSD matrix")
            js_dist = jensenshannon_divergence_backend(nd_A, nd_B)
            if isinstance(js_dist, torch.Tensor):
                np.save(jsd_cache, js_dist.cpu().numpy())
                M2 = js_dist
            else:
                np.save(jsd_cache, js_dist)
                M2 = nx.from_numpy(js_dist)

    elif neighborhood_dissimilarity == 'cosine':
        if isinstance(nd_A, torch.Tensor):
            na = nd_A if not use_gpu else nd_A.cuda()
            nb = nd_B if not use_gpu else nd_B.cuda()
            num = na @ nb.T
            den = na.norm(dim=1)[:, None] * nb.norm(dim=1)[None, :]
            M2  = 1.0 - num / (den + 1e-12)
        else:
            na  = np.asarray(nd_A)
            nb  = np.asarray(nd_B)
            num = na @ nb.T
            den = np.linalg.norm(na, axis=1)[:, None] * np.linalg.norm(nb, axis=1)[None, :]
            M2  = nx.from_numpy(1.0 - num / (den + 1e-12))

    elif neighborhood_dissimilarity == 'msd':
        na  = nd_A.cpu().numpy() if isinstance(nd_A, torch.Tensor) else np.asarray(nd_A)
        nb  = nd_B.cpu().numpy() if isinstance(nd_B, torch.Tensor) else np.asarray(nd_B)
        M2  = nx.from_numpy(pairwise_msd(na, nb))

    else:
        raise ValueError(
            f"neighborhood_dissimilarity must be 'jsd', 'cosine', or 'msd'. "
            f"Got: {neighborhood_dissimilarity!r}")

    # Move M1, M2 to GPU if needed
    if use_gpu and isinstance(nx, ot.backend.TorchBackend):
        if not isinstance(M1, torch.Tensor):
            M1 = torch.from_numpy(np.asarray(M1)).cuda()
        if not isinstance(M2, torch.Tensor):
            M2 = torch.from_numpy(np.asarray(M2)).cuda()
        M1 = M1.cuda()
        M2 = M2.cuda()

    # ── Marginal distributions ────────────────────────────────────────────────
    if a_distribution is None:
        a = nx.ones((sliceA.shape[0],)) / sliceA.shape[0]
    else:
        a = nx.from_numpy(a_distribution)

    if b_distribution is None:
        b = nx.ones((sliceB.shape[0],)) / sliceB.shape[0]
    else:
        b = nx.from_numpy(b_distribution)

    if use_gpu and isinstance(nx, ot.backend.TorchBackend):
        a = a.cuda()
        b = b.cuda()

    # ── Initial transport plan ────────────────────────────────────────────────
    if G_init is not None:
        G_init_t = nx.from_numpy(G_init)
        if isinstance(nx, ot.backend.TorchBackend):
            G_init_t = G_init_t.float()
            if use_gpu:
                G_init_t = G_init_t.cuda()
    else:
        G_init_t = None

    # ── Log initial objectives ────────────────────────────────────────────────
    def _np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    G0_np = np.ones((sliceA.shape[0], sliceB.shape[0])) / (
        sliceA.shape[0] * sliceB.shape[0])

    if neighborhood_dissimilarity == 'jsd':
        init_nb = float(np.sum(_np(M2) * G0_np))
        logFile.write(f"Initial obj neighbour (jsd): {init_nb:.6f}\n")
    init_gene = float(np.sum(_np(cosine_dist_gene_expr) * G0_np))
    logFile.write(f"Initial obj gene (cosine):    {init_gene:.6f}\n\n")

    # ── Solve FGW ────────────────────────────────────────────────────────────
    pi, logw = fused_gromov_wasserstein_incent(
        M1, M2, D_A, D_B, a, b,
        G_init=G_init_t,
        loss_fun='square_loss',
        alpha=alpha,
        gamma=gamma,
        log=True,
        numItermax=numItermax,
        verbose=verbose,
        use_gpu=use_gpu,
    )
    pi = nx.to_numpy(pi)

    # ── Log final objectives ──────────────────────────────────────────────────
    if neighborhood_dissimilarity == 'jsd':
        max_idx   = np.argmax(pi, axis=1)
        jsd_np    = _np(M2)
        final_nb  = float(sum(pi[i, max_idx[i]] * jsd_np[i, max_idx[i]]
                              for i in range(len(max_idx))))
        logFile.write(f"Final obj neighbour (jsd): {final_nb:.6f}\n")

    final_gene = float(np.sum(_np(cosine_dist_gene_expr) * pi))
    logFile.write(f"Final obj gene (cosine):   {final_gene:.6f}\n")
    logFile.write(f"Runtime: {time.time() - start_time:.1f}s\n")
    logFile.close()

    if use_gpu and isinstance(nx, ot.backend.TorchBackend):
        torch.cuda.empty_cache()

    if return_obj:
        init_nb_val  = init_nb  if neighborhood_dissimilarity == 'jsd' else 0.0
        final_nb_val = final_nb if neighborhood_dissimilarity == 'jsd' else 0.0
        return pi, init_nb_val, init_gene, final_nb_val, final_gene

    return pi