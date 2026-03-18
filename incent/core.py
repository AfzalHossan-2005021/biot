import os
import ot
import time
import torch
import datetime
import numpy as np
import pandas as pd

from tqdm import tqdm
from anndata import AnnData
from numpy.typing import NDArray
from typing import Optional, Tuple, Union

from .utils import to_dense_array, extract_data_matrix, jensenshannon_divergence_backend, pairwise_msd


def identify_target_hemisphere(sliceA, sliceB, nd_A, nd_B, threshold=0.05):
    """
    Determine which spatial half of sliceB the cells of sliceA best match,
    using population-level niche fingerprint comparison.
    Returns: 'left', 'right', or 'both' (if no clear signal).
    """
    coords_B = sliceB.obsm['spatial']
    
    # Step 1: Find the bilateral symmetry axis of B
    # PCA on spatial coordinates: PC1 = longest axis = left-right axis
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(coords_B)
    pc1_scores = pca.transform(coords_B)[:, 0]
    midline = np.median(pc1_scores)  # Midline = median along PC1
    
    # Step 2: Split B into two halves
    left_mask  = pc1_scores <= midline
    right_mask = pc1_scores >  midline
    
    # Step 3: Population-level niche fingerprints
    # Average over all cells — integrates out individual cell noise
    mu_A     = nd_A.mean(axis=0)
    mu_B_L   = nd_B[left_mask].mean(axis=0)
    mu_B_R   = nd_B[right_mask].mean(axis=0)
    
    # Normalize to probability distributions
    mu_A   = mu_A   / mu_A.sum()
    mu_B_L = mu_B_L / mu_B_L.sum()
    mu_B_R = mu_B_R / mu_B_R.sum()
    
    # Step 4: JSD comparison at population level
    from scipy.spatial.distance import jensenshannon
    jsd_left  = jensenshannon(mu_A, mu_B_L)
    jsd_right = jensenshannon(mu_A, mu_B_R)
    
    # Step 5: Decision with uncertainty band
    ratio = abs(jsd_left - jsd_right) / (jsd_left + jsd_right + 1e-12)
    if ratio < threshold:
        return 'both', left_mask, right_mask, pc1_scores, midline
    elif jsd_left < jsd_right:
        return 'left', left_mask, right_mask, pc1_scores, midline
    else:
        return 'right', left_mask, right_mask, pc1_scores, midline


def hemisphere_aware_G0(sliceA, sliceB, hemisphere, left_mask, right_mask):
    """
    Build an asymmetric initialization G0 that concentrates transport mass
    onto the identified target hemisphere of B.
    High weight on target hemisphere cells, near-zero on other hemisphere.
    """
    n_A = sliceA.shape[0]
    n_B = sliceB.shape[0]
    
    G0 = np.ones((n_A, n_B)) / (n_A * n_B)
    
    if hemisphere == 'left':
        # Amplify target hemisphere, suppress other
        G0[:, left_mask]  *= 10.0
        G0[:, right_mask] *= 0.01
    elif hemisphere == 'right':
        G0[:, right_mask] *= 10.0
        G0[:, left_mask]  *= 0.01
    # 'both': uniform initialization unchanged
    
    # Renormalize to valid transport plan
    G0 /= G0.sum()
    return G0


def spatial_coherence_cost(sliceB, pc1_scores, midline, hemisphere, lambda_coh=0.5):
    """
    Returns M_coh: (n_B,) cost vector — penalty for each B cell being
    on the wrong side of the midline given the identified hemisphere.
    Applied as an additive term to M1: M1 += lambda_coh * M_coh[None, :]
    Broadcasting means all cells in A share the same lateral penalty for each B cell.
    """
    if hemisphere == 'both':
        return np.zeros(sliceB.shape[0])
    
    # Soft penalty: sigmoid of signed distance from midline,
    # pointing toward the wrong hemisphere
    signed_dist = pc1_scores - midline  # positive = right half
    
    if hemisphere == 'left':
        # Penalize cells in B that are on the right side
        # Penalty increases with distance from midline into wrong territory
        M_coh = np.maximum(0, signed_dist)  # 0 on left, positive on right
    else:  # 'right'
        M_coh = np.maximum(0, -signed_dist) # 0 on right, positive on left
    
    # Normalize to [0,1]
    if M_coh.max() > 0:
        M_coh /= M_coh.max()
    
    return M_coh  # shape (n_B,) — broadcast over all cells in A

# ── NEW: Phase 1 ─────────────────────────────────────────────────────────────
def joint_anatomical_embedding(nd_A, nd_B, sigma=None, n_components=15):
    """
    Joint diffusion map on neighborhood distributions of two slices.
    Returns cluster labels for all n_A + n_B cells.
    sigma=None → estimated as median pairwise JSD (bandwidth heuristic).
    """
    from sklearn.neighbors import BallTree
    import scipy.sparse as sp
    from sklearn.cluster import KMeans

    X = np.vstack([nd_A, nd_B])         # (n_A+n_B) x K
    n = X.shape[0]

    # Pairwise JSD — reuse jensenshannon_divergence_backend
    D = jensenshannon_divergence_backend(X, X)   # (n x n) full matrix

    if sigma is None:
        sigma = np.median(D[D > 0])

    K = np.exp(-D**2 / sigma**2)
    # Row-normalize to Markov matrix
    row_sums = K.sum(axis=1, keepdims=True)
    P = K / row_sums

    # Top eigenvectors (diffusion coordinates)
    from scipy.sparse.linalg import eigs
    vals, vecs = eigs(P, k=n_components+1, which='LM')
    vals, vecs = vals[1:].real, vecs[:, 1:].real   # drop trivial component
    Phi = vecs * vals[np.newaxis, :]               # scale by eigenvalues

    # Leiden clustering on diffusion coordinates
    # (use leidenalg if available, else fall back to KMeans)
    try:
        import leidenalg, igraph
        import sklearn.neighbors
        knn = sklearn.neighbors.NearestNeighbors(n_neighbors=15).fit(Phi)
        adj = knn.kneighbors_graph(mode='connectivity')
        g = igraph.Graph.Adjacency(adj.toarray().tolist())
        partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
        labels = np.array(partition.membership)
    except ImportError:
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=20, n_init=10).fit_predict(Phi)

    return labels, Phi


# ── NEW: Phase 2 ─────────────────────────────────────────────────────────────
def adaptive_marginals(labels, n_A, n_B, eps=0.05):
    """
    Given cluster labels for all n_A+n_B cells,
    return adaptive uniform marginals a (n_A,) and b (n_B,)
    where only cells in shared clusters have nonzero weight.
    """
    labels_A = labels[:n_A]
    labels_B = labels[n_A:]

    clusters = np.unique(labels)
    shared_clusters = set()
    for r in clusters:
        frac_A = np.sum(labels_A == r) / max(np.sum(labels == r), 1)
        if eps < frac_A < 1 - eps:
            shared_clusters.add(r)

    mask_A = np.array([labels_A[i] in shared_clusters for i in range(n_A)], dtype=float)
    mask_B = np.array([labels_B[j] in shared_clusters for j in range(n_B)], dtype=float)

    if mask_A.sum() == 0 or mask_B.sum() == 0:
        # Fallback: no shared clusters found → use uniform (full overlap assumed)
        return np.ones(n_A)/n_A, np.ones(n_B)/n_B, set()

    a = mask_A / mask_A.sum()
    b = mask_B / mask_B.sum()
    return a, b, shared_clusters


# ── MODIFIED: cluster coherence cost ─────────────────────────────────────────
def cluster_coherence_cost(labels_A, labels_B):
    """Binary mismatch matrix: 0 if same cluster, 1 otherwise."""
    return (labels_A[:, None] != labels_B[None, :]).astype(np.float64)


def find_mnn_anchors(M_bio, k=15):
    """
    Discover mutual nearest-neighbor anchors between A and B from a
    biological cost matrix M_bio (lower is better).
    """
    M_bio = np.asarray(M_bio, dtype=np.float64)
    k = int(max(1, min(k, M_bio.shape[0], M_bio.shape[1])))

    nn_A_to_B = np.argsort(M_bio, axis=1)[:, :k]
    nn_B_to_A = np.argsort(M_bio, axis=0)[:k, :].T

    anchors = []
    for i in range(M_bio.shape[0]):
        for j in nn_A_to_B[i]:
            if i in nn_B_to_A[j]:
                anchors.append((i, int(j)))
    return anchors


def ransac_rigid_transform(coords_A, coords_B, anchors, n_iter=500, inlier_thresh=None):
    """
    Fit a 2D rigid transform x_B ≈ R*x_A + t using RANSAC over anchor pairs.
    Returns R, t, and inlier indices into the anchor list.
    """
    from scipy.linalg import svd

    if len(anchors) < 2:
        return np.eye(2), np.zeros(2), np.array([], dtype=int)

    pts_A = np.array([coords_A[i] for i, _ in anchors], dtype=np.float64)
    pts_B = np.array([coords_B[j] for _, j in anchors], dtype=np.float64)

    if inlier_thresh is None:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=2).fit(coords_B)
        dists, _ = nn.kneighbors(coords_B)
        inlier_thresh = 5.0 * np.median(dists[:, 1])

    def fit_rigid(pA, pB):
        cA, cB = pA.mean(0), pB.mean(0)
        H = (pA - cA).T @ (pB - cB)
        U, _, Vt = svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = cB - R @ cA
        return R, t

    best_inliers = np.array([], dtype=int)
    best_R, best_t = None, None

    n_iter = int(max(1, n_iter))
    for _ in range(n_iter):
        idx = np.random.choice(len(anchors), size=2, replace=False)
        R, t = fit_rigid(pts_A[idx], pts_B[idx])
        residuals = np.linalg.norm(pts_B - (pts_A @ R.T + t), axis=1)
        inliers = np.where(residuals < inlier_thresh)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            if len(inliers) >= 2:
                best_R, best_t = fit_rigid(pts_A[inliers], pts_B[inliers])

    if best_R is None:
        best_R, best_t = fit_rigid(pts_A, pts_B)
        best_inliers = np.arange(len(anchors))

    return best_R, best_t, best_inliers


def spatial_deviation_cost(coords_A, coords_B, R, t, sigma=None):
    """
    Spatial deviation matrix M_dev(i,j) = ||x_B_j - (R*x_A_i + t)||^2 / sigma^2.
    """
    T_coords_A = coords_A @ R.T + t
    diff = T_coords_A[:, None, :] - coords_B[None, :, :]
    sq_dist = np.sum(diff ** 2, axis=2)

    if sigma is None:
        nz = sq_dist[sq_dist > 0]
        sigma = np.sqrt(np.median(nz)) if nz.size else 1.0

    return sq_dist / (sigma ** 2 + 1e-8)


def refine_transform_from_plan(pi, coords_A, coords_B, n_top=500):
    """
    Re-estimate rigid transform from top-confidence transport pairs.
    """
    from scipy.linalg import svd

    pi = np.asarray(pi, dtype=np.float64)
    flat = pi.ravel()
    if flat.size == 0:
        return np.eye(2), np.zeros(2)

    n_top = int(max(2, min(n_top, flat.size)))
    top_idx = np.argpartition(flat, -n_top)[-n_top:]
    rows = top_idx // pi.shape[1]
    cols = top_idx % pi.shape[1]
    weights = flat[top_idx]
    w_sum = np.sum(weights)
    if w_sum <= 0:
        return np.eye(2), np.zeros(2)

    pA = coords_A[rows]
    pB = coords_B[cols]
    w = weights / w_sum

    cA = np.sum(w[:, None] * pA, axis=0)
    cB = np.sum(w[:, None] * pB, axis=0)
    H = (pA - cA).T @ np.diag(w) @ (pB - cB)
    U, _, Vt = svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cB - R @ cA

    return R, t


def pairwise_align(
    sliceA: AnnData, 
    sliceB: AnnData, 
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    radius: float,
    filePath: str,
    use_rep: Optional[str] = None, 
    G_init = None, 
    a_distribution = None, 
    b_distribution = None, 
    norm: bool = False, 
    numItermax: int = 6000, 
    backend = ot.backend.TorchBackend(), 
    use_gpu: bool = False, 
    return_obj: bool = False,
    verbose: bool = False, 
    gpu_verbose: bool = True, 
    sliceA_name: Optional[str] = None,
    sliceB_name: Optional[str] = None,
    overwrite = False,
    neighborhood_dissimilarity: str='jsd',
    **kwargs) -> Union[NDArray[np.floating], Tuple[NDArray[np.floating], float, float, float, float]]:
    """

    This method is written by Anup Bhowmik, CSE, BUET

    Calculates and returns optimal alignment of two slices of single cell MERFISH data. 
    
    Args:
        sliceA: Slice A to align.
        sliceB: Slice B to align.
        alpha: weight for spatial distance
        beta: weight for cell type one-hot encoding cost
        gamma: weight for neighborhood expression distance (e.g., JSD)
        radius: spatial radius (Euclidean distance) defining the local neighborhood of a cell.
        filePath: Absolute or relative directory path used for caching distance matrices and results.
        use_rep: If ``None``, uses ``slice.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``slice.obsm[use_rep]``.
        G_init (array-like, optional): Initial mapping to be used in FGW-OT, otherwise default is uniform mapping.
        a_distribution (array-like, optional): Distribution of sliceA spots, otherwise default is uniform.
        b_distribution (array-like, optional): Distribution of sliceB spots, otherwise default is uniform.
        norm: If ``True``, scales spatial distances such that neighboring spots are at distance 1. Otherwise, spatial distances remain unchanged.
        numItermax: Max number of iterations during FGW-OT.
        backend: Type of backend to run calculations. For list of backends available on system: ``ot.backend.get_backend_list()``.
        use_gpu: If ``True``, use gpu. Otherwise, use cpu. Currently we only have gpu support for Pytorch.
        return_obj: If ``True``, additionally returns objective function output of FGW-OT.
        verbose: If ``True``, FGW-OT is verbose.
        gpu_verbose: If ``True``, print whether gpu is being used to user.
        sliceA_name: Optional string identifier for slice A caching.
        sliceB_name: Optional string identifier for slice B caching.
        overwrite: If ``True``, forces recalculation of distance matrices ignoring cache.
        neighborhood_dissimilarity: Name of measure for neighborhood comparisons (e.g., ``'jsd'`` for Jensen-Shannon Divergence).

    Returns:
        - Alignment of spots.

        If ``return_obj = True``, additionally returns:
        
        - Objective function output of cost 
    """

    start_time = time.time()

    if not os.path.exists(filePath):
        os.makedirs(filePath)

    logFile = open(f"{filePath}/log.txt", "w")

    logFile.write(f"pairwise_align_INCENT\n")
    currDateTime = datetime.datetime.now()

    # logFile.write(f"{currDateTime.date()}, {currDateTime.strftime("%I:%M %p")} BDT, {currDateTime.strftime("%A")} \n")

    logFile.write(f"{currDateTime}\n")
    logFile.write(f"sliceA_name: {sliceA_name}, sliceB_name: {sliceB_name}\n")
   

    logFile.write(f"alpha: {alpha}\n")
    logFile.write(f"beta: {beta}\n")
    logFile.write(f"gamma: {gamma}\n")
    logFile.write(f"radius: {radius}\n")


    
    # Determine if gpu or cpu is being used
    if use_gpu:
        if torch.cuda.is_available():
            backend = ot.backend.TorchBackend()
            if gpu_verbose:
                print("GPU is requested and available, using gpu.")
        else:
            use_gpu = False
            backend = ot.backend.NumpyBackend()
            if gpu_verbose:
                print("GPU is requested but not available, resorting to torch cpu.")
    else:
        backend = ot.backend.NumpyBackend()
        if gpu_verbose:
            print("Using selected backend cpu. If you want to use gpu, set use_gpu = True.")
    
    # check if slices are valid
    for s in [sliceA, sliceB]:
        if not len(s):
            raise ValueError(f"Found empty `AnnData`:\n{s}.")

    
    # Backend
    nx = backend

    # Filter to shared genes
    shared_genes = sliceA.var_names.intersection(sliceB.var_names)
    if len(shared_genes) == 0:
        raise ValueError("No shared genes between the two slices.")
    sliceA = sliceA[:, shared_genes]
    sliceB = sliceB[:, shared_genes]


    # Filter to shared cell types
    # This is needed for the cell-type mismatch penalty, and also ensures that the neighborhood distributions are comparable (same set of cell types).
    shared_cell_types = pd.Index(sliceA.obs['cell_type_annot']).unique().intersection(pd.Index(sliceB.obs['cell_type_annot']).unique())
    if len(shared_cell_types) == 0:
        raise ValueError("No shared cell types between the two slices.")
    sliceA = sliceA[sliceA.obs['cell_type_annot'].isin(shared_cell_types)]
    sliceB = sliceB[sliceB.obs['cell_type_annot'].isin(shared_cell_types)]

    
    # Calculate spatial distances
    coordinatesA = sliceA.obsm['spatial'].copy()
    coordinatesB = sliceB.obsm['spatial'].copy()
    coordinatesA = nx.from_numpy(coordinatesA)
    coordinatesB = nx.from_numpy(coordinatesB)
    
    if isinstance(nx,ot.backend.TorchBackend):
        coordinatesA = coordinatesA.float()
        coordinatesB = coordinatesB.float()
    D_A = ot.dist(coordinatesA, coordinatesA, metric='euclidean')
    D_B = ot.dist(coordinatesB, coordinatesB, metric='euclidean')

    # Calculate gene expression dissimilarity
    # filePath = '/content/drive/MyDrive/Thesis_data_anup/local_data'
    cosine_dist_gene_expr = cosine_distance(sliceA, sliceB, sliceA_name, sliceB_name, filePath, use_rep = use_rep, use_gpu = use_gpu, nx = nx, beta = beta, overwrite=overwrite)

    # ── Explicit cell-type mismatch penalty ──────────────────────────────
    # Binary matrix: 0 for same type, 1 for different type.
    # Added to M1 so it enters the FW gradient directly → strong cell-type signal.

    _lab_A = np.asarray(sliceA.obs['cell_type_annot'].values)
    _lab_B = np.asarray(sliceB.obs['cell_type_annot'].values)
    M_celltype = (_lab_A[:, None] != _lab_B[None, :]).astype(np.float64)

    if isinstance(cosine_dist_gene_expr, torch.Tensor):
        M_celltype_t = torch.from_numpy(M_celltype).to(cosine_dist_gene_expr.device)
        M1 = (1 - beta) * cosine_dist_gene_expr + beta * M_celltype_t
    else:
        M1_combined = (1 - beta) * cosine_dist_gene_expr + beta * M_celltype
        M1 = nx.from_numpy(M1_combined)

    logFile.write(f"[cell_type_penalty] beta={beta}, M_celltype shape={M_celltype.shape}\n")


    # jensenshannon_divergence_backend actually returns jensen shannon distance
    # neighborhood_distribution_slice_1, neighborhood_distribution_slice_1 will be pre computed

    if os.path.exists(f"{filePath}/neighborhood_distribution_{sliceA_name}.npy") and not overwrite:
        print("Loading precomputed neighborhood distribution of slice A")
        neighborhood_distribution_sliceA = np.load(f"{filePath}/neighborhood_distribution_{sliceA_name}.npy")
    else:
        print("Calculating neighborhood distribution of slice A")
        neighborhood_distribution_sliceA = neighborhood_distribution(sliceA, radius = radius)


        neighborhood_distribution_sliceA += 0.01 # for avoiding zero division error
        # print("Saving neighborhood distribution of slice A")
        # np.save(f"{filePath}/neighborhood_distribution_{sliceA_name}.npy", neighborhood_distribution_sliceA)


    if os.path.exists(f"{filePath}/neighborhood_distribution_{sliceB_name}.npy") and not overwrite:
        print("Loading precomputed neighborhood distribution of slice B")
        neighborhood_distribution_sliceB = np.load(f"{filePath}/neighborhood_distribution_{sliceB_name}.npy")
    else:
        print("Calculating neighborhood distribution of slice B")
        neighborhood_distribution_sliceB = neighborhood_distribution(sliceB, radius = radius)


        neighborhood_distribution_sliceB += 0.01 # for avoiding zero division error
        # print("Saving neighborhood distribution of slice B")
        # np.save(f"{filePath}/neighborhood_distribution_{sliceB_name}.npy", neighborhood_distribution_sliceB)


    if ('numpy' in str(type(neighborhood_distribution_sliceA))) and use_gpu:
        neighborhood_distribution_sliceA = torch.from_numpy(neighborhood_distribution_sliceA)
    if ('numpy' in str(type(neighborhood_distribution_sliceB))) and use_gpu:
        neighborhood_distribution_sliceB = torch.from_numpy(neighborhood_distribution_sliceB)

    if use_gpu:
        neighborhood_distribution_sliceA = neighborhood_distribution_sliceA.cuda()
        neighborhood_distribution_sliceB = neighborhood_distribution_sliceB.cuda()

    if neighborhood_dissimilarity == 'jsd':
        if os.path.exists(f"{filePath}/js_dist_neighborhood_{sliceA_name}_{sliceB_name}.npy") and not overwrite:
            print("Loading precomputed JSD of neighborhood distribution for slice A and slice B")
            js_dist_neighborhood = np.load(f"{filePath}/js_dist_neighborhood_{sliceA_name}_{sliceB_name}.npy")
            if use_gpu and isinstance(nx, ot.backend.TorchBackend):
                js_dist_neighborhood = torch.from_numpy(js_dist_neighborhood).cuda()
        else:
            print("Calculating JSD of neighborhood distribution for slice A and slice B")

            js_dist_neighborhood = jensenshannon_divergence_backend(neighborhood_distribution_sliceA, neighborhood_distribution_sliceB)

  
        if isinstance(js_dist_neighborhood, torch.Tensor):
            M2 = js_dist_neighborhood
            if use_gpu and js_dist_neighborhood.device.type != 'cuda':
                M2 = M2.cuda()
        else:
            M2 = nx.from_numpy(js_dist_neighborhood)

    elif neighborhood_dissimilarity == 'cosine':
        if isinstance(neighborhood_distribution_sliceA, torch.Tensor) or isinstance(neighborhood_distribution_sliceB, torch.Tensor):
            ndA = neighborhood_distribution_sliceA
            ndB = neighborhood_distribution_sliceB
            if not isinstance(ndA, torch.Tensor):
                ndA = torch.from_numpy(np.asarray(ndA))
            if not isinstance(ndB, torch.Tensor):
                ndB = torch.from_numpy(np.asarray(ndB))
            if use_gpu:
                ndA = ndA.cuda()
                ndB = ndB.cuda()
            numerator = ndA @ ndB.T
            denom = ndA.norm(dim=1)[:, None] * ndB.norm(dim=1)[None, :]
            cosine_dist_neighborhood = 1 - numerator / denom
            M2 = cosine_dist_neighborhood
        else:
            ndA = np.asarray(neighborhood_distribution_sliceA)
            ndB = np.asarray(neighborhood_distribution_sliceB)
            numerator = ndA @ ndB.T
            denom = np.linalg.norm(ndA, axis=1)[:, None] * np.linalg.norm(ndB, axis=1)[None, :]
            cosine_dist_neighborhood = 1 - numerator / denom
            M2 = nx.from_numpy(cosine_dist_neighborhood)

    elif neighborhood_dissimilarity == 'msd':
        if isinstance(neighborhood_distribution_sliceA, torch.Tensor):
            ndA = neighborhood_distribution_sliceA.detach().cpu().numpy()
        else:
            ndA = np.asarray(neighborhood_distribution_sliceA)
        if isinstance(neighborhood_distribution_sliceB, torch.Tensor):
            ndB = neighborhood_distribution_sliceB.detach().cpu().numpy()
        else:
            ndB = np.asarray(neighborhood_distribution_sliceB)

        msd_neighborhood = pairwise_msd(ndA, ndB)
        M2 = nx.from_numpy(msd_neighborhood)

    else:
        raise ValueError(
            "Invalid neighborhood_dissimilarity. Expected one of {'jsd','cosine','msd'}; "
            f"got {neighborhood_dissimilarity!r}."
        )
    
    # Compute neighborhood distributions (existing code)
    nd_A = neighborhood_distribution_sliceA.detach().cpu().numpy() if isinstance(neighborhood_distribution_sliceA, torch.Tensor) else np.asarray(neighborhood_distribution_sliceA)
    nd_B = neighborhood_distribution_sliceB.detach().cpu().numpy() if isinstance(neighborhood_distribution_sliceB, torch.Tensor) else np.asarray(neighborhood_distribution_sliceB)
    
    # ─── LAYER 2: Hemisphere identification ───────────────────────────────
    hemisphere, left_mask, right_mask, pc1_B, midline_B = \
        identify_target_hemisphere(sliceA, sliceB, nd_A, nd_B)
    
    logFile.write(f"Hemisphere identification: {hemisphere}\n")
    # logFile.write(f"JSD to L: {jsd_left:.4f}, JSD to R: {jsd_right:.4f}\n")

    # ─── LAYER 1: Shared-scale normalization ──────────────────────────────
    # CRITICAL FIX: replace independent max-norm with shared-scale norm
    # Old (wrong):
    #   D_A /= nx.max(D_A)
    #   D_B /= nx.max(D_B)
    # New (correct):
    scale = nx.max(D_B)   # Normalize BOTH by the scale of the larger structure
    D_A /= scale
    D_B /= scale

    if isinstance(nx,ot.backend.TorchBackend) and use_gpu:
        D_A = D_A.cuda()
        D_B = D_B.cuda()

    # ─── LAYER 3: Spatial coherence cost ──────────────────────────────────
    M_coh = spatial_coherence_cost(sliceB, pc1_B, midline_B, hemisphere, 
                                    lambda_coh=0.5)
    if isinstance(cosine_dist_gene_expr, torch.Tensor):
        M_coh_t = torch.from_numpy(M_coh).float().to(cosine_dist_gene_expr.device)
        M1 = M1 + 0.5 * M_coh_t[None, :]
    else:
        M1 = M1 + 0.5 * M_coh[np.newaxis, :]

    # ─── LAYER 2 continued: hemisphere-aware initialization ───────────────
    G_init_hemi = hemisphere_aware_G0(sliceA, sliceB, hemisphere, 
                                       left_mask, right_mask)
    
    labels, Phi = joint_anatomical_embedding(
    neighborhood_distribution_sliceA, neighborhood_distribution_sliceB)

    a_distribution, b_distribution, shared_clusters = adaptive_marginals(labels, sliceA.shape[0], sliceB.shape[0])

    labels_A = labels[:sliceA.shape[0]]
    labels_B = labels[sliceA.shape[0]:]
    M_clust  = cluster_coherence_cost(labels_A, labels_B)

    # M1 is now: gene_expr + beta*celltype + delta*cluster_coherence
    M1 += delta*M_clust

    if isinstance(nx,ot.backend.TorchBackend) and use_gpu:
        if not isinstance(M1, torch.Tensor):
            M1 = nx.from_numpy(M1)
        if not isinstance(M2, torch.Tensor):
            M2 = nx.from_numpy(M2)
        M1 = M1.cuda()
        M2 = M2.cuda()
    
    # init distributions
    if a_distribution is None:
        # uniform distribution, a = array([1/n, 1/n, ...])
        a = nx.ones((sliceA.shape[0],))/sliceA.shape[0]
    else:
        a = nx.from_numpy(a_distribution)
        
    if b_distribution is None:
        b = nx.ones((sliceB.shape[0],))/sliceB.shape[0]
    else:
        b = nx.from_numpy(b_distribution)

    if isinstance(nx,ot.backend.TorchBackend) and use_gpu:
        a = a.cuda()
        b = b.cuda()
    
    if norm:
        # Heritage PASTE flag: scaled min distance to 1. 
        # Replaced globally by max-normalization [0,1] at distance calculation for stability.
        pass
    
    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    # Run OT (BIOT): MNN anchors -> RANSAC rigid init -> EM with linear OT
    coords_A_np = np.asarray(sliceA.obsm['spatial'].copy(), dtype=np.float64)
    coords_B_np = np.asarray(sliceB.obsm['spatial'].copy(), dtype=np.float64)

    M1_np = _to_np(M1).astype(np.float64)
    M2_np = _to_np(M2).astype(np.float64)
    M_bio = M1_np + gamma * M2_np

    a_np = _to_np(a).astype(np.float64)
    b_np = _to_np(b).astype(np.float64)
    a_np = np.clip(a_np, 0.0, None)
    b_np = np.clip(b_np, 0.0, None)
    a_np = a_np / (np.sum(a_np) + 1e-12)
    b_np = b_np / (np.sum(b_np) + 1e-12)

    k_mnn = int(kwargs.get('k_mnn', 15))
    n_ransac = int(kwargs.get('n_ransac', 500))
    em_iters = int(kwargs.get('em_iters', 5))
    alpha_dev = float(kwargs.get('alpha_dev', alpha))
    alpha_dev = float(np.clip(alpha_dev, 0.0, 1.0))

    anchors = find_mnn_anchors(M_bio, k=k_mnn)
    logFile.write(f"MNN anchors found: {len(anchors)}\n")

    if len(anchors) < 6:
        logFile.write("Warning: too few MNN anchors, using biological OT only\n")
        pi = ot.emd(a_np, b_np, M_bio.astype(np.float64))
    else:
        R, t, inliers = ransac_rigid_transform(coords_A_np, coords_B_np, anchors, n_iter=n_ransac)
        logFile.write(f"RANSAC inliers: {len(inliers)}/{len(anchors)}\n")
        logFile.write(f"Translation t: {t}\n")

        inlier_fraction = len(inliers) / max(len(anchors), 1)
        if inlier_fraction < 0.3:
            # Weak geometric consensus: avoid forcing a potentially wrong spatial transform.
            logFile.write("Low RANSAC confidence: slice A near midline, using biological OT only\n")
            pi = ot.emd(a_np, b_np, M_bio.astype(np.float64))
        else:
            if inlier_fraction < 0.5:
                alpha_dev_adjusted = alpha_dev * (inlier_fraction / 0.5)
                logFile.write(
                    f"Moderate RANSAC confidence: inlier_fraction={inlier_fraction:.4f}, "
                    f"alpha_dev adjusted {alpha_dev:.4f} -> {alpha_dev_adjusted:.4f}\n"
                )
            else:
                alpha_dev_adjusted = alpha_dev

            bio_scale = np.percentile(M_bio, 95)
            if not np.isfinite(bio_scale) or bio_scale <= 0:
                bio_scale = np.max(M_bio) + 1e-8
            M_bio_norm = M_bio / (bio_scale + 1e-8)

            pi = None
            for em_iter in range(max(1, em_iters)):
                M_dev = spatial_deviation_cost(coords_A_np, coords_B_np, R, t)
                dev_scale = np.percentile(M_dev, 95)
                if not np.isfinite(dev_scale) or dev_scale <= 0:
                    dev_scale = np.max(M_dev) + 1e-8
                M_dev_norm = M_dev / (dev_scale + 1e-8)

                M_total = (1.0 - alpha_dev_adjusted) * M_bio_norm + alpha_dev_adjusted * M_dev_norm
                pi = ot.emd(a_np, b_np, M_total.astype(np.float64))

                R_new, t_new = refine_transform_from_plan(pi, coords_A_np, coords_B_np, n_top=min(500, pi.size))
                t_change = np.linalg.norm(t_new - t)
                R_change = np.linalg.norm(R_new - R, ord='fro')
                logFile.write(f"EM iter {em_iter}: |Dt|={t_change:.6f}, |DR|={R_change:.6f}\n")

                R, t = R_new, t_new
                if t_change < 1.0 and R_change < 1e-3:
                    break

            if pi is None:
                pi = ot.emd(a_np, b_np, M_bio.astype(np.float64))

    G_np = np.ones((a.shape[0], b.shape[0]), dtype=np.float64) / (a.shape[0] * b.shape[0])

    if neighborhood_dissimilarity == 'jsd':
        initial_obj_neighbor = np.sum(_to_np(js_dist_neighborhood) * G_np)
    if neighborhood_dissimilarity == 'msd':
        initial_obj_neighbor = np.sum(_to_np(msd_neighborhood) * G_np)
    elif neighborhood_dissimilarity == 'cosine':
        initial_obj_neighbor = np.sum(_to_np(cosine_dist_neighborhood) * G_np)

    initial_obj_gene = np.sum(_to_np(cosine_dist_gene_expr) * G_np)

    if neighborhood_dissimilarity == 'jsd':
        # print(f"Initial objective neighbor (jsd): {initial_obj_neighbor}")
        logFile.write(f"Initial objective neighbor (jsd): {initial_obj_neighbor}\n")

    elif neighborhood_dissimilarity == 'cosine':
        # print(f"Initial objective neighbor (cosine_dist): {initial_obj_neighbor_cos}")
        logFile.write(f"Initial objective neighbor (cosine_dist): {initial_obj_neighbor}\n")
    elif neighborhood_dissimilarity == 'msd':
        # print(f"Initial objective neighbor (msd): {initial_obj_neighbor}")
        logFile.write(f"Initial objective neighbor (mean sq distance): {initial_obj_neighbor}\n")

    # print(f"Initial objective gene expr (cosine_dist): {initial_obj_gene}")
    logFile.write(f"Initial objective (cosine_dist): {initial_obj_gene}\n")
    

    pi = np.asarray(pi, dtype=np.float64)

    if neighborhood_dissimilarity == 'jsd':
        max_indices = np.argmax(pi, axis=1)
        # multiply each value of max_indices from pi_mat with the corresponding js_dist entry
        jsd_error = np.zeros(max_indices.shape)
        _dist_np = _to_np(js_dist_neighborhood)
        for i in range(len(max_indices)):
            jsd_error[i] = pi[i][max_indices[i]] * _dist_np[i][max_indices[i]]

        final_obj_neighbor = np.sum(jsd_error)
    elif neighborhood_dissimilarity == 'msd':
        final_obj_neighbor = np.sum(_to_np(msd_neighborhood)*pi)

    elif neighborhood_dissimilarity == 'cosine':
        max_indices = np.argmax(pi, axis=1)
        # multiply each value of max_indices from pi_mat with the corresponding js_dist entry
        cos_error = np.zeros(max_indices.shape)
        _dist_np = _to_np(cosine_dist_neighborhood)
        for i in range(len(max_indices)):
            cos_error[i] = pi[i][max_indices[i]] * _dist_np[i][max_indices[i]]

        final_obj_neighbor = np.sum(cos_error)


    final_obj_gene = np.sum(_to_np(cosine_dist_gene_expr) * pi)

    if neighborhood_dissimilarity == 'jsd':
        logFile.write(f"Final objective neighbor (jsd): {final_obj_neighbor}\n")
        # print(f"Final objective neighbor (jsd): {final_obj_neighbor}\n")
    elif neighborhood_dissimilarity == 'cosine':
        logFile.write(f"Final objective neighbor (cosine_dist): {final_obj_neighbor}\n")
        # print(f"Final objective neighbor (cosine_dist): {final_obj_neighbor}\n")

    logFile.write(f"Final objective gene expr(cosine_dist): {final_obj_gene}\n")
    # print(f"Final objective (cosine_dist): {final_obj_gene}\n")
    

    logFile.write(f"Runtime: {str(time.time() - start_time)} seconds\n")
    # print(f"Runtime: {str(time.time() - start_time)} seconds\n")
    logFile.write(f"---------------------------------------------\n\n\n")

    logFile.close()

    # new code ends

    if isinstance(backend,ot.backend.TorchBackend) and use_gpu:
        torch.cuda.empty_cache()

    if return_obj:
        return pi, initial_obj_neighbor, initial_obj_gene, final_obj_neighbor, final_obj_gene
    
    return pi


def neighborhood_distribution(curr_slice, radius):
    """
    This method is added by Anup Bhowmik
    Args:
        curr_slice: Slice to get niche distribution for.
        pairwise_distances: Pairwise distances between cells of a slice.
        radius: Radius of the niche.

    Returns:
        niche_distribution: Niche distribution for the slice.
    """

    cell_types = np.array(curr_slice.obs['cell_type_annot'].astype(str))
    unique_cell_types = np.unique(cell_types)
    cell_type_to_index = {ct: i for i, ct in enumerate(unique_cell_types)}
    
    source_coords = curr_slice.obsm['spatial']
    n_cells = curr_slice.shape[0]
    
    cells_within_radius = np.zeros((n_cells, len(unique_cell_types)), dtype=float)

    # Use BallTree instead of full O(n^2) distance matrix for memory & speed scalability
    from sklearn.neighbors import BallTree
    tree = BallTree(source_coords)
    neighbor_lists = tree.query_radius(source_coords, r=radius)

    for i in tqdm(range(n_cells), desc="Computing neighborhood distribution"):
        neighbors = neighbor_lists[i]
        for ind in neighbors:
            ct = cell_types[ind]
            cells_within_radius[i][cell_type_to_index[ct]] += 1
            
    # CRITICAL FIX: Normalize to probability distributions before computing JSD
    row_sums = cells_within_radius.sum(axis=1, keepdims=True)
    # Avoid division by zero for isolated cells
    row_sums[row_sums == 0] = 1 
    cells_within_radius = cells_within_radius / row_sums

    return cells_within_radius


def cosine_distance(sliceA, sliceB, sliceA_name, sliceB_name, filePath, use_rep = None, use_gpu = False, nx = ot.backend.NumpyBackend(), beta = 0.8, overwrite = False):
    from sklearn.metrics.pairwise import cosine_distances
    import os

    A_X, B_X = nx.from_numpy(to_dense_array(extract_data_matrix(sliceA,use_rep))), nx.from_numpy(to_dense_array(extract_data_matrix(sliceB,use_rep)))

    if isinstance(nx,ot.backend.TorchBackend) and use_gpu:
        A_X = A_X.cuda()
        B_X = B_X.cuda()

   
    s_A = A_X + 0.01
    s_B = B_X + 0.01

    fileName = f"{filePath}/cosine_dist_gene_expr_{sliceA_name}_{sliceB_name}.npy"
    
    if os.path.exists(fileName) and not overwrite:
        print("Loading precomputed Cosine distance of gene expression for slice A and slice B")
        cosine_dist_gene_expr = np.load(fileName)
        if use_gpu and isinstance(nx, ot.backend.TorchBackend):
            cosine_dist_gene_expr = torch.from_numpy(cosine_dist_gene_expr).cuda()
    else:
        print("Calculating cosine dist of gene expression for slice A and slice B")

        if isinstance(s_A, torch.Tensor) and isinstance(s_B, torch.Tensor):
            # Calculate manually using PyTorch to stay on GPU
            s_A_norm = s_A / s_A.norm(dim=1)[:, None]
            s_B_norm = s_B / s_B.norm(dim=1)[:, None]
            cosine_dist_gene_expr = 1 - torch.mm(s_A_norm, s_B_norm.T)
            np.save(fileName, cosine_dist_gene_expr.cpu().detach().numpy())
        else:
            from sklearn.metrics.pairwise import cosine_distances
            cosine_dist_gene_expr = cosine_distances(s_A, s_B)
            np.save(fileName, cosine_dist_gene_expr)

    return cosine_dist_gene_expr
