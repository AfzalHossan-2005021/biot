"""
register_incent.py
==================
Robust MERFISH slice alignment for serial sections from the same animal.
Works for any organ, any number of regions, any symmetry.

Pipeline:
  1. Verify scale (NN-distance ratio) — should be ~1.0 for same platform
  2. RANSAC rigid transform from cell-type centroid correspondences
  3. Overlap detection via spatial BallTree query
  4. Local biological matching within registered spatial neighborhoods
  5. EM refinement of translation only
"""

import os, time, datetime, warnings
import numpy as np
from scipy.linalg import svd
from sklearn.neighbors import BallTree
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import jensenshannon
import scipy.sparse as sp


# ════════════════════════════════════════════════════════════════════════════════
# 1.  SCALE VERIFICATION
# ════════════════════════════════════════════════════════════════════════════════

def verify_scale(coords_A, coords_B, k=5, n_sample=3000, tol=0.25, verbose=True):
    """
    Verify that both slices are in the same physical units.
    Uses the median k-th nearest-neighbour distance as a characteristic length.

    Returns scale_ratio = median_NN_B / median_NN_A.
    For same-platform data this should be in [1-tol, 1+tol].
    If not, A's coords are rescaled before registration.

    Parameters
    ----------
    tol : float
        Fractional tolerance. If |ratio - 1| > tol, rescale A.
    """
    def med_nn(coords, k, n_sample):
        n = len(coords)
        if n > n_sample:
            idx = np.random.choice(n, n_sample, replace=False)
            coords = coords[idx]
        tree = BallTree(coords)
        d, _ = tree.query(coords, k=k + 1)   # k+1 because self is included
        return float(np.median(d[:, k]))

    dA = med_nn(coords_A, k, n_sample)
    dB = med_nn(coords_B, k, n_sample)

    if dA < 1e-9:
        return 1.0
    ratio = dB / dA

    if verbose:
        status = "OK" if abs(ratio - 1.0) <= tol else "RESCALING"
        print(f"[Scale] median-NN(A)={dA:.2f}  median-NN(B)={dB:.2f}  "
              f"ratio={ratio:.4f}  [{status}]")

    return float(ratio)


# ════════════════════════════════════════════════════════════════════════════════
# 2.  CELL-TYPE CENTROID LANDMARKS
# ════════════════════════════════════════════════════════════════════════════════

def cell_type_landmarks(coords, cell_types, min_cells=15):
    """
    Compute the spatial centroid of each cell type.

    Returns dict: {cell_type_name -> (2,) centroid array}
    Only cell types with >= min_cells cells are included.
    """
    landmarks = {}
    for ct in np.unique(cell_types):
        mask = cell_types == ct
        if mask.sum() >= min_cells:
            landmarks[ct] = coords[mask].mean(axis=0)
    return landmarks


# ════════════════════════════════════════════════════════════════════════════════
# 3.  RIGID TRANSFORM (SVD, closed form)
# ════════════════════════════════════════════════════════════════════════════════

def fit_rigid_2d(p_src, p_dst):
    """
    Fit rotation R and translation t such that p_dst ≈ R @ p_src.T + t
    from >= 2 point correspondences. Enforces det(R)=+1.

    Parameters
    ----------
    p_src, p_dst : (N, 2) float arrays  N >= 2

    Returns
    -------
    R : (2, 2)  rotation matrix
    t : (2,)    translation vector
    """
    assert len(p_src) >= 2, "Need at least 2 correspondences."
    cA = p_src.mean(0)
    cB = p_dst.mean(0)
    H  = (p_src - cA).T @ (p_dst - cB)
    U, _, Vt = svd(H)
    R = Vt.T @ U.T
    # Fix reflection (det=-1) if SVD produces one
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = cB - R @ cA
    return R, t


def apply_rigid(coords, R, t):
    """Apply rigid transform: out[i] = R @ coords[i] + t."""
    return coords @ R.T + t


# ════════════════════════════════════════════════════════════════════════════════
# 4.  RANSAC OVER CENTROID CORRESPONDENCES
# ════════════════════════════════════════════════════════════════════════════════

def ransac_from_centroids(lm_A, lm_B, shared_ct,
                           n_iter=5000, inlier_thresh=None, verbose=True):
    """
    RANSAC rigid transform from cell-type centroid correspondences.

    lm_A, lm_B   : dicts {ct_name -> (2,) centroid}
    shared_ct     : list of cell type names present in both
    inlier_thresh : distance threshold (physical units); auto if None

    Returns
    -------
    R, t          : best rigid transform  (None, None if failed)
    inlier_mask   : bool array over shared_ct
    meta          : dict with diagnostics
    """
    # Build correspondence arrays — only types present in both landmark sets
    valid = [ct for ct in shared_ct if ct in lm_A and ct in lm_B]
    if len(valid) < 2:
        return None, None, np.array([], dtype=bool), {}

    pA = np.array([lm_A[ct] for ct in valid])   # (K, 2)
    pB = np.array([lm_B[ct] for ct in valid])   # (K, 2)
    K  = len(valid)

    if inlier_thresh is None:
        # Adaptive: 15% of the median pairwise inter-centroid distance in B
        pdB = np.linalg.norm(pB[:, None, :] - pB[None, :, :], axis=-1)
        med = np.median(pdB[pdB > 0])
        inlier_thresh = max(0.15 * float(med), 1.0)

    best_R, best_t     = None, None
    best_inliers_idx   = np.array([], dtype=int)
    best_score         = -np.inf

    rng = np.random.default_rng(42)

    for _ in range(n_iter):
        idx = rng.choice(K, size=min(2, K), replace=False)
        try:
            R_c, t_c = fit_rigid_2d(pA[idx], pB[idx])
        except Exception:
            continue

        res  = np.linalg.norm(apply_rigid(pA, R_c, t_c) - pB, axis=1)
        inl  = np.where(res < inlier_thresh)[0]
        # Score: inlier count, tie-break by sum of inverse residuals
        score = len(inl) + (1.0 / (res[inl].sum() + 1e-9) if len(inl) else 0)

        if score > best_score and len(inl) > 0:
            best_score       = score
            best_inliers_idx = inl
            if len(inl) >= 2:
                best_R, best_t = fit_rigid_2d(pA[inl], pB[inl])
            else:
                best_R, best_t = R_c, t_c

    if best_R is None:
        return None, None, np.zeros(K, dtype=bool), {}

    # Final refit on all inliers
    if len(best_inliers_idx) >= 2:
        best_R, best_t = fit_rigid_2d(pA[best_inliers_idx], pB[best_inliers_idx])

    inlier_mask   = np.zeros(K, dtype=bool)
    inlier_mask[best_inliers_idx] = True
    inlier_frac   = inlier_mask.sum() / K

    angle_deg = float(np.degrees(np.arctan2(best_R[1, 0], best_R[0, 0])))

    meta = dict(
        K=K, n_inliers=int(inlier_mask.sum()),
        inlier_frac=inlier_frac,
        angle_deg=angle_deg,
        inlier_thresh=inlier_thresh,
        valid_ct=valid,
        pA=pA, pB=pB,
    )

    if verbose:
        print(f"    RANSAC: {inlier_mask.sum()}/{K} inliers  "
              f"({100*inlier_frac:.0f}%)  "
              f"angle={angle_deg:.1f}°  thresh={inlier_thresh:.2f}")

    return best_R, best_t, inlier_mask, meta


# ════════════════════════════════════════════════════════════════════════════════
# 5.  FULL REGISTRATION  (4 reflections × RANSAC)
# ════════════════════════════════════════════════════════════════════════════════

REFLECTIONS = [(1, 1), (-1, 1), (1, -1), (-1, -1)]


def register_slices(sliceA, sliceB,
                    min_ct_cells=15,
                    n_ransac=5000,
                    scale_tol=0.25,
                    verbose=True):
    """
    Find the rigid transform (reflection + rotation + translation) that
    brings sliceA's coordinates into sliceB's physical coordinate frame.

    No FFT. No density maps. No grid normalization.
    Operates entirely in physical coordinate units.

    Parameters
    ----------
    sliceA, sliceB : AnnData   must have .obsm['spatial'] and
                               .obs['cell_type_annot']
    min_ct_cells   : int       min cells per cell type to use as landmark
    n_ransac       : int       RANSAC iterations per reflection
    scale_tol      : float     if |scale_ratio-1| > tol, rescale A coords

    Returns
    -------
    coords_A_reg : (n_A, 2)  A's cells in B's physical frame
    coords_B     : (n_B, 2)  B's cells (unchanged)
    transform    : dict       all parameters for inversion/logging
    """
    ct_A   = np.array(sliceA.obs['cell_type_annot'].astype(str))
    ct_B   = np.array(sliceB.obs['cell_type_annot'].astype(str))
    cA_raw = sliceA.obsm['spatial'].copy().astype(float)
    cB_raw = sliceB.obsm['spatial'].copy().astype(float)

    # ── Step 1: Scale verification ───────────────────────────────────────────
    scale = verify_scale(cA_raw, cB_raw, tol=scale_tol, verbose=verbose)
    cA_scaled = cA_raw * scale

    # ── Step 2: Landmarks in B ────────────────────────────────────────────────
    lm_B = cell_type_landmarks(cB_raw, ct_B, min_cells=min_ct_cells)
    shared_ct = sorted(set(np.unique(ct_A)) & set(lm_B.keys()))

    if len(shared_ct) < 3:
        raise ValueError(
            f"Only {len(shared_ct)} shared cell types have >= {min_ct_cells} cells. "
            "Lower min_ct_cells or check that slices share cell types.")

    if verbose:
        print(f"[Reg] {len(shared_ct)} shared cell types  "
              f"|  testing {len(REFLECTIONS)} reflections...")

    # ── Step 3: RANSAC over 4 reflections ────────────────────────────────────
    best = dict(score=-np.inf)

    for flip_x, flip_y in REFLECTIONS:
        cA_flip = cA_scaled * np.array([flip_x, flip_y])
        lm_A    = cell_type_landmarks(cA_flip, ct_A, min_cells=min_ct_cells)

        if verbose:
            print(f"  flip=({flip_x:+d},{flip_y:+d})")

        R, t, inl_mask, meta = ransac_from_centroids(
            lm_A, lm_B, shared_ct, n_iter=n_ransac, verbose=verbose)

        if R is None:
            continue

        score = meta['n_inliers'] + meta['inlier_frac']
        if score > best['score']:
            best = dict(
                flip=(flip_x, flip_y),
                R=R, t=t,
                inlier_mask=inl_mask,
                score=score,
                scale=scale,
                **meta,
            )

    if best['score'] == -np.inf:
        raise RuntimeError(
            "Registration failed for all reflections. "
            "Check that slices share cell types and overlap spatially.")

    if verbose:
        print(f"\n[Reg] ✓  flip={best['flip']}  "
              f"angle={best['angle_deg']:.1f}°  "
              f"scale={best['scale']:.4f}  "
              f"inliers={best['n_inliers']}/{best['K']}  "
              f"({100*best['inlier_frac']:.0f}%)")

    # ── Step 4: Apply best transform to all cells in A ───────────────────────
    flip_x, flip_y = best['flip']
    cA_reg = apply_rigid(cA_raw * best['scale'] * np.array([flip_x, flip_y]),
                         best['R'], best['t'])

    return cA_reg, cB_raw.copy(), best


# ════════════════════════════════════════════════════════════════════════════════
# 6.  OVERLAP DETECTION
# ════════════════════════════════════════════════════════════════════════════════

def detect_overlap(coords_A_reg, coords_B,
                   radius=None, percentile=80, verbose=True):
    """
    Determine which A cells fall within the spatial coverage of B.

    radius : float or None.  If None: 3 × (percentile-th NN dist in B).

    Returns
    -------
    overlap_mask : (n_A,) bool   True = this A cell has B neighbors = overlapping
    radius       : float         the radius used
    """
    if radius is None:
        tree_tmp = BallTree(coords_B)
        d, _     = tree_tmp.query(coords_B, k=2)
        radius   = 3.0 * float(np.percentile(d[:, 1], percentile))

    tree_B       = BallTree(coords_B)
    counts       = tree_B.query_radius(coords_A_reg, r=radius, count_only=True)
    overlap_mask = counts > 0

    if verbose:
        n_ov = overlap_mask.sum()
        n_A  = len(coords_A_reg)
        print(f"[Overlap] radius={radius:.2f}  "
              f"overlapping={n_ov}/{n_A} ({100*n_ov/n_A:.1f}%)")

    return overlap_mask, radius


# ════════════════════════════════════════════════════════════════════════════════
# 7.  LOCAL BIOLOGICAL MATCHING
# ════════════════════════════════════════════════════════════════════════════════

def local_match(sliceA, sliceB,
                coords_A_reg, coords_B,
                nd_A, nd_B,
                cosine_dist,
                overlap_mask,
                radius,
                gamma=0.5,
                soft_temp=0.3,
                verbose=True):
    """
    For each overlapping A cell, assign to the most biologically similar
    B cell within the registered spatial radius.

    Non-overlapping A cells get zero weight (all-zero row in pi).

    soft_temp = 0   → hard argmin
    soft_temp = 1   → proportional to e^{-cost/mean_cost}

    Returns pi : (n_A, n_B) float32
    """
    n_A, n_B = sliceA.shape[0], sliceB.shape[0]
    pi       = np.zeros((n_A, n_B), dtype=np.float32)
    tree_B   = BallTree(coords_B)

    n_matched = 0
    for i in np.where(overlap_mask)[0]:
        cands = tree_B.query_radius([coords_A_reg[i]], r=radius)[0]
        if len(cands) == 0:
            continue

        # Biological cost for cell i vs each candidate j
        cost = np.array([
            (1.0 - gamma) * float(cosine_dist[i, j])
            + gamma * float(jensenshannon(nd_A[i] + 1e-9,
                                          nd_B[j] + 1e-9))
            for j in cands
        ], dtype=np.float64)

        if soft_temp < 1e-6:
            # Hard assignment
            best_j    = cands[np.argmin(cost)]
            pi[i, best_j] = 1.0 / n_A
        else:
            # Soft assignment
            span  = cost.max() - cost.min() + 1e-12
            tau   = soft_temp * span + 1e-12
            logw  = -cost / tau
            logw -= logw.max()
            w     = np.exp(logw)
            w    /= w.sum()
            for m, j in enumerate(cands):
                pi[i, j] = float(w[m]) / n_A

        n_matched += 1

    if verbose:
        print(f"[Match] matched={n_matched}  "
              f"non-overlapping={overlap_mask.size - n_matched}")

    return pi


# ════════════════════════════════════════════════════════════════════════════════
# 8.  EM TRANSLATION REFINEMENT
# ════════════════════════════════════════════════════════════════════════════════

def em_refine_translation(pi, coords_A_reg, coords_B,
                           sliceA, sliceB,
                           nd_A, nd_B, cosine_dist,
                           overlap_mask, radius, gamma,
                           n_iter=5, verbose=True):
    """
    EM loop: re-estimate the residual translation from the current transport
    plan, update coords_A_reg, re-run local matching.  Converges in 3-5 iters.
    """
    coords_reg = coords_A_reg.copy()

    for it in range(n_iter):
        # E-step: extract high-confidence matched pairs from pi
        row_max    = pi.max(axis=1)                    # (n_A,)
        threshold  = np.percentile(row_max[row_max > 0], 75)
        conf_mask  = (row_max >= threshold) & overlap_mask

        if conf_mask.sum() < 10:
            break

        matched_j  = pi[conf_mask].argmax(axis=1)     # best B cell per A cell
        pA         = coords_reg[conf_mask]
        pB         = coords_B[matched_j]
        w          = row_max[conf_mask]
        w         /= w.sum()

        # Weighted residual translation
        delta_t    = float((w * (pB[:, 0] - pA[:, 0])).sum()), \
                     float((w * (pB[:, 1] - pA[:, 1])).sum())
        delta_norm = float(np.linalg.norm(delta_t))

        coords_reg = coords_reg + np.array(delta_t)

        if verbose:
            print(f"[EM {it+1}] Δt=({delta_t[0]:.3f}, {delta_t[1]:.3f})  "
                  f"|Δt|={delta_norm:.4f}")

        if delta_norm < 0.5:      # converged (in physical units)
            break

        # M-step: re-run local matching with updated coords
        pi = local_match(sliceA, sliceB, coords_reg, coords_B,
                         nd_A, nd_B, cosine_dist, overlap_mask,
                         radius, gamma, soft_temp=0.3, verbose=False)

    return pi, coords_reg


# ════════════════════════════════════════════════════════════════════════════════
# 9.  NEIGHBOURHOOD DISTRIBUTION  (unchanged from INCENT)
# ════════════════════════════════════════════════════════════════════════════════

def neighborhood_distribution(curr_slice, radius):
    from tqdm import tqdm
    cell_types  = np.array(curr_slice.obs['cell_type_annot'].astype(str))
    unique_ct   = np.unique(cell_types)
    ct2idx      = {c: i for i, c in enumerate(unique_ct)}
    coords      = curr_slice.obsm['spatial']
    n           = curr_slice.shape[0]
    tree        = BallTree(coords)
    nbrs        = tree.query_radius(coords, r=radius)
    dist        = np.zeros((n, len(unique_ct)), dtype=np.float32)
    for i in tqdm(range(n), desc="Niche"):
        for idx in nbrs[i]:
            dist[i, ct2idx[cell_types[idx]]] += 1.0
    rs = dist.sum(1, keepdims=True)
    rs[rs == 0] = 1.0
    return dist / rs


# ════════════════════════════════════════════════════════════════════════════════
# 10. MASTER FUNCTION
# ════════════════════════════════════════════════════════════════════════════════

def pairwise_align(sliceA, sliceB,
                   gamma          = 0.5,
                   radius         = None,
                   min_ct_cells   = 15,
                   n_ransac       = 5000,
                   search_radius  = None,
                   soft_temp      = 0.3,
                   em_iters       = 5,
                   filePath       = './output',
                   use_rep        = None,
                   sliceA_name    = 'A',
                   sliceB_name    = 'B',
                   return_extra   = False,
                   verbose        = True,
                   **kwargs):
    """
    Register and align two MERFISH slices from the same animal.

    Parameters
    ----------
    gamma         : weight of niche-JSD vs gene-cosine. 0 = genes only.
    radius        : neighbourhood radius for niche distribution
                    (physical units). If None: 3× median NN distance in A.
    min_ct_cells  : min cells per type to use as registration landmark.
    n_ransac      : RANSAC iterations per reflection candidate.
    search_radius : spatial matching radius (physical units).
                    If None: auto from B's NN distances.
    soft_temp     : matching softness [0=hard, 1=soft]. 0.3 works well.
    em_iters      : EM translation refinement iterations.
    return_extra  : if True return (pi, coords_A_reg, coords_B, transform).

    Returns
    -------
    pi            : (n_A, n_B) float32  transport plan
    """
    os.makedirs(filePath, exist_ok=True)
    log_p = f"{filePath}/{sliceA_name}_to_{sliceB_name}.log"
    log   = open(log_p, 'w')
    log.write(f"pairwise_align  {sliceA_name} → {sliceB_name}\n")
    log.write(f"Started: {datetime.datetime.now()}\n\n")
    t0 = time.time()

    # ── 0.  Shared genes and cell types ─────────────────────────────────────
    shared_genes = sliceA.var_names.intersection(sliceB.var_names)
    if not len(shared_genes):
        raise ValueError("No shared genes.")
    sA = sliceA[:, shared_genes].copy()
    sB = sliceB[:, shared_genes].copy()

    shared_ct = set(sA.obs['cell_type_annot']) & set(sB.obs['cell_type_annot'])
    if not shared_ct:
        raise ValueError("No shared cell types.")
    sA = sA[sA.obs['cell_type_annot'].isin(shared_ct)].copy()
    sB = sB[sB.obs['cell_type_annot'].isin(shared_ct)].copy()

    n_A, n_B = sA.shape[0], sB.shape[0]
    log.write(f"n_A={n_A}  n_B={n_B}  "
              f"shared_genes={len(shared_genes)}  shared_ct={len(shared_ct)}\n\n")

    if verbose:
        print(f"[Align] {sliceA_name}({n_A}) → {sliceB_name}({n_B})  "
              f"genes={len(shared_genes)}  cell_types={len(shared_ct)}")

    # ── 1.  Auto-estimate niche radius from A's NN distances ─────────────────
    if radius is None:
        tree_tmp = BallTree(sA.obsm['spatial'])
        d, _     = tree_tmp.query(sA.obsm['spatial'], k=6)
        radius   = float(np.median(d[:, 5]) * 5)
        if verbose:
            print(f"[Align] Auto niche radius: {radius:.2f}")

    # ── 2.  Biological cost matrices ─────────────────────────────────────────
    def dense(X):
        return X.toarray() if sp.issparse(X) else np.asarray(X)

    aX = dense(sA.X if use_rep is None else sA.obsm[use_rep]).astype(np.float32) + 0.01
    bX = dense(sB.X if use_rep is None else sB.obsm[use_rep]).astype(np.float32) + 0.01
    cos_dist = cosine_distances(aX, bX).astype(np.float32)

    nd_A = neighborhood_distribution(sA, radius).astype(np.float32) + 0.01
    nd_B = neighborhood_distribution(sB, radius).astype(np.float32) + 0.01

    t1 = time.time()
    log.write(f"Biological costs: {t1-t0:.1f}s\n")

    # ── 3.  Spatial registration ──────────────────────────────────────────────
    if verbose:
        print(f"[Align] Registering coordinate frames...")

    coords_A_reg, coords_B, transform = register_slices(
        sA, sB,
        min_ct_cells=min_ct_cells,
        n_ransac=n_ransac,
        verbose=verbose)

    t2 = time.time()
    log.write(f"Registration: flip={transform['flip']}  "
              f"angle={transform['angle_deg']:.2f}°  "
              f"scale={transform['scale']:.4f}  "
              f"inliers={transform['n_inliers']}/{transform['K']}  "
              f"time={t2-t1:.1f}s\n")

    # ── 4.  Overlap detection ─────────────────────────────────────────────────
    if search_radius is None:
        # Auto: 4× median NN distance in B
        tree_tmp2  = BallTree(coords_B)
        d2, _      = tree_tmp2.query(coords_B, k=2)
        search_radius = float(np.median(d2[:, 1]) * 4.0)
        if verbose:
            print(f"[Align] Auto search radius: {search_radius:.2f}")

    overlap_mask, sr = detect_overlap(coords_A_reg, coords_B,
                                       radius=search_radius, verbose=verbose)

    # ── 5.  Initial local biological matching ─────────────────────────────────
    if verbose:
        print(f"[Align] Local biological matching (γ={gamma})...")

    pi = local_match(sA, sB, coords_A_reg, coords_B,
                     nd_A, nd_B, cos_dist,
                     overlap_mask, search_radius,
                     gamma=gamma, soft_temp=soft_temp, verbose=verbose)

    t3 = time.time()

    # ── 6.  EM translation refinement ────────────────────────────────────────
    if verbose:
        print(f"[Align] EM refinement ({em_iters} iters)...")

    pi, coords_A_reg = em_refine_translation(
        pi, coords_A_reg, coords_B,
        sA, sB, nd_A, nd_B, cos_dist,
        overlap_mask, search_radius, gamma,
        n_iter=em_iters, verbose=verbose)

    t4 = time.time()
    log.write(f"Matching+EM: {t4-t3:.1f}s\n")
    log.write(f"Total: {t4-t0:.1f}s\n")
    log.close()

    if verbose:
        print(f"[Align] Done in {t4-t0:.1f}s  log→ {log_p}")

    if return_extra:
        return pi, coords_A_reg, coords_B, transform
    return pi


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def visualize_alignment(pi, sliceA, sliceB,
                         coords_A_reg, coords_B,
                         transform,
                         top_k=400,
                         save_path=None):
    """
    Four-panel alignment visualization.

    Panel 1: A raw coords    (shows original frame)
    Panel 2: B raw coords    (shows original frame)
    Panel 3: A registered into B's frame, overlaid on B
             — THIS is the alignment quality panel
    Panel 4: Registration diagnostic — centroid inliers/outliers
    """
    ct_B     = np.array(sliceB.obs['cell_type_annot'].values)
    ct_A     = np.array(sliceA.obs['cell_type_annot'].values)
    raw_A    = sliceA.obsm['spatial']
    raw_B    = sliceB.obsm['spatial']

    # Shared cell types for colour coding
    uniq    = np.unique(ct_B)
    ct2idx  = {c: i for i, c in enumerate(uniq)}
    cmap20  = plt.get_cmap('tab20', len(uniq))

    # Top-k pairs by weight
    flat    = pi.flatten()
    top_idx = np.argsort(flat)[-top_k:]
    rows    = top_idx // pi.shape[1]
    cols    = top_idx %  pi.shape[1]
    w_norm  = flat[top_idx] / (flat[top_idx].max() + 1e-12)

    fig = plt.figure(figsize=(18, 8))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

    # ── Panel 1: A raw ───────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0])
    colors_A_raw = [cmap20(ct2idx.get(c, 0)) if c in ct2idx else (0.5,0.5,0.5,1)
                    for c in ct_A]
    ax.scatter(*raw_A.T, s=1, c=colors_A_raw, alpha=0.6, rasterized=True)
    ax.set_title('Slice A\n(raw frame)', fontsize=10)
    ax.set_aspect('equal'); ax.axis('off')

    # ── Panel 2: B raw ───────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1])
    colors_B = [cmap20(ct2idx[c]) for c in ct_B]
    ax.scatter(*raw_B.T, s=1, c=colors_B, alpha=0.4, rasterized=True)
    ax.set_title('Slice B\n(raw frame)', fontsize=10)
    ax.set_aspect('equal'); ax.axis('off')

    # ── Panel 3: A registered on top of B ────────────────────────────────────
    ax = fig.add_subplot(gs[2])
    # B as background
    ax.scatter(*coords_B.T, s=1, c='#c8c8c8', alpha=0.25,
               rasterized=True, label='B')
    # A in B's frame, coloured by cell type
    colors_A_reg = [cmap20(ct2idx.get(c, 0)) if c in ct2idx else (0.5,0.5,0.5,1)
                    for c in ct_A]
    ax.scatter(*coords_A_reg.T, s=3, c=colors_A_reg, alpha=0.8,
               rasterized=True, label='A (registered)')
    # Correspondence lines
    for idx in range(len(rows)):
        i, j = rows[idx], cols[idx]
        ax.plot([coords_A_reg[i, 0], coords_B[j, 0]],
                [coords_A_reg[i, 1], coords_B[j, 1]],
                'k-', alpha=float(w_norm[idx]) * 0.5, lw=0.4)
    ax.set_title('Alignment\n(A registered into B frame)', fontsize=10)
    ax.set_aspect('equal'); ax.axis('off')

    # NOTE: if A is a partial slice, it will correctly appear
    # as a subregion of B — this is expected, not a bug.

    # ── Panel 4: Registration diagnostic ─────────────────────────────────────
    ax = fig.add_subplot(gs[3])
    pA      = transform.get('pA')    # centroid positions in A (flipped+scaled frame)
    pB      = transform.get('pB')    # centroid positions in B frame
    inl     = transform.get('inlier_mask', np.ones(len(pA), dtype=bool)
                             if pA is not None else None)

    if pA is not None and pB is not None:
        # Apply the found R,t to pA to bring into B frame
        R, t = transform['R'], transform['t']
        flip = np.array(transform['flip'])
        # pA is already in the "flip+scaled" frame that RANSAC used
        pA_B = apply_rigid(pA, R, t)

        ax.scatter(*pB[inl].T,   s=40, c='#1D9E75', marker='o',
                   zorder=3, label='inlier centroid (B)')
        ax.scatter(*pA_B[inl].T, s=40, c='#534AB7', marker='^',
                   zorder=3, label='inlier centroid (A→B)')
        for k in np.where(inl)[0]:
            ax.plot([pA_B[k,0], pB[k,0]], [pA_B[k,1], pB[k,1]],
                    '#1D9E75', lw=0.8, alpha=0.7)

        if (~inl).any():
            ax.scatter(*pB[~inl].T,   s=40, c='#EF9F27', marker='o',
                       alpha=0.4, label='outlier (B)')
            ax.scatter(*pA_B[~inl].T, s=40, c='#D85A30', marker='^',
                       alpha=0.4, label='outlier (A→B)')

        ax.set_title(f'Registration diagnostic\n'
                     f'{inl.sum()}/{len(inl)} inlier cell-type centroids\n'
                     f'angle={transform["angle_deg"]:.1f}°  '
                     f'flip={transform["flip"]}', fontsize=9)
        ax.legend(fontsize=7, markerscale=1.5)
        ax.set_aspect('equal'); ax.axis('off')
    else:
        ax.text(0.5, 0.5, 'No diagnostic\navailable',
                ha='center', va='center', transform=ax.transAxes)

    plt.suptitle(
        'INCENT-Register alignment\n'
        'Panel 3 key: A cells coloured by cell type, overlaid on B (grey).\n'
        'If partial overlap: A correctly appears as a SUBREGION of B.',
        fontsize=9, y=1.01)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

