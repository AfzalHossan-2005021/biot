"""
sff_incent.py  —  SFF-INCENT: Spatial Frequency Fingerprinting for MERFISH alignment
=======================================================================================
Drop-in replacement for pairwise_align().

Key design decisions:
  1. No GW, no FGW, no raw-coordinate distance matrices.
  2. Registration is solved first, at population level, via image registration.
  3. Cell matching is solved second, locally, using biology.
  4. General: works for any organ, any number of regions, any symmetry.
"""

import os, time, datetime
import numpy as np
from scipy.fft          import fft2, ifft2, fftshift
from scipy.ndimage      import gaussian_filter, rotate as ndrotate, zoom as ndzoom, map_coordinates
from sklearn.neighbors  import BallTree
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance   import jensenshannon


# ═══════════════════════════════════════════════════════════════════════════════
# Module 1 — Density map construction
# ═══════════════════════════════════════════════════════════════════════════════

def build_density_map(coords, cell_types, grid_size=128, sigma_px=4):
    """
    Convert a cell point cloud into a multi-channel 2D KDE density image.

    Parameters
    ----------
    coords     : (n, 2) float   raw spatial coordinates (any scale/frame)
    cell_types : (n,)   str     cell type label per cell
    grid_size  : int            resolution H=W of the output grid
    sigma_px   : float          Gaussian KDE bandwidth in grid pixels

    Returns
    -------
    img     : (K, H, W) float32    K = number of unique cell types
    ct_names: (K,)      ndarray    cell type names (channel labels)
    affine  : dict                 maps grid pixels ↔ original coords
                                   keys: mn, max_span, scale, offset_x, offset_y
    """
    ct_names = np.unique(cell_types)
    K = len(ct_names)
    ct_idx = {c: i for i, c in enumerate(ct_names)}

    mn   = coords.min(axis=0)
    span = coords.max(axis=0) - mn
    max_span = span.max()
    if max_span < 1e-9:
        max_span = 1.0

    # Keep aspect ratio: fit longest axis to (grid_size - 2*margin) pixels
    margin = 4
    scale = (grid_size - 2 * margin) / max_span
    # Center the tissue inside the grid
    offset = margin + (grid_size - 2*margin - span * scale) / 2.0

    affine = dict(mn=mn, max_span=max_span, scale=scale,
                  offset=offset, grid_size=grid_size)

    img = np.zeros((K, grid_size, grid_size), dtype=np.float32)
    for i in range(len(coords)):
        # Map physical coord → grid pixel
        gx = int(np.clip(round((coords[i, 0] - mn[0]) * scale + offset[0]),
                         0, grid_size - 1))
        gy = int(np.clip(round((coords[i, 1] - mn[1]) * scale + offset[1]),
                         0, grid_size - 1))
        img[ct_idx[cell_types[i]], gy, gx] += 1.0

    for k in range(K):
        img[k] = gaussian_filter(img[k].astype(np.float64),
                                 sigma=sigma_px).astype(np.float32)
    return img, ct_names, affine


def grid_to_phys(gx, gy, affine):
    """Convert grid pixel coordinates back to physical coordinates."""
    x = (gx - affine['offset'][0]) / affine['scale'] + affine['mn'][0]
    y = (gy - affine['offset'][1]) / affine['scale'] + affine['mn'][1]
    return x, y


def phys_to_grid(coords, affine):
    """Convert physical coordinates to grid pixel coordinates."""
    gx = (coords[:, 0] - affine['mn'][0]) * affine['scale'] + affine['offset'][0]
    gy = (coords[:, 1] - affine['mn'][1]) * affine['scale'] + affine['offset'][1]
    return np.stack([gx, gy], axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Module 2 — Phase correlation primitives
# ═══════════════════════════════════════════════════════════════════════════════

def _phase_corr_2d(FA, FB):
    """
    Normalized cross-power spectrum between two FFTs.
    Returns the real-part IFFT (phase correlation map), fftshift-centered.
    Peak location gives the integer pixel shift.
    """
    cross = np.conj(FA) * FB
    denom = np.abs(cross) + 1e-12
    return np.real(ifft2(cross / denom))


def phase_correlate_multichannel(imgA, imgB, weights=None):
    """
    Multi-channel phase correlation.

    imgA, imgB : (K, H, W) float arrays (same K, H, W)
    weights    : (K,) channel weights; default = sqrt of signal product

    Returns
    -------
    tx, ty    : integer pixel offsets  (B = shifted version of A by (tx, ty))
    peak_val  : float   confidence (higher = better)
    corr_map  : (H, W)  full correlation map for diagnostics
    """
    K, H, W = imgA.shape
    corr_acc = np.zeros((H, W), dtype=np.float64)

    for k in range(K):
        FA = fft2(imgA[k].astype(np.float64))
        FB = fft2(imgB[k].astype(np.float64))
        w  = float(np.sqrt(imgA[k].sum() * imgB[k].sum()) + 1e-12) \
             if weights is None else float(weights[k])
        corr_acc += w * _phase_corr_2d(FA, FB)

    corr_map  = fftshift(corr_acc)
    fy, fx    = np.unravel_index(np.argmax(corr_map), corr_map.shape)
    peak_val  = corr_map[fy, fx]

    # Convert fftshift-centered indices to signed offsets
    ty = fy - H // 2
    tx = fx - W // 2

    return int(tx), int(ty), float(peak_val), corr_map


def _log_polar(mag, num_angles=360, num_radii=None):
    """
    Log-polar transform of a magnitude spectrum image (already fftshifted).
    Used for Fourier-Mellin rotation/scale estimation.
    """
    H, W     = mag.shape
    num_radii = num_radii or (min(H, W) // 2)
    cy, cx   = H / 2.0, W / 2.0
    max_r    = min(cx, cy) * 0.7   # avoid wrap-around

    theta = np.linspace(0, np.pi, num_angles, endpoint=False)
    log_r = np.linspace(0, np.log(max_r + 1e-9), num_radii)
    r     = np.exp(log_r)

    R, Th = np.meshgrid(r, theta, indexing='ij')     # (num_radii, num_angles)
    xs    = cx + R * np.cos(Th)
    ys    = cy + R * np.sin(Th)

    return map_coordinates(mag, [ys.ravel(), xs.ravel()],
                           order=1, mode='constant', cval=0.0
                           ).reshape(num_radii, num_angles).astype(np.float32)


def fourier_mellin_rotation_scale(imgA_sum, imgB_sum, num_angles=360):
    """
    Estimate rotation and scale between two single-channel images via
    Fourier-Mellin (log-polar phase correlation on magnitude spectra).

    Returns
    -------
    angle_deg   : float  rotation of A relative to B (degrees)
    scale_ratio : float  scale of A relative to B
    confidence  : float
    """
    H, W    = imgA_sum.shape
    window  = np.outer(np.hanning(H), np.hanning(W))

    FA_mag  = np.abs(fftshift(fft2(imgA_sum * window))) + 1e-12
    FB_mag  = np.abs(fftshift(fft2(imgB_sum * window))) + 1e-12

    lpA     = _log_polar(FA_mag, num_angles=num_angles)
    lpB     = _log_polar(FB_mag, num_angles=num_angles)

    # Phase correlate: tx → log-radial (scale), ty → angular (rotation)
    tx, ty, conf, _ = phase_correlate_multichannel(
        lpA[None], lpB[None], weights=None)

    num_radii = lpA.shape[0]
    max_r     = np.log(min(H, W) / 2 * 0.7 + 1e-9)
    scale_ratio = float(np.exp(tx * max_r / max(num_radii, 1)))
    angle_deg   = float(ty * 180.0 / num_angles)

    return angle_deg, scale_ratio, conf


# ═══════════════════════════════════════════════════════════════════════════════
# Module 3 — Image-space transform helpers
# ═══════════════════════════════════════════════════════════════════════════════

def apply_rotation_stack(img, angle_deg):
    """Rotate (K,H,W) stack by angle_deg (degrees), keeping image size."""
    K = img.shape[0]
    out = np.zeros_like(img)
    for k in range(K):
        out[k] = ndrotate(img[k], angle=angle_deg, reshape=False,
                          mode='constant', cval=0.0)
    return out


def apply_scale_stack(img, scale_ratio):
    """Scale (K,H,W) stack by scale_ratio, centred, output same H×W."""
    if abs(scale_ratio - 1.0) < 0.005:
        return img.copy()
    K, H, W = img.shape
    out = np.zeros_like(img)
    for k in range(K):
        z = ndzoom(img[k], scale_ratio, order=1)
        sh, sw = z.shape
        if sh >= H and sw >= W:
            y0 = (sh - H) // 2; x0 = (sw - W) // 2
            out[k] = z[y0:y0+H, x0:x0+W]
        else:
            y0 = (H - sh) // 2; x0 = (W - sw) // 2
            out[k, y0:y0+sh, x0:x0+sw] = z
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Module 4 — Full transform search over 4 reflections
# ═══════════════════════════════════════════════════════════════════════════════

REFLECTIONS = [(1, 1), (-1, 1), (1, -1), (-1, -1)]


def find_transform(sliceA, sliceB, grid_size=128, sigma_px=4,
                   num_angles=360, verbose=True):
    """
    Find the full rigid transform (reflection, rotation, scale, translation)
    that registers sliceA into sliceB's coordinate frame.

    Uses multi-channel cell-type density maps + Fourier-Mellin transform.
    No biological cost matrices used here — pure spatial frequency matching.

    Returns
    -------
    best      : dict   all transform parameters + confidence
    coords_A_reg : (n_A, 2)  A's coords in B's physical frame
    coords_B     : (n_B, 2)  B's coords (unchanged)
    """
    ct_A   = np.array(sliceA.obs['cell_type_annot'].astype(str))
    ct_B   = np.array(sliceB.obs['cell_type_annot'].astype(str))
    cA_raw = sliceA.obsm['spatial'].copy().astype(float)
    cB_raw = sliceB.obsm['spatial'].copy().astype(float)

    # Build B density map once (reference frame)
    imgB, ct_B_names, affB = build_density_map(
        cB_raw, ct_B, grid_size=grid_size, sigma_px=sigma_px)
    imgB_sum = imgB.sum(axis=0)

    best = dict(confidence=-np.inf)

    for flip_x, flip_y in REFLECTIONS:
        cA_flip = cA_raw * np.array([flip_x, flip_y])

        imgA, ct_A_names, affA = build_density_map(
            cA_flip, ct_A, grid_size=grid_size, sigma_px=sigma_px)
        imgA_sum = imgA.sum(axis=0)

        # Shared cell types
        shared_ct = sorted(set(ct_A_names) & set(ct_B_names))
        if not shared_ct:
            continue
        idxA = [np.where(ct_A_names == c)[0][0] for c in shared_ct]
        idxB = [np.where(ct_B_names == c)[0][0] for c in shared_ct]
        imgA_sh = imgA[idxA]
        imgB_sh = imgB[idxB]

        # Step 1: Fourier-Mellin → rotation + scale
        angle, scale, rs_conf = fourier_mellin_rotation_scale(
            imgA_sum, imgB_sum, num_angles=num_angles)

        # Fourier-Mellin has a 180° ambiguity — test both
        for angle_candidate in [angle, angle + 180.0]:
            # Apply rotation and scale to A's density map
            imgA_rs = apply_rotation_stack(imgA_sh, angle_candidate)
            imgA_rs = apply_scale_stack(imgA_rs, scale)

            # Step 2: Phase correlation → translation
            tx, ty, peak_val, corr_map = phase_correlate_multichannel(
                imgA_rs, imgB_sh)

            if verbose:
                print(f"  flip=({flip_x:+d},{flip_y:+d})  "
                      f"angle={angle_candidate:+7.1f}°  "
                      f"scale={scale:.3f}  "
                      f"tx={tx:+4d}  ty={ty:+4d}  "
                      f"peak={peak_val:.4f}")

            if peak_val > best['confidence']:
                best = dict(
                    flip=(flip_x, flip_y),
                    angle=angle_candidate,
                    scale=scale,
                    tx_px=tx, ty_px=ty,
                    confidence=peak_val,
                    affA=affA, affB=affB,
                )

    if best['confidence'] == -np.inf:
        raise RuntimeError("SFF registration failed: no shared cell types found.")

    if verbose:
        print(f"\n>>> Best: flip={best['flip']}  "
              f"angle={best['angle']:.1f}°  scale={best['scale']:.3f}  "
              f"tx={best['tx_px']}  ty={best['ty_px']}  "
              f"conf={best['confidence']:.4f}\n")

    coords_A_reg = _apply_transform_to_coords(cA_raw, best)
    return best, coords_A_reg, cB_raw.copy()


def _apply_transform_to_coords(coords_A_raw, transform):
    """
    Apply the discovered image-space transform to physical cell coordinates.

    The forward transform in grid space is:
        1. Reflect:   cA_flip  = cA * (flip_x, flip_y)
        2. To grid A: g = (cA_flip - mn_A) * scale_A + offset_A
        3. Rotate:    g_rot = R(angle) @ (g - center) + center
        4. Scale:     g_sc  = (g_rot - center) * scale_ratio + center
        5. Translate: g_t   = g_sc + (tx_px, ty_px)
        6. To phys B: cB    = (g_t - offset_B) / scale_B + mn_B
    """
    flip_x, flip_y   = transform['flip']
    angle_rad        = np.deg2rad(transform['angle'])
    scale_ratio      = transform['scale']
    tx, ty           = transform['tx_px'], transform['ty_px']
    affA             = transform['affA']
    affB             = transform['affB']
    G                = affA['grid_size']
    cx = cy          = G / 2.0

    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]])

    # Step 1: reflect
    cA = coords_A_raw * np.array([flip_x, flip_y])

    # Step 2: to grid-A pixels
    gA = ((cA - affA['mn']) * affA['scale']
          + affA['offset'])                       # (n,2) in [0, G-1]

    # Step 3: rotate around grid centre
    gA_c     = gA - np.array([cx, cy])
    gA_rot   = gA_c @ R.T + np.array([cx, cy])   # R.T because coords are row vectors

    # Step 4: scale around grid centre
    gA_sc    = (gA_rot - np.array([cx, cy])) * scale_ratio + np.array([cx, cy])

    # Step 5: translate
    gA_t     = gA_sc + np.array([tx, ty])

    # Step 6: to B physical coords
    coords_A_reg = (gA_t - affB['offset']) / affB['scale'] + affB['mn']
    return coords_A_reg


# ═══════════════════════════════════════════════════════════════════════════════
# Module 5 — Overlap detection + local biological matching
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_search_radius(coords_B, percentile=75):
    """
    Estimate a good search radius from B's cell density.
    Use the `percentile`-th nearest-neighbor distance × 3.
    Gives a radius that typically contains 5–15 cells.
    """
    tree = BallTree(coords_B)
    dists, _ = tree.query(coords_B, k=2)     # k=2: first is self (dist=0)
    nn_dists = dists[:, 1]
    return float(np.percentile(nn_dists, percentile) * 3.0)


def local_biological_matching(sliceA, sliceB,
                               coords_A_reg, coords_B,
                               nd_A, nd_B,
                               cosine_dist_AB,
                               gamma=0.5,
                               search_radius=None,
                               soft_temp=0.5,
                               verbose=True):
    """
    Match each cell i in A to cells in B within a spatial radius,
    using a biological cost = (1-gamma)*cosine_gene + gamma*JSD_niche.

    Non-overlapping cells (no B neighbors within radius) are left unmatched.

    soft_temp : softmax temperature (as fraction of local cost range).
                0 → hard argmin,  1 → fully soft.  Default 0.5 works well.

    Returns
    -------
    pi          : (n_A, n_B) float32   transport plan (rows sum to 1/n_A for matched cells)
    unmatched   : list[int]            indices of A cells with no B neighbor in radius
    """
    n_A, n_B = sliceA.shape[0], sliceB.shape[0]

    if search_radius is None:
        search_radius = estimate_search_radius(coords_B)
        if verbose:
            print(f"  Auto search radius: {search_radius:.2f} (same units as coords_B)")

    tree_B = BallTree(coords_B)
    pi     = np.zeros((n_A, n_B), dtype=np.float32)
    unmatched = []

    for i in range(n_A):
        cands = tree_B.query_radius([coords_A_reg[i]], r=search_radius)[0]

        if len(cands) == 0:
            unmatched.append(i)
            continue

        # Biological cost for cell i vs each candidate j
        bio = np.array([
            (1.0 - gamma) * float(cosine_dist_AB[i, j])
            + gamma * float(jensenshannon(nd_A[i] + 1e-9, nd_B[j] + 1e-9))
            for j in cands
        ])

        # Soft assignment via temperature-scaled softmax
        span = bio.max() - bio.min() + 1e-12
        tau  = soft_temp * span + 1e-12
        logw = -bio / tau
        logw -= logw.max()
        w    = np.exp(logw)
        w   /= w.sum()

        for m, j in enumerate(cands):
            pi[i, j] = float(w[m]) / n_A     # normalise by n_A for global plan

    if verbose:
        n_matched = n_A - len(unmatched)
        print(f"  Matched: {n_matched}/{n_A} "
              f"({100*n_matched/n_A:.1f}%)   "
              f"Non-overlapping: {len(unmatched)}")

    return pi, unmatched


# ═══════════════════════════════════════════════════════════════════════════════
# Module 6 — Neighbourhood distribution (reused from INCENT)
# ═══════════════════════════════════════════════════════════════════════════════

def neighborhood_distribution(curr_slice, radius):
    """Exact copy of INCENT's neighbourhood_distribution. No changes needed."""
    from tqdm import tqdm
    cell_types = np.array(curr_slice.obs['cell_type_annot'].astype(str))
    unique_ct  = np.unique(cell_types)
    ct2idx     = {c: i for i, c in enumerate(unique_ct)}
    coords     = curr_slice.obsm['spatial']
    n          = curr_slice.shape[0]

    tree = BallTree(coords)
    nbrs = tree.query_radius(coords, r=radius)

    dist = np.zeros((n, len(unique_ct)), dtype=float)
    for i in tqdm(range(n), desc="Niche distribution"):
        for idx in nbrs[i]:
            dist[i, ct2idx[cell_types[idx]]] += 1.0

    row_sums = dist.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return dist / row_sums


# ═══════════════════════════════════════════════════════════════════════════════
# Module 7 — Master pairwise_align replacement
# ═══════════════════════════════════════════════════════════════════════════════

def pairwise_align(sliceA, sliceB,
                   # biological parameters
                   gamma       = 0.5,
                   radius      = 100,
                   # registration parameters
                   grid_size   = 128,
                   sigma_px    = 4,
                   num_angles  = 360,
                   # matching parameters
                   search_radius = None,
                   soft_temp     = 0.5,
                   # bookkeeping
                   filePath     = './sff_output',
                   use_rep      = None,
                   sliceA_name  = 'A',
                   sliceB_name  = 'B',
                   return_extra = False,
                   verbose      = True,
                   **kwargs):
    """
    SFF-INCENT  —  drop-in replacement for INCENT's pairwise_align().

    Parameters
    ----------
    sliceA, sliceB : AnnData
        Must have .obsm['spatial'], .obs['cell_type_annot'], .X gene expression.
    gamma          : float [0,1]
        Weight of niche-JSD vs gene-cosine in biological cost. 0 = genes only.
    radius         : float
        Neighbourhood radius for niche distribution (same unit as spatial coords).
    grid_size      : int
        Density map resolution. 128 is good for ~15k cells. Use 64 for speed.
    sigma_px       : float
        KDE bandwidth in grid pixels. Larger = smoother, more robust registration.
    search_radius  : float or None
        Spatial search radius for cell matching (physical units of B).
        If None: auto-estimated as 3 × 75th-percentile NN-distance in B.
    soft_temp      : float [0,1]
        Softmax temperature for local assignment. 0 = hard argmin, 1 = soft.
    return_extra   : bool
        If True, return (pi, coords_A_reg, coords_B, transform_dict).

    Returns
    -------
    pi             : (n_A, n_B) ndarray   transport plan
    [optional]     : coords_A_reg, coords_B, transform
    """
    os.makedirs(filePath, exist_ok=True)
    log_path = f"{filePath}/sff_{sliceA_name}_{sliceB_name}.log"
    log = open(log_path, 'w')
    log.write(f"SFF-INCENT  |  {sliceA_name} → {sliceB_name}\n")
    log.write(f"Started: {datetime.datetime.now()}\n\n")
    t0 = time.time()

    # ── 0. Pre-processing: shared genes and cell types (INCENT standard) ─────
    shared_genes = sliceA.var_names.intersection(sliceB.var_names)
    if len(shared_genes) == 0:
        raise ValueError("No shared genes.")
    sA = sliceA[:, shared_genes].copy()
    sB = sliceB[:, shared_genes].copy()

    shared_ct = set(sA.obs['cell_type_annot']) & set(sB.obs['cell_type_annot'])
    if len(shared_ct) == 0:
        raise ValueError("No shared cell types.")
    sA = sA[sA.obs['cell_type_annot'].isin(shared_ct)]
    sB = sB[sB.obs['cell_type_annot'].isin(shared_ct)]

    n_A, n_B = sA.shape[0], sB.shape[0]
    log.write(f"n_A={n_A}  n_B={n_B}  "
              f"shared_genes={len(shared_genes)}  shared_ct={len(shared_ct)}\n")

    # ── 1. Biological cost matrices ───────────────────────────────────────────
    import scipy.sparse as sp
    def to_dense(X):
        return X.toarray() if sp.issparse(X) else np.asarray(X)

    A_X = to_dense(sA.X if use_rep is None else sA.obsm[use_rep]) + 0.01
    B_X = to_dense(sB.X if use_rep is None else sB.obsm[use_rep]) + 0.01
    cosine_dist = cosine_distances(A_X, B_X).astype(np.float32)

    nd_A = neighborhood_distribution(sA, radius).astype(np.float32) + 0.01
    nd_B = neighborhood_distribution(sB, radius).astype(np.float32) + 0.01

    t1 = time.time()
    log.write(f"Biological costs: {t1-t0:.1f}s\n")
    if verbose:
        print(f"[SFF] Biological costs done ({t1-t0:.1f}s)")

    # ── 2. Spatial registration (no biology used here) ───────────────────────
    if verbose:
        print(f"[SFF] Running Fourier-Mellin registration "
              f"(grid={grid_size}, σ={sigma_px}px)...")

    transform, coords_A_reg, coords_B = find_transform(
        sA, sB,
        grid_size=grid_size,
        sigma_px=sigma_px,
        num_angles=num_angles,
        verbose=verbose)

    t2 = time.time()
    log.write(f"Registration: flip={transform['flip']}  "
              f"angle={transform['angle']:.2f}°  "
              f"scale={transform['scale']:.4f}  "
              f"tx={transform['tx_px']}  ty={transform['ty_px']}  "
              f"conf={transform['confidence']:.4f}  "
              f"time={t2-t1:.1f}s\n")
    if verbose:
        print(f"[SFF] Registration done ({t2-t1:.1f}s)")

    # ── 3. Local biological matching ──────────────────────────────────────────
    if verbose:
        print(f"[SFF] Local biological matching...")

    pi, unmatched = local_biological_matching(
        sA, sB,
        coords_A_reg, coords_B,
        nd_A, nd_B,
        cosine_dist,
        gamma=gamma,
        search_radius=search_radius,
        soft_temp=soft_temp,
        verbose=verbose)

    t3 = time.time()
    log.write(f"Matching: unmatched={len(unmatched)}/{n_A}  time={t3-t2:.1f}s\n")
    log.write(f"Total: {t3-t0:.1f}s\n")
    log.close()

    if verbose:
        print(f"[SFF] Done. Total: {t3-t0:.1f}s")

    if return_extra:
        return pi, coords_A_reg, coords_B, transform
    return pi


# ═══════════════════════════════════════════════════════════════════════════════
# Module 8 — Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_alignment(pi, sliceA, sliceB,
                   coords_A_reg, coords_B,
                   top_k=300, figsize=(14, 6), save_path=None):
    """
    Two-panel alignment plot.

    Left:  raw coordinates — shows the pre-registration problem.
    Right: registered coordinates — shows the alignment quality.
           Lines connect top-k highest-weight (i,j) pairs in π.
           A cells coloured by matched B cell type → anatomical correctness check.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    ct_B   = np.array(sliceB.obs['cell_type_annot'].values)
    uniq   = np.unique(ct_B)
    ct2col = {c: i for i, c in enumerate(uniq)}
    cmap   = plt.get_cmap('tab20', len(uniq))

    # Top-k pairs by transport weight
    flat     = pi.flatten()
    top_flat = np.argsort(flat)[-top_k:]
    rows     = top_flat // pi.shape[1]
    cols     = top_flat %  pi.shape[1]
    weights  = flat[top_flat]
    w_norm   = weights / (weights.max() + 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ── Left: raw ──────────────────────────────────────────────────────────
    ax = axes[0]
    raw_A = sliceA.obsm['spatial']
    raw_B = sliceB.obsm['spatial']
    ax.scatter(*raw_B.T, s=1, c='#9FE1CB', alpha=0.4, rasterized=True, label='B')
    ax.scatter(*raw_A.T, s=1, c='#AFA9EC', alpha=0.4, rasterized=True, label='A')
    ax.set_title('Raw coordinates (before registration)', fontsize=10)
    ax.legend(markerscale=5, fontsize=8)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── Right: registered ─────────────────────────────────────────────────
    ax = axes[1]
    ax.scatter(*coords_B.T, s=1, c='#9FE1CB', alpha=0.3, rasterized=True)

    # Colour A cells by matched B cell type
    matched_j = np.argmax(pi, axis=1)
    colors    = np.array([ct2col[ct_B[j]] for j in matched_j])
    ax.scatter(*coords_A_reg.T, s=3, c=colors,
               cmap='tab20', vmin=0, vmax=len(uniq),
               alpha=0.8, rasterized=True,
               label='A (coloured by matched B cell type)')

    # Correspondence lines for top-k pairs
    for idx in range(len(rows)):
        i, j = rows[idx], cols[idx]
        ax.plot([coords_A_reg[i, 0], coords_B[j, 0]],
                [coords_A_reg[i, 1], coords_B[j, 1]],
                'k-', alpha=float(w_norm[idx]) * 0.6,
                linewidth=0.4)

    ax.set_title('Registered (A coloured by matched B cell type)', fontsize=10)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


def plot_density_maps(sliceA, sliceB, transform,
                      grid_size=128, sigma_px=4, n_show=4):
    """
    Diagnostic: show density map channels and the phase correlation map.
    Helps verify that registration is correct before cell matching.
    """
    import matplotlib.pyplot as plt

    ct_A = np.array(sliceA.obs['cell_type_annot'].astype(str))
    ct_B = np.array(sliceB.obs['cell_type_annot'].astype(str))

    imgB, ct_B_names, affB = build_density_map(
        sliceB.obsm['spatial'], ct_B,
        grid_size=grid_size, sigma_px=sigma_px)

    flip_x, flip_y = transform['flip']
    cA_flip = sliceA.obsm['spatial'] * np.array([flip_x, flip_y])
    imgA, ct_A_names, affA = build_density_map(
        cA_flip, ct_A, grid_size=grid_size, sigma_px=sigma_px)

    shared = sorted(set(ct_A_names) & set(ct_B_names))[:n_show]

    fig, axes = plt.subplots(3, n_show, figsize=(n_show * 3, 9))
    for k, ct in enumerate(shared):
        ia = np.where(ct_A_names == ct)[0][0]
        ib = np.where(ct_B_names == ct)[0][0]

        # Apply stored transform to A image
        aimg = apply_rotation_stack(imgA[[ia]], transform['angle'])[0]
        aimg = apply_scale_stack(aimg[None], transform['scale'])[0]

        axes[0, k].imshow(imgA[ia], cmap='hot', origin='lower')
        axes[0, k].set_title(f'A: {ct[:15]}', fontsize=8)
        axes[0, k].axis('off')

        axes[1, k].imshow(imgB[ib], cmap='hot', origin='lower')
        axes[1, k].set_title(f'B: {ct[:15]}', fontsize=8)
        axes[1, k].axis('off')

        axes[2, k].imshow(aimg, cmap='hot', origin='lower')
        axes[2, k].set_title('A (registered)', fontsize=8)
        axes[2, k].axis('off')

    plt.suptitle('Density map channels: A raw | B | A registered', fontsize=10)
    plt.tight_layout()
    plt.show()