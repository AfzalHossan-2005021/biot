"""
visualize.py — Alignment visualization and diagnostics for INCENT
==================================================================
Run diagnose() first to understand your data, then plot_alignment()
to inspect results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def diagnose(sliceA, sliceB):
    """
    Print key statistics to help choose alpha and verify the normalization fix.

    Run this BEFORE calling pairwise_align.
    """
    from sklearn.neighbors import BallTree

    cA = sliceA.obsm['spatial'].astype(float)
    cB = sliceB.obsm['spatial'].astype(float)

    print("=" * 55)
    print("COORDINATE RANGES")
    print(f"  A: x=[{cA[:,0].min():.1f}, {cA[:,0].max():.1f}]  "
          f"y=[{cA[:,1].min():.1f}, {cA[:,1].max():.1f}]")
    print(f"  B: x=[{cB[:,0].min():.1f}, {cB[:,0].max():.1f}]  "
          f"y=[{cB[:,1].min():.1f}, {cB[:,1].max():.1f}]")

    diam_A = np.linalg.norm(cA.max(0) - cA.min(0))
    diam_B = np.linalg.norm(cB.max(0) - cB.min(0))
    print(f"\nSPATIAL EXTENTS (pairwise distance range after normalization)")
    print(f"  A diameter: {diam_A:.1f}")
    print(f"  B diameter: {diam_B:.1f}")
    print(f"  Ratio A/B:  {diam_A/diam_B:.4f}  "
          f"← D_A will span [0, {diam_A/diam_B:.4f}] after fix")
    print(f"              D_B will span [0, 1.0000] after fix")

    dA = BallTree(cA).query(cA, k=2)[0][:, 1]
    dB = BallTree(cB).query(cB, k=2)[0][:, 1]
    print(f"\nMEDIAN NN DISTANCE")
    print(f"  A: {np.median(dA):.2f}   B: {np.median(dB):.2f}   "
          f"ratio: {np.median(dA)/np.median(dB):.3f}  (1.0 = same platform)")

    ctA = set(sliceA.obs['cell_type_annot'])
    ctB = set(sliceB.obs['cell_type_annot'])
    print(f"\nCELL TYPE OVERLAP")
    print(f"  A: {len(ctA)}  B: {len(ctB)}  shared: {len(ctA & ctB)}")

    # Spatial overlap of raw coordinate boxes
    ox = max(0.0, min(cA[:,0].max(), cB[:,0].max())
                - max(cA[:,0].min(), cB[:,0].min()))
    oy = max(0.0, min(cA[:,1].max(), cB[:,1].max())
                - max(cA[:,1].min(), cB[:,1].min()))
    print(f"\nRAW COORDINATE BOX OVERLAP")
    print(f"  x overlap: {ox:.1f}   y overlap: {oy:.1f}")
    if ox > 0 and oy > 0:
        print("  → Frames OVERLAP in raw space. "
              "Spatial GW signal is valid. Use alpha > 0.")
    else:
        print("  → Frames do NOT overlap in raw space.\n"
              "  → But D_A / D_B are INTERNAL distances — unaffected by origin.\n"
              "  → Spatial GW signal is STILL VALID. Use alpha > 0.\n"
              "  → The shared-scale fix (D_A /= max(D_B)) handles this correctly.")

    print(f"\nRECOMMENDED SETTINGS")
    print(f"  alpha  = 0.4–0.6  (GW spatial weight)")
    print(f"  gamma  = 0.3–0.5  (neighbourhood dissimilarity weight)")
    print(f"  radius = {np.median(dA) * 10:.0f}–{np.median(dA) * 20:.0f}  "
          f"(neighbourhood radius, ~10–20× median NN dist)")
    print("=" * 55)


def check_normalization(D_A, D_B, nx):
    """
    Quick sanity check after computing and normalizing D_A, D_B.
    Call immediately after the normalization block.
    """
    maxA = float(nx.max(D_A))
    maxB = float(nx.max(D_B))
    print(f"\nNORMALIZATION CHECK")
    print(f"  D_A max: {maxA:.6f}  (should be < 1.0)")
    print(f"  D_B max: {maxB:.6f}  (should be 1.0)")
    print(f"  Ratio:   {maxA:.6f}")
    ok = maxB > 0.999 and maxA < 1.0 - 1e-6
    print(f"  Status:  {'✓ CORRECT' if ok else '✗ WRONG — check normalization'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Alignment plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_alignment(pi, sliceA, sliceB, top_k=300,
                   figsize=(14, 6), save_path=None,
                   title=None):
    """
    Two-panel alignment visualization.

    Panel 1 — Overlay:
        A cells (coloured by cell type) drawn over B cells (grey).
        Because slices are from different axial depths, A should appear
        as a spatially coherent SUBREGION of B.
        If A looks spatially coherent → registration/alignment is correct.
        If A looks scattered randomly → alignment has failed.

    Panel 2 — Correspondences:
        Top-k highest-weight (i, j) pairs connected by lines.
        Spatially short lines → good local matching.
        Long crossing lines → something is wrong.

    Notes
    -----
    This visualization uses RAW coordinates of both slices.
    Because the two slices were imaged at completely different physical
    positions (millions of units apart), the raw coordinates cannot be
    directly overlaid.

    The correct way to overlay them is:
        1. Run pairwise_align() to get pi.
        2. Infer the rigid transform from pi using infer_transform().
        3. Call plot_alignment_registered() with coords_A_reg.

    Alternatively, set normalize_coords=True below to center both
    slices at the origin before plotting (loses absolute position
    information but makes the overlay readable).
    """
    ct_B  = np.array(sliceB.obs['cell_type_annot'].values)
    ct_A  = np.array(sliceA.obs['cell_type_annot'].values)
    uniq  = np.unique(np.concatenate([ct_A, ct_B]))
    ct2i  = {c: i for i, c in enumerate(uniq)}
    cmap  = plt.get_cmap('tab20', len(uniq))

    # Center both slices at origin for overlay readability
    cA = sliceA.obsm['spatial'].astype(float)
    cB = sliceB.obsm['spatial'].astype(float)
    cA = cA - cA.mean(0)
    cB = cB - cB.mean(0)

    # Top-k pairs
    flat    = pi.flatten()
    top_idx = np.argsort(flat)[-top_k:]
    rows    = top_idx // pi.shape[1]
    cols    = top_idx %  pi.shape[1]
    w_norm  = flat[top_idx] / (flat[top_idx].max() + 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ── Panel 1: overlay ─────────────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(*cB.T, s=1, c='#d4d4d4', alpha=0.3,
               rasterized=True, label='B')
    colors_A = [cmap(ct2i[c]) for c in ct_A]
    ax.scatter(*cA.T, s=4, c=colors_A, alpha=0.85,
               rasterized=True, label='A (by cell type)')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Overlay (both centred)\nA=coloured by cell type, B=grey',
                 fontsize=10)
    ax.legend(markerscale=4, fontsize=7, loc='upper right',
              handles=[plt.scatter([], [], s=8, c=[cmap(ct2i[c])], label=c)
                       for c in sorted(ct2i)],
              ncol=2)

    # ── Panel 2: correspondences ─────────────────────────────────────────────
    ax = axes[1]
    ax.scatter(*cB.T, s=1, c='#d4d4d4', alpha=0.2, rasterized=True)
    ax.scatter(*cA.T, s=2, c=colors_A,  alpha=0.5, rasterized=True)

    for k in range(len(rows)):
        i, j = int(rows[k]), int(cols[k])
        ax.plot([cA[i, 0], cB[j, 0]],
                [cA[i, 1], cB[j, 1]],
                'k-', lw=0.4, alpha=float(w_norm[k]) * 0.6)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Top-{top_k} correspondences\n'
                 '(opacity ∝ transport weight)', fontsize=10)

    suptitle = title or 'INCENT alignment (coords centred for display)'
    plt.suptitle(suptitle, fontsize=10, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")
    plt.show()


def plot_alignment_registered(pi, sliceA, sliceB,
                               coords_A_reg, top_k=300,
                               figsize=(14, 6), save_path=None):
    """
    Same as plot_alignment but uses pre-registered coordinates for A.
    coords_A_reg : (n_A, 2) — A's coordinates mapped into B's frame
                   (obtain from infer_transform()).
    """
    ct_B  = np.array(sliceB.obs['cell_type_annot'].values)
    ct_A  = np.array(sliceA.obs['cell_type_annot'].values)
    uniq  = np.unique(np.concatenate([ct_A, ct_B]))
    ct2i  = {c: i for i, c in enumerate(uniq)}
    cmap  = plt.get_cmap('tab20', len(uniq))

    cB      = sliceB.obsm['spatial'].astype(float)
    cA_reg  = np.asarray(coords_A_reg)

    flat    = pi.flatten()
    top_idx = np.argsort(flat)[-top_k:]
    rows    = top_idx // pi.shape[1]
    cols    = top_idx %  pi.shape[1]
    w_norm  = flat[top_idx] / (flat[top_idx].max() + 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    colors_A = [cmap(ct2i[c]) for c in ct_A]

    ax = axes[0]
    ax.scatter(*cB.T,     s=1, c='#d4d4d4', alpha=0.3,
               rasterized=True, label='B')
    ax.scatter(*cA_reg.T, s=4, c=colors_A,  alpha=0.85,
               rasterized=True, label='A (registered)')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('A registered into B frame\n(A is a spatial subregion of B)',
                 fontsize=10)

    ax = axes[1]
    ax.scatter(*cB.T,     s=1, c='#d4d4d4', alpha=0.2, rasterized=True)
    ax.scatter(*cA_reg.T, s=2, c=colors_A,  alpha=0.5, rasterized=True)
    for k in range(len(rows)):
        i, j = int(rows[k]), int(cols[k])
        ax.plot([cA_reg[i, 0], cB[j, 0]],
                [cA_reg[i, 1], cB[j, 1]],
                'k-', lw=0.4, alpha=float(w_norm[k]) * 0.6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Top-{top_k} correspondences', fontsize=10)

    plt.suptitle('INCENT alignment — registered coordinates', fontsize=10, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Infer spatial transform from transport plan
# ─────────────────────────────────────────────────────────────────────────────

def infer_transform(pi, sliceA, sliceB, top_frac=0.15):
    """
    Infer the rigid transform (R, t) that maps sliceA's coordinates
    into sliceB's coordinate frame, using the high-confidence cell
    correspondences in the transport plan pi.

    Parameters
    ----------
    pi       : (n_A, n_B) transport plan from pairwise_align()
    top_frac : fraction of highest-weight pairs to use (default 15%)

    Returns
    -------
    R            : (2, 2) rotation matrix
    t            : (2,)   translation vector
    coords_A_reg : (n_A, 2) A coordinates in B's frame
    """
    from scipy.linalg import svd

    cA = sliceA.obsm['spatial'].astype(float)
    cB = sliceB.obsm['spatial'].astype(float)

    # Pick top pairs by weight
    flat    = pi.flatten()
    n_top   = max(50, int(top_frac * (pi > 0).sum()))
    n_top   = min(n_top, len(flat))
    top_idx = np.argsort(flat)[-n_top:]
    rows    = top_idx // pi.shape[1]
    cols    = top_idx %  pi.shape[1]
    weights = flat[top_idx].astype(float)
    weights = weights / weights.sum()

    pA = cA[rows]   # (n_top, 2)
    pB = cB[cols]   # (n_top, 2)

    # Weighted centroids
    cA_bar = (weights[:, None] * pA).sum(0)
    cB_bar = (weights[:, None] * pB).sum(0)

    # Weighted SVD
    H        = (pA - cA_bar).T @ np.diag(weights) @ (pB - cB_bar)
    U, _, Vt = svd(H)
    R        = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = cB_bar - R @ cA_bar

    coords_A_reg = cA @ R.T + t

    angle = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    print(f"Inferred transform:")
    print(f"  Rotation: {angle:.2f}°")
    print(f"  Translation: ({t[0]:.2f}, {t[1]:.2f})")
    print(f"  Based on {n_top} high-confidence cell pairs")

    return R, t, coords_A_reg