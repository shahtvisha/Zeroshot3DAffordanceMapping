"""
zeroshot3D/geometry.py
Geometry-aware affordance priors — the novel contribution.

Core idea: CLIP knows what things look like, but not what
shapes afford which actions. A flat horizontal surface
affords placing. A consistent-normal region affords grasping.
A concave upward surface affords pouring into.

We encode this physical common sense as per-point geometric
scores, then multiply with CLIP scores. A point needs BOTH
to score highly — it must look right AND be shaped right.

No training required. These are hand-specified physics priors.
"""

import numpy as np
from scipy.spatial import KDTree


# ─────────────────────────────────────────────
# Step 1: compute local geometric properties
# ─────────────────────────────────────────────

def compute_local_geometry(points: np.ndarray,
                            k_neighbors: int = 20) -> dict:
    """
    For each point, compute local geometric descriptors using
    its k nearest neighbors via PCA on the local covariance matrix.

    This is the standard approach in 3D perception — the same math
    used in ICP, PointNet++, and most classical point cloud methods.

    Returns a dict of (N,) or (N,3) arrays:
      normals:    (N, 3)  estimated surface normal
      planarity:  (N,)    how flat the local neighborhood is [0,1]
      linearity:  (N,)    how edge-like (elongated) [0,1]
      scattering: (N,)    how noisy/disordered [0,1]
      curvature:  (N,)    surface curvature estimate [0,1]
    """
    N = len(points)
    tree = KDTree(points)

    normals    = np.zeros((N, 3), dtype=np.float32)
    planarity  = np.zeros(N, dtype=np.float32)
    linearity  = np.zeros(N, dtype=np.float32)
    scattering = np.zeros(N, dtype=np.float32)
    curvature  = np.zeros(N, dtype=np.float32)

    # Query all neighbors at once for efficiency
    _, indices = tree.query(points, k=k_neighbors + 1)

    for i in range(N):
        neighbors = points[indices[i, 1:]]  # exclude self

        # Center the neighborhood
        centered = neighbors - neighbors.mean(axis=0)

        # PCA via covariance matrix
        cov = (centered.T @ centered) / k_neighbors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # eigh returns ascending order — λ0 ≤ λ1 ≤ λ2
        l0, l1, l2 = eigenvalues
        total = l0 + l1 + l2 + 1e-8

        # Normal = eigenvector of smallest eigenvalue
        normals[i] = eigenvectors[:, 0]

        # Shape descriptors (Westin et al. 1997 — standard in 3D literature)
        linearity[i]  = (l2 - l1) / total      # high → edge/cylinder
        planarity[i]  = (l1 - l0) / total      # high → flat surface
        scattering[i] = l0 / total              # high → disordered/noisy
        curvature[i]  = l0 / total              # same as scattering here

    # Orient normals consistently (toward positive Z = "up" convention)
    flip = normals[:, 2] < 0
    normals[flip] *= -1

    return {
        "normals":    normals,
        "planarity":  planarity,
        "linearity":  linearity,
        "scattering": scattering,
        "curvature":  curvature,
    }


# ─────────────────────────────────────────────
# Step 2: query-conditioned geometric priors
# ─────────────────────────────────────────────

def geometric_prior_score(geom: dict, query: str) -> np.ndarray:
    """
    Map a text query to a geometric score per point.

    This encodes physical common sense: the geometry that affords
    "grasping" is different from the geometry that affords "placing."

    Args:
        geom:  output of compute_local_geometry()
        query: natural language affordance query

    Returns:
        scores: (N,) float32 in [0, 1]
                High = this point's geometry matches the affordance
    """
    query_lower = query.lower()

    normals    = geom["normals"]
    planarity  = geom["planarity"]
    linearity  = geom["linearity"]
    scattering = geom["scattering"]
    curvature  = geom["curvature"]

    # ── Grasp / pick up / grab ──────────────────────────────────────────
    # Graspable surfaces: consistent normals (not chaotic), some curvature
    # (curved objects are easier to grip than perfectly flat ones),
    # not too scattered (noisy regions are unreliable).
    if any(w in query_lower for w in ["grasp", "pick", "grab", "hold",
                                        "grip", "lift", "take"]):
        normal_consistency = 1.0 - scattering   # low scatter = consistent normals
        slight_curve = 1.0 - np.abs(curvature - 0.15) / 0.15  # peak at ~15% curvature
        slight_curve = np.clip(slight_curve, 0, 1)
        score = 0.6 * normal_consistency + 0.4 * slight_curve

    # ── Place / put / set down ──────────────────────────────────────────
    # Placeable surfaces: flat (high planarity), horizontal (normal points up).
    # The dot product with [0,0,1] measures "upward-facingness."
    elif any(w in query_lower for w in ["place", "put", "set", "rest",
                                          "lay", "deposit"]):
        flatness    = planarity                          # how planar
        upward_face = np.clip(normals[:, 2], 0, 1)      # dot with up vector
        score = 0.5 * flatness + 0.5 * upward_face

    # ── Pour / fill / pour into ─────────────────────────────────────────
    # Target for pouring: concave (bowl-shaped) AND facing upward.
    # We approximate concavity by finding points whose normals diverge
    # from their neighbors (normals point outward from a bowl's rim).
    elif any(w in query_lower for w in ["pour", "fill", "empty into",
                                          "add to", "drop in"]):
        upward_face  = np.clip(normals[:, 2], 0, 1)
        # High planarity in a bowl's interior, high scattering at the rim
        containment  = planarity * upward_face
        score = containment

    # ── Push / press / slide ────────────────────────────────────────────
    # Pushable surfaces: vertical face (normal horizontal), flat and stable.
    elif any(w in query_lower for w in ["push", "press", "slide",
                                          "shove", "button"]):
        horizontal_normal = np.sqrt(normals[:, 0]**2 + normals[:, 1]**2)
        score = 0.6 * horizontal_normal + 0.4 * planarity

    # ── Open / close (drawers, doors, lids) ─────────────────────────────
    elif any(w in query_lower for w in ["open", "close", "pull",
                                          "drawer", "door", "lid"]):
        # Edges and handles: high linearity, some normal consistency
        score = 0.7 * linearity + 0.3 * (1.0 - scattering)

    # ── Insert / plug in ────────────────────────────────────────────────
    elif any(w in query_lower for w in ["insert", "plug", "slot",
                                          "attach", "connect"]):
        # Openings: locally linear (the edge of a hole), facing a consistent dir
        score = 0.5 * linearity + 0.5 * (1.0 - scattering)

    # ── Default: prefer clean, non-noisy geometry ────────────────────────
    else:
        score = 1.0 - scattering

    # Normalize to [0, 1]
    score = np.clip(score, 0, 1).astype(np.float32)
    s_min, s_max = score.min(), score.max()
    score = (score - s_min) / (s_max - s_min + 1e-8)

    return score


# ─────────────────────────────────────────────
# Step 3: fusion
# ─────────────────────────────────────────────

def fuse_scores(semantic: np.ndarray, geometric: np.ndarray,
                mode: str = "multiply",
                geo_weight: float = 0.4) -> np.ndarray:
    """
    Combine semantic (CLIP) and geometric scores.

    multiply is the key design choice here:
      - If EITHER score is near zero, the final score is near zero
      - A point needs to look right (semantic) AND be shaped right (geometric)
      - Addition would let a very high semantic score compensate for
        wrong geometry — which is exactly the failure mode we're fixing

    Args:
        mode: "multiply" (recommended) | "weighted_sum" | "harmonic"
        geo_weight: only used for weighted_sum
    """
    sem = np.clip(semantic,  0, 1).astype(np.float32)
    geo = np.clip(geometric, 0, 1).astype(np.float32)

    if mode == "multiply":
        # Geometric mean variant — softer than raw multiply
        fused = np.sqrt(sem * geo)

    elif mode == "weighted_sum":
        fused = (1 - geo_weight) * sem + geo_weight * geo

    elif mode == "harmonic":
        # Harmonic mean — requires both to be high
        fused = 2 * sem * geo / (sem + geo + 1e-8)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Final normalization
    f_min, f_max = fused.min(), fused.max()
    fused = (fused - f_min) / (f_max - f_min + 1e-8)

    return fused.astype(np.float32)