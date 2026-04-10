"""
experiments/run_scannet.py
Evaluation on ScanNet for zero-shot 3D affordance mapping.

ScanNet doesn't have per-point affordance labels the way LASO does.
Instead we use it in two ways:

  1. QUALITATIVE — run our method on real indoor scenes and save
     visualizations. This is what goes in your writeup and portfolio.

  2. PROXY EVALUATION — ScanNet200 has per-point semantic labels
     (chair, table, door, etc.). We use these as a proxy: if a point
     belongs to a semantically relevant object class for the query
     affordance, we treat it as a soft ground truth. E.g. for "sit",
     the ground truth region is all points labeled "chair" or "sofa".

Download the mesh files (smallest option, ~5MB per scene):
    python download-scannet.py -o data/scannet --id scene0000_00 --type _vh_clean_2.ply

Download with semantic labels for proxy eval:
    python download-scannet.py -o data/scannet --id scene0000_00 --type _vh_clean_2.labels.ply

Download the label map (needed once for ScanNet200 class names):
    python download-scannet.py -o data/scannet --label_map

Run:
    python experiments/run_scannet.py --scan scene0000_00
    python experiments/run_scannet.py --scan scene0000_00 --affordance support --save
    python experiments/run_scannet.py --all --save          # all downloaded scenes
    python experiments/run_scannet.py --qualitative         # just save visuals, no metrics
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import (
    compute_text_similarity,
    generate_synthetic_features,
    generate_synthetic_text_embedding,
)
from src.geometry import (
    compute_local_geometry,
    fuse_scores,
    geometric_prior_score,
)
from src.pointcloud import voxel_downsample

SCANNET_DIR = ROOT / "data" / "scannet" / "scans"

# ── Affordance queries (same as run_laso) ─────────────────────────────────────

AFFORDANCE_QUERIES = {
    "grasp":   "the graspable region suitable for picking up with fingers",
    "support": "the flat horizontal surface that supports objects on top",
    "contain": "the hollow region that can contain objects inside",
    "sit":     "the seat surface for a person to sit on",
    "open":    "the handle or edge region for opening by pulling",
    "move":    "the part to push or drag to move the object",
    "lay":     "the flat surface for laying an object down on",
}

# ── ScanNet200 class → affordance mapping (proxy ground truth) ────────────────
# Maps ScanNet200 label names to which affordances they support.
# Used to compute proxy metrics when no per-point affordance labels exist.

CLASS_AFFORDANCE_MAP = {
    # sit
    "chair":          ["sit"],
    "sofa":           ["sit", "lay"],
    "armchair":       ["sit"],
    "stool":          ["sit"],
    "bench":          ["sit", "lay"],
    "office chair":   ["sit"],
    # support / place on
    "table":          ["support", "grasp"],
    "desk":           ["support"],
    "coffee table":   ["support"],
    "dining table":   ["support"],
    "counter":        ["support"],
    "shelf":          ["support", "contain"],
    "nightstand":     ["support"],
    "dresser":        ["support", "contain"],
    # contain
    "trash can":      ["contain"],
    "box":            ["contain"],
    "basket":         ["contain"],
    "bag":            ["contain", "grasp"],
    "bowl":           ["contain"],
    "cup":            ["grasp", "contain"],
    # open / move
    "door":           ["open", "move"],
    "cabinet":        ["open"],
    "drawer":         ["open"],
    "refrigerator":   ["open"],
    # grasp
    "bottle":         ["grasp", "contain", "pour"],
    "plant":          ["grasp"],
    "book":           ["grasp"],
    "laptop":         ["grasp", "move"],
    "keyboard":       ["grasp", "move"],
    # lay
    "bed":            ["lay", "sit"],
    "couch":          ["lay", "sit"],
    "mat":            ["lay"],
    "rug":            ["lay"],
}


# ── PLY loader ────────────────────────────────────────────────────────────────

def load_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Load a ScanNet PLY file.

    Returns:
        points: (N, 3) float32  XYZ
        colors: (N, 3) float32  RGB in [0,1]
        labels: (N,)  int32 | None  — ScanNet200 label IDs if present
    """
    with open(path, "rb") as f:
        # Parse ASCII header
        header, properties, n_verts = [], [], 0
        in_vertex = False
        binary_le = True

        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            header.append(line)
            if line == "end_header":
                break
            if line.startswith("element vertex"):
                n_verts   = int(line.split()[-1])
                in_vertex = True
            elif line.startswith("element") and "vertex" not in line:
                in_vertex = False
            elif line.startswith("property") and in_vertex:
                parts = line.split()
                properties.append((parts[1], parts[2]))  # (dtype, name)
            elif "binary_big_endian" in line:
                binary_le = False

        dtype_map = {
            "float": "f4", "float32": "f4",
            "double": "f8", "float64": "f8",
            "uchar": "u1", "uint8": "u1",
            "char": "i1", "int8": "i1",
            "short": "i2", "int16": "i2",
            "ushort": "u2", "uint16": "u2",
            "int": "i4", "int32": "i4",
            "uint": "u4", "uint32": "u4",
        }
        dt = np.dtype([(name, dtype_map.get(pt, "f4"))
                       for pt, name in properties])
        if not binary_le:
            dt = dt.newbyteorder(">")

        raw = np.frombuffer(f.read(n_verts * dt.itemsize), dtype=dt)

    points = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float32)

    if "red" in raw.dtype.names:
        colors = np.stack([raw["red"], raw["green"], raw["blue"]],
                           axis=1).astype(np.float32) / 255.0
    elif "r" in raw.dtype.names:
        colors = np.stack([raw["r"], raw["g"], raw["b"]],
                           axis=1).astype(np.float32) / 255.0
    else:
        colors = np.ones((len(points), 3), dtype=np.float32) * 0.6

    # Semantic labels — present in _vh_clean_2.labels.ply
    labels = None
    for lname in ("label", "objectId", "object_id"):
        if lname in raw.dtype.names:
            labels = raw[lname].astype(np.int32)
            break

    return points, colors, labels


def load_label_map(data_dir: Path = SCANNET_DIR) -> dict[int, str]:
    """
    Load the ScanNet200 label map: label_id → class_name.
    File: scannetv2-labels.combined.tsv (download with --label_map flag)
    """
    tsv_path = data_dir / "scannetv2-labels.combined.tsv"
    if not tsv_path.exists():
        print(f"[scannet] Label map not found at {tsv_path}")
        print(f"  Download: python download-scannet.py -o data/scannet --label_map")
        return {}

    label_map = {}
    with open(tsv_path) as f:
        header = f.readline().strip().split("\t")
        id_col   = header.index("id")
        name_col = header.index("raw_category") if "raw_category" in header \
                   else header.index("category")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > max(id_col, name_col):
                try:
                    label_map[int(parts[id_col])] = parts[name_col].lower()
                except ValueError:
                    pass
    print(f"[scannet] Loaded label map: {len(label_map)} classes")
    return label_map


def load_scan(scan_id: str,
              max_points: int = 30_000,
              voxel_size: float = 0.02) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Load and downsample a ScanNet scan.
    Tries the labeled PLY first, falls back to clean mesh.
    """
    scan_dir = SCANNET_DIR / scan_id
    if not scan_dir.exists():
        raise FileNotFoundError(
            f"Scan not found: {scan_dir}\n"
            f"Download: python download-scannet.py -o data/scannet --id {scan_id} "
            f"--type _vh_clean_2.ply"
        )

    # Prefer labeled version for proxy eval
    labeled_ply = scan_dir / f"{scan_id}_vh_clean_2.labels.ply"
    clean_ply   = scan_dir / f"{scan_id}_vh_clean_2.ply"

    if labeled_ply.exists():
        print(f"[scannet] Loading labeled PLY: {labeled_ply.name}")
        points, colors, labels = load_ply(labeled_ply)
    elif clean_ply.exists():
        print(f"[scannet] Loading clean PLY: {clean_ply.name}")
        points, colors, labels = load_ply(clean_ply)
    else:
        raise FileNotFoundError(
            f"No PLY file found in {scan_dir}.\n"
            f"Download: python download-scannet.py -o data/scannet --id {scan_id} "
            f"--type _vh_clean_2.ply"
        )

    print(f"[scannet] Raw: {len(points):,} points")

    # Voxel downsample — keeps labels consistent
    if voxel_size > 0 and len(points) > max_points:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd_down   = pcd.voxel_down_sample(voxel_size)

        points_down = np.asarray(pcd_down.points, dtype=np.float32)
        colors_down = np.asarray(pcd_down.colors, dtype=np.float32)

        # Re-assign labels: for each downsampled point, use nearest original label
        if labels is not None:
            from scipy.spatial import KDTree
            tree   = KDTree(points)
            _, idx = tree.query(points_down)
            labels_down = labels[idx]
        else:
            labels_down = None

        points, colors, labels = points_down, colors_down, labels_down

    # Final random cap if still too large
    if len(points) > max_points:
        idx    = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
        labels = labels[idx] if labels is not None else None

    print(f"[scannet] Downsampled: {len(points):,} points  "
          f"(labels={'yes' if labels is not None else 'no'})")
    return points, colors, labels


# ── Proxy ground truth from semantic labels ────────────────────────────────────

def build_proxy_gt(labels: np.ndarray, affordance: str,
                   label_map: dict[int, str]) -> np.ndarray:
    """
    Build a binary GT mask using semantic class labels as proxy.
    A point is "positive" if its class name supports the affordance.

    Returns:
        gt_mask: (N,) bool
    """
    if labels is None or not label_map:
        return None

    gt_mask = np.zeros(len(labels), dtype=bool)
    for label_id, class_name in label_map.items():
        if affordance in CLASS_AFFORDANCE_MAP.get(class_name, []):
            gt_mask |= (labels == label_id)

    return gt_mask


# ── Scoring methods ────────────────────────────────────────────────────────────

def run_clip_only(points, colors, query):
    feats    = generate_synthetic_features(points, colors)
    text_emb = generate_synthetic_text_embedding(query)
    return compute_text_similarity(feats, text_emb)


def run_geometry_only(points, colors, query):
    geom = compute_local_geometry(points, k_neighbors=10)
    return geometric_prior_score(geom, query)


def run_geometry_aware(points, colors, query):
    feats    = generate_synthetic_features(points, colors)
    text_emb = generate_synthetic_text_embedding(query)
    clip_sc  = compute_text_similarity(feats, text_emb)
    geom     = compute_local_geometry(points, k_neighbors=10)
    geo_sc   = geometric_prior_score(geom, query)
    return fuse_scores(clip_sc, geo_sc, mode="multiply")


METHODS = {
    "clip_only":      run_clip_only,
    "geometry_only":  run_geometry_only,
    "geometry_aware": run_geometry_aware,
}


# ── Metrics ───────────────────────────────────────────────────────────────────

def precision_at_k(scores, gt_mask, k):
    k = min(k, len(scores))
    return float(gt_mask[np.argsort(scores)[-k:]].sum()) / k


def iou_score(scores, gt_mask, threshold=0.5):
    norm  = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    pred  = norm > threshold
    inter = (pred & gt_mask).sum()
    union = (pred | gt_mask).sum()
    return float(inter) / (union + 1e-8)


# ── Visualization ─────────────────────────────────────────────────────────────

def save_visualization(scan_id: str, affordance: str,
                        points: np.ndarray, colors: np.ndarray,
                        scores: np.ndarray, method: str):
    """
    Save a top-down 2D projection of the affordance heatmap as PNG.
    Does not require a display — works headless on a server.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("[viz] matplotlib not installed, skipping visualization")
        return

    out_dir = ROOT / "results" / "figures" / "scannet"
    out_dir.mkdir(parents=True, exist_ok=True)

    norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             facecolor="#0f1117")
    fig.suptitle(f"{scan_id} · {affordance} · {method}",
                 color="white", fontsize=12, y=1.01)

    for ax, (pts_col, title) in zip(axes, [
        (colors,                                     "RGB"),
        (cm.plasma(norm_scores)[:, :3],              f"Affordance: {affordance}"),
    ]):
        ax.set_facecolor("#0f1117")
        ax.scatter(points[:, 0], points[:, 2],
                   c=pts_col, s=0.3, linewidths=0)
        ax.set_title(title, color="white", fontsize=10)
        ax.set_aspect("equal")
        ax.axis("off")

    out_path = out_dir / f"{scan_id}_{affordance}_{method}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="#0f1117")
    plt.close(fig)
    print(f"[viz] Saved → {out_path}")


# ── Single scan evaluation ────────────────────────────────────────────────────

def evaluate_scan(scan_id: str,
                  affordances: list[str] = None,
                  save: bool = False,
                  max_points: int = 30_000) -> dict:

    if affordances is None:
        affordances = list(AFFORDANCE_QUERIES.keys())

    print(f"\n{'='*60}")
    print(f"Scan: {scan_id}")
    print(f"{'='*60}")

    points, colors, labels = load_scan(scan_id, max_points=max_points)
    label_map = load_label_map()

    results = {}

    for affordance in affordances:
        query   = AFFORDANCE_QUERIES[affordance]
        gt_mask = build_proxy_gt(labels, affordance, label_map)
        has_gt  = gt_mask is not None and gt_mask.sum() > 10

        print(f"\n  Affordance: '{affordance}'  "
              f"GT points: {gt_mask.sum() if has_gt else 'N/A (no labels)'}")

        aff_results = {}
        for mname, fn in METHODS.items():
            scores = fn(points, colors, query)

            row = {"method": mname}
            if has_gt:
                row["P@50"]  = precision_at_k(scores, gt_mask, 50)
                row["P@100"] = precision_at_k(scores, gt_mask, 100)
                row["IoU"]   = iou_score(scores, gt_mask)

            aff_results[mname] = row

            marker = "  ◄" if mname == "geometry_aware" else ""
            if has_gt:
                print(f"    {mname:<22} P@50={row['P@50']:.3f}  "
                      f"IoU={row['IoU']:.3f}{marker}")
            else:
                print(f"    {mname:<22} (qualitative only){marker}")

            if save:
                save_visualization(scan_id, affordance,
                                   points, colors, scores, mname)

        results[affordance] = aff_results

    return results


# ── Multi-scan evaluation ─────────────────────────────────────────────────────

def evaluate_all(affordances: list[str] = None,
                 max_scans: int = None,
                 save: bool = False) -> dict:

    if not SCANNET_DIR.exists():
        print(f"[scannet] Directory not found: {SCANNET_DIR}")
        print(f"  Download scans to data/scannet/ first.")
        return {}

    scan_ids = sorted(
        d.name for d in SCANNET_DIR.iterdir()
        if d.is_dir() and d.name.startswith("scene")
    )

    if not scan_ids:
        print(f"[scannet] No scans found in {SCANNET_DIR}")
        return {}

    if max_scans:
        scan_ids = scan_ids[:max_scans]

    print(f"[scannet] Found {len(scan_ids)} scans. Evaluating...")

    all_results = {}
    for scan_id in scan_ids:
        try:
            all_results[scan_id] = evaluate_scan(
                scan_id, affordances=affordances, save=save
            )
        except FileNotFoundError as e:
            print(f"  [skip] {scan_id}: {e}")

    # Aggregate across scans
    if all_results:
        _print_aggregate(all_results)

    return all_results


def _print_aggregate(all_results: dict):
    """Print mean metrics across all scans."""
    from collections import defaultdict

    print(f"\n{'='*60}")
    print("AGGREGATE (mean across scans with proxy GT)")
    print(f"{'='*60}")

    accum = defaultdict(lambda: defaultdict(list))
    for scan_results in all_results.values():
        for aff, aff_res in scan_results.items():
            for mname, row in aff_res.items():
                if "P@50" in row:
                    accum[mname]["P@50"].append(row["P@50"])
                    accum[mname]["IoU"].append(row["IoU"])

    print(f"{'Method':<22} {'P@50':>8} {'IoU':>8} {'n':>5}")
    print(f"{'-'*45}")
    for mname, metrics in accum.items():
        if metrics["P@50"]:
            marker = "  ◄" if mname == "geometry_aware" else ""
            print(f"{mname:<22} "
                  f"{np.mean(metrics['P@50']):>8.4f} "
                  f"{np.mean(metrics['IoU']):>8.4f} "
                  f"{len(metrics['P@50']):>5}{marker}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ScanNet evaluation for zero-shot affordance mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scan", type=str, default=None,
                        help="Single scan ID (e.g. scene0000_00)")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all downloaded scans")
    parser.add_argument("--affordance", nargs="+", default=None,
                        help="Affordances to evaluate (default: all)")
    parser.add_argument("--max-scans", type=int, default=None,
                        help="Cap number of scans for --all mode")
    parser.add_argument("--max-points", type=int, default=30_000,
                        help="Max points per scan (default: 30000)")
    parser.add_argument("--save", action="store_true",
                        help="Save PNG visualizations to results/figures/scannet/")
    parser.add_argument("--qualitative", action="store_true",
                        help="Save visuals only, skip metrics (no label map needed)")
    parser.add_argument("--list", action="store_true",
                        help="List downloaded scans and exit")
    args = parser.parse_args()

    if args.list:
        scans = sorted(
            d.name for d in SCANNET_DIR.iterdir()
            if d.is_dir() and d.name.startswith("scene")
        ) if SCANNET_DIR.exists() else []
        print(f"Downloaded scans ({len(scans)}):")
        for s in scans:
            print(f"  {s}")
        sys.exit(0)

    affordances = args.affordance or list(AFFORDANCE_QUERIES.keys())

    if args.qualitative and args.scan:
        # Just visualize, no metrics
        points, colors, _ = load_scan(args.scan, max_points=args.max_points)
        for aff in affordances:
            query  = AFFORDANCE_QUERIES[aff]
            scores = run_geometry_aware(points, colors, query)
            save_visualization(args.scan, aff, points, colors, scores,
                               "geometry_aware")

    elif args.scan:
        results = evaluate_scan(
            args.scan,
            affordances=affordances,
            save=args.save,
            max_points=args.max_points,
        )
        if args.save:
            out = ROOT / "results" / "tables" / f"scannet_{args.scan}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved → {out}")

    elif args.all:
        evaluate_all(
            affordances=affordances,
            max_scans=args.max_scans,
            save=args.save,
        )

    else:
        parser.print_help()