import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")   
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.linewidth":   0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "figure.dpi":       150,
})

METHOD_COLORS = {
    "(A) CLIP only":       "#6B7280",
    "(B) Geometry only":   "#3B82F6",
    "(C) Add fusion":      "#F59E0B",
    "(D) Multiply (ours)": "#10B981",
    "(E) Harmonic fusion": "#8B5CF6",
}



def apply_heatmap_to_image(rgb: np.ndarray, scores: np.ndarray,
                            alpha: float = 0.6) -> np.ndarray:
    """Overlay a jet heatmap on an RGB image projection of the point cloud."""
    import matplotlib.cm as cm
    norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    heat = cm.plasma(norm)[:, :3]   # (N, 3) RGB from plasma colormap

    # We can't directly overlay — project to a 2D grid
    # For visualization, bin points into a top-down 2D grid
    H = W = 128
    grid_rgb  = np.ones((H, W, 3), dtype=np.float32) * 0.15
    grid_heat = np.ones((H, W, 3), dtype=np.float32) * 0.15
    grid_cnt  = np.zeros((H, W), dtype=np.int32)

    pts = np.load("/tmp/_pts_tmp.npy") if Path("/tmp/_pts_tmp.npy").exists() \
          else None

    return grid_rgb, grid_heat   # caller handles projection


def project_top_down(points: np.ndarray, colors: np.ndarray,
                      scores: np.ndarray,
                      resolution: int = 128) -> tuple:
    """
    Project a 3D point cloud to a top-down 2D image for visualization.
    Returns (rgb_img, heatmap_img) both as (H, W, 3) float32 arrays.
    """
    import matplotlib.cm as cm

    H = W = resolution
    rgb_img  = np.zeros((H, W, 3), dtype=np.float32)
    heat_img = np.zeros((H, W, 3), dtype=np.float32)
    cnt      = np.zeros((H, W), dtype=np.int32)

    xy = points[:, :2]
    xy_min, xy_max = xy.min(0), xy.max(0)
    px = ((xy - xy_min) / (xy_max - xy_min + 1e-8) * (W - 1)).astype(int)
    px = np.clip(px, 0, W - 1)

    norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    heat_rgb = cm.plasma(norm_scores)[:, :3]

    for i in range(len(points)):
        u, v = px[i, 0], px[i, 1]
        rgb_img[v, u]  += colors[i]
        heat_img[v, u] += heat_rgb[i]
        cnt[v, u] += 1

    mask = cnt > 0
    rgb_img[mask]  /= cnt[mask, np.newaxis]
    heat_img[mask] /= cnt[mask, np.newaxis]

    return np.clip(rgb_img, 0, 1), np.clip(heat_img, 0, 1)



def figure_comparison(n_examples: int = 4, affordance: str = "grasp"):
    """
    Main qualitative figure: for each example object, show
    [RGB projection] [CLIP-only] [Geometry-aware (ours)] [Ground Truth]
    """
    from experiments.run_laso import (
        LASODataset, AFFORDANCE_QUERIES,
        run_clip_only, run_geometry_aware,
    )

    query   = AFFORDANCE_QUERIES.get(affordance, f"the {affordance} region")
    dataset = LASODataset(affordance=affordance)

    fig, axes = plt.subplots(
        n_examples, 4,
        figsize=(12, 3 * n_examples),
        gridspec_kw={"wspace": 0.05, "hspace": 0.3}
    )

    col_labels = ["RGB", "CLIP-only", "Ours (geometry-aware)", "Ground truth"]
    for ax, label in zip(axes[0], col_labels):
        ax.set_title(label, fontsize=10, fontweight="bold", pad=4)

    found = 0
    for i in range(len(dataset)):
        if found >= n_examples:
            break
        points, colors, gt_mask, shape_id, aff = dataset[i]
        if gt_mask.sum() < 5:
            continue

        clip_sc  = run_clip_only(points, colors, query)
        ours_sc  = run_geometry_aware(points, colors, query)
        gt_sc    = gt_mask.astype(np.float32)

        rgb_img, _          = project_top_down(points, colors, clip_sc)
        _, clip_heat        = project_top_down(points, colors, clip_sc)
        _, ours_heat        = project_top_down(points, colors, ours_sc)
        _, gt_heat          = project_top_down(points, colors, gt_sc)

        row = axes[found]
        for ax, img in zip(row, [rgb_img, clip_heat, ours_heat, gt_heat]):
            ax.imshow(img, interpolation="bilinear")
            ax.axis("off")
        row[0].set_ylabel(shape_id[:12], fontsize=8, rotation=0, ha="right",
                          labelpad=4)

        found += 1

    fig.suptitle(f"Affordance: '{affordance}'  |  Query: \"{query}\"",
                 fontsize=11, y=1.01)

    out = FIGURES_DIR / f"comparison_{affordance}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[figures] Saved → {out}")



def figure_ablation_bars(results: dict = None, affordance: str = "grasp"):
    """
    Bar chart comparing all ablation variants on P@50.
    If results not provided, runs the ablation first.
    """
    if results is None:
        from experiments.ablation import run_ablation
        results = run_ablation(affordances=[affordance], n_objects=20)
        results = results.get(affordance, {})

    if not results:
        print("[figures] No ablation results to plot.")
        return

    variants = list(results.keys())
    p50_vals = [results[v]["P@50"] for v in variants]
    colors   = [METHOD_COLORS.get(v, "#9CA3AF") for v in variants]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(variants)), p50_vals, color=colors,
                  width=0.6, edgecolor="white", linewidth=0.5)

    # Highlight our method
    for bar, v in zip(bars, variants):
        if "ours" in v.lower():
            bar.set_edgecolor("#064E3B")
            bar.set_linewidth(2)

    # Value labels on top of bars
    for bar, val in zip(bars, p50_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([v.replace("(", "").replace(")", "")
                         for v in variants],
                        rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Precision@50", fontsize=11)
    ax.set_title(f"Ablation study — '{affordance}' affordance", fontsize=12)
    ax.set_ylim(0, max(p50_vals) * 1.15)

    # Baseline reference line
    if results:
        baseline = results.get("(A) CLIP only", {}).get("P@50", 0)
        if baseline > 0:
            ax.axhline(baseline, color="#6B7280", linestyle="--",
                       linewidth=0.8, alpha=0.6, label="CLIP-only baseline")
            ax.legend(fontsize=9)

    fig.tight_layout()
    out = FIGURES_DIR / f"ablation_{affordance}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] Saved → {out}")



def figure_pr_curves(affordance: str = "grasp", n_objects: int = 20):
    """
    Precision-recall curves at varying thresholds.
    A method with a higher area under the PR curve is uniformly better.
    """
    from experiments.run_laso import (
        LASODataset, AFFORDANCE_QUERIES,
        run_clip_only, run_geometry_aware,
    )
    from experiments.ablation import variant_geometry_only

    query   = AFFORDANCE_QUERIES.get(affordance, f"the {affordance} region")
    dataset = LASODataset(affordance=affordance)
    n       = min(n_objects, len(dataset))

    thresholds = np.linspace(0.1, 0.95, 30)

    methods_pr = {
        "CLIP-only":           {"fn": run_clip_only,       "prec": [], "rec": []},
        "Geometry-only":       {"fn": variant_geometry_only,"prec": [], "rec": []},
        "Geometry-aware (ours)":{"fn": run_geometry_aware, "prec": [], "rec": []},
    }

    for t in thresholds:
        for mname, mdata in methods_pr.items():
            precs, recs = [], []
            for i in range(n):
                points, colors, gt_mask, shape_id, aff = dataset[i]
                if gt_mask.sum() == 0:
                    continue
                scores = mdata["fn"](points, colors, query)
                norm   = (scores - scores.min()) / \
                         (scores.max() - scores.min() + 1e-8)
                pred   = norm > t
                tp = (pred & gt_mask).sum()
                fp = (pred & ~gt_mask).sum()
                fn_ = (~pred & gt_mask).sum()
                precs.append(tp / (tp + fp + 1e-8))
                recs.append(tp / (tp + fn_ + 1e-8))
            mdata["prec"].append(float(np.mean(precs)) if precs else 0)
            mdata["rec"].append(float(np.mean(recs)) if recs else 0)

    fig, ax = plt.subplots(figsize=(6, 5))

    style = {"CLIP-only": ("--", "#6B7280"),
             "Geometry-only": (":", "#3B82F6"),
             "Geometry-aware (ours)": ("-", "#10B981")}

    for mname, mdata in methods_pr.items():
        ls, color = style[mname]
        lw = 2.5 if "ours" in mname else 1.5
        ax.plot(mdata["rec"], mdata["prec"],
                linestyle=ls, color=color, linewidth=lw, label=mname)

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(f"Precision-Recall — '{affordance}'", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    out = FIGURES_DIR / f"pr_curve_{affordance}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] Saved → {out}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure",
                        choices=["comparison", "ablation", "pr", "all"],
                        default="all")
    parser.add_argument("--affordance", default="grasp")
    parser.add_argument("--n-objects",  type=int, default=20)
    args = parser.parse_args()

    if args.figure in ("comparison", "all"):
        figure_comparison(n_examples=4, affordance=args.affordance)

    if args.figure in ("ablation", "all"):
        figure_ablation_bars(affordance=args.affordance)

    if args.figure in ("pr", "all"):
        figure_pr_curves(affordance=args.affordance,
                         n_objects=args.n_objects)

    print(f"\nAll figures saved to {FIGURES_DIR}/")