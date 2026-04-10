import argparse
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_laso import (
    LASODataset, AFFORDANCE_QUERIES,
    precision_at_k, iou, sim,
    run_clip_only, run_geometry_only,
)
from src.features import (
    generate_synthetic_features,
    generate_synthetic_text_embedding,
    compute_text_similarity,
)
from src.geometry import (
    compute_local_geometry,
    geometric_prior_score,
    fuse_scores,
)


# ── Ablation variants ─────────────────────────────────────────────────────────

def variant_clip_only(points, colors, query):
    return run_clip_only(points, colors, query)


def variant_geometry_only(points, colors, query):
    return run_geometry_only(points, colors, query)


def variant_additive(points, colors, query):
    feats    = generate_synthetic_features(points, colors)
    text_emb = generate_synthetic_text_embedding(query)
    clip_sc  = compute_text_similarity(feats, text_emb)
    geom     = compute_local_geometry(points)
    geo_sc   = geometric_prior_score(geom, query)
    return fuse_scores(clip_sc, geo_sc, mode="weighted_sum", geo_weight=0.4)


def variant_multiply(points, colors, query):
    feats    = generate_synthetic_features(points, colors)
    text_emb = generate_synthetic_text_embedding(query)
    clip_sc  = compute_text_similarity(feats, text_emb)
    geom     = compute_local_geometry(points)
    geo_sc   = geometric_prior_score(geom, query)
    return fuse_scores(clip_sc, geo_sc, mode="multiply")


def variant_harmonic(points, colors, query):
    feats    = generate_synthetic_features(points, colors)
    text_emb = generate_synthetic_text_embedding(query)
    clip_sc  = compute_text_similarity(feats, text_emb)
    geom     = compute_local_geometry(points)
    geo_sc   = geometric_prior_score(geom, query)
    return fuse_scores(clip_sc, geo_sc, mode="harmonic")


ABLATION_VARIANTS = {
    "(A) CLIP only":           variant_clip_only,
    "(B) Geometry only":       variant_geometry_only,
    "(C) Add fusion":          variant_additive,
    "(D) Multiply (ours)":     variant_multiply,   # ← full method
    "(E) Harmonic fusion":     variant_harmonic,
}


# ── k-neighbor sensitivity ────────────────────────────────────────────────────

def ablation_k_neighbors(points, colors, gt_mask, query,
                          k_values=(5, 10, 20, 30, 50)):
    """
    Show sensitivity of geometric priors to the k_neighbors hyperparameter.
    Ideally results should be stable across a range — that's a good sign.
    """
    feats    = generate_synthetic_features(points, colors)
    text_emb = generate_synthetic_text_embedding(query)
    clip_sc  = compute_text_similarity(feats, text_emb)

    print(f"\n  k-neighbor sensitivity (query='{query}'):")
    print(f"  {'k':>5} {'P@50':>8} {'IoU':>8}")
    for k in k_values:
        geom   = compute_local_geometry(points, k_neighbors=k)
        geo_sc = geometric_prior_score(geom, query)
        fused  = fuse_scores(clip_sc, geo_sc, mode="multiply")
        p50 = precision_at_k(fused, gt_mask, min(50, len(points)))
        i_  = iou(fused, gt_mask)
        print(f"  {k:>5} {p50:>8.4f} {i_:>8.4f}")


# ── Query sensitivity ─────────────────────────────────────────────────────────

def ablation_query_variants(points, colors, gt_mask,
                             canonical_query: str,
                             query_variants: list[str]):
    """
    Show that results are stable across different phrasings of the same query.
    This matters because zero-shot methods can be brittle to query wording.
    """
    print(f"\n  Query sensitivity (affordance: '{canonical_query}'):")
    print(f"  {'Query':<40} {'P@50':>8} {'IoU':>8}")

    for query in [canonical_query] + query_variants:
        feats    = generate_synthetic_features(points, colors)
        text_emb = generate_synthetic_text_embedding(query)
        clip_sc  = compute_text_similarity(feats, text_emb)
        geom     = compute_local_geometry(points)
        geo_sc   = geometric_prior_score(geom, query)
        fused    = fuse_scores(clip_sc, geo_sc, mode="multiply")

        p50 = precision_at_k(fused, gt_mask, min(50, len(points)))
        i_  = iou(fused, gt_mask)
        marker = "  ← canonical" if query == canonical_query else ""
        print(f"  {query:<40} {p50:>8.4f} {i_:>8.4f}{marker}")


# ── Main ablation ─────────────────────────────────────────────────────────────

def run_ablation(affordances: list[str] = None,
                 n_objects: int = 30,
                 data_dir: str = "data/laso",
                 save_results: bool = False):

    if affordances is None:
        affordances = ["grasp", "support", "contain"]

    all_results = {}

    for affordance in affordances:
        query   = AFFORDANCE_QUERIES.get(affordance, f"the {affordance} region")
        dataset = LASODataset(data_dir=data_dir, affordance=affordance)
        n       = min(n_objects, len(dataset))

        print(f"\n{'='*65}")
        print(f"Ablation: '{affordance}'  (n={n} objects)")
        print(f"{'='*65}")

        accum = {v: {"p50": [], "iou": [], "sim": []}
                 for v in ABLATION_VARIANTS}

        for i in range(n):
            points, colors, gt_mask, shape_id, aff = dataset[i]
            if gt_mask.sum() == 0:
                continue

            for vname, fn in ABLATION_VARIANTS.items():
                try:
                    scores = fn(points, colors, query)
                except Exception as e:
                    continue
                k = min(50, len(points))
                accum[vname]["p50"].append(precision_at_k(scores, gt_mask, k))
                accum[vname]["iou"].append(iou(scores, gt_mask))
                accum[vname]["sim"].append(sim(scores, gt_mask))

        # Print table
        print(f"\n{'Variant':<26} {'P@50':>8} {'IoU':>8} {'SIM':>8}")
        print(f"{'-'*52}")
        results_aff = {}
        for vname in ABLATION_VARIANTS:
            if not accum[vname]["p50"]:
                continue
            p50_ = float(np.mean(accum[vname]["p50"]))
            iou_ = float(np.mean(accum[vname]["iou"]))
            sim_ = float(np.mean(accum[vname]["sim"]))
            marker = "  ◄ full method" if "ours" in vname else ""
            print(f"{vname:<26} {p50_:>8.4f} {iou_:>8.4f} {sim_:>8.4f}{marker}")
            results_aff[vname] = {"P@50": p50_, "IoU": iou_, "SIM": sim_}

        all_results[affordance] = results_aff

        # Run secondary ablations on first object with a valid GT mask
        for i in range(n):
            points, colors, gt_mask, shape_id, aff = dataset[i]
            if gt_mask.sum() > 10:
                ablation_k_neighbors(points, colors, gt_mask, query)
                ablation_query_variants(
                    points, colors, gt_mask,
                    canonical_query=query,
                    query_variants=[
                        f"region for {affordance}",
                        f"where to {affordance}",
                        f"the best spot to {affordance} this object",
                        affordance,
                    ]
                )
                break

    if save_results:
        out_path = Path("results/tables/ablation.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved → {out_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--affordances", nargs="+",
                        default=["grasp", "support", "contain"])
    parser.add_argument("--n-objects", type=int, default=30)
    parser.add_argument("--data-dir", default="data/laso")
    parser.add_argument("--save-results", action="store_true")
    args = parser.parse_args()

    run_ablation(
        affordances=args.affordances,
        n_objects=args.n_objects,
        data_dir=args.data_dir,
        save_results=args.save_results,
    )