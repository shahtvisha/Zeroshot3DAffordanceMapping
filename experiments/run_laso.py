import argparse
import json
import sys
import pickle
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from zeroshot3d.features import (
    generate_synthetic_features,
    generate_synthetic_text_embedding,
    compute_text_similarity,
)
from zeroshot3d.geometry import (
    compute_local_geometry,
    geometric_prior_score,
    fuse_scores,
)


# Affordance → language mapping

AFFORDANCE_QUERIES = {
    "grasp": "the graspable region suitable for picking up",
    "contain": "the region that can contain or hold objects inside",
    "support": "the flat supporting surface to place objects on",
    "wrap-grasp": "the region suitable for wrapping fingers around",
    "lay": "the flat surface to lay something on",
    "sit": "the surface suitable for sitting on",
    "display": "the surface for displaying objects",
    "open": "the handle or region for opening",
    "pour": "the region for pouring liquid",
    "move": "the part to push or move the object",
}


# Metrics

def precision_at_k(scores, gt_mask, k):
    top_idx = np.argsort(scores)[-k:]
    return float(gt_mask[top_idx].sum()) / k


def iou(scores, gt_mask, threshold=0.5):
    norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    pred = norm > threshold
    inter = (pred & gt_mask).sum()
    union = (pred | gt_mask).sum()
    return float(inter) / (union + 1e-8)


def sim(scores, gt_mask):
    norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    gt_f = gt_mask.astype(np.float32)
    inter = np.minimum(norm, gt_f).sum()
    union = np.maximum(norm, gt_f).sum()
    return float(inter) / (union + 1e-8)


# Dataset

class LASODataset:
    def __init__(self, data_dir="data/laso", split="test", affordance=None):
        self.data_dir = Path(data_dir)
        self.split = split

        # anno_path = self.data_dir / f"anno_{split}.pkl"
        # obj_path = self.data_dir / f"object_{split}.pkl"
        anno_path = Path("data/laso/anno_test.pkl")
        obj_path = Path("data/laso/objects_test.pkl")

        if not anno_path.exists() or not obj_path.exists():
            raise FileNotFoundError(
                f"Missing dataset files:\n{anno_path}\n{obj_path}"
            )

        print(f"[LASO] Loading {anno_path}")
        with open(anno_path, "rb") as f:
            self.annotations = pickle.load(f)

        print(f"[LASO] Loading {obj_path}")
        with open(obj_path, "rb") as f:
            self.objects = pickle.load(f)

        if affordance:
            self.annotations = [
                a for a in self.annotations
                if a["affordance"] == affordance
            ]

        print(f"[LASO] {len(self.annotations)} samples loaded")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations[idx]

        shape_id = sample["shape_id"]
        affordance = sample["affordance"]
        mask = sample["mask"].astype(bool)

        obj = self.objects[shape_id]

        # --- Handle multiple possible formats ---
        if isinstance(obj, dict):
            if "points" in obj:
                points = obj["points"]
            elif "xyz" in obj:
                points = obj["xyz"]
            else:
                raise ValueError(f"Unknown object format keys: {obj.keys()}")

            colors = obj.get("colors", np.ones_like(points) * 0.5)

        else:
            pc = np.array(obj)
            points = pc[:, :3]
            colors = (
                pc[:, 3:6] / 255.0
                if pc.shape[1] >= 6
                else np.ones((len(pc), 3)) * 0.5
            )

        points = points.astype(np.float32)
        colors = colors.astype(np.float32)

        if len(points) != len(mask):
            raise ValueError(f"Point/mask mismatch for {shape_id}")

        return points, colors, mask, shape_id, affordance


# Methods

def run_clip_only(points, colors, query):
    feats = generate_synthetic_features(points, colors)
    text = generate_synthetic_text_embedding(query)
    return compute_text_similarity(feats, text)


def run_geometry_only(points, colors, query):
    geom = compute_local_geometry(points, k_neighbors=20)
    return geometric_prior_score(geom, query)


def run_geometry_aware(points, colors, query):
    feats = generate_synthetic_features(points, colors)
    text = generate_synthetic_text_embedding(query)

    clip_scores = compute_text_similarity(feats, text)

    geom = compute_local_geometry(points, k_neighbors=20)
    geo_scores = geometric_prior_score(geom, query)

    return fuse_scores(clip_scores, geo_scores, mode="multiply")


METHODS = {
    "clip_only": run_clip_only,
    "geometry_only": run_geometry_only,
    "geometry_aware": run_geometry_aware,
}


# Evaluation


def evaluate(data_dir="data/laso",
             split="test",
             affordance=None,
             methods=None,
             save_results=False):

    if methods is None:
        methods = list(METHODS.keys())

    dataset = LASODataset(data_dir, split, affordance)

    metric_accum = {
        m: {"p50": [], "p100": [], "iou": [], "sim": []}
        for m in methods
    }

    for i in range(len(dataset)):
        points, colors, gt_mask, shape_id, aff = dataset[i]

        if gt_mask.sum() == 0:
            continue

        query = AFFORDANCE_QUERIES.get(
            aff, f"the {aff} region"
        )

        for method in methods:
            fn = METHODS[method]

            try:
                scores = fn(points, colors, query)
            except Exception as e:
                print(f"[warn] {method} failed on {shape_id}: {e}")
                continue

            k50 = min(50, len(points))
            k100 = min(100, len(points))

            metric_accum[method]["p50"].append(
                precision_at_k(scores, gt_mask, k50)
            )
            metric_accum[method]["p100"].append(
                precision_at_k(scores, gt_mask, k100)
            )
            metric_accum[method]["iou"].append(
                iou(scores, gt_mask)
            )
            metric_accum[method]["sim"].append(
                sim(scores, gt_mask)
            )

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(dataset)}] processed")

    # Aggregate
    results = {}
    for method in methods:
        m = metric_accum[method]
        if not m["p50"]:
            continue

        results[method] = {
            "P@50": float(np.mean(m["p50"])),
            "P@100": float(np.mean(m["p100"])),
            "IoU": float(np.mean(m["iou"])),
            "SIM": float(np.mean(m["sim"])),
            "n": len(m["p50"]),
        }

    print_results(results)

    if save_results:
        out = Path("results/tables/laso_results.json")
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Saved → {out}")

    return results


def print_results(results):
    print("\n" + "=" * 65)
    print("LASO Evaluation")
    print("=" * 65)
    print(f"{'Method':<22} {'P@50':>7} {'P@100':>7} {'IoU':>7} {'SIM':>7} {'n':>5}")
    print("-" * 65)

    for name, r in results.items():
        marker = " ◄" if name == "geometry_aware" else ""
        print(f"{name:<22} {r['P@50']:>7.4f} {r['P@100']:>7.4f} "
              f"{r['IoU']:>7.4f} {r['SIM']:>7.4f} {r['n']:>5}{marker}")

    print("=" * 65 + "\n")


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", default="data/laso")
    parser.add_argument("--split", default="test")
    parser.add_argument("--affordance", default=None)
    parser.add_argument("--save-results", action="store_true")

    args = parser.parse_args()

    evaluate(
        data_dir=args.data_dir,
        split=args.split,
        affordance=args.affordance,
        save_results=args.save_results,
    )