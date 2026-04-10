"""
web/app.py
FastAPI backend for the portfolio demo.

Endpoints:
    POST /api/score             — run affordance scoring on uploaded point cloud
    GET  /api/sample/{name}     — load a precomputed demo scene
    GET  /api/scannet/{scan_id} — load a ScanNet scene by ID (e.g. scene0000_00)
    GET  /api/affordances       — list available affordance queries
    GET  /health                — health check

Run locally:
    pip install fastapi "uvicorn[standard]" python-multipart
    uvicorn web.app:app --reload --port 8000

ScanNet data directory (set via env var SCANNET_DIR or defaults to data/scannet/):
    data/scannet/
        scene0000_00/
            scene0000_00_vh_clean_2.ply   ← dense mesh (use this)
            scene0000_00.txt              ← metadata
        scene0001_00/
            ...
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import io
import json
import os
import pickle
import struct
import sys
import time
import zlib
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Fix sys.path once at module level ─────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
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
from src.pointcloud import (
    generate_synthetic_scene,
    voxel_downsample,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

STATIC_DIR  = Path(__file__).parent / "static"
DEMO_DIR    = Path(__file__).parent / "demo_scenes"
SCANNET_DIR = Path(os.environ.get("SCANNET_DIR", ROOT / "data" / "scannet" / "scans"))

# ── Scene configuration ───────────────────────────────────────────────────────
# Each scene has its own seed and object count so selecting "Mug" shows
# a single mug, "Chair" shows a single chair, etc.
SCENE_CONFIG = {
    "mug":    {"seed": 0,  "n_objects": 1},
    "chair":  {"seed": 10, "n_objects": 1},
    "bottle": {"seed": 20, "n_objects": 1},
    "table":  {"seed": 30, "n_objects": 1},
}

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Preparing demo scenes...")
    try:
        _prepare_demo_scenes()
        print("[startup] Ready.")
    except Exception as e:
        print(f"[startup] Warning: scene prep failed ({e}). Will generate on demand.")
    yield


app = FastAPI(
    title="Zero-Shot 3D Affordance Mapper",
    description="Geometry-aware zero-shot affordance mapping — CLIP + geometric priors",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Affordance queries ────────────────────────────────────────────────────────

AFFORDANCES = {
    "grasp":      "the graspable region suitable for picking up with fingers",
    "contain":    "the hollow region that can contain objects inside",
    "support":    "the flat horizontal surface that supports objects on top",
    "wrap-grasp": "the cylindrical region for wrapping a full hand around",
    "open":       "the handle or edge region for opening by pulling",
    "pour":       "the spout or opening for pouring liquid",
    "move":       "the part to push or drag to move the object",
    "sit":        "the seat surface for sitting on",
    "lay":        "the flat surface for laying objects down",
}

# ── ScanNet loader ────────────────────────────────────────────────────────────

def load_scannet_scene(scan_id: str,
                        max_points: int = 50_000) -> tuple[np.ndarray, np.ndarray]:
    scan_dir = SCANNET_DIR / scan_id
    ply_path = scan_dir / f"{scan_id}_vh_clean_2.ply"

    if not ply_path.exists():
        raise FileNotFoundError(
            f"ScanNet scene not found: {ply_path}\n"
            f"Download with: python download-scannet.py -o data/scannet --id {scan_id}"
        )

    points, colors = _parse_ply(ply_path)

    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    print(f"[scannet] Loaded {scan_id}: {len(points):,} points")
    return points.astype(np.float32), colors.astype(np.float32)


def _parse_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        n_vertices = 0
        properties = []
        in_vertex = False
        binary_little = True

        for line in header_lines:
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
                in_vertex = True
            elif line.startswith("element") and "vertex" not in line:
                in_vertex = False
            elif line.startswith("property") and in_vertex:
                parts = line.split()
                properties.append((parts[1], parts[2]))
            elif "binary_big_endian" in line:
                binary_little = False

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

        dt = np.dtype([
            (name, dtype_map.get(ptype, "f4"))
            for ptype, name in properties
        ])

        if not binary_little:
            dt = dt.newbyteorder(">")

        raw = np.frombuffer(f.read(n_vertices * dt.itemsize), dtype=dt)

    points = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float32)

    if "red" in raw.dtype.names:
        colors = np.stack([raw["red"], raw["green"], raw["blue"]],
                           axis=1).astype(np.float32) / 255.0
    elif "r" in raw.dtype.names:
        colors = np.stack([raw["r"], raw["g"], raw["b"]],
                           axis=1).astype(np.float32) / 255.0
    else:
        colors = np.ones((len(points), 3), dtype=np.float32) * 0.6

    return points, colors


def list_scannet_scenes() -> list[str]:
    if not SCANNET_DIR.exists():
        return []
    return sorted(
        d.name for d in SCANNET_DIR.iterdir()
        if d.is_dir() and d.name.startswith("scene")
    )


# ── Core scoring ──────────────────────────────────────────────────────────────

def score_pointcloud(points: np.ndarray, colors: np.ndarray,
                     affordance: str,
                     method: str = "geometry_aware") -> dict:
    query = AFFORDANCES.get(affordance, f"the {affordance} region")
    t0    = time.time()

    feats    = generate_synthetic_features(points, colors)
    text_emb = generate_synthetic_text_embedding(query)
    clip_sc  = compute_text_similarity(feats, text_emb)

    if method in ("geometry_aware", "geometry_only"):
        k      = min(20, len(points) - 1)
        geom   = compute_local_geometry(points, k_neighbors=k)
        geo_sc = geometric_prior_score(geom, query)

    if method == "clip_only":
        scores = clip_sc
    elif method == "geometry_only":
        scores = geo_sc
    else:
        scores = fuse_scores(clip_sc, geo_sc, mode="multiply")

    scores   = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    top_mask = scores > 0.7
    centroid = (points[top_mask].mean(axis=0).tolist()
                if top_mask.sum() > 0 else points.mean(axis=0).tolist())

    return {
        "scores":    scores.tolist(),
        "points":    points.tolist(),
        "colors":    colors.tolist(),
        "centroid":  centroid,
        "n_points":  len(points),
        "n_top":     int(top_mask.sum()),
        "method":    method,
        "affordance": affordance,
        "query":     query,
        "elapsed_s": round(time.time() - t0, 3),
    }


# ── Demo scene cache ──────────────────────────────────────────────────────────

def _prepare_demo_scenes():
    print("running prepare demo scenes")
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    for name, cfg in SCENE_CONFIG.items():
        print(f"Processing: {name}")
        path = DEMO_DIR / f"{name}.pkl"
        # FIX: invalidate old cache entries that used n_objects=4
        if path.exists():
            try:
                with open(path, "rb") as f:
                    cached = pickle.load(f)
                # If the cache was built with n_objects=4 it will have far more
                # points than a single-object scene — regenerate it.
                if len(cached.get("points", [])) > 8000:
                    print(f"[startup] Stale cache for {name}, regenerating…")
                    path.unlink()
            except Exception:
                path.unlink(missing_ok=True)

        if not path.exists():
            print(f"Generating: {name}")
            pts, col, _ = generate_synthetic_scene(
                n_objects=cfg["n_objects"],   # ← FIX: 1 object per scene
                seed=cfg["seed"],
            )
            pts, col = voxel_downsample(pts, col, voxel_size=0.02)
            with open(path, "wb") as f:
                pickle.dump({"points": pts.tolist(), "colors": col.tolist()}, f)
            print(f"[startup] Cached: {name}")
    print("Done")


def _load_demo_scene(name: str) -> tuple[np.ndarray, np.ndarray]:
    path = DEMO_DIR / f"{name}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            data = pickle.load(f)
        return np.array(data["points"]), np.array(data["colors"])

    # Generate on the fly if cache missing
    cfg = SCENE_CONFIG.get(name, {"seed": 42, "n_objects": 1})
    pts, col, _ = generate_synthetic_scene(
        n_objects=cfg["n_objects"],
        seed=cfg["seed"],
    )
    pts, col = voxel_downsample(pts, col, voxel_size=0.02)
    return pts, col


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    scenes = list_scannet_scenes()
    return {
        "status":         "ok",
        "version":        "1.0.0",
        "scannet_dir":    str(SCANNET_DIR),
        "scannet_scenes": len(scenes),
        "scannet_ids":    scenes[:10],
    }


@app.get("/api/affordances")
def list_affordances():
    return {"affordances": list(AFFORDANCES.keys()), "queries": AFFORDANCES}


@app.get("/api/scenes")
def list_scenes():
    return {
        "synthetic": list(SCENE_CONFIG.keys()),
        "scannet":   list_scannet_scenes(),
    }


@app.get("/api/sample/{scene_name}")
def get_sample(scene_name: str,
               affordance: str = "grasp",
               method: str = "geometry_aware"):
    try:
        points, colors = _load_demo_scene(scene_name)
        return JSONResponse(score_pointcloud(points, colors, affordance, method))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scannet/{scan_id}")
def get_scannet(scan_id: str,
                affordance: str = "grasp",
                method: str = "geometry_aware",
                max_points: int = 5000):
    import re
    if not re.fullmatch(r"scene\d{4}_\d{2}", scan_id):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scan_id format '{scan_id}'. Expected: scene0000_00"
        )
    try:
        points, colors = load_scannet_scene(scan_id, max_points=max_points)
        result = score_pointcloud(points, colors, affordance, method)
        result["scan_id"] = scan_id
        return JSONResponse(result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/score")
async def score_upload(
    file:       UploadFile = File(...),
    affordance: str        = Form("grasp"),
    method:     str        = Form("geometry_aware"),
    max_points: int        = Form(5000),
):
    content = await file.read()
    suffix  = Path(file.filename).suffix.lower()

    try:
        if suffix == ".npy":
            arr    = np.load(io.BytesIO(content))
            points = arr[:, :3].astype(np.float32)
            colors = (arr[:, 3:6] / 255.0 if arr.shape[1] >= 6
                      else np.full((len(arr), 3), 0.6, dtype=np.float32))

        elif suffix == ".ply":
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)
            points, colors = _parse_ply(tmp_path)
            tmp_path.unlink(missing_ok=True)

        elif suffix == ".pkl":
            data   = pickle.loads(content)
            points = np.array(data["points"], dtype=np.float32)
            colors = np.array(data.get("colors", np.full((len(points), 3), 0.6)),
                              dtype=np.float32)
            if colors.max() > 1.01:
                colors /= 255.0

        else:
            raise ValueError(f"Unsupported: {suffix}. Use .npy, .ply, or .pkl")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File parse error: {e}")

    if len(points) > max_points:
        idx    = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    try:
        return JSONResponse(score_pointcloud(points, colors, affordance, method))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/compare/{scene_name}")
def compare_methods(scene_name: str,
                    affordance: str = "grasp",
                    source: str = "synthetic"):
    try:
        if source == "scannet":
            import re
            if not re.fullmatch(r"scene\d{4}_\d{2}", scene_name):
                raise ValueError("Invalid scan_id format")
            points, colors = load_scannet_scene(scene_name)
        else:
            points, colors = _load_demo_scene(scene_name)

        results = {}
        for method in ("clip_only", "geometry_only", "geometry_aware"):
            r = score_pointcloud(points, colors, affordance, method)
            results[method] = {"scores": r["scores"], "elapsed_s": r["elapsed_s"]}

        results["points"]   = points.tolist()
        results["colors"]   = colors.tolist()
        results["n_points"] = len(points)
        return JSONResponse(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))