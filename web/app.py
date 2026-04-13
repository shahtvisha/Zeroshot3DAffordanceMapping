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

# ── Lifespan (replaces deprecated @app.on_event("startup")) ───────────────────
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once on startup before the server accepts requests."""
    print("[startup] Preparing demo scenes...")
    try:
        _prepare_demo_scenes()
        print("[startup] Ready.")
    except Exception as e:
        # Don't crash the server if scene prep fails — endpoints will
        # generate scenes on-the-fly as fallback
        print(f"[startup] Warning: scene prep failed ({e}). Will generate on demand.")
    yield
    # anything after yield runs on shutdown — nothing needed here
 
# ── App ───────────────────────────────────────────────────────────────────────
 
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
    """
    Load a ScanNet scene from its _vh_clean_2.ply file.
 
    ScanNet naming: scene%04d_%02d  e.g. scene0000_00
    Expected file:  data/scannet/{scan_id}/{scan_id}_vh_clean_2.ply
 
    The .ply is a binary PLY with vertex properties: x y z nx ny nz r g b
    We parse it manually to avoid requiring open3d or plyfile as hard deps.
 
    Returns:
        points: (N, 3) float32  XYZ in meters
        colors: (N, 3) float32  RGB in [0, 1]
    """
    scan_dir = SCANNET_DIR / scan_id
    ply_path = scan_dir / f"{scan_id}_vh_clean_2.ply"
 
    if not ply_path.exists():
        raise FileNotFoundError(
            f"ScanNet scene not found: {ply_path}\n"
            f"Download with: python download-scannet.py -o data/scannet --id {scan_id}"
        )
 
    points, colors = _parse_ply(ply_path)
 
    # Downsample large scenes for API responsiveness
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
 
    print(f"[scannet] Loaded {scan_id}: {len(points):,} points")
    return points.astype(np.float32), colors.astype(np.float32)
 
 
def _parse_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a binary PLY file without external dependencies.
    Handles the standard ScanNet vertex layout: x y z nx ny nz r g b
    """
    with open(path, "rb") as f:
        # Read ASCII header
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            header_lines.append(line)
            if line == "end_header":
                break
 
        # Parse header for vertex count and property order
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
                properties.append((parts[1], parts[2]))   # (dtype_str, name)
            elif "binary_big_endian" in line:
                binary_little = False
 
        # Map PLY dtype strings to numpy
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
 
    # Extract XYZ
    points = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float32)
 
    # Extract RGB — ScanNet stores as uint8 r g b
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
    """Return all downloaded scan IDs sorted."""
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
    """Run affordance scoring and return result dict."""
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
 
 
# ── Demo scene generators — distinct recognizable objects ─────────────────────
 
def _make_mug(n: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Mug: cylinder body + smaller cylinder handle protruding from the side.
    Body is terracotta red, handle is slightly lighter.
    """
    rng = np.random.default_rng(seed)
    pts, col = [], []
 
    # ── Body: hollow cylinder surface (not filled) ──
    n_body = int(n * 0.70)
    theta  = rng.uniform(0, 2 * np.pi, n_body)
    height = rng.uniform(0.0, 0.12, n_body)
    r_body = 0.038 + rng.normal(0, 0.001, n_body)
    x = r_body * np.cos(theta)
    z = r_body * np.sin(theta)
    pts.append(np.stack([x, height, z], axis=1))
    body_col = np.array([[0.82, 0.28, 0.18]] * n_body) + rng.normal(0, 0.02, (n_body, 3))
    col.append(body_col)
 
    # ── Base disc ──
    n_base = int(n * 0.08)
    r_b    = rng.uniform(0, 0.038, n_base)
    t_b    = rng.uniform(0, 2 * np.pi, n_base)
    pts.append(np.stack([r_b*np.cos(t_b), np.zeros(n_base), r_b*np.sin(t_b)], axis=1))
    col.append(np.array([[0.70, 0.22, 0.14]] * n_base) + rng.normal(0, 0.01, (n_base, 3)))
 
    # ── Handle: small curved tube on the right side ──
    n_handle = n - n_body - n_base
    # Arc from y=0.03 to y=0.09, protruding in +x direction
    arc    = rng.uniform(0.2, np.pi - 0.2, n_handle)   # arc angle
    handle_r = 0.028   # radius of the arc
    offset_x = 0.038   # flush with body surface
    hx = offset_x + handle_r * np.cos(arc)
    hy = 0.03 + 0.06 * (arc - 0.2) / (np.pi - 0.4)    # runs up the side
    hz = rng.normal(0, 0.005, n_handle)                 # thin tube
    pts.append(np.stack([hx, hy, hz], axis=1))
    col.append(np.array([[0.88, 0.38, 0.26]] * n_handle) + rng.normal(0, 0.02, (n_handle, 3)))
 
    P = np.concatenate(pts).astype(np.float32)
    C = np.clip(np.concatenate(col), 0, 1).astype(np.float32)
    return P, C
 
 
def _make_chair(n: int = 2500, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Chair: seat (flat box), backrest (tall flat panel), four cylindrical legs.
    """
    rng = np.random.default_rng(seed)
    pts, col = [], []
    beige  = np.array([0.76, 0.70, 0.60])
    dark   = np.array([0.35, 0.28, 0.20])
 
    def box_surface(cx, cy, cz, w, h, d, color, count):
        """Sample points on surface of an axis-aligned box."""
        # 6 faces, weighted by area
        faces = []
        # top/bottom (XZ plane)
        for y_off in [0, h]:
            ux = rng.uniform(cx-w/2, cx+w/2, count//6)
            uz = rng.uniform(cz-d/2, cz+d/2, count//6)
            faces.append(np.stack([ux, np.full_like(ux, cy+y_off), uz], axis=1))
        # front/back (XY plane)
        for z_off in [-d/2, d/2]:
            ux = rng.uniform(cx-w/2, cx+w/2, count//6)
            uy = rng.uniform(cy, cy+h, count//6)
            faces.append(np.stack([ux, uy, np.full_like(ux, cz+z_off)], axis=1))
        # left/right (YZ plane)
        for x_off in [-w/2, w/2]:
            uy = rng.uniform(cy, cy+h, count//6)
            uz = rng.uniform(cz-d/2, cz+d/2, count//6)
            faces.append(np.stack([np.full_like(uy, cx+x_off), uy, uz], axis=1))
        p = np.concatenate(faces)
        c = np.tile(color, (len(p), 1)) + rng.normal(0, 0.02, (len(p), 3))
        return p, c
 
    # Seat
    p, c = box_surface(0, 0.45, 0, 0.42, 0.05, 0.40, beige, n//3)
    pts.append(p); col.append(c)
 
    # Backrest
    p, c = box_surface(0, 0.50, -0.19, 0.40, 0.42, 0.04, beige, n//4)
    pts.append(p); col.append(c)
 
    # Four legs: cylinders
    leg_positions = [(-0.17, -0.19), (0.17, -0.19), (-0.17, 0.17), (0.17, 0.17)]
    n_leg = (n - n//3 - n//4) // 4
    for lx, lz in leg_positions:
        t  = rng.uniform(0, 2*np.pi, n_leg)
        hy = rng.uniform(0.0, 0.45, n_leg)
        lr = 0.016 + rng.normal(0, 0.001, n_leg)
        p  = np.stack([lx + lr*np.cos(t), hy, lz + lr*np.sin(t)], axis=1)
        c  = np.tile(dark, (n_leg,1)) + rng.normal(0, 0.015, (n_leg, 3))
        pts.append(p); col.append(c)
 
    P = np.concatenate(pts).astype(np.float32)
    C = np.clip(np.concatenate(col), 0, 1).astype(np.float32)
    return P, C
 
 
def _make_bottle(n: int = 2000, seed: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Bottle: wide cylindrical body tapering to a narrow neck.
    """
    rng = np.random.default_rng(seed)
    pts, col = [], []
    green = np.array([0.15, 0.55, 0.30])
 
    heights = rng.uniform(0, 0.26, n)
    # Radius profile: wide at base, narrows at shoulder (h=0.18), then neck (h>0.20)
    def radius_at(h):
        r = np.where(h < 0.17, 0.035,                       # body
            np.where(h < 0.20, 0.035 - (h-0.17)/0.03*0.020, # shoulder taper
                               0.015))                        # neck
        return r + rng.normal(0, 0.001, len(h))
 
    r  = radius_at(heights)
    th = rng.uniform(0, 2*np.pi, n)
    x  = r * np.cos(th)
    z  = r * np.sin(th)
    pts.append(np.stack([x, heights, z], axis=1))
    c   = np.tile(green, (n,1)) + rng.normal(0, 0.025, (n,3))
    col.append(c)
 
    # Base disc
    n_base = n // 8
    r_b = rng.uniform(0, 0.035, n_base)
    t_b = rng.uniform(0, 2*np.pi, n_base)
    pts.append(np.stack([r_b*np.cos(t_b), np.zeros(n_base), r_b*np.sin(t_b)], axis=1))
    col.append(np.tile(green*0.8, (n_base,1)))
 
    P = np.concatenate(pts).astype(np.float32)
    C = np.clip(np.concatenate(col), 0, 1).astype(np.float32)
    return P, C
 
 
def _make_table(n: int = 2500, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Table: flat rectangular top surface + four legs.
    """
    rng = np.random.default_rng(seed)
    pts, col = [], []
    wood  = np.array([0.72, 0.52, 0.32])
    dark  = np.array([0.40, 0.28, 0.18])
 
    # Tabletop — surface points only (top face dominant)
    n_top = int(n * 0.45)
    tx = rng.uniform(-0.55, 0.55, n_top)
    tz = rng.uniform(-0.30, 0.30, n_top)
    ty = np.full(n_top, 0.75) + rng.normal(0, 0.002, n_top)
    pts.append(np.stack([tx, ty, tz], axis=1))
    col.append(np.tile(wood, (n_top,1)) + rng.normal(0, 0.02, (n_top,3)))
 
    # Tabletop sides (thin slab)
    n_side = int(n * 0.10)
    for axis in ['x', 'z']:
        for sign in [-1, 1]:
            k = n_side // 4
            if axis == 'x':
                pts.append(np.stack([
                    np.full(k, sign * 0.55) + rng.normal(0,.001,k),
                    rng.uniform(0.73, 0.75, k),
                    rng.uniform(-0.30, 0.30, k)], axis=1))
            else:
                pts.append(np.stack([
                    rng.uniform(-0.55, 0.55, k),
                    rng.uniform(0.73, 0.75, k),
                    np.full(k, sign * 0.30) + rng.normal(0,.001,k)], axis=1))
            col.append(np.tile(wood*0.9, (k,1)) + rng.normal(0,0.015,(k,3)))
 
    # Four legs
    leg_pos = [(-0.48, -0.24), (0.48, -0.24), (-0.48, 0.24), (0.48, 0.24)]
    n_leg   = (n - n_top - n_side) // 4
    for lx, lz in leg_pos:
        t  = rng.uniform(0, 2*np.pi, n_leg)
        hy = rng.uniform(0.0, 0.73, n_leg)
        lr = 0.022 + rng.normal(0, 0.001, n_leg)
        pts.append(np.stack([lx + lr*np.cos(t), hy, lz + lr*np.sin(t)], axis=1))
        col.append(np.tile(dark, (n_leg,1)) + rng.normal(0, 0.015, (n_leg,3)))
 
    P = np.concatenate(pts).astype(np.float32)
    C = np.clip(np.concatenate(col), 0, 1).astype(np.float32)
    return P, C
 
 
SCENE_BUILDERS = {
    "mug":    _make_mug,
    "chair":  _make_chair,
    "bottle": _make_bottle,
    "table":  _make_table,
}
 
 
def _prepare_demo_scenes():
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    for name, builder in SCENE_BUILDERS.items():
        path = DEMO_DIR / f"{name}.pkl"
        # Always regenerate — delete old tabletop-scene caches
        pts, col = builder()
        with open(path, "wb") as f:
            pickle.dump({"points": pts.tolist(), "colors": col.tolist()}, f)
        print(f"[startup] Built scene: {name} ({len(pts):,} pts)")
 
 
def _load_demo_scene(name: str) -> tuple[np.ndarray, np.ndarray]:
    path = DEMO_DIR / f"{name}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            data = pickle.load(f)
        return np.array(data["points"]), np.array(data["colors"])
    # Build on the fly if cache missing
    builder = SCENE_BUILDERS.get(name, _make_mug)
    return builder()
 
 
# ── Endpoints ─────────────────────────────────────────────────────────────────
 
@app.get("/health")
def health():
    scenes = list_scannet_scenes()
    return {
        "status":         "ok",
        "version":        "1.0.0",
        "scannet_dir":    str(SCANNET_DIR),
        "scannet_scenes": len(scenes),
        "scannet_ids":    scenes[:10],   # first 10 as preview
    }
 
 
@app.get("/api/affordances")
def list_affordances():
    return {"affordances": list(AFFORDANCES.keys()), "queries": AFFORDANCES}
 
 
@app.get("/api/scenes")
def list_scenes():
    """List all available scenes: synthetic demos + downloaded ScanNet."""
    return {
        "synthetic": ["mug", "chair", "bottle", "table"],
        "scannet":   list_scannet_scenes(),
    }
 
 
@app.get("/api/sample/{scene_name}")
def get_sample(scene_name: str,
               affordance: str = "grasp",
               method: str = "geometry_aware"):
    """Score a synthetic demo scene."""
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
    """
    Score a ScanNet scene by scan ID.
 
    Example: GET /api/scannet/scene0000_00?affordance=support&method=geometry_aware
 
    The scan must already be downloaded to data/scannet/{scan_id}/.
    """
    # Validate scan_id format to prevent path traversal
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
    """
    Upload a point cloud file and run affordance scoring.
 
    Accepted formats:
        .npy  — (N, 3) XYZ  or  (N, 6) XYZ+RGB (uint8)
        .ply  — binary or ASCII PLY with x y z [r g b] vertices
        .pkl  — dict with keys 'points' and optionally 'colors'
    """
    content = await file.read()
    suffix  = Path(file.filename).suffix.lower()
 
    try:
        if suffix == ".npy":
            arr    = np.load(io.BytesIO(content))
            points = arr[:, :3].astype(np.float32)
            colors = (arr[:, 3:6] / 255.0 if arr.shape[1] >= 6
                      else np.full((len(arr), 3), 0.6, dtype=np.float32))
 
        elif suffix == ".ply":
            # Write to a temp file so _parse_ply can seek
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
    """
    Run all three methods on the same scene — used by the comparison view.
    source: 'synthetic' | 'scannet'
    """
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