
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from typing import Literal
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_clip_model = None
_clip_processor = None
_dino_model = None
_dino_processor = None


def load_clip(model_name: str = "openai/clip-vit-large-patch14"):
    """
    Load CLIP. First call downloads ~900MB, subsequent calls use cache.
    Recommended: clip-vit-large-patch14 (best quality)
                 clip-vit-base-patch32  (faster, smaller)
    """
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        print(f"[features] Loading CLIP: {model_name}  (first run downloads weights)")
        _clip_processor = CLIPProcessor.from_pretrained(model_name)
        _clip_model = CLIPModel.from_pretrained(model_name).to(DEVICE)
        _clip_model.eval()
        print(f"[features] CLIP loaded on {DEVICE}")
    return _clip_model, _clip_processor


def load_dino(model_name: str = "facebook/dinov2-base"):
    """
    Load DINOv2. Better spatial features than CLIP but no text encoder.
    Use DINOv2 features for geometric consistency, CLIP for text matching.
    """
    global _dino_model, _dino_processor
    if _dino_model is None:
        from transformers import AutoImageProcessor, AutoModel
        print(f"[features] Loading DINOv2: {model_name}")
        _dino_processor = AutoImageProcessor.from_pretrained(model_name)
        _dino_model = AutoModel.from_pretrained(model_name).to(DEVICE)
        _dino_model.eval()
        print(f"[features] DINOv2 loaded on {DEVICE}")
    return _dino_model, _dino_processor


def extract_clip_dense_features(
    image: np.ndarray,
    patch_size: int = 14,
    model_name: str = "openai/clip-vit-large-patch14",
) -> np.ndarray:
    """
    Extract per-patch CLIP features from an image.

    CLIP processes images as grids of patches. We extract the intermediate
    patch tokens (not the [CLS] token) to get a spatial feature map.

    Args:
        image:      (H, W, 3) uint8 RGB image
        patch_size: must match the model (14 for ViT-L/14, 32 for ViT-B/32)

    Returns:
        features: (H_patches, W_patches, D) float32 feature map
                  For ViT-L/14 on 224×224: (16, 16, 768)
    """
    model, processor = load_clip(model_name)

    pil_img = Image.fromarray(image)
    inputs = processor(images=pil_img, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        # Hook into the vision encoder to get patch tokens
        vision_outputs = model.vision_model(
            pixel_values=inputs["pixel_values"],
            output_hidden_states=True,
        )
        # Last hidden state: (1, num_patches+1, D)
        # Index 0 is [CLS], rest are spatial patches
        patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]  # (1, N, D)

    patch_tokens = patch_tokens.squeeze(0)  # (N, D)
    patch_tokens = F.normalize(patch_tokens, dim=-1)

    # Infer grid size from number of patches
    N, D = patch_tokens.shape
    grid_size = int(N ** 0.5)
    assert grid_size * grid_size == N, f"Expected square patch grid, got {N} patches"

    features = patch_tokens.cpu().numpy().reshape(grid_size, grid_size, D)
    return features.astype(np.float32)


def extract_dino_dense_features(
    image: np.ndarray,
    model_name: str = "facebook/dinov2-base",
) -> np.ndarray:
    """
    Extract per-patch DINOv2 features. DINOv2 gives better spatial
    correspondence than CLIP — use it when you need geometric precision.

    Returns:
        features: (H_patches, W_patches, D) float32
    """
    model, processor = load_dino(model_name)

    pil_img = Image.fromarray(image)
    inputs = processor(images=pil_img, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # skip [CLS]

    patch_tokens = patch_tokens.squeeze(0)
    patch_tokens = F.normalize(patch_tokens, dim=-1)

    N, D = patch_tokens.shape
    grid_size = int(N ** 0.5)
    features = patch_tokens.cpu().numpy().reshape(grid_size, grid_size, D)
    return features.astype(np.float32)


def embed_text_clip(
    texts: list[str],
    model_name: str = "openai/clip-vit-large-patch14",
) -> np.ndarray:
    """
    Embed a list of text strings with CLIP's text encoder.

    Returns:
        embeddings: (len(texts), D) float32, L2-normalized
    """
    model, processor = load_clip(model_name)

    inputs = processor(text=texts, return_tensors="pt",
                        padding=True, truncation=True).to(DEVICE)

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = F.normalize(text_features, dim=-1)

    return text_features.cpu().numpy().astype(np.float32)


def project_points_to_image(
    points: np.ndarray,
    intrinsics,
    image_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points (camera frame) back onto the image plane.

    Returns:
        pixel_coords: (N, 2) int32  — (u, v) pixel for each point
        valid_mask:   (N,)   bool   — True if point projects inside the image
    """
    H, W = image_hw
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]

    # Perspective projection
    u = (X / (Z + 1e-8)) * intrinsics.fx + intrinsics.cx
    v = (Y / (Z + 1e-8)) * intrinsics.fy + intrinsics.cy

    u_int = np.round(u).astype(np.int32)
    v_int = np.round(v).astype(np.int32)

    valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H) & (Z > 0)

    pixel_coords = np.stack([u_int, v_int], axis=1)
    return pixel_coords, valid


def lift_features_to_3d(
    points: np.ndarray,
    feature_map: np.ndarray,
    intrinsics,
    image_hw: tuple[int, int],
    model_input_size: int = 224,
) -> np.ndarray:
    """
    Core Day 2 function: assign a CLIP/DINO feature vector to every 3D point
    by projecting it onto the 2D feature map.

    Args:
        points:           (N, 3) XYZ in camera frame
        feature_map:      (H_f, W_f, D) dense feature grid from CLIP/DINOv2
        intrinsics:       camera intrinsics
        image_hw:         (H, W) of the ORIGINAL image (before model resize)
        model_input_size: size the image was resized to for the model (224)

    Returns:
        point_features: (N, D) float32 — one feature vector per 3D point
                        invalid points get a zero vector
    """
    N = len(points)
    H_f, W_f, D = feature_map.shape
    H_img, W_img = image_hw

    pixel_coords, valid = project_points_to_image(points, intrinsics, image_hw)

   
    # (feature map is H_f × W_f, original image is H_img × W_img)
    scale_u = W_f / W_img
    scale_v = H_f / H_img

    feat_u = np.clip((pixel_coords[:, 0] * scale_u).astype(np.int32), 0, W_f - 1)
    feat_v = np.clip((pixel_coords[:, 1] * scale_v).astype(np.int32), 0, H_f - 1)

    point_features = np.zeros((N, D), dtype=np.float32)
    point_features[valid] = feature_map[feat_v[valid], feat_u[valid]]

    print(f"[features] Lifted features: {valid.sum():,}/{N:,} valid points  "
          f"({100*valid.mean():.1f}% coverage)")

    return point_features


def compute_text_similarity(
    point_features: np.ndarray,
    text_embedding: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between each point's feature and a text embedding.

    Args:
        point_features: (N, D) L2-normalized point features
        text_embedding: (D,) L2-normalized text embedding (single query)

    Returns:
        scores: (N,) float32 similarity in [-1, 1], normalized to [0, 1]
    """
    norms = np.linalg.norm(point_features, axis=1, keepdims=True)
    safe_features = point_features / (norms + 1e-8)

    scores = safe_features @ text_embedding  
    scores = (scores + 1.0) / 2.0

    return scores.astype(np.float32)



def generate_synthetic_features(points: np.ndarray,
                                  colors: np.ndarray,
                                  feature_dim: int = 768,
                                  seed: int = 0) -> np.ndarray:
    """
    Generate fake CLIP-like features for testing the pipeline end-to-end
    before you hook up real CLIP. Features are based on point color,
    so semantically similar colored regions get similar features.
    """
    rng = np.random.default_rng(seed)
    N = len(points)

    color_feat = np.repeat(colors, feature_dim // 3 + 1, axis=1)[:, :feature_dim]

    noise = rng.normal(0, 0.1, (N, feature_dim)).astype(np.float32)
    features = color_feat.astype(np.float32) + noise

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / (norms + 1e-8)

    return features


def generate_synthetic_text_embedding(query: str,
                                        feature_dim: int = 768,
                                        seed: int = 99) -> np.ndarray:
    """
    Fake text embedding for testing. In the real pipeline this is replaced
    by embed_text_clip(). Seeded so same query → same embedding.
    """
    rng = np.random.default_rng(abs(hash(query)) % (2**32))
    emb = rng.normal(0, 1, feature_dim).astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    return emb


def clip_score_pointcloud(
    points: np.ndarray,
    image: np.ndarray,
    intrinsics,
    query_text: str,
    backbone: Literal["clip", "dino"] = "clip",
    model_name: str = "openai/clip-vit-large-patch14",
) -> np.ndarray:
    """
    End-to-end Day 2 pipeline:
      image + 3D points + text query → per-point affordance scores

    Args:
        points:     (N, 3) XYZ in camera frame
        image:      (H, W, 3) uint8 RGB
        intrinsics: CameraIntrinsics
        query_text: natural language query e.g. "grasping region of the mug"

    Returns:
        scores: (N,) float32 in [0, 1]
    """
    H, W = image.shape[:2]

    print(f"[features] Query: '{query_text}'")

    if backbone == "clip":
        feature_map = extract_clip_dense_features(image, model_name=model_name)
    else:
        feature_map = extract_dino_dense_features(image)

    print(f"[features] Feature map: {feature_map.shape}")

    text_emb = embed_text_clip([query_text], model_name=model_name)[0]

    point_features = lift_features_to_3d(
        points, feature_map, intrinsics, image_hw=(H, W)
    )

    scores = compute_text_similarity(point_features, text_emb)

    print(f"[features] Score range: [{scores.min():.3f}, {scores.max():.3f}]  "
          f"mean={scores.mean():.3f}")

    return scores


# Quick self-test

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.pointcloud import generate_synthetic_scene, voxel_downsample

    print("Day 2 test: synthetic feature lifting...")

    points, colors, labels = generate_synthetic_scene()
    pts_down, col_down = voxel_downsample(points, colors, voxel_size=0.015)

    print("\n[Test A] Synthetic features (no CLIP needed):")
    feats = generate_synthetic_features(pts_down, col_down)
    text_emb = generate_synthetic_text_embedding("grasping region")
    scores = compute_text_similarity(feats, text_emb)
    print(f"  Scores: {scores.shape}  range [{scores.min():.3f}, {scores.max():.3f}]")

    if "--real" in sys.argv:
        print("\n[Test B] Real CLIP features:")
        fake_image = (col_down * 255).astype(np.uint8).reshape(
            int(len(col_down)**0.5), -1, 3
        )[:224, :224]  

        from src.features import CameraIntrinsics
        intrinsics = CameraIntrinsics()
        scores_real = clip_score_pointcloud(
            pts_down[:224*224], fake_image, intrinsics,
            query_text="where to grasp the red object"
        )
        print(f"  Real CLIP scores: {scores_real.shape}")

    print("\nDay 2 test complete. Run with --real to test actual CLIP.")