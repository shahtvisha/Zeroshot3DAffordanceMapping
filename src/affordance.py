import base64
import json
import os
import numpy as np
from pathlib import Path
from PIL import Image
import io

from dotenv import load_dotenv
load_dotenv()


SYSTEM_PROMPT = """You are a spatial reasoning assistant for a 3D robotic perception system.
Given an image of a scene and a user query about affordances (possible interactions),
you describe the relevant region in precise visual and spatial terms.

Your description will be used as a CLIP text embedding to find matching 3D points,
so describe the target region in terms of:
- Visual appearance (color, texture, shape, material)
- Geometric properties (flat surface, curved edge, cylindrical, concave)
- Spatial location in the scene (left, right, top, center)
- Why this region affords the requested interaction

Be specific and concrete. Respond only with the JSON format requested."""


def query_gpt4o_affordance(
    image: np.ndarray,
    user_query: str,
    scene_description: str = "",
    model: str = "gpt-4o",
) -> dict:
    """
    Ask GPT-4o to reason about where in the scene the affordance is located.

    Args:
        image:             (H, W, 3) uint8 RGB image of the scene
        user_query:        e.g. "where should I grasp this object?"
        scene_description: optional text description of what's in the scene

    Returns:
        dict with keys:
          - affordance_description: rich text description for CLIP embedding
          - region_description:     where in the image (for debugging)
          - confidence:             LLM's confidence 0-1
          - reasoning:              chain-of-thought
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Set OPENAI_API_KEY in your .env file or environment.\n"
            "Get a key at: https://platform.openai.com/api-keys"
        )

    client = OpenAI(api_key=api_key)

    # Encode image to base64
    pil_img = Image.fromarray(image)
    # Resize for API efficiency — 512px is enough for spatial reasoning
    pil_img.thumbnail((512, 512), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    scene_ctx = f"\nScene context: {scene_description}" if scene_description else ""

    user_message = f"""Query: {user_query}{scene_ctx}

Analyze this scene and respond with ONLY a JSON object in this exact format:
{{
  "affordance_description": "<rich visual+geometric description of the target region, 2-4 sentences>",
  "region_description": "<where in the image: e.g. 'center-left, red cylindrical object'>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<1-2 sentence explanation of why this region affords the interaction>"
}}"""

    print(f"[affordance] Querying {model}: '{user_query}'")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": user_message},
                ],
            },
        ],
        max_tokens=400,
        temperature=0.1,  # low temp for consistent spatial reasoning
    )

    raw = response.choices[0].message.content.strip()

    # Parse JSON response
    try:
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
    except json.JSONDecodeError:
        print(f"[affordance] Warning: Could not parse JSON, using raw response")
        result = {
            "affordance_description": raw,
            "region_description": "unknown",
            "confidence": 0.5,
            "reasoning": raw,
        }

    print(f"[affordance] LLM description: {result.get('affordance_description', '')[:100]}...")
    print(f"[affordance] Region: {result.get('region_description')}  "
          f"Confidence: {result.get('confidence')}")

    return result


def query_local_llm_affordance(
    image: np.ndarray,
    user_query: str,
    model_name: str = "llava-hf/llava-1.5-7b-hf",
) -> dict:
    """
    Fallback: use a local LLaVA model instead of GPT-4o (no API key needed).
    Requires ~14GB VRAM. Good for offline / no-API-key situations.
    """
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        raise ImportError("Run: pip install transformers")

    prompt = (
        f"<image>\nUSER: {user_query}\n"
        "Describe in detail the visual region of the image that affords this interaction. "
        "Include color, shape, texture, and spatial location.\nASSISTANT:"
    )

    pipe = hf_pipeline(
        "image-to-text",
        model=model_name,
        device=0 if __import__("torch").cuda.is_available() else -1,
    )
    pil_img = Image.fromarray(image)
    result = pipe(pil_img, prompt=prompt, max_new_tokens=200)
    description = result[0]["generated_text"]

    return {
        "affordance_description": description,
        "region_description": "local LLM — no region metadata",
        "confidence": 0.7,
        "reasoning": description,
    }


def compute_llm_guided_scores(
    point_features: np.ndarray,
    llm_result: dict,
    clip_model_name: str = "openai/clip-vit-large-patch14",
) -> np.ndarray:

    from zeroshot3d.features import embed_text_clip, compute_text_similarity

    description = llm_result.get("affordance_description", "")
    if not description:
        raise ValueError("LLM result missing affordance_description")

    # Embed the rich LLM description
    text_emb = embed_text_clip([description], model_name=clip_model_name)[0]

    # Score each point
    scores = compute_text_similarity(point_features, text_emb)
    return scores


def combine_scores(
    clip_scores: np.ndarray,
    llm_scores: np.ndarray,
    clip_weight: float = 0.4,
    llm_weight: float = 0.6,
) -> np.ndarray:
    """
    Weighted combination of direct CLIP scores and LLM-guided scores.
    LLM gets higher weight because its descriptions are richer.
    """
    assert abs(clip_weight + llm_weight - 1.0) < 1e-5
    combined = clip_weight * clip_scores + llm_weight * llm_scores
    return combined.astype(np.float32)


def get_top_affordance_points(
    points: np.ndarray,
    scores: np.ndarray,
    top_k: int = 50,
    threshold: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the highest-affordance region.

    Returns:
        top_points: (K, 3) the top-scoring 3D points
        top_scores: (K,)  their scores
    """
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    # Apply threshold first, then take top-k
    above_thresh = scores_norm > threshold
    if above_thresh.sum() < top_k:
        above_thresh = np.zeros_like(above_thresh, dtype=bool)
        top_idx = np.argsort(scores_norm)[-top_k:]
        above_thresh[top_idx] = True

    top_points = points[above_thresh]
    top_scores = scores_norm[above_thresh]

    centroid = top_points.mean(axis=0)
    print(f"[affordance] Top region: {len(top_points)} points  "
          f"centroid=({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})")

    return top_points, top_scores



def zero_shot_affordance_map(
    points: np.ndarray,
    colors: np.ndarray,
    image: np.ndarray,
    intrinsics,
    query: str,
    use_llm: bool = True,
    llm_backend: str = "gpt4o",   # "gpt4o" | "local"
    clip_model: str = "openai/clip-vit-large-patch14",
    clip_weight: float = 0.4,
    llm_weight: float = 0.6,
) -> dict:
    from zeroshot3d.features import (
        extract_clip_dense_features, embed_text_clip,
        lift_features_to_3d, compute_text_similarity,
    )

    H, W = image.shape[:2]

    print(f"\n[pipeline] Step 1/4: Extracting CLIP features...")
    feature_map = extract_clip_dense_features(image, model_name=clip_model)

    print(f"[pipeline] Step 2/4: Lifting features to 3D...")
    point_features = lift_features_to_3d(
        points, feature_map, intrinsics, image_hw=(H, W)
    )

    print(f"[pipeline] Step 3/4: Computing CLIP scores for '{query}'...")
    query_emb = embed_text_clip([query], model_name=clip_model)[0]
    clip_scores = compute_text_similarity(point_features, query_emb)

    result = {
        "scores": clip_scores,
        "clip_scores": clip_scores,
        "llm_scores": None,
        "llm_result": None,
        "top_points": None,
    }

    if use_llm:
        print(f"[pipeline] Step 4/4: LLM spatial reasoning...")
        try:
            if llm_backend == "gpt4o":
                llm_result = query_gpt4o_affordance(image, query)
            else:
                llm_result = query_local_llm_affordance(image, query)

            llm_scores = compute_llm_guided_scores(
                point_features, llm_result, clip_model_name=clip_model
            )
            final_scores = combine_scores(clip_scores, llm_scores,
                                          clip_weight, llm_weight)
            result.update({
                "scores": final_scores,
                "llm_scores": llm_scores,
                "llm_result": llm_result,
            })
        except Exception as e:
            print(f"[pipeline] LLM step failed ({e}), using CLIP-only scores.")
    else:
        print(f"[pipeline] Step 4/4: Skipped (LLM disabled, CLIP-only mode)")

    # ── Top region extraction ──
    top_pts, top_sc = get_top_affordance_points(points, result["scores"])
    result["top_points"] = top_pts

    return result



def mock_affordance_map(points: np.ndarray, colors: np.ndarray,
                         query: str = "grasp") -> dict:
    """
    Fully synthetic affordance map for pipeline wiring tests.
    Scores are based on geometry only (no models needed).
    """
    from zeroshot3d.features import (
        generate_synthetic_features, generate_synthetic_text_embedding,
        compute_text_similarity,
    )

    feats = generate_synthetic_features(points, colors)
    text_emb = generate_synthetic_text_embedding(query)
    scores = compute_text_similarity(feats, text_emb)

    top_pts, _ = get_top_affordance_points(points, scores)

    return {
        "scores": scores,
        "clip_scores": scores,
        "llm_scores": None,
        "llm_result": {"affordance_description": f"[mock] {query}",
                        "confidence": 0.5},
        "top_points": top_pts,
    }


# Quick self-test

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from zeroshot3d.pointcloud import generate_synthetic_scene, voxel_downsample

    print("Day 3 test: mock affordance map...")
    points, colors, labels = generate_synthetic_scene()

    pts_down, col_down = voxel_downsample(points, colors, voxel_size=0.015)

    result = mock_affordance_map(pts_down, col_down, query="grasp the red mug")
    scores = result["scores"]
    print(f"  Scores: {scores.shape}  range [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"  Top region: {len(result['top_points'])} points")
    print("Day 3 mock test complete.")