import numpy as np
from pathlib import Path
from PIL import Image
import open3d as o3d


class CameraIntrinsics:
    """
    Holds the pinhole camera parameters needed to unproject depth → 3D.
    Default values match a standard RealSense D435 at 640×480.
    Swap these out if you're using a different sensor or dataset.
    """
    def __init__(self, fx=600.0, fy=600.0, cx=320.0, cy=240.0,
                 width=640, height=480):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height

    @classmethod
    def for_scannet(cls):
        """ScanNet dataset standard intrinsics."""
        return cls(fx=577.87, fy=577.87, cx=319.5, cy=239.5,
                   width=640, height=480)

    @classmethod
    def for_realsense_720p(cls):
        return cls(fx=920.0, fy=920.0, cx=640.0, cy=360.0,
                   width=1280, height=720)

def depth_to_pointcloud(
    color_image: np.ndarray,
    depth_image: np.ndarray,
    intrinsics: CameraIntrinsics,
    depth_scale: float = 1000.0,
    depth_max: float = 3.0,
    depth_min: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unproject a depth image into a 3D point cloud.

    Args:
        color_image:  (H, W, 3) uint8 RGB image
        depth_image:  (H, W)    float or uint16 depth in mm (if depth_scale=1000)
        intrinsics:   camera pinhole parameters
        depth_scale:  divide raw depth by this to get meters (1000 for mm→m)
        depth_max:    clip points farther than this (meters)
        depth_min:    clip points closer than this (meters)

    Returns:
        points: (N, 3) float32 XYZ in camera space (meters)
        colors: (N, 3) float32 RGB in [0, 1]
    """
    H, W = depth_image.shape

    # Build pixel coordinate grids
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)  # (H, W) each

    depth_m = depth_image.astype(np.float32) / depth_scale

    valid = (depth_m > depth_min) & (depth_m < depth_max)

    X = (uu - intrinsics.cx) / intrinsics.fx * depth_m
    Y = (vv - intrinsics.cy) / intrinsics.fy * depth_m
    Z = depth_m

    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)  # (H*W, 3)
    colors = color_image.reshape(-1, 3).astype(np.float32) / 255.0  # (H*W, 3)
    valid_flat = valid.reshape(-1)

    points = points[valid_flat]
    colors = colors[valid_flat]

    return points, colors


def load_rgbd_from_arrays(color_path: str, depth_path: str,
                           intrinsics: CameraIntrinsics,
                           depth_scale: float = 1000.0):
    """
    Load from image files. Depth can be:
      - 16-bit PNG  (standard, values in mm → divide by 1000)
      - 32-bit float PNG
      - .npy file
    """
    color_path = Path(color_path)
    depth_path = Path(depth_path)

    color_img = np.array(Image.open(color_path).convert("RGB"))

    if depth_path.suffix == ".npy":
        depth_img = np.load(depth_path)
    else:
        depth_pil = Image.open(depth_path)
        depth_img = np.array(depth_pil)

    if color_img.shape[:2] != depth_img.shape[:2]:
        h, w = depth_img.shape[:2]
        color_pil = Image.fromarray(color_img).resize((w, h), Image.BILINEAR)
        color_img = np.array(color_pil)

    return depth_to_pointcloud(color_img, depth_img, intrinsics, depth_scale)


def load_rgbd_from_npy(color_npy: str, depth_npy: str,
                        intrinsics: CameraIntrinsics,
                        depth_scale: float = 1.0):
    """Load color and depth both from .npy arrays (depth already in meters)."""
    color = np.load(color_npy)
    depth = np.load(depth_npy)
    return depth_to_pointcloud(color, depth, intrinsics, depth_scale)



def generate_synthetic_scene(n_objects: int = 4,
                               points_per_object: int = 1200,
                               seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic tabletop scene for testing without a real sensor.

    Returns:
        points:   (N, 3)  XYZ coordinates
        colors:   (N, 3)  RGB in [0, 1]
        labels:   (N,)    integer object label (0 = table, 1+ = objects)
    """
    rng = np.random.default_rng(seed)
    all_points, all_colors, all_labels = [], [], []

    n_table = 4000
    table_x = rng.uniform(-0.8, 0.8, n_table)
    table_y = rng.uniform(-0.5, 0.5, n_table)
    table_z = rng.uniform(-0.005, 0.005, n_table)
    table_pts = np.stack([table_x, table_y, table_z], axis=1)
    table_color = np.tile([0.72, 0.60, 0.45], (n_table, 1))  # wood brown
    table_color += rng.normal(0, 0.02, table_color.shape)
    all_points.append(table_pts)
    all_colors.append(np.clip(table_color, 0, 1))
    all_labels.append(np.zeros(n_table, dtype=int))

    objects = [
        ( 0.0,   0.0,  0.15, 0.04, [0.9, 0.2, 0.2]),   # red mug (grasp target)
        ( 0.3,   0.1,  0.08, 0.06, [0.2, 0.5, 0.9]),   # blue bowl
        (-0.25,  0.15, 0.20, 0.03, [0.3, 0.8, 0.3]),   # green bottle
        ( 0.1,  -0.2,  0.05, 0.08, [0.9, 0.85, 0.3]),  # yellow plate
    ][:n_objects]

    for i, (cx, cy, h, r, color) in enumerate(objects):
        # Cylinder approximation
        theta = rng.uniform(0, 2 * np.pi, points_per_object)
        radii = rng.uniform(0, r, points_per_object)
        x = cx + radii * np.cos(theta)
        y = cy + radii * np.sin(theta)
        z = rng.uniform(0.0, h, points_per_object)
        pts = np.stack([x, y, z], axis=1)
        col = np.tile(color, (points_per_object, 1))
        col += rng.normal(0, 0.03, col.shape)
        all_points.append(pts)
        all_colors.append(np.clip(col, 0, 1))
        all_labels.append(np.full(points_per_object, i + 1, dtype=int))

    points = np.concatenate(all_points, axis=0).astype(np.float32)
    colors = np.concatenate(all_colors, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0)

    return points, colors, labels


def save_synthetic_scene(save_dir: str = "data/synthetic"):
    """Generate and save a synthetic scene to disk."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    points, colors, labels = generate_synthetic_scene()
    np.save(save_dir / "points.npy", points)
    np.save(save_dir / "colors.npy", colors)
    np.save(save_dir / "labels.npy", labels)
    print(f"[pointcloud] Saved synthetic scene → {save_dir}  ({len(points):,} points)")
    return points, colors, labels



def voxel_downsample(points: np.ndarray, colors: np.ndarray,
                      voxel_size: float = 0.01):
    """Downsample for faster processing. Typical voxel_size: 0.01–0.05 m."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pts_down = np.asarray(pcd_down.points, dtype=np.float32)
    col_down = np.asarray(pcd_down.colors, dtype=np.float32)
    print(f"[pointcloud] Downsampled: {len(points):,} → {len(pts_down):,} points")
    return pts_down, col_down


def estimate_normals(points: np.ndarray, colors: np.ndarray,
                     radius: float = 0.05, max_nn: int = 30):
    """Estimate surface normals. Used by some affordance methods."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    normals = np.asarray(pcd.normals, dtype=np.float32)
    return normals


def colormap_jet(t: np.ndarray) -> np.ndarray:
    """Map scalar [0,1] → RGB via jet colormap. Returns (N,3)."""
    t = np.clip(t, 0, 1)
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    return np.stack([r, g, b], axis=1)


def colormap_plasma(t: np.ndarray) -> np.ndarray:
    """Plasma colormap — looks better in screenshots than jet."""
    t = np.clip(t, 0, 1)
    r = np.clip(0.05 + 2.0 * t - 0.5 * t**2, 0, 1)
    g = np.clip(-0.1 + 1.2 * t - 1.5 * t**2 + 0.6 * t**3, 0, 1) * 0.6
    b = np.clip(0.6 - 2.0 * t + 2.0 * t**2, 0, 1)
    return np.stack([r, g, b], axis=1)


def make_affordance_pcd(points: np.ndarray, scores: np.ndarray,
                         colormap: str = "jet") -> o3d.geometry.PointCloud:
    """Build an Open3D PointCloud colored by affordance scores."""
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    fn = colormap_jet if colormap == "jet" else colormap_plasma
    colors = fn(scores_norm)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def visualize_pointcloud(points: np.ndarray, colors: np.ndarray,
                          window_name: str = "Point Cloud",
                          point_size: float = 2.0):
    """Show a colored point cloud interactively."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    o3d.visualization.draw_geometries(
        [pcd, frame],
        window_name=window_name,
        width=1200, height=800,
    )


def visualize_affordance(points: np.ndarray, colors: np.ndarray,
                          scores: np.ndarray,
                          query: str = "",
                          colormap: str = "plasma",
                          point_size: float = 2.5,
                          threshold: float = None):
    """
    Show the affordance heatmap side-by-side with the original RGB cloud.

    Args:
        threshold: if set, paint points above threshold bright red instead of heatmap
    """
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    pcd_rgb = o3d.geometry.PointCloud()
    pcd_rgb.points = o3d.utility.Vector3dVector(points)
    pcd_rgb.colors = o3d.utility.Vector3dVector(colors)

    x_offset = (points[:, 0].max() - points[:, 0].min()) * 1.3
    offset_pts = points.copy()
    offset_pts[:, 0] += x_offset

    if threshold is not None:
        afford_colors = colors.copy() * 0.3  
        high = scores_norm > threshold
        afford_colors[high] = [1.0, 0.15, 0.15]  
    else:
        fn = colormap_jet if colormap == "jet" else colormap_plasma
        afford_colors = fn(scores_norm)

    pcd_afford = o3d.geometry.PointCloud()
    pcd_afford.points = o3d.utility.Vector3dVector(offset_pts)
    pcd_afford.colors = o3d.utility.Vector3dVector(afford_colors)

    title = f"RGB (left) | Affordance: '{query}' (right)" if query else "RGB | Affordance"
    o3d.visualization.draw_geometries(
        [pcd_rgb, pcd_afford],
        window_name=title,
        width=1600, height=800,
    )


def save_affordance_render(points: np.ndarray, scores: np.ndarray,
                            output_path: str = "results/affordance.png",
                            colormap: str = "plasma"):
    """Save an offscreen render — use this for README screenshots."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    pcd = make_affordance_pcd(points, scores, colormap)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1200, height=800)
    vis.add_geometry(pcd)

    render_opt = vis.get_render_option()
    render_opt.point_size = 3.0
    render_opt.background_color = np.array([0.05, 0.05, 0.05])

    ctr = vis.get_view_control()
    ctr.set_zoom(0.75)
    ctr.set_front([0.0, -0.3, -1.0])
    ctr.set_up([0.0, -1.0, 0.0])

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_path)
    vis.destroy_window()
    print(f"[pointcloud] Saved render → {output_path}")


# Quick self-test

if __name__ == "__main__":
    print("Day 1 test: generating synthetic scene...")
    points, colors, labels = generate_synthetic_scene()
    print(f"  Points: {points.shape}  Colors: {colors.shape}")

    pts_down, col_down = voxel_downsample(points, colors, voxel_size=0.01)

    print("  Showing point cloud — close the window to continue.")
    visualize_pointcloud(pts_down, col_down, "Synthetic Tabletop Scene")

    fake_scores = pts_down[:, 2]
    visualize_affordance(pts_down, col_down, fake_scores,
                         query="where to grasp?", colormap="plasma")
    print("Day 1 complete.")