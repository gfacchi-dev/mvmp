import open3d as o3d
import os as os_module
import json
import logging
import numpy as np
import scipy
import tempfile
import shutil
from sklearn.cluster import DBSCAN

from .mp_utils import *
from .rendering import *

from .triangles import TRIANGLES
IMG_SIZE = 720

logger = logging.getLogger("mvmp")


def hpr_mesh_based(mesh, eye=[0, 0, 0]):
    """Hidden Point Removal using mesh-based raycasting (vectorized)."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    pcd_points = mesh.vertex.positions.numpy()
    eye = np.array(eye, dtype=np.float32)

    distance_vectors = pcd_points - eye
    norms = np.linalg.norm(distance_vectors, axis=1, keepdims=True)
    directions = distance_vectors / norms

    eyes = np.tile(eye, (len(pcd_points), 1))
    rays = np.hstack([eyes, directions]).astype(np.float32)

    cast_results = scene.cast_rays(rays)

    hit_distances = cast_results["t_hit"].numpy()
    visibility_mask = (hit_distances + 0.00001) >= norms.squeeze()

    return np.where(visibility_mask)


def perspective_rays_directions(img_landmarks, size, intrinsic):
    """Compute perspective ray directions (vectorized)."""
    landmarks_array = np.asarray(img_landmarks, dtype=np.float32)
    scaled = landmarks_array * size
    homogeneous = np.hstack([scaled[:, :2], np.ones((len(scaled), 1))])

    inv_intrinsic = np.linalg.inv(intrinsic)
    rays = homogeneous @ inv_intrinsic.T

    norms = np.linalg.norm(rays, axis=1, keepdims=True)
    return rays / norms


def cluster_consensus(points, eps=0.01):
    """Find largest cluster using DBSCAN, return its centroid.
    
    More robust than median for handling outliers from poor ray intersections.
    
    Args:
        points: Array of 3D points
        eps: Maximum distance between points in a cluster (scaled to normalized mesh)
    
    Returns:
        Centroid of the largest cluster
    """
    points = np.asarray(points)
    
    if len(points) < 3:
        return np.median(points, axis=0)
    
    db = DBSCAN(eps=eps, min_samples=2).fit(points)
    labels = db.labels_
    
    # Find largest cluster (excluding noise points labeled -1)
    valid_labels = labels[labels >= 0]
    if len(valid_labels) == 0:
        # No clusters found, fall back to median
        return np.median(points, axis=0)
    
    unique, counts = np.unique(valid_labels, return_counts=True)
    largest_cluster = unique[np.argmax(counts)]
    cluster_points = points[labels == largest_cluster]
    
    return np.mean(cluster_points, axis=0)


def __predict(meshes, projections_number, args=None, camera_angles=None, verbose=True, debug_output_dir=None, camera_distance_multiplier=1.0):
    """Core prediction function.

    Args:
        meshes: Prepared mesh dict from meshes_setup()
        projections_number: Number of random camera projections
        args: Optional args object (CLI compatibility)
        camera_angles: Optional list of (yaw, pitch) tuples in degrees.
                       If provided, projections_number is ignored.
        verbose: If True, print progress to stdout
        debug_output_dir: Optional path to save debug renders (plain + landmarks)
        camera_distance_multiplier: Multiplier for camera distance (default: 1.0, use <1.0 to get closer)
    """
    mesh = meshes["mesh"]
    mesh_t = meshes["tensor"]

    has_textures = hasattr(mesh, 'textures') and len(mesh.textures) > 0
    has_vertex_colors = mesh.has_vertex_colors()

    if not has_textures and not has_vertex_colors:
        logger.warning("Mesh has no vertex colors or textures, using default gray")
        mesh.paint_uniform_color([0.7, 0.7, 0.7])

    detector = detectorInit()

    configure_headless_rendering()
    renderer = o3d.visualization.rendering.OffscreenRenderer(IMG_SIZE, IMG_SIZE)
    renderer.scene.set_background([0, 0, 0, 1])

    # For textured meshes, use read_triangle_model + add_model to preserve UV mapping.
    # add_geometry() ignores triangle_uvs, producing scrambled textures.
    _tmpdir = None
    if has_textures:
        _tmpdir = tempfile.mkdtemp()
        tmp_obj = os_module.path.join(_tmpdir, "mesh.obj")
        o3d.io.write_triangle_mesh(tmp_obj, mesh)
        model = o3d.io.read_triangle_model(tmp_obj)
        renderer.scene.add_model("face", model)
    else:
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        renderer.scene.add_geometry("face", mesh, mat)

    FOV_DEG = 50.0
    fov_rad = np.radians(FOV_DEG)

    bbox = mesh.get_axis_aligned_bounding_box()
    mesh_radius = np.linalg.norm(bbox.get_extent()) / 2
    camera_distance = (mesh_radius / np.tan(fov_rad / 2)) * camera_distance_multiplier

    renderer.scene.camera.set_projection(
        FOV_DEG, 1.0,
        camera_distance * 0.01, camera_distance * 10,
        o3d.visualization.rendering.Camera.FovType.Vertical
    )

    f = (IMG_SIZE / 2) / np.tan(fov_rad / 2)
    intr_mat = np.array([[f, 0, IMG_SIZE / 2],
                         [0, f, IMG_SIZE / 2],
                         [0, 0, 1]])

    # Precompute inverse intrinsic (used every iteration)
    inv_intr_mat = np.linalg.inv(intr_mat)

    # Build camera rotations from user-provided angles or random sampling
    if camera_angles is not None:
        camera_angles = np.asarray(camera_angles, dtype=np.float64)
        x_rots = np.radians(camera_angles[:, 1])  # pitch
        y_rots = np.radians(camera_angles[:, 0])  # yaw
        projections_number = len(camera_angles)
    else:
        # Yaw: -60 to +60 degrees, Pitch: -50 to +20 degrees
        y_rots = np.random.uniform(-np.pi/3, np.pi/3, projections_number)
        x_rots = np.random.uniform(-5*np.pi/18, np.pi/9, projections_number)

    camera_rots = [
        np.asarray(o3d.geometry.get_rotation_matrix_from_axis_angle([x, y, 0]))
        for x, y in zip(x_rots, y_rots)
    ]

    views = {i: [] for i in range(478)}
    camera_data = []
    landmark_candidates = {i: [] for i in range(478)}
    detection_count = 0

    # Create debug output directory if requested
    if debug_output_dir:
        os_module.makedirs(debug_output_dir, exist_ok=True)

    if verbose:
        print(f"Projecting {projections_number} views...", flush=True)

    for idx, camera_r in enumerate(camera_rots):
        if verbose and idx > 0 and idx % 50 == 0:
            print(f"  {idx}/{projections_number} ({detection_count} detections)", flush=True)

        # Get yaw and pitch for this camera
        yaw_deg = np.degrees(y_rots[idx])
        pitch_deg = np.degrees(x_rots[idx])

        camera_pos = (camera_r @ np.array([0, 0, 1])) * camera_distance
        up = camera_r @ np.array([0, 1, 0])
        renderer.scene.camera.look_at([0, 0, 0], camera_pos.tolist(), up.tolist())

        img = np.asarray(renderer.render_to_image())[:, :, :3]

        # Save plain render if debug mode
        if debug_output_dir:
            from PIL import Image
            Image.fromarray(img).save(
                os_module.path.join(debug_output_dir, f"render_yaw{yaw_deg:+06.1f}_pitch{pitch_deg:+06.1f}.png")
            )

        detection_result = detector.detect(mpImage(img))

        if not detection_result.face_landmarks:
            continue

        detection_count += 1

        # Save render with landmarks if debug mode
        if debug_output_dir:
            from PIL import Image, ImageDraw
            img_debug = img.copy()
            pil_img = Image.fromarray(img_debug)
            draw = ImageDraw.Draw(pil_img)

            # Draw 2D landmarks
            for landmark in detection_result.face_landmarks[0]:
                x = int(landmark.x * IMG_SIZE)
                y = int(landmark.y * IMG_SIZE)
                draw.ellipse([x-2, y-2, x+2, y+2], fill=(0, 255, 0))

            pil_img.save(
                os_module.path.join(debug_output_dir, f"landmarks_yaw{yaw_deg:+06.1f}_pitch{pitch_deg:+06.1f}.png")
            )

        mp_mesh = o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor([[p.x, -p.y, -p.z] for p in detection_result.face_landmarks[0]], dtype=o3d.core.Dtype.Float32),
            o3d.core.Tensor(TRIANGLES)
        )
        mp_mesh.translate(-mp_mesh.get_axis_aligned_bounding_box().get_center().numpy())

        visible_points = hpr_mesh_based(mp_mesh, [0, 0, 1])

        landmarks = [[p.x, p.y, 0] for p in detection_result.face_landmarks[0]]
        visible_indices = visible_points[0] if isinstance(visible_points, tuple) else visible_points
        landmarks = np.asarray(landmarks)[visible_indices]

        # Vectorized ray computation
        scaled = landmarks * IMG_SIZE
        homogeneous = np.hstack([scaled[:, :2], np.ones((len(scaled), 1))])
        persp_rays = homogeneous @ inv_intr_mat.T
        persp_rays /= np.linalg.norm(persp_rays, axis=1, keepdims=True)

        world_rays = (persp_rays * [1, -1, -1]) @ np.linalg.inv(camera_r)

        camera_data.append(camera_pos.copy())

        for i, landmark_idx in enumerate(visible_indices):
            views[landmark_idx].append(
                np.asarray(
                    [*camera_pos, *world_rays[i]],
                    dtype=np.float32
                )
            )

    del renderer

    detected_landmarks = sum(1 for rays in views.values() if len(rays) > 0)

    if verbose:
        print(f"  Detected faces in {detection_count}/{projections_number} views, {detected_landmarks}/478 landmarks covered", flush=True)

    def _cleanup():
        if _tmpdir is not None:
            shutil.rmtree(_tmpdir, ignore_errors=True)

    if detection_count == 0:
        _cleanup()
        raise RuntimeError("No face detected in any projection. Check mesh orientation or increase projections.")

    min_required = max(10, int(projections_number * 0.1))
    if detected_landmarks < min_required:
        _cleanup()
        raise RuntimeError(f"Only {detected_landmarks} landmarks detected, minimum required: {min_required}. Try increasing projections.")

    # Raycasting — batch all rays together for efficiency
    if verbose:
        print(f"Raycasting {detected_landmarks} landmarks...", flush=True)

    landmarks_3d = []
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    for i, rays in views.items():
        if len(rays) == 0:
            continue

        rays = np.asarray(rays)
        ans = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))

        hits = ans['t_hit'].numpy()
        origins = rays[:, :3]
        directions = rays[:, 3:]

        points = origins + directions * hits.reshape(-1, 1)

        valid_mask = np.isfinite(hits)
        valid_points = points[valid_mask]
        for pt in valid_points:
            landmark_candidates[i].append(pt)

        if len(valid_points) > 0:
            # Use DBSCAN clustering for robust outlier rejection
            consensus = cluster_consensus(valid_points, eps=0.01)
            landmarks_3d.append(consensus.tolist())

    if len(landmarks_3d) == 0:
        _cleanup()
        raise RuntimeError("No landmarks could be triangulated from raycasting.")

    if len(landmarks_3d) < 100:
        logger.warning(f"Only {len(landmarks_3d)}/478 landmarks detected, results may be incomplete")

    # Find closest vertex for each landmark using KDTree (much faster than cdist)
    mesh_path = meshes["mesh_path"]
    abs_mesh_path = os_module.path.abspath(mesh_path)

    actual_mesh = o3d.io.read_triangle_mesh(abs_mesh_path, enable_post_processing=True)

    scale = meshes["transform_scale"]
    center = meshes["transform_center"]
    actual_mesh.scale(scale, center)
    actual_mesh.translate(-center)

    tree = scipy.spatial.KDTree(np.asarray(actual_mesh.vertices))
    _, closest_vertices_ids = tree.query(np.asarray(landmarks_3d))
    closest_vertices_ids = [int(x) for x in closest_vertices_ids]

    # Convert landmarks back to original scale
    original_scale = 1.0 / scale
    landmarks_3d = (np.asarray(landmarks_3d) * original_scale + center).tolist()

    if verbose:
        print(f"Done. {len(landmarks_3d)} landmarks detected.", flush=True)

    _cleanup()

    return landmarks_3d, closest_vertices_ids, camera_data, landmark_candidates


def predict(args):
    """Wrapper for CLI usage."""
    return __predict(args.meshes, args.projections_number, args)
