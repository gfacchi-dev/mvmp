"""
Core prediction module — pyrender (headless) + trimesh for rendering,
raycasting, and mesh I/O.  MediaPipe provides 2D face landmarks;
zone-based views back-project them to 3D on the mesh surface.
"""
import os as os_module
import logging
import json
import numpy as np
import scipy


from .mp_utils import *
from .triangles import TRIANGLES

IMG_SIZE = 720
FOV_DEG = 50.0
logger = logging.getLogger("mvmp")

# ── pyrender / headless setup ─────────────────────────────────────────────
os_module.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import pyglet
pyglet.options["shadow_window"] = False
import pyrender
import pyrr
import trimesh


def _camera_pose(eye, target=(0, 0, 0), up=(0, 1, 0)):
    """Return a pyrender camera pose (4×4) from eye → target."""
    M = pyrr.Matrix44.look_at(
        eye=np.asarray(eye, dtype=float),
        target=np.asarray(target, dtype=float),
        up=np.asarray(up, dtype=float),
    )
    return np.linalg.inv(np.asarray(M.T, dtype=np.float64))


# ── Fibonacci helpers (unchanged) ─────────────────────────────────────────
def _fibonacci_sphere(n):
    phi = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(n)
    y = 1.0 - (i / max(n - 1, 1)) * 2.0
    r = np.sqrt(np.clip(1.0 - y * y, 0, None))
    theta = phi * i
    return np.column_stack([r * np.cos(theta), y, r * np.sin(theta)])


def _fibonacci_neighbors(directions, k=6):
    dirs = np.asarray(directions)
    dots = dirs @ dirs.T
    neighbors = []
    for i in range(len(dirs)):
        order = np.argsort(-dots[i])
        neighbors.append(order[1:k+1].tolist())
    return neighbors


def _face_dir_world(cam_dir, face_dir_cam):
    d = np.asarray(cam_dir, dtype=float)
    d /= np.linalg.norm(d)
    forward_world = -d
    up_ref = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(d, up_ref)) > 0.95:
        up_ref = np.array([1.0, 0.0, 0.0])
    up_world = up_ref - np.dot(up_ref, d) * d
    up_world /= np.linalg.norm(up_world)
    right_world = np.cross(up_world, forward_world)
    right_world /= np.linalg.norm(right_world)
    up_world = np.cross(forward_world, right_world)
    R = np.column_stack([right_world, up_world, forward_world])
    face_dir_cam = np.asarray(face_dir_cam, dtype=float)
    n = np.linalg.norm(face_dir_cam)
    if n > 1e-10:
        face_dir_cam = face_dir_cam / n
    fd_world = R @ face_dir_cam
    fd_world /= np.linalg.norm(fd_world)
    return fd_world


def _extract_face_pose(face_matrix):
    M = np.asarray(face_matrix, dtype=np.float64)
    R = M[:3, :3].copy()
    R /= np.linalg.norm(R, axis=0, keepdims=True)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    from scipy.spatial.transform import Rotation
    yaw, pitch, roll = Rotation.from_matrix(R).as_euler("YXZ", degrees=True)
    return float(yaw), float(pitch), float(roll), R[:, 2]


def _frontal_score(yaw, pitch, roll, face_dir_cam=None):
    sigma = np.array([20.0, 15.0, 12.0])
    scaled = np.array([yaw, pitch, roll]) / sigma
    return float(np.exp(-0.5 * np.dot(scaled, scaled)))


def _rotation_align(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a /= np.linalg.norm(a); b /= np.linalg.norm(b)
    v = np.cross(a, b); c = float(np.dot(a, b))
    nv = np.linalg.norm(v)
    if nv < 1e-10:
        if c > 0:
            return np.eye(3)
        # anti-parallel: rotate 180° around any axis perpendicular to a
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp -= np.dot(perp, a) * a
        perp /= np.linalg.norm(perp)
        from scipy.spatial.transform import Rotation as _Rot
        return _Rot.from_rotvec(perp * np.pi).as_matrix()
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / nv ** 2)


# ── Mesh loading helpers ──────────────────────────────────────────────────
def _load_trimesh(mesh_path):
    """Load a mesh file via trimesh (handles BMP, JPG, PNG textures)."""
    m = trimesh.load(mesh_path, force="mesh")
    if not isinstance(m, trimesh.Trimesh):
        raise RuntimeError(f"trimesh could not load {mesh_path} as a single mesh")
    return m


# ── Rendering helpers ─────────────────────────────────────────────────────
def _prepare_renderer(mesh, cam_dist):
    """Create a pyrender scene + OffscreenRenderer for a trimesh."""
    py_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene()
    scene.add(py_mesh)
    camera = pyrender.PerspectiveCamera(
        yfov=np.radians(FOV_DEG), aspectRatio=1.0,
        znear=cam_dist * 0.01, zfar=cam_dist * 10,
    )
    cam_node = scene.add(camera)
    renderer = pyrender.OffscreenRenderer(IMG_SIZE, IMG_SIZE)
    return renderer, scene, cam_node


def _render_view(renderer, scene, cam_node, pose):
    """Render the scene with the given camera pose; return RGB uint8 image."""
    scene.set_pose(cam_node, pose)
    color, _ = renderer.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
    return np.ascontiguousarray(color[:, :, :3])





def _perspective_rays_directions(img_landmarks, size, intrinsic):
    """Vectorized perspective ray directions."""
    la = np.asarray(img_landmarks, dtype=np.float64)
    scaled = la * size
    h = np.hstack([scaled[:, :2], np.ones((len(scaled), 1))])
    inv_K = np.linalg.inv(intrinsic)
    rays = h @ inv_K.T
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return rays





# ── Auto-align phase ──────────────────────────────────────────────────────
def _auto_align_phase(renderer, scene, camera, detector, cam_dist, tri_mesh,
                       n_fibonacci=100, verbose=True, debug_output_dir=None,
                       min_neighbor_support=3, max_score_isolation=3.0,
                       max_direction_deviation=30.0, min_direction_support=2):
    fibonacci_dirs = _fibonacci_sphere(n_fibonacci)
    neighbors = _fibonacci_neighbors(fibonacci_dirs, k=6)
    probe_results = []
    detection_by_fib = [None] * n_fibonacci
    weighted_sum = np.zeros(3)
    total_weight = 0.0

    if verbose:
        print(f"\n{'='*60}")
        print(f" AUTO-ALIGN PHASE: {n_fibonacci} Fibonacci-sphere probes")
        print(f"{'='*60}")
        print(f"  #  |     yaw      pitch     roll    |  score  |  camera direction (world)")

    for idx, d in enumerate(fibonacci_dirs):
        d = np.asarray(d, dtype=float)
        d /= np.linalg.norm(d)

        eye = d * cam_dist
        up_ref = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(d, up_ref)) > 0.95:
            up_ref = np.array([1.0, 0.0, 0.0])
        up = up_ref - np.dot(up_ref, d) * d
        up /= np.linalg.norm(up)

        pose = _camera_pose(eye, target=(0, 0, 0), up=up)
        img = _render_view(renderer, scene, camera, pose)
        detection_result = detector.detect(mpImage(img))

        if not detection_result.face_landmarks:
            if verbose and len(probe_results) < 20:
                print(f"  {idx:3d} | {'no detection':>35s}  |         |  ({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f})")
            continue

        yaw_d = pitch_d = roll_d = 0.0
        score = 0.0
        face_dir_cam = np.array([0.0, 0.0, 1.0])
        lms = detection_result.face_landmarks[0]

        if (detection_result.facial_transformation_matrixes and
                len(detection_result.facial_transformation_matrixes) > 0):
            fm = np.asarray(detection_result.facial_transformation_matrixes[0])
            yaw_d, pitch_d, roll_d, face_dir_cam = _extract_face_pose(fm)
            score = _frontal_score(yaw_d, pitch_d, roll_d, face_dir_cam=face_dir_cam)
        else:
            xs = [l.x for l in lms]; ys = [l.y for l in lms]
            bbox_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            mid_x = (lms[133].x + lms[362].x) / 2.0
            mid_y = np.mean(ys)
            centering = max(0.0, 1.0 - abs(mid_x - 0.5) * 2.0) * max(0.0, 1.0 - abs(mid_y - 0.5) * 2.0)
            score = bbox_area * centering

        if score < 1e-6:
            continue

        detection_by_fib[idx] = len(probe_results)
        probe_results.append({
            "dir": d.tolist(), "yaw": yaw_d, "pitch": pitch_d, "roll": roll_d,
            "score": score, "fib_idx": idx,
            "face_dir_cam": face_dir_cam.tolist() if isinstance(face_dir_cam, np.ndarray) else face_dir_cam,
        })
        weighted_sum += d * score
        total_weight += score

        if verbose:
            marker = " ←TOP" if score == max((r["score"] for r in probe_results), default=0) else ""
            print(f"  {idx:3d} | {yaw_d:+7.1f}° {pitch_d:+7.1f}° {roll_d:+7.1f}°  | {score:7.4f}  |  "
                  f"({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f}){marker}")

    if total_weight == 0:
        if verbose:
            print("  NO FACE DETECTED in any Fibonacci probe — skipping auto-align.")
        return None, None, probe_results

    # ── Neighbour-support filter ────────────────────────────────────────
    for r in probe_results:
        fib_idx = r["fib_idx"]
        r["neighbor_support"] = sum(
            1 for j in neighbors[fib_idx] if detection_by_fib[j] is not None
        )

    raw_sorted = sorted(probe_results, key=lambda r: r["score"], reverse=True)
    validated = [r for r in raw_sorted if r["neighbor_support"] >= min_neighbor_support]

    if not validated:
        if verbose:
            print(f"  WARNING: no probe passed neighbour-support filter "
                  f"(min={min_neighbor_support}), falling back to raw best.")
        validated = raw_sorted[:1]

    def _neighbor_scores(pr):
        fib_idx = pr["fib_idx"]
        scores = []
        for j in neighbors[fib_idx]:
            if detection_by_fib[j] is not None:
                scores.append(probe_results[detection_by_fib[j]]["score"])
        return scores

    def _select_best(candidates):
        for c in candidates:
            n_scores = _neighbor_scores(c)
            if len(n_scores) < 2:
                return c
            median_s = float(np.median(n_scores))
            if median_s < 1e-9:
                return c
            ratio = c["score"] / median_s
            if ratio > max_score_isolation:
                if verbose:
                    print(f"  Rejecting fib_idx={c['fib_idx']:3d} (score {c['score']:.4f}, "
                          f"isolation ratio {ratio:.2f} > {max_score_isolation})")
                continue
            best_fd = _face_dir_world(c["dir"], c["face_dir_cam"])
            agree = 0
            n_neighbors = 0
            for j in neighbors[c["fib_idx"]]:
                if detection_by_fib[j] is not None:
                    n_neighbors += 1
                    nb = probe_results[detection_by_fib[j]]
                    nb_fd = _face_dir_world(nb["dir"], nb["face_dir_cam"])
                    dot = np.clip(np.dot(best_fd, nb_fd), -1.0, 1.0)
                    ang = float(np.degrees(np.arccos(abs(dot))))
                    if ang < max_direction_deviation:
                        agree += 1
            if agree < min_direction_support:
                if verbose:
                    print(f"  Rejecting fib_idx={c['fib_idx']:3d} (score {c['score']:.4f}, "
                          f"direction agreement {agree}/{n_neighbors} < {min_direction_support})")
                continue
            return c
        if verbose:
            print("  WARNING: no candidate passed all filters, falling back to raw best.")
        return candidates[0]

    best_probe = _select_best(validated)
    face_dir_best = np.asarray(best_probe["dir"], dtype=float)
    face_dir_best /= np.linalg.norm(face_dir_best)

    if verbose:
        top_n = min(10, len(probe_results))
        top = sorted(probe_results, key=lambda r: r["score"], reverse=True)[:top_n]
        print(f"\n  Top {top_n} probes by frontal score:")
        print(f"  {'rank':<5} {'yaw':>7} {'pitch':>7} {'roll':>7} {'score':>8}  direction")
        for rank, r in enumerate(top, 1):
            d = r["dir"]
            print(f"  {rank:<5} {r['yaw']:+6.1f}° {r['pitch']:+6.1f}° {r['roll']:+6.1f}° {r['score']:8.4f}  "
                  f"({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f})")
        print(f"\n  Total probes: {n_fibonacci}  |  Detections: {len(probe_results)}")
        print(f"  Best-probe face direction (world): "
              f"({face_dir_best[0]:+.4f}, {face_dir_best[1]:+.4f}, {face_dir_best[2]:+.4f})")
        print(f"  → Aligning this to world +Z")

    if debug_output_dir and probe_results:
        os_module.makedirs(debug_output_dir, exist_ok=True)
        report = {
            "n_fibonacci_probes": n_fibonacci,
            "n_detections": len(probe_results),
            "face_direction": face_dir_best.tolist(),
            "probes": probe_results,
        }
        with open(os_module.path.join(debug_output_dir, "auto_align_report.json"), "w") as f:
            json.dump(report, f, indent=2)

    # Also save the best probe's render image if a debug dir was given
    if debug_output_dir:
        os_module.makedirs(debug_output_dir, exist_ok=True)
        best_dir = np.asarray(best_probe["dir"])

    # ── Bisection refinement ───────────────────────────────────────────
    # Refine the face direction by bisecting yaw and pitch to drive
    # the face's detected yaw/pitch toward zero.
    from scipy.spatial.transform import Rotation as Rot

    # Convert best camera direction → initial yaw/pitch
    d0 = face_dir_best.copy()
    yaw0 = np.arctan2(d0[0], np.sqrt(d0[1]**2 + d0[2]**2))
    pitch0 = np.arctan2(d0[1], d0[2])

    def _score_at(yaw_rad, pitch_rad, roll_rad=0.0):
        """Render from (yaw, pitch, roll) and return frontal score.
        Roll rotates the camera's up vector around the view direction."""
        Ry = Rot.from_euler("Y", yaw_rad).as_matrix()
        Rx = Rot.from_euler("X", -pitch_rad).as_matrix()
        R_view = Rx @ Ry
        eye = (R_view @ np.array([0, 0, 1])) * cam_dist
        up_base = R_view @ np.array([0, 1, 0])
        view_dir = R_view @ np.array([0, 0, 1])
        R_roll = Rot.from_rotvec(view_dir * roll_rad).as_matrix()
        up_vec = R_roll @ up_base
        pose = _camera_pose(eye, up=up_vec)
        img = _render_view(renderer, scene, camera, pose)
        result = detector.detect(mpImage(img))
        if not result.face_landmarks:
            return -1.0
        if (result.facial_transformation_matrixes and
                len(result.facial_transformation_matrixes) > 0):
            y, p, r, fdc = _extract_face_pose(
                np.asarray(result.facial_transformation_matrixes[0]))
            return _frontal_score(y, p, r)
        return 0.0

    # Nelder-Mead simplex optimisation over (yaw, pitch, roll)
    # to maximise frontal score (= minimise negative score)
    from scipy.optimize import minimize

    def _loss(params):
        y, p, r = params
        s = _score_at(y, p, r)
        # Small penalty for large roll (keep face upright)
        return -s + 0.001 * abs(r)

    x0 = np.array([yaw0, pitch0, 0.0])
    init_simplex = np.array([
        x0,
        x0 + [np.radians(15), 0, 0],
        x0 + [0, np.radians(15), 0],
        x0 + [0, 0, np.radians(15)],
    ])

    res = minimize(_loss, x0, method='Nelder-Mead',
                   options={'xatol': np.radians(0.5), 'fatol': 1e-4,
                            'maxiter': 50, 'initial_simplex': init_simplex})
    cur_yaw, cur_pitch, cur_roll = res.x
    final_score = -res.fun

    # Convert refined yaw/pitch/roll → camera direction and alignment rotation
    Ry_ref = Rot.from_euler("Y", cur_yaw).as_matrix()
    Rx_ref = Rot.from_euler("X", -cur_pitch).as_matrix()
    R_view = Rx_ref @ Ry_ref
    face_dir_refined = R_view @ np.array([0, 0, 1])
    face_dir_refined /= np.linalg.norm(face_dir_refined)

    R_face = _rotation_align(face_dir_refined, np.array([0.0, 0.0, 1.0]))
    R_roll = Rot.from_rotvec(np.array([0, 0, 1]) * (-cur_roll)).as_matrix()
    R_align = R_roll @ R_face

    if verbose:
        print(f"  Nelder-Mead refined ({res.nit} iters, {res.nfev} evals): "
              f"yaw={np.degrees(cur_yaw):.1f}° pitch={np.degrees(cur_pitch):.1f}° "
              f"roll={np.degrees(cur_roll):.1f}° score={final_score:.4f}")
        print(f"  Refined face direction: ({face_dir_refined[0]:+.4f}, "
              f"{face_dir_refined[1]:+.4f}, {face_dir_refined[2]:+.4f})")

    # Save refinement debug renders
    if debug_output_dir:
        from PIL import Image as PILImage
        for label, y, p, r in [("probe_best", yaw0, pitch0, 0.0),
                                 ("refined", cur_yaw, cur_pitch, cur_roll)]:
            Ry = Rot.from_euler("Y", y).as_matrix()
            Rx = Rot.from_euler("X", -p).as_matrix()
            Rv = Rx @ Ry
            eye = (Rv @ np.array([0, 0, 1])) * cam_dist
            up_base = Rv @ np.array([0, 1, 0])
            vd = Rv @ np.array([0, 0, 1])
            Rr = Rot.from_rotvec(vd * r).as_matrix()
            pose = _camera_pose(eye, up=Rr @ up_base)
            img = _render_view(renderer, scene, camera, pose)
            PILImage.fromarray(img).save(
                os_module.path.join(debug_output_dir, f"{label}.png"))

    # ── Two-step alignment: forward direction first, then roll ────────────
    # Step 1: apply R_face so the face now looks straight at +Z
    tri_mesh.vertices = (R_face @ tri_mesh.vertices.T).T
    scene.remove_node(list(scene.mesh_nodes)[0])
    scene.add(pyrender.Mesh.from_trimesh(tri_mesh, smooth=False))

    # Step 2: measure the residual roll directly from the front camera.
    # cur_roll was detected from the refined (off-axis) camera, so it can
    # diverge from the true front-view roll when pitch is large (HEADSPACE).
    front_pose = _camera_pose(np.array([0., 0., cam_dist]),
                              target=(0., 0., 0.), up=(0., 1., 0.))
    img_front = _render_view(renderer, scene, camera, front_pose)
    result_front = detector.detect(mpImage(img_front))

    roll_front_deg = float(np.degrees(cur_roll))  # fallback
    if (result_front.face_landmarks and
            result_front.facial_transformation_matrixes and
            len(result_front.facial_transformation_matrixes) > 0):
        _, _, roll_front_deg, _ = _extract_face_pose(
            np.asarray(result_front.facial_transformation_matrixes[0]))

    # Step 3: apply roll correction around world +Z
    R_roll = Rot.from_rotvec(np.array([0., 0., 1.]) * np.radians(-roll_front_deg)).as_matrix()
    R_align = R_roll @ R_face
    tri_mesh.vertices = (R_roll @ tri_mesh.vertices.T).T

    if verbose:
        print(f"  Roll (front view): {roll_front_deg:.1f}°  "
              f"(Nelder-Mead had: {np.degrees(cur_roll):.1f}°)")
        print(f"  Rotation applied to mesh. Face should now face +Z with zero roll.\n")

    return R_align, face_dir_best, probe_results


# ── Main prediction ───────────────────────────────────────────────────────
def __predict(meshes, verbose=True, debug_output_dir=None,
              camera_distance_multiplier=1.0, auto_orient=True,
              n_fibonacci=100, min_neighbor_support=3,
              max_score_isolation=3.0, max_direction_deviation=30.0,
              min_direction_support=2):
    mesh_path = meshes["mesh_path"]
    tri_mesh = meshes["trimesh"]

    meshes["orientation_R"] = None

    detector = detectorInit()

    # ── Renderer setup ──────────────────────────────────────────────────
    cam_dist = 2.0 * camera_distance_multiplier
    renderer, scene, camera = _prepare_renderer(tri_mesh, cam_dist)

    # Camera intrinsic matrix (same as Open3D version)
    fov_rad = np.radians(FOV_DEG)
    f = (IMG_SIZE / 2) / np.tan(fov_rad / 2)
    intr_mat = np.array([[f, 0, IMG_SIZE / 2],
                         [0, f, IMG_SIZE / 2],
                         [0, 0, 1]], dtype=np.float64)
    inv_intr_mat = np.linalg.inv(intr_mat)

    # ── Auto-align phase ────────────────────────────────────────────────
    if auto_orient:
        R_align, face_dir_avg, probe_results = _auto_align_phase(
            renderer, scene, camera, detector, cam_dist, tri_mesh,
            n_fibonacci=n_fibonacci, verbose=verbose,
            debug_output_dir=debug_output_dir,
            min_neighbor_support=min_neighbor_support,
            max_score_isolation=max_score_isolation,
            max_direction_deviation=max_direction_deviation,
            min_direction_support=min_direction_support,
        )
        if R_align is not None:
            meshes["orientation_R"] = R_align
            # Rebuild pyrender mesh for the rotated trimesh
            scene.remove_node(list(scene.mesh_nodes)[0])
            py_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=False)
            scene.add(py_mesh)

    # ── Build zone-based camera poses ──────────────────────────────────────
    from .zone_config import ZONE_CAMERAS, ZONE_LANDMARKS, ZONE_NAMES
    from scipy.spatial.transform import Rotation

    zone_ids = sorted(ZONE_CAMERAS.keys())
    poses = []
    cam_dirs = []
    cam_rotations = []
    yaw_degs = []
    pitch_degs = []

    for zone_id in zone_ids:
        yaw_deg, pitch_deg = ZONE_CAMERAS[zone_id]
        y = np.radians(yaw_deg)
        x = np.radians(pitch_deg)
        R_yaw = Rotation.from_euler("Y", y).as_matrix()
        R_pitch = Rotation.from_euler("X", -x).as_matrix()
        cam_R = R_pitch @ R_yaw
        eye = (cam_R @ np.array([0, 0, 1])) * cam_dist
        up = cam_R @ np.array([0, 1, 0])
        poses.append(_camera_pose(eye, up=up))
        cam_dirs.append(eye / cam_dist)
        cam_rotations.append(cam_R)
        yaw_degs.append(yaw_deg)
        pitch_degs.append(pitch_deg)

    # ── Projection loop ─────────────────────────────────────────────────
    views = {i: [] for i in range(478)}
    camera_positions = []
    landmark_candidates = {i: [] for i in range(478)}
    detection_count = 0

    if debug_output_dir:
        os_module.makedirs(debug_output_dir, exist_ok=True)

    if verbose:
        print(f"Projecting {len(poses)} zone-based views...", flush=True)

    for idx, (pose, zone_id) in enumerate(zip(poses, zone_ids)):
        zone_lm_set = set(ZONE_LANDMARKS[zone_id])
        zone_name = ZONE_NAMES[zone_id]
        yaw_deg = yaw_degs[idx]
        pitch_deg = pitch_degs[idx]

        img = _render_view(renderer, scene, camera, pose)

        if debug_output_dir:
            from PIL import Image
            Image.fromarray(img).save(
                os_module.path.join(debug_output_dir,
                    f"render_{zone_name}_yaw{yaw_deg:+06.1f}_pitch{pitch_deg:+06.1f}.png")
            )

        detection_result = detector.detect(mpImage(img))
        if not detection_result.face_landmarks:
            if verbose:
                print(f"  Zone {zone_name:>15s}: no detection", flush=True)
            continue

        detection_count += 1

        if debug_output_dir:
            from PIL import Image, ImageDraw
            pil_img = Image.fromarray(img.copy())
            draw = ImageDraw.Draw(pil_img)
            for lm in detection_result.face_landmarks[0]:
                px = int(lm.x * IMG_SIZE); py = int(lm.y * IMG_SIZE)
                draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(0, 255, 0))
            pil_img.save(
                os_module.path.join(debug_output_dir,
                    f"landmarks_{zone_name}_yaw{yaw_deg:+06.1f}_pitch{pitch_deg:+06.1f}.png")
            )

        all_landmarks = detection_result.face_landmarks[0]
        landmarks_2d = np.array([[p.x, p.y, 0] for p in all_landmarks], dtype=np.float64)

        persp_rays = _perspective_rays_directions(landmarks_2d, IMG_SIZE, intr_mat)

        world_rays = (persp_rays * [1, -1, -1]) @ np.linalg.inv(cam_rotations[idx])

        camera_pos = cam_dirs[idx] * cam_dist
        camera_positions.append(camera_pos.copy())

        for lm_idx in range(478):
            if lm_idx not in zone_lm_set:
                continue
            lm = all_landmarks[lm_idx]
            px = max(0, min(int(lm.x * IMG_SIZE), IMG_SIZE - 1))
            py_img = max(0, min(int(lm.y * IMG_SIZE), IMG_SIZE - 1))
            if np.all(img[py_img, px] > 240):
                continue
            views[lm_idx].append(np.concatenate([camera_pos, world_rays[lm_idx]]).astype(np.float32))

    del renderer

    detected_landmarks = sum(1 for rays in views.values() if len(rays) > 0)

    if verbose:
        print(f"  Detected faces in {detection_count}/{len(poses)} zone views, "
              f"{detected_landmarks}/478 landmarks covered", flush=True)

    if detection_count == 0:
        raise RuntimeError("No face detected in any zone view.")
    if detected_landmarks < 50:
        logger.warning(f"Only {detected_landmarks}/478 landmarks detected across all zone views.")

    # ── Raycast landmarks to mesh surface ───────────────────────────────
    if verbose:
        print(f"Raycasting {detected_landmarks} landmarks...", flush=True)

    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(tri_mesh)

    landmarks_3d_norm = {}
    for i, rays in views.items():
        if len(rays) == 0:
            continue
        rays = np.asarray(rays, dtype=np.float64)
        origins = rays[:, :3]
        directions = rays[:, 3:6]

        locations, index_ray, _ = intersector.intersects_location(
            ray_origins=origins,
            ray_directions=directions,
            multiple_hits=False,
        )

        if len(index_ray) > 0:
            pt = origins[index_ray[0]] + directions[index_ray[0]] * np.linalg.norm(locations[0] - origins[index_ray[0]])
            landmark_candidates[i].append(pt)
            landmarks_3d_norm[i] = pt

    if len(landmarks_3d_norm) == 0:
        raise RuntimeError("No landmarks could be triangulated from raycasting.")
    if len(landmarks_3d_norm) < 100:
        logger.warning(f"Only {len(landmarks_3d_norm)}/478 landmarks detected, results may be incomplete")

    # ── KDTree vertex lookup (in normalized space) ──────────────────────
    detected_indices = sorted(landmarks_3d_norm.keys())
    lm_array_norm = np.array([landmarks_3d_norm[k] for k in detected_indices], dtype=np.float64)

    tree = scipy.spatial.KDTree(np.asarray(tri_mesh.vertices, dtype=np.float64))
    _, nearest = tree.query(lm_array_norm)
    closest_vertices_ids = {k: int(v) for k, v in zip(detected_indices, nearest)}

    # ── Denormalize back to original coordinates ────────────────────────
    center = meshes["transform_center"]
    scale = meshes["transform_scale"]
    orientation_R = meshes.get("orientation_R")

    lm_array = lm_array_norm * scale
    if orientation_R is not None:
        lm_array = (orientation_R.T @ lm_array.T).T
    lm_array = lm_array + center

    landmarks_3d = {k: pt.tolist() for k, pt in zip(detected_indices, lm_array)}

    if verbose:
        print(f"Done. {len(landmarks_3d)} landmarks detected.", flush=True)

    return landmarks_3d, closest_vertices_ids, camera_positions, landmark_candidates
