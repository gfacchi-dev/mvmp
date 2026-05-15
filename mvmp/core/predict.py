"""
Core prediction module — moderngl (headless EGL/OSMesa) + trimesh for rendering,
raycasting, and mesh I/O.  MediaPipe provides 2D face landmarks;
zone-based views back-project them to 3D on the mesh surface.
"""
import os as os_module
import ctypes.util as _ctypes_util
import logging
import json
import numpy as np
import scipy
import moderngl
from typing import Callable

from .mp_utils import *

IMG_SIZE = 720
FOV_DEG = 50.0
logger = logging.getLogger("mvmp")

import trimesh

# ── GLSL shaders ──────────────────────────────────────────────────────────
_VERT = """
#version 330 core
in vec3 in_position;
in vec2 in_texcoord;
uniform mat4 mvp;
out vec2 v_texcoord;
void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_texcoord = in_texcoord;
}
"""

_FRAG = """
#version 330 core
uniform sampler2D texture0;
in vec2 v_texcoord;
out vec4 f_color;
void main() {
    f_color = texture(texture0, v_texcoord);
}
"""


# ── Math helpers ──────────────────────────────────────────────────────────
def _look_at(eye, target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)):
    """Return 4×4 view matrix (world→camera, column-vector convention)."""
    e = np.asarray(eye,    dtype=np.float32)
    t = np.asarray(target, dtype=np.float32)
    u = np.asarray(up,     dtype=np.float32)
    f = e - t;  f /= np.linalg.norm(f)          # +Z axis (camera looks down −Z)
    r = np.cross(u, f); r /= np.linalg.norm(r)  # +X axis
    u = np.cross(f, r)                           # corrected +Y axis
    return np.array([
        [r[0], r[1], r[2], -float(np.dot(r, e))],
        [u[0], u[1], u[2], -float(np.dot(u, e))],
        [f[0], f[1], f[2], -float(np.dot(f, e))],
        [0.0,  0.0,  0.0,  1.0],
    ], dtype=np.float32)


def _projection(fov_deg, aspect, near, far):
    """OpenGL perspective projection matrix (column-vector, row-major numpy)."""
    f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
    return np.array([
        [f / aspect, 0.0, 0.0,                        0.0                    ],
        [0.0,        f,   0.0,                        0.0                    ],
        [0.0,        0.0, (far + near) / (near - far), 2*far*near/(near - far)],
        [0.0,        0.0, -1.0,                       0.0                    ],
    ], dtype=np.float32)


# ── Renderer ──────────────────────────────────────────────────────────────
class _Renderer:
    """Headless OpenGL renderer (moderngl, EGL or OSMesa)."""

    def __init__(self, img_size, cam_dist, fov_deg):
        self.img_size = img_size
        import platform
        system = platform.system()
        has_egl = bool(_ctypes_util.find_library("EGL"))
        if system == "Linux":
            backends = ['egl', 'osmesa'] if has_egl else ['osmesa', 'egl']
        else:
            backends = []  # macOS/Windows: let moderngl pick (CGL / WGL)

        ctx = None
        last_exc = None
        for backend in backends:
            try:
                ctx = moderngl.create_context(standalone=True, backend=backend)
                break
            except Exception as exc:
                last_exc = exc
        if ctx is None:
            try:
                ctx = moderngl.create_context(standalone=True)
            except Exception as exc:
                last_exc = exc
                _osmesa_hint = (
                    "\nOn Linux without a GPU, install OSMesa: "
                    "apt install libosmesa6  (Ubuntu/Debian) or "
                    "dnf install mesa-libOSMesa  (Fedora/RHEL)"
                ) if system == "Linux" else ""
                raise RuntimeError(
                    f"Could not create an OpenGL context for headless rendering "
                    f"({last_exc}).{_osmesa_hint}"
                ) from exc
        self._ctx = ctx

        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.enable(moderngl.CULL_FACE)

        self._tex = self._ctx.texture((img_size, img_size), 4)
        self._fbo = self._ctx.framebuffer(
            color_attachments=[self._tex],
            depth_attachment=self._ctx.depth_texture((img_size, img_size)),
        )
        self._prog = self._ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG)

        near = cam_dist * 0.01
        far  = cam_dist * 10.0
        self._proj = _projection(fov_deg, 1.0, near, far)

        self._vao  = None
        self._vbo  = None
        self._gl_tex = None

    def upload_mesh(self, mesh):
        """Upload (or re-upload after rotation) trimesh geometry to the GPU."""
        if self._vao is not None:
            self._vao.release()
        if self._vbo is not None:
            self._vbo.release()

        # Expand by faces (no index buffer, matching pyrender smooth=False)
        verts = mesh.vertices[mesh.faces].reshape(-1, 3).astype(np.float32)
        has_uv = (hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None
                  and len(mesh.visual.uv) == len(mesh.vertices))
        if has_uv:
            uvs = mesh.visual.uv[mesh.faces].reshape(-1, 2).astype(np.float32)
        else:
            uvs = np.zeros((len(verts), 2), dtype=np.float32)

        vertex_data = np.hstack([verts, uvs]).astype(np.float32)
        self._vbo = self._ctx.buffer(vertex_data.tobytes())
        self._vao = self._ctx.vertex_array(
            self._prog,
            [(self._vbo, '3f 2f', 'in_position', 'in_texcoord')],
        )

        # Texture — upload once; re-uploading on mesh rotation is not needed
        if self._gl_tex is None:
            mat = getattr(mesh.visual, 'material', None)
            img_pil = None
            if mat is not None:
                img_pil = getattr(mat, 'image', None) or getattr(mat, 'baseColorTexture', None)
            if img_pil is not None:
                img_arr = np.flipud(np.array(img_pil.convert('RGBA')))
                h, w = img_arr.shape[:2]
                self._gl_tex = self._ctx.texture((w, h), 4, img_arr.tobytes())
                self._gl_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            else:
                self._gl_tex = self._ctx.texture((1, 1), 4, b'\xff\xff\xff\xff')

    def render(self, eye, target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)):
        """Render from a camera position; return RGB uint8 array (H, W, 3)."""
        view = _look_at(eye, target, up)
        mvp  = (self._proj @ view).astype(np.float32)

        self._fbo.use()
        self._ctx.viewport = (0, 0, self.img_size, self.img_size)
        self._ctx.clear(1.0, 1.0, 1.0, 1.0)

        self._prog['mvp'].write(mvp.T.tobytes())   # column-major for GLSL
        self._gl_tex.use(0)
        self._prog['texture0'] = 0
        self._vao.render(moderngl.TRIANGLES)

        # fbo.read() is unreliable with RGBA attachments; read via texture object
        raw = np.frombuffer(self._tex.read(), dtype=np.uint8).reshape(
            self.img_size, self.img_size, 4)
        return np.ascontiguousarray(np.flipud(raw[:, :, :3]))

    def release(self):
        try:
            if self._vao:      self._vao.release()
            if self._vbo:      self._vbo.release()
            if self._gl_tex:   self._gl_tex.release()
            if self._tex:      self._tex.release()
            if self._fbo:      self._fbo.release()
            if self._ctx:      self._ctx.release()
        except Exception:
            pass

    def __del__(self):
        self.release()


# ── Fibonacci helpers ─────────────────────────────────────────────────────
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
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp -= np.dot(perp, a) * a
        perp /= np.linalg.norm(perp)
        from scipy.spatial.transform import Rotation as _Rot
        return _Rot.from_rotvec(perp * np.pi).as_matrix()
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / nv ** 2)


# ── Auto-align phase ──────────────────────────────────────────────────────
def _auto_align_phase(renderer, detector, cam_dist, tri_mesh,
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

        img = renderer.render(eye, target=(0.0, 0.0, 0.0), up=up)
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

    # ── Bisection refinement ───────────────────────────────────────────
    from scipy.spatial.transform import Rotation as Rot

    d0 = face_dir_best.copy()
    yaw0   = np.arctan2(d0[0], np.sqrt(d0[1]**2 + d0[2]**2))
    pitch0 = np.arctan2(d0[1], d0[2])

    def _score_at(yaw_rad, pitch_rad, roll_rad=0.0):
        Ry = Rot.from_euler("Y", yaw_rad).as_matrix()
        Rx = Rot.from_euler("X", -pitch_rad).as_matrix()
        R_view = Rx @ Ry
        eye = (R_view @ np.array([0, 0, 1])) * cam_dist
        up_base = R_view @ np.array([0, 1, 0])
        view_dir = R_view @ np.array([0, 0, 1])
        R_roll = Rot.from_rotvec(view_dir * roll_rad).as_matrix()
        up_vec = R_roll @ up_base
        img = renderer.render(eye, up=up_vec)
        result = detector.detect(mpImage(img))
        if not result.face_landmarks:
            return -1.0
        if (result.facial_transformation_matrixes and
                len(result.facial_transformation_matrixes) > 0):
            y, p, r, _ = _extract_face_pose(
                np.asarray(result.facial_transformation_matrixes[0]))
            return _frontal_score(y, p, r)
        return 0.0

    from scipy.optimize import minimize

    def _loss(params):
        y, p, r = params
        s = _score_at(y, p, r)
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
            img = renderer.render(eye, up=Rr @ up_base)
            PILImage.fromarray(img).save(
                os_module.path.join(debug_output_dir, f"{label}.png"))

    # ── Two-step alignment ─────────────────────────────────────────────
    tri_mesh.vertices = (R_face @ tri_mesh.vertices.T).T
    renderer.upload_mesh(tri_mesh)

    front_eye = np.array([0., 0., cam_dist])
    img_front = renderer.render(front_eye)
    result_front = detector.detect(mpImage(img_front))

    roll_front_deg = float(np.degrees(cur_roll))
    if (result_front.face_landmarks and
            result_front.facial_transformation_matrixes and
            len(result_front.facial_transformation_matrixes) > 0):
        _, _, roll_front_deg, _ = _extract_face_pose(
            np.asarray(result_front.facial_transformation_matrixes[0]))

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
              min_direction_support=2,
              progress_callback: Callable[[float, str], None] | None = None):
    tri_mesh = meshes["trimesh"]

    meshes["orientation_R"] = None

    detector = detectorInit()

    cam_dist = 2.0 * camera_distance_multiplier
    renderer = _Renderer(IMG_SIZE, cam_dist, FOV_DEG)
    renderer.upload_mesh(tri_mesh)

    # Camera intrinsic matrix (kept for raycasting — unchanged)
    fov_rad  = np.radians(FOV_DEG)
    f        = (IMG_SIZE / 2) / np.tan(fov_rad / 2)
    intr_mat = np.array([[f, 0, IMG_SIZE / 2],
                         [0, f, IMG_SIZE / 2],
                         [0, 0, 1]], dtype=np.float64)

    # ── Auto-align phase ────────────────────────────────────────────────
    if auto_orient:
        R_align, _, _ = _auto_align_phase(
            renderer, detector, cam_dist, tri_mesh,
            n_fibonacci=n_fibonacci, verbose=verbose,
            debug_output_dir=debug_output_dir,
            min_neighbor_support=min_neighbor_support,
            max_score_isolation=max_score_isolation,
            max_direction_deviation=max_direction_deviation,
            min_direction_support=min_direction_support,
        )
        if R_align is not None:
            meshes["orientation_R"] = R_align
            renderer.upload_mesh(tri_mesh)  # mesh rotated inside _auto_align_phase

    # ── Build zone-based camera poses ──────────────────────────────────────
    from .zone_config import ZONE_CAMERAS, ZONE_LANDMARKS, ZONE_NAMES
    from scipy.spatial.transform import Rotation

    zone_ids = sorted(ZONE_CAMERAS.keys())
    eyes = []
    ups  = []
    cam_dirs     = []
    cam_rotations = []
    yaw_degs  = []
    pitch_degs = []

    for zone_id in zone_ids:
        yaw_deg, pitch_deg = ZONE_CAMERAS[zone_id]
        y = np.radians(yaw_deg)
        x = np.radians(pitch_deg)
        R_yaw   = Rotation.from_euler("Y", y).as_matrix()
        R_pitch = Rotation.from_euler("X", -x).as_matrix()
        cam_R   = R_pitch @ R_yaw
        eye = (cam_R @ np.array([0, 0, 1])) * cam_dist
        up  = cam_R @ np.array([0, 1, 0])
        eyes.append(eye)
        ups.append(up)
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
        print(f"Projecting {len(eyes)} zone-based views...", flush=True)

    if progress_callback is not None:
        progress_callback(0.1, "Rendering viewpoints")

    for idx, zone_id in enumerate(zone_ids):
        zone_lm_set = set(ZONE_LANDMARKS[zone_id])
        zone_name   = ZONE_NAMES[zone_id]
        yaw_deg     = yaw_degs[idx]
        pitch_deg   = pitch_degs[idx]

        img = renderer.render(eyes[idx], up=ups[idx])

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
            draw    = ImageDraw.Draw(pil_img)
            for lm in detection_result.face_landmarks[0]:
                px = int(lm.x * IMG_SIZE); py = int(lm.y * IMG_SIZE)
                draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(0, 255, 0))
            pil_img.save(
                os_module.path.join(debug_output_dir,
                    f"landmarks_{zone_name}_yaw{yaw_deg:+06.1f}_pitch{pitch_deg:+06.1f}.png")
            )

        all_landmarks = detection_result.face_landmarks[0]
        landmarks_2d  = np.array([[p.x, p.y, 0] for p in all_landmarks], dtype=np.float64)

        persp_rays = _perspective_rays_directions(landmarks_2d, IMG_SIZE, intr_mat)
        world_rays = (persp_rays * [1, -1, -1]) @ np.linalg.inv(cam_rotations[idx])

        camera_pos = cam_dirs[idx] * cam_dist
        camera_positions.append(camera_pos.copy())

        for lm_idx in range(478):
            if lm_idx not in zone_lm_set:
                continue
            lm = all_landmarks[lm_idx]
            px   = max(0, min(int(lm.x * IMG_SIZE), IMG_SIZE - 1))
            py_img = max(0, min(int(lm.y * IMG_SIZE), IMG_SIZE - 1))
            if np.all(img[py_img, px] > 240):
                continue
            views[lm_idx].append(np.concatenate([camera_pos, world_rays[lm_idx]]).astype(np.float32))

        if progress_callback is not None:
            pct = 0.1 + (idx + 1) / len(zone_ids) * 0.5
            progress_callback(pct, f"Detecting landmarks (camera {idx + 1}/{len(zone_ids)})")

    renderer.release()

    detected_landmarks = sum(1 for rays in views.values() if len(rays) > 0)

    if verbose:
        print(f"  Detected faces in {detection_count}/{len(eyes)} zone views, "
              f"{detected_landmarks}/478 landmarks covered", flush=True)

    if detection_count == 0:
        raise RuntimeError("No face detected in any zone view.")
    if detected_landmarks < 50:
        logger.warning(f"Only {detected_landmarks}/478 landmarks detected across all zone views.")

    # ── Raycast landmarks to mesh surface ───────────────────────────────
    if progress_callback is not None:
        progress_callback(0.85, "Triangulating")

    if verbose:
        print(f"Raycasting {detected_landmarks} landmarks...", flush=True)

    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(tri_mesh)

    landmarks_3d_norm = {}
    for i, rays in views.items():
        if len(rays) == 0:
            continue
        rays = np.asarray(rays, dtype=np.float64)
        origins    = rays[:, :3]
        directions = rays[:, 3:6]

        locations, index_ray, _ = intersector.intersects_location(
            ray_origins=origins,
            ray_directions=directions,
            multiple_hits=False,
        )

        if len(index_ray) > 0:
            pt = origins[index_ray[0]] + directions[index_ray[0]] * np.linalg.norm(
                locations[0] - origins[index_ray[0]])
            landmark_candidates[i].append(pt)
            landmarks_3d_norm[i] = pt

    if len(landmarks_3d_norm) == 0:
        raise RuntimeError("No landmarks could be triangulated from raycasting.")
    if len(landmarks_3d_norm) < 100:
        logger.warning(f"Only {len(landmarks_3d_norm)}/478 landmarks detected, results may be incomplete")

    # ── KDTree vertex lookup ─────────────────────────────────────────────
    detected_indices = sorted(landmarks_3d_norm.keys())
    lm_array_norm    = np.array([landmarks_3d_norm[k] for k in detected_indices], dtype=np.float64)

    tree = scipy.spatial.KDTree(np.asarray(tri_mesh.vertices, dtype=np.float64))
    _, nearest = tree.query(lm_array_norm)
    closest_vertices_ids = {k: int(v) for k, v in zip(detected_indices, nearest)}

    # ── Denormalize back to original coordinates ─────────────────────────
    center       = meshes["transform_center"]
    scale        = meshes["transform_scale"]
    orientation_R = meshes.get("orientation_R")

    lm_array = lm_array_norm * scale
    if orientation_R is not None:
        lm_array = (orientation_R.T @ lm_array.T).T
    lm_array = lm_array + center

    landmarks_3d = {k: pt.tolist() for k, pt in zip(detected_indices, lm_array)}

    if verbose:
        print(f"Done. {len(landmarks_3d)} landmarks detected.", flush=True)

    if progress_callback is not None:
        progress_callback(0.95, "Completed")

    return landmarks_3d, closest_vertices_ids, camera_positions, landmark_candidates


def _perspective_rays_directions(img_landmarks, size, intrinsic):
    la = np.asarray(img_landmarks, dtype=np.float64)
    scaled = la * size
    h = np.hstack([scaled[:, :2], np.ones((len(scaled), 1))])
    inv_K = np.linalg.inv(intrinsic)
    rays = h @ inv_K.T
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return rays
