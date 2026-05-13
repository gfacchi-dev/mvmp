"""
Zone-based camera configuration for deterministic multi-view landmark detection.

The 478 MediaPipe landmarks are partitioned into 5 zones by Voronoi graph-BFS
seeded from anatomically known regions (eye, brow, iris, lips, nose).

Each zone is assigned a fixed camera direction.  After auto-align the face
faces +Z, so:
  yaw  > 0  →  camera at +X side  →  looks at face's −X = subject's LEFT
  yaw  < 0  →  camera at −X side  →  looks at face's +X = subject's RIGHT
  pitch > 0  →  camera at +Y side  →  looks downward, sees top of forehead
  pitch < 0  →  camera at −Y side  →  looks upward, sees chin

The left/right labelling assumes the standard orientation preserved by auto-
align (face's left ear at −X in world space).  If your dataset has the opposite
handedness, negate the yaw signs in ZONE_CAMERAS.
"""
from __future__ import annotations
import collections
import numpy as np

from .triangles import TRIANGLES
from .mesh_face_connections import (
    FACEMESH_LEFT_EYE,      FACEMESH_LEFT_EYEBROW,
    FACEMESH_RIGHT_EYE,     FACEMESH_RIGHT_EYEBROW,
    FACEMESH_LEFT_IRIS,     FACEMESH_RIGHT_IRIS,
    FACEMESH_LIPS,          FACEMESH_NOSE,
)

N_LM = 478

# ── Zone IDs ─────────────────────────────────────────────────────────────
FRONT         = 0
LEFT_QUARTER  = 1   # subject's left  (camera yaw = +35°)
RIGHT_QUARTER = 2   # subject's right (camera yaw = −35°)
UPPER         = 3   # forehead / top  (camera pitch = +15°)
LOWER         = 4   # chin / jaw      (camera pitch = −15°)

# (yaw_deg, pitch_deg) for each zone camera
ZONE_CAMERAS: dict[int, tuple[float, float]] = {
    FRONT:         (  0.0,   0.0),
    LEFT_QUARTER:  (+35.0,   0.0),
    RIGHT_QUARTER: (-35.0,   0.0),
    UPPER:         (  0.0, +15.0),
    LOWER:         (  0.0, -15.0),
}

ZONE_NAMES = {
    FRONT:         "front",
    LEFT_QUARTER:  "left_quarter",
    RIGHT_QUARTER: "right_quarter",
    UPPER:         "upper",
    LOWER:         "lower",
}


# ── helpers ───────────────────────────────────────────────────────────────

def _verts(edge_set: frozenset) -> set[int]:
    out: set[int] = set()
    for u, v in edge_set:
        out.add(u)
        out.add(v)
    return out


def _bfs_dist(seeds: set[int], adj: list[set[int]]) -> np.ndarray:
    dist = np.full(N_LM, np.inf)
    q: collections.deque[int] = collections.deque()
    for s in seeds:
        if s < N_LM:
            dist[s] = 0.0
            q.append(s)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if np.isinf(dist[v]):
                dist[v] = dist[u] + 1.0
                q.append(v)
    return dist


# ── zone assignment ───────────────────────────────────────────────────────

def _build_zones() -> tuple[np.ndarray, dict[int, list[int]]]:
    # adjacency list from face mesh triangles
    adj: list[set[int]] = [set() for _ in range(N_LM)]
    for a, b, c in TRIANGLES:
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))

    # seed sets — include iris pupil centres (468, 473) not in edge connections
    left_seeds = _verts(FACEMESH_LEFT_EYE | FACEMESH_LEFT_EYEBROW | FACEMESH_LEFT_IRIS)
    left_seeds.add(473)   # left iris pupil centre

    right_seeds = _verts(FACEMESH_RIGHT_EYE | FACEMESH_RIGHT_EYEBROW | FACEMESH_RIGHT_IRIS)
    right_seeds.add(468)  # right iris pupil centre

    center_seeds = _verts(FACEMESH_LIPS | FACEMESH_NOSE)

    # superior/inferior oval landmarks — used only for top/bottom split
    top_seeds    = {10, 338, 297, 332, 284, 251, 389}
    bottom_seeds = {152, 148, 175, 377, 400, 369, 136, 150}

    d_left   = _bfs_dist(left_seeds,   adj)
    d_right  = _bfs_dist(right_seeds,  adj)
    d_center = _bfs_dist(center_seeds, adj)
    d_top    = _bfs_dist(top_seeds,    adj)
    d_bottom = _bfs_dist(bottom_seeds, adj)

    zones = np.full(N_LM, FRONT, dtype=np.int32)

    for i in range(N_LM):
        dl = d_left[i]
        dr = d_right[i]
        dc = d_center[i]
        dt = d_top[i]
        db = d_bottom[i]

        # top/bottom take priority — they win even over lateral BFS proximity
        if dt < dl and dt < dr and dt < dc and dt < db:
            zones[i] = UPPER
        elif db < dl and db < dr and db < dc and db < dt:
            zones[i] = LOWER
        elif min(dl, dr) < dc:
            zones[i] = LEFT_QUARTER if dl <= dr else RIGHT_QUARTER
        # else stays FRONT

    zone_lms: dict[int, list[int]] = {
        z: [i for i in range(N_LM) if zones[i] == z]
        for z in ZONE_CAMERAS
    }
    return zones, zone_lms


# Computed once at import time
LANDMARK_ZONES, ZONE_LANDMARKS = _build_zones()
