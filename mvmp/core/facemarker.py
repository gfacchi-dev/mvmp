"""
Facemarker: Main API class for 3D face landmark detection
"""
import os
import json
import logging
import numpy as np

from .predict import __predict as _predict_impl
from .io_utils import import_mesh, meshes_setup

logger = logging.getLogger("mvmp")


class FacemarkerResult:
    """Container for face landmark detection results."""

    def __init__(self, landmarks_3d, closest_vertices_ids, camera_data=None,
                 landmark_candidates=None, transform_params=None):
        self.landmarks_3d = landmarks_3d
        self.closest_vertices_ids = closest_vertices_ids
        self.camera_data = camera_data
        self.landmark_candidates = landmark_candidates
        self.transform_params = transform_params

    def __repr__(self):
        n = len(self.landmarks_3d) if self.landmarks_3d is not None else 0
        return f"FacemarkerResult({n} landmarks, {len(self.closest_vertices_ids)} vertex indices)"

    def to_dict(self):
        """Convert result to dictionary format."""
        data = {
            "coordinates": {str(k): v for k, v in self.landmarks_3d.items()},
            "closest_vertex_indexes": {str(k): v for k, v in self.closest_vertices_ids.items()},
        }

        if self.camera_data:
            data["cameras"] = [
                pos.tolist() if hasattr(pos, 'tolist') else pos
                for pos in self.camera_data
            ]

        return data

    def save_json(self, output_path):
        """Save results to JSON file."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)


class Facemarker:
    """
    Detect facial landmarks on 3D meshes using zone-based multi-view projection.

    Five anatomically-defined zone cameras cover the full face:
    front, left quarter, right quarter, upper (forehead), lower (chin/jaw).
    Each camera is responsible only for its assigned MediaPipe landmarks,
    avoiding cross-zone confusion near occlusion boundaries.

    Example:
        >>> marker = Facemarker()
        >>> result = marker.predict("path/to/mesh.obj")
        >>> result.save_json("landmarks.json")
        >>> print(result)
        FacemarkerResult(478 landmarks, 478 vertex indices)
    """

    def __init__(self, verbose=True, debug_output_dir=None,
                 camera_distance_multiplier=1.0, auto_orient=True,
                 n_fibonacci=100, min_neighbor_support=3,
                 max_score_isolation=3.0, max_direction_deviation=30.0,
                 min_direction_support=2, allow_missing_texture=False):
        """
        Args:
            verbose: Print progress messages (default: True)
            debug_output_dir: Optional path to save debug renders per zone
            camera_distance_multiplier: Multiplier for camera distance (default: 1.0)
            auto_orient: If True (default), run Fibonacci-sphere auto-alignment to face +Z
            n_fibonacci: Number of Fibonacci probes for auto-alignment (default: 100)
            min_neighbor_support: Min neighbouring probes that must detect a face (default: 3)
            max_score_isolation: Max score ratio vs neighbour median (default: 3.0)
            max_direction_deviation: Max degrees between face direction and consensus (default: 30.0)
            min_direction_support: Min top-N probes agreeing on face direction (default: 2)
            allow_missing_texture: If True only warn instead of raising on missing textures (default: False)
        """
        self.verbose = verbose
        self.debug_output_dir = debug_output_dir
        self.camera_distance_multiplier = camera_distance_multiplier
        self.auto_orient = auto_orient
        self.n_fibonacci = n_fibonacci
        self.min_neighbor_support = min_neighbor_support
        self.max_score_isolation = max_score_isolation
        self.max_direction_deviation = max_direction_deviation
        self.min_direction_support = min_direction_support
        self.allow_missing_texture = allow_missing_texture

        # Suppress trimesh log spam
        logging.getLogger("trimesh").setLevel(logging.WARNING)

    def predict(self, mesh_path):
        """
        Detect facial landmarks on a mesh.

        Args:
            mesh_path: Path to 3D mesh file (supports .obj, .ply, .stl, .gltf, .glb, .off, etc.)

        Returns:
            FacemarkerResult with landmarks_3d, closest_vertices_ids, etc.
        """
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        meshes = import_mesh(mesh_path,
                             allow_missing_texture=self.allow_missing_texture)
        meshes = meshes_setup(meshes)

        landmarks_3d, closest_vertices_ids, camera_data, landmark_candidates = _predict_impl(
            meshes, verbose=self.verbose,
            debug_output_dir=self.debug_output_dir,
            camera_distance_multiplier=self.camera_distance_multiplier,
            auto_orient=self.auto_orient,
            n_fibonacci=self.n_fibonacci,
            min_neighbor_support=self.min_neighbor_support,
            max_score_isolation=self.max_score_isolation,
            max_direction_deviation=self.max_direction_deviation,
            min_direction_support=self.min_direction_support,
        )

        if landmarks_3d is None or closest_vertices_ids is None:
            raise RuntimeError("Landmark detection failed")

        return FacemarkerResult(
            landmarks_3d=landmarks_3d,
            closest_vertices_ids=closest_vertices_ids,
            camera_data=camera_data,
            landmark_candidates=landmark_candidates,
            transform_params=None
        )
