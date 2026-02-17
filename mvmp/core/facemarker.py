"""
Facemarker: Main API class for 3D face landmark detection
"""
import os
import json
import logging
import numpy as np
import open3d as o3d

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
            "coordinates": self.landmarks_3d.tolist() if hasattr(self.landmarks_3d, 'tolist') else self.landmarks_3d,
            "closest_vertex_indexes": self.closest_vertices_ids,
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
    Detect facial landmarks on 3D meshes.

    Example:
        >>> marker = Facemarker()
        >>> result = marker.predict("path/to/mesh.obj", projections=100)
        >>> result.save_json("landmarks.json")
        >>> print(result)
        FacemarkerResult(478 landmarks, 478 vertex indices)

        # With custom camera angles
        >>> marker = Facemarker(camera_angles=[(0, 0), (10, -5), (-15, 10)])
        >>> result = marker.predict("path/to/mesh.obj")
    """

    def __init__(self, projections=100, camera_angles=None, verbose=True):
        """
        Args:
            projections: Number of random projections (default: 100).
                         Ignored if camera_angles is provided.
            camera_angles: Optional list of (yaw, pitch) tuples in degrees.
                          Example: [(0, 0), (10, -5), (-15, 10)]
                          Yaw = left/right rotation, Pitch = up/down rotation.
            verbose: Print progress messages (default: True)
        """
        self.projections = projections
        self.camera_angles = camera_angles
        self.verbose = verbose

        # Suppress Open3D warnings during mesh I/O
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

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

        meshes = import_mesh(mesh_path)
        meshes = meshes_setup(meshes, auto_align=True)

        landmarks_3d, closest_vertices_ids, camera_data, landmark_candidates = _predict_impl(
            meshes, self.projections, camera_angles=self.camera_angles, verbose=self.verbose
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
