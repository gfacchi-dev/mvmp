import open3d as o3d
import json
import numpy as np
import logging

logger = logging.getLogger("mvmp")


def import_mesh(filename):
    import os

    # Convert to absolute path so Open3D can find texture files
    abs_path = os.path.abspath(filename)

    try:
        textured_mesh = o3d.io.read_triangle_mesh(abs_path, enable_post_processing=False)
    except Exception as e:
        raise RuntimeError(f"Error importing mesh: {e}")

    if not textured_mesh.vertices:
        raise RuntimeError(f"Error loading mesh: no vertices found in {filename}")

    has_textures = len(textured_mesh.textures) > 0 if hasattr(textured_mesh, 'textures') else False

    if not has_textures and len(textured_mesh.vertex_colors) == 0:
        logger.warning("Mesh has no colors or textures, using uniform gray")
        textured_mesh.paint_uniform_color([0.7, 0.7, 0.7])

    if len(textured_mesh.vertex_normals) == 0:
        textured_mesh.compute_vertex_normals()

    if len(textured_mesh.triangle_normals) == 0:
        textured_mesh.compute_triangle_normals()

    return {
        "mesh": textured_mesh,
        "mesh_path": filename,
        "tensor": None,
    }


def meshes_setup(meshes, auto_align=True):
    """Normalize, center, and prepare mesh for rendering/raycasting.

    Args:
        meshes: Dict with mesh and path from import_mesh()
        auto_align: If True, attempt automatic face alignment

    Returns:
        Updated dict with normalized mesh, tensor mesh, and transform params
    """
    mesh = meshes["mesh"]

    if len(mesh.vertex_normals) == 0:
        mesh.compute_vertex_normals()

    if len(mesh.triangle_normals) == 0:
        mesh.compute_triangle_normals()

    # Normalize mesh into a unit cube
    scale = max(mesh.get_max_bound() - mesh.get_min_bound())
    center = mesh.get_axis_aligned_bounding_box().get_center()

    mesh.scale(1/scale, center)
    mesh.translate(-center)

    # Convert to tensor mesh for efficient raycasting
    mesh_t = o3d.t.geometry.TriangleMesh(
        device=o3d.core.Device("CUDA:0" if o3d.core.cuda.is_available() else "CPU:0")
    ).from_legacy(mesh)

    meshes.update({
        "tensor": mesh_t,
        "transform_scale": 1/scale,
        "transform_center": center
    })

    return meshes


def save_facemarks_json(input_path, landmarks_3d, closest_vertices_ids, json_path, camera_data=None):
    data = {
        "model": input_path,
        "coordinates": landmarks_3d.tolist() if hasattr(landmarks_3d, 'tolist') else landmarks_3d,
        "closest_vertex_indexes": closest_vertices_ids
    }

    if camera_data:
        data["cameras"] = [
            {
                "position": cam["position"].tolist() if hasattr(cam["position"], 'tolist') else cam["position"],
                "direction": cam["direction"].tolist() if hasattr(cam["direction"], 'tolist') else cam["direction"]
            }
            for cam in camera_data
        ]

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
