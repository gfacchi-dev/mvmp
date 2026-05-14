import json
import os
import numpy as np
import logging
import trimesh

logger = logging.getLogger("mvmp")

_MTL_TEX_KEYS = {"map_kd", "map_ka", "map_ks", "map_ns", "map_d",
                 "map_bump", "bump", "disp", "norm", "map_pr", "map_pm"}


def _check_obj_textures(obj_path):
    """Parse OBJ → MTL → texture files and return a list of MissingTexture warnings.

    Each item is a dict with keys: mtl, referenced, found_alt (path of a
    same-stem file if one exists, else None).
    """
    obj_dir = os.path.dirname(obj_path)
    warnings = []

    # collect mtllib references from the OBJ
    mtl_paths = []
    try:
        with open(obj_path, "r", errors="replace") as fh:
            for line in fh:
                if line.startswith("mtllib "):
                    name = line.split(None, 1)[1].strip()
                    mtl_paths.append(os.path.join(obj_dir, name))
    except OSError:
        return warnings

    for mtl_path in mtl_paths:
        if not os.path.exists(mtl_path):
            warnings.append({"mtl": mtl_path, "referenced": mtl_path,
                             "found_alt": None, "kind": "missing_mtl"})
            continue
        try:
            with open(mtl_path, "r", errors="replace") as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    if parts[0].lower() in _MTL_TEX_KEYS:
                        tex_name = parts[-1]
                        tex_path = os.path.join(obj_dir, tex_name)
                        if not os.path.exists(tex_path):
                            # look for an image file with the same stem
                            _IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp",
                                         ".tga", ".tiff", ".tif", ".exr"}
                            stem = os.path.splitext(tex_name)[0]
                            alt = next(
                                (os.path.join(obj_dir, f)
                                 for f in os.listdir(obj_dir)
                                 if os.path.splitext(f)[0] == stem
                                 and os.path.splitext(f)[1].lower() in _IMG_EXTS
                                 and os.path.isfile(os.path.join(obj_dir, f))),
                                None,
                            )
                            warnings.append({
                                "mtl": mtl_path,
                                "referenced": tex_path,
                                "found_alt": alt,
                                "kind": "missing_texture",
                            })
        except OSError:
            continue

    return warnings


def import_mesh(filename, allow_missing_texture=False):
    """Load a mesh via trimesh (supports OBJ, PLY, STL, GLTF, etc. with textures).

    Args:
        filename: Path to the mesh file.
        allow_missing_texture: If False (default), raise RuntimeError when the
            mesh has a material referencing a texture but no image was loaded.
            If True, only log a warning and continue.
    """
    abs_path = os.path.abspath(filename)

    if abs_path.lower().endswith(".obj"):
        for w in _check_obj_textures(abs_path):
            if w["kind"] == "missing_mtl":
                logger.warning("MTL file not found: %s", w["referenced"])
            else:
                alt = f" (found same-stem alt: {w['found_alt']})" if w["found_alt"] else ""
                logger.warning("Texture missing: %s%s", w["referenced"], alt)

    try:
        mesh = trimesh.load(abs_path, force="mesh")
    except Exception as e:
        raise RuntimeError(f"Error importing mesh: {e}")

    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"trimesh did not return a single mesh for {filename}")

    if len(mesh.vertices) == 0:
        raise RuntimeError(f"No vertices found in {filename}")

    # ── Texture loaded? ──────────────────────────────────────────────────
    has_texture = (
        hasattr(mesh.visual, "material")
        and mesh.visual.material is not None
        and mesh.visual.material.image is not None
    )
    if not has_texture:
        msg = (
            f"Mesh loaded but texture image is missing — "
            f"MediaPipe needs textured renders ({filename})"
        )
        if allow_missing_texture:
            logger.warning(msg)
        else:
            raise RuntimeError(msg)

    return {
        "mesh_path": filename,
        "trimesh": mesh,
    }


def meshes_setup(meshes):
    """Normalise mesh to unit sphere, store transform params."""
    mesh = meshes["trimesh"]

    # Compute normalisation
    center = mesh.vertices.mean(axis=0)
    mesh.vertices = mesh.vertices - center
    scale = np.linalg.norm(mesh.vertices, axis=1, ord=2).max()
    if scale > 0:
        mesh.vertices = mesh.vertices / scale

    meshes.update({
        "transform_center": center.astype(np.float64),
        "transform_scale": float(scale),
        "trimesh": mesh,
    })
    return meshes


def save_facemarks_json(input_path, landmarks_3d, closest_vertices_ids, json_path, camera_data=None):
    data = {
        "model": input_path,
        "coordinates": landmarks_3d.tolist() if hasattr(landmarks_3d, 'tolist') else landmarks_3d,
        "closest_vertex_indexes": closest_vertices_ids,
    }
    if camera_data:
        data["cameras"] = [
            pos.tolist() if hasattr(pos, 'tolist') else pos
            for pos in camera_data
        ]
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
