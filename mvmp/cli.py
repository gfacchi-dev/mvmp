"""
CLI entry point for MVMP (3D Multi-View MediaPipe)
"""
import os
import sys
import json
import numpy as np

from .arg_parser import parse_args
from .core.predict import __predict
from .core.io_utils import import_mesh, meshes_setup


def main():
    """Main CLI entry point"""
    args = parse_args()

    # Supported mesh formats
    MESH_EXTENSIONS = ('.obj', '.ply', '.stl', '.gltf', '.glb', '.off')

    to_process = []
    if os.path.isdir(args.path):
        for r, _, files in os.walk(args.path):
            for f in files:
                if f.lower().endswith(MESH_EXTENSIONS):
                    to_process.append(os.path.join(r, f))
    else:
        if not args.path.lower().endswith(MESH_EXTENSIONS):
            print(f"Error: {args.path} is not a supported mesh file.")
            print(f"Supported formats: {', '.join(MESH_EXTENSIONS)}")
            sys.exit(1)
        to_process.append(args.path)

    for file in to_process:
        meshes = import_mesh(file)
        # Auto-align è abilitato di default, disabilitato con --no-auto-align se esiste
        auto_align = not getattr(args, 'no_auto_align', False)
        args.meshes = meshes_setup(meshes, auto_align=auto_align)

        landmarks_3d, closest_vertices_ids, camera_data, landmark_candidates = __predict(
            args.meshes, args.projections_number, args
        )

        if landmarks_3d is None or closest_vertices_ids is None:
            print(f"\nSkipping {file} due to detection failure.\n")
            continue

        # Usa output_path se specificato, altrimenti crea directory output/
        if args.output_path:
            directory = args.output_path
        else:
            directory = os.path.join(os.getcwd(), "output")
            os.makedirs(directory, exist_ok=True)

        file_name = os.path.splitext(os.path.basename(args.path))[0]
        json_file = os.path.join(directory, f"{file_name}_landmarks.json")

        # Ottieni i transform params
        transform_params = {
            "center": args.meshes["transform_center"].tolist(),
            "scale": float(args.meshes["transform_scale"])
        }

        data = {
            "model": file_name,
            "normalized_coordinates": landmarks_3d.tolist() if hasattr(landmarks_3d, 'tolist') else landmarks_3d,
            "closest_vertex_indexes": closest_vertices_ids,
            "transform_params": transform_params
        }

        # Aggiungi camera_data se disponibile
        if camera_data:
            data["cameras"] = [
                pos.tolist() if hasattr(pos, 'tolist') else pos
                for pos in camera_data
            ]

        with open(json_file, "w") as f:
            json.dump(data, f, indent=4)

        print(f"\n=== Landmarks saved in {json_file} ===")


if __name__ == "__main__":
    main()
