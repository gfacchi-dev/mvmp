"""
CLI entry point for MVMP (3D Multi-View MediaPipe)
"""
import os
import sys
import json

from .arg_parser import parse_args
from .core.facemarker import Facemarker


def main():
    args = parse_args()

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

    marker = Facemarker(
        debug_output_dir=args.debug,
        camera_distance_multiplier=args.camera_distance,
        auto_orient=not args.no_auto_orient,
    )

    for file in to_process:
        print(f"Processing {file}...")
        try:
            result = marker.predict(file)
        except Exception as e:
            print(f"  Error: {e}")
            continue

        out_dir = args.output_path or os.path.join(os.getcwd(), "output")
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(file))[0]
        json_path = os.path.join(out_dir, f"{stem}_landmarks.json")
        result.save_json(json_path)
        print(f"  → {json_path}")


if __name__ == "__main__":
    main()
