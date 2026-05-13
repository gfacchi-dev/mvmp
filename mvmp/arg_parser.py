import argparse
def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect facial landmarks on 3D meshes using MediaPipe + zone-based multi-view projection. Supports .obj, .ply, .stl, .gltf, .glb, .off."
    )

    parser.add_argument("path", help=(
        "Path to a single mesh file or a folder to be scanned recursively."
    ))
    parser.add_argument(
        "-o", "--output-path",
        type=str,
        help="Optional output directory. Default: ./output/"
    )
    parser.add_argument(
        "--debug",
        type=str,
        default=None,
        help="Save debug renders and auto-align report to this directory."
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=1.0,
        help="Camera distance multiplier (default: 1.0)."
    )
    parser.add_argument(
        "--no-auto-orient",
        action="store_true",
        help="Disable Fibonacci-sphere auto-alignment."
    )
    return parser.parse_args()
