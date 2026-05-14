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
    parser.add_argument(
        "--n-fibonacci",
        type=int,
        default=100,
        help="Number of Fibonacci-sphere probes for auto-alignment (default: 100)."
    )
    parser.add_argument(
        "--min-neighbor-support",
        type=int,
        default=3,
        help="Min neighbouring probes that must detect a face for a probe to be trusted (default: 3)."
    )
    parser.add_argument(
        "--max-score-isolation",
        type=float,
        default=3.0,
        help="Max score ratio vs neighbour median before probe is rejected as isolated spike (default: 3.0)."
    )
    parser.add_argument(
        "--max-direction-deviation",
        type=float,
        default=30.0,
        help="Max degrees between a probe's face direction and neighbour consensus (default: 30.0)."
    )
    parser.add_argument(
        "--min-direction-support",
        type=int,
        default=2,
        help="Min neighbour probes that must agree on face direction (default: 2)."
    )
    parser.add_argument(
        "--allow-missing-texture",
        action="store_true",
        help="Only warn (do not abort) when a mesh has no texture image loaded."
    )
    return parser.parse_args()
