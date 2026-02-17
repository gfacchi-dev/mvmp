import argparse
def parse_args():
    parser = argparse.ArgumentParser(
        description="Process a single mesh file or all mesh files in a folder or children folders. Supports .obj, .ply, .stl, .gltf, .glb, .off formats."
    )

    parser.add_argument("path", nargs='?', default="/var/datasets/LAFAS-84Aligned/raw", help=(
        "Path to a single mesh file or a folder to be accessed recursively. Default: /var/datasets/LAFAS-84Aligned/raw"
    ))
    
    parser.add_argument(
        "-p", "--projections-number",
        type=int,
        default=500,
        help="Number of projections to be calculated. Default is 500."
    )
    
    parser.add_argument(
        "-o", "--output-path",
        type=str,
        help="Optional. If set, results will be saved to this path. If not set, the output will be saved inside the mesh file folder."
    )
    
    parser.add_argument(
        "-r", "--render",
        action="store_true",
        help="If --render is set the result will be rendered on an external window."
    )

    parser.add_argument(
        "-d", "--double",
        action="store_true",
        help="If --double is set the mesh will be processed twice, the first time to adjust its alignment to the world frame, the second to actually predict the landmarks."
    )

    return parser.parse_args()
