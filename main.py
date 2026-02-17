import os, sys, json
from mvmp.arg_parser import parse_args
from mvmp.core.predict import *
from mvmp.core.io_utils import import_mesh, meshes_setup
import numpy as np
if __name__ == "__main__":
    args = parse_args()

    to_process = []
    if os.path.isdir(args.path):

        for r,_,files in os.walk(args.path):
            for f in files:
                if f.lower().endswith(".obj"):
                    to_process.append(os.path.join(r,f))

    else:

        if not args.path.endswith(".obj"):
            print(f"Error: {args.path} is not an .obj file. Only .obj meshes are supported")
            sys.exit(1)

        to_process.append(args.path)


    for file in to_process:
        meshes = import_mesh(file)
        # Auto-align è abilitato di default, disabilitato con --no-auto-align
        auto_align = not getattr(args, 'no_auto_align', False)
        args.meshes = meshes_setup(meshes, auto_align=auto_align)

        landmarks_3d, closest_vertices_ids, camera_data, landmark_candidates = predict(args)
        
        if landmarks_3d is None or closest_vertices_ids is None:
            print(f"\nSkipping {file} due to detection failure.\n")
            continue

### OUTPUT
        # Salva il percorso completo prima di sovrascrivere
        mesh_path = file
        
        # Usa output_path se specificato, altrimenti crea directory output/
        if args.output_path:
            directory = args.output_path
        else:
            directory = os.path.join(os.getcwd(), "output")
            os.makedirs(directory, exist_ok=True)
        
        file = os.path.splitext(os.path.basename(args.path))[0]
        json_file = os.path.join(directory, f"{file}_landmarks.json")

        # Ottieni i transform params per la visualizzazione (come nel backend)
        transform_params = {
            "center": args.meshes["transform_center"].tolist(),
            "scale": float(args.meshes["transform_scale"])
        }

        data = {
            "model": file,
            "normalized coordinates": landmarks_3d.tolist() if hasattr(landmarks_3d, 'tolist') else landmarks_3d,
            "closest vertex indexes": closest_vertices_ids,
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
        
        print()
