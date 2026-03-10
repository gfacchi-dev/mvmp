# MVMP: 3D Multi-View MediaPipe

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Framework](https://img.shields.io/badge/Framework-Python_3.11-yellow)](https://www.python.org/downloads/release/python-3110/) [![Face Landmarker](https://img.shields.io/badge/Model-MediaPipe_Face_Landmarker-red)](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)

## Description

MVMP (Multi-View MediaPipe) is a lightweight tool for 3D facial landmark detection on static textured meshes. It renders multiple camera views of the mesh, detects 2D landmarks with MediaPipe, and backprojects them into 3D space through DBSCAN-based consensus triangulation. The result is 478 facial landmarks aligned with the 3D mesh geometry, with robust outlier rejection.

**Supported mesh formats:** .obj, .ply, .stl, .gltf, .glb, .off

<!--![alt text](./img/pipelineOverview.png)-->
<img src="./img/pipelineOverview.png">

## Installation

```bash
pip install mvmp
```

The MediaPipe Face Landmarker model is bundled in the package.

### From Source

```bash
git clone https://github.com/gfacchi-dev/mvmp.git
cd mvmp
pip install .
```

## Usage

### Python API

```python
from mvmp import Facemarker

# Create a detector
marker = Facemarker()

# Detect landmarks on a mesh
result = marker.predict("path/to/mesh.obj")
print(result)  # FacemarkerResult(478 landmarks, 478 vertex indices)

# Access results
landmarks_3d = result.landmarks_3d          # list of [x, y, z] coordinates (original scale)
vertex_indices = result.closest_vertices_ids  # closest mesh vertex per landmark

# Save to JSON
result.save_json("landmarks.json")
```

#### More projections = more accuracy

```python
marker = Facemarker(projections=500)
result = marker.predict("mesh.obj")
```

#### Custom camera angles

Instead of random projections, specify exact (yaw, pitch) angles in degrees:

```python
marker = Facemarker(camera_angles=[
    (0, 0),       # front view
    (30, 0),      # 30 degrees right
    (-30, 0),     # 30 degrees left
    (0, -20),     # looking up
    (0, 15),      # looking down
])
result = marker.predict("mesh.obj")
```

#### Process multiple meshes

```python
marker = Facemarker(projections=200)

for mesh_path in mesh_files:
    result = marker.predict(mesh_path)
    result.save_json(f"output/{mesh_path.stem}.json")
```

#### Quiet mode

```python
marker = Facemarker(verbose=False)
result = marker.predict("mesh.obj")
```

### Command Line

```bash
mvmp path/to/mesh.obj -p 100 -o output/

# Process all mesh files in a directory (supports .obj, .ply, .stl, .gltf, .glb, .off)
mvmp meshes/ -p 200 -o results/
```

**Arguments:**
- `path`: Path to mesh file or directory
- `-p, --projections-number`: Number of projections (default: 500)
- `-o, --output-path`: Output directory

### Output Format

JSON output contains coordinates at the original mesh scale:

```json
{
  "coordinates": [[x, y, z], ...],
  "closest_vertex_indexes": [idx1, idx2, ...]
}
```

### Results
<!--![alt text](./img/results.png)-->
<img src="pipelineOverview.png">

## Contributing

1. Fork the repository and create a feature branch.
2. Make your changes with clear commit messages.
3. Open a pull request.

## License

[MIT License](LICENSE)

## Contact

Questions or suggestions? Open an issue on [GitHub](https://github.com/gfacchi-dev/mvmp/issues).
