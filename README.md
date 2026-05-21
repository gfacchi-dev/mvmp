# MVMP: 3D Multi-View MediaPipe

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Framework](https://img.shields.io/badge/Framework-Python_3.11-yellow)](https://www.python.org/downloads/release/python-3110/) [![Face Landmarker](https://img.shields.io/badge/Model-MediaPipe_Face_Landmarker-red)](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)

## Description

MVMP detects 478 MediaPipe-compatible 3D facial landmarks on textured meshes. 
It renders the mesh from 5 deterministic zone-based viewpoints (front, left/right sides, 
upper/lower), detects 2D landmarks with MediaPipe, and back-projects each to the mesh 
surface via raycasting. An optional Fibonacci-sphere auto-alignment phase orients the face 
toward +Z before detection.

**Supported mesh formats:** .obj, .ply, .stl, .gltf, .glb, .off

## System Requirements

| Platform | OpenGL backend | Notes |
|---|---|---|
| Linux + GPU | EGL (auto-detected) | Works out of the box |
| Linux headless (no GPU) | OSMesa | `apt install libosmesa6` (Ubuntu/Debian) · `dnf install mesa-libOSMesa` (Fedora/RHEL) |
| macOS | CGL (system OpenGL) | OpenGL deprecated since macOS 10.14 but still functional |
| Windows | WGL | Requires a GPU; pure CPU/headless not supported |

**Python:** 3.11 – 3.12 fully supported. Python 3.13 may work but mediapipe wheels are not yet universally available for that version.

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

marker = Facemarker()
result = marker.predict("path/to/mesh.obj")
print(result)  # FacemarkerResult(478 landmarks, 478 vertex indices)

landmarks_3d = result.landmarks_3d              # dict[int, [x, y, z]]
vertex_indices = result.closest_vertices_ids    # dict[int, int]

result.save_json("landmarks.json")
```

#### Quiet mode / debug output

```python
marker = Facemarker(verbose=False)

# Save per-zone renders and auto-align report
marker = Facemarker(debug_output_dir="debug/")
```

#### Camera distance

```python
# Move cameras closer (0.5×) or farther (2×)
marker = Facemarker(camera_distance_multiplier=0.8)
```

#### Disable auto-alignment

```python
# Skip Fibonacci-sphere alignment (mesh assumed to already face +Z)
marker = Facemarker(auto_orient=False)
```

### Command Line

```bash
mvmp path/to/mesh.obj -o output/

# Process a folder
mvmp meshes/ -o results/

# With debug renders
mvmp mesh.obj --debug debug_output/

# Closer camera, no auto-align
mvmp mesh.obj --camera-distance 0.8 --no-auto-orient
```

**Arguments:**
- `path`: Mesh file or directory
- `-o, --output-path`: Output directory (default: `./output/`)
- `--debug`: Save debug renders and auto-align report to this directory
- `--camera-distance`: Camera distance multiplier (default: 1.0)
- `--no-auto-orient`: Skip Fibonacci-sphere auto-alignment

### Output Format

```json
{
  "coordinates": {"0": [x, y, z], ...},
  "closest_vertex_indexes": {"0": 12345, ...}
}
```

## Contributing

1. Fork the repository and create a feature branch.
2. Make your changes with clear commit messages.
3. Open a pull request.

## Citation
If you use MVMP in your research or software, please cite it:

```bibtex
@software{Facchi_mvmp,
  author = {Facchi, G. M.},
  title  = {{mvmp}},
  url    = {https://github.com/gfacchi-dev/mvmp}
}
```

## License

[MIT License](LICENSE)

## Contact

Questions or suggestions? Open an issue on [GitHub](https://github.com/gfacchi-dev/mvmp/issues).
