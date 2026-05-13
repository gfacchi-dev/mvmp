"""Export mesh+landmarks as GLB asset or Plotly HTML, colored by zone."""
import os, sys, numpy as np, trimesh
from pathlib import Path

from mvmp.core.zone_config import LANDMARK_ZONES, ZONE_NAMES

ZONE_COLORS_RGB = {
    0: (255, 255, 255),   # front: white
    1: (0, 255, 255),      # left_quarter: cyan
    2: (255, 0, 255),      # right_quarter: magenta
    3: (255, 255, 0),      # upper: yellow
    4: (0, 255, 0),        # lower: lime
}
ZONE_COLORS_HEX = {
    0: 'white', 1: 'cyan', 2: 'magenta', 3: 'yellow', 4: 'lime',
}

def export_glb(mesh_path, lmks, out_path):
    mesh = trimesh.load(mesh_path, force='mesh')
    mesh.visual.face_colors = [180, 180, 180, 255]
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name='head')
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.5)
    for i, pt in enumerate(lmks):
        s = sphere.copy()
        s.apply_translation(pt)
        c = ZONE_COLORS_RGB[int(LANDMARK_ZONES[i])]
        s.visual.vertex_colors = [*c, 255]
        scene.add_geometry(s, node_name=f'lmk_{i}')
    scene.export(out_path)
    print(f'GLB → {out_path}')

def export_html(mesh_path, lmks, out_path, vids=None):
    import plotly.graph_objects as go
    mesh = trimesh.load(mesh_path, force='mesh')
    fig = go.Figure()
    n = min(50000, len(mesh.vertices))
    idx = np.random.choice(len(mesh.vertices), n, replace=False)
    fig.add_trace(go.Scatter3d(
        x=mesh.vertices[idx,0], y=mesh.vertices[idx,1], z=mesh.vertices[idx,2],
        mode='markers', marker=dict(size=1, color='lightgray', opacity=0.3), name='mesh'
    ))
    all_indices = np.arange(478)
    for z in sorted(ZONE_NAMES):
        mask = LANDMARK_ZONES == z
        zi = all_indices[mask]
        hover = []
        for i in zi:
            v = vids[str(i)] if vids else '?'
            hover.append(f'lmk {i}<br>vert {v}<br>{ZONE_NAMES[z]}')
        fig.add_trace(go.Scatter3d(
            x=lmks[mask,0], y=lmks[mask,1], z=lmks[mask,2],
            mode='markers', marker=dict(size=4, color=ZONE_COLORS_HEX[z]),
            name=ZONE_NAMES[z], text=hover, hoverinfo='text'
        ))
    fig.update_layout(scene=dict(aspectmode='data'), title=os.path.basename(mesh_path))
    fig.write_html(out_path)
    print(f'HTML → {out_path}')

if __name__ == '__main__':
    from mvmp import Facemarker
    mesh_path = sys.argv[1] if len(sys.argv) > 1 else None
    if mesh_path is None:
        print("Usage: uv run python visualize.py <mesh.obj>")
        sys.exit(1)
    marker = Facemarker()
    result = marker.predict(mesh_path)
    lmks = np.array(list(result.to_dict()['coordinates'].values()))
    vids = result.to_dict()['closest_vertex_indexes']
    d = Path('debug_output')
    d.mkdir(exist_ok=True)
    stem = Path(mesh_path).stem
    export_glb(mesh_path, lmks, str(d / f'{stem}_zones.glb'))
    export_html(mesh_path, lmks, str(d / f'{stem}_zones.html'), vids)
