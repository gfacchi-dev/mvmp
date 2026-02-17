import numpy as np
import open3d as o3d
import os


def configure_headless_rendering():
    """Configure for headless rendering via EGL — no X11 or Xvfb required.

    Open3D's OffscreenRenderer uses EGL on Linux, which renders directly
    on the GPU (or CPU via softpipe) without a display server.
    Call this before creating an OffscreenRenderer instance.
    """
    if "DISPLAY" not in os.environ:
        os.environ.setdefault("EGL_PLATFORM", "surfaceless")


def ensure_display():
    """Ensure a display is available for interactive windows (e.g. render_result).

    Not needed for headless prediction — the OffscreenRenderer handles that.
    Only required if you want to open an interactive Open3D window.
    """
    import subprocess

    if "DISPLAY" in os.environ and os.environ["DISPLAY"]:
        return

    print("No display detected, starting Xvfb...")
    try:
        display_num = 99
        proc = subprocess.Popen(
            ["Xvfb", f":{display_num}", "-screen", "0", "1024x768x24", "-ac"]
        )
        os.environ["DISPLAY"] = f":{display_num}"
        print(f"Xvfb started on display :{display_num} (pid={proc.pid})")
    except FileNotFoundError:
        raise RuntimeError(
            "No display found and Xvfb is not installed. "
            "Install it with: sudo apt install xvfb"
        )


def render_result(actual_mesh, closest_vertices_ids):
    if os.environ.get("DISPLAY", "") == ":99":
        print("Cannot display with virtual display, ignoring --render flag\n")
        return

    closest_vertices = np.asarray(actual_mesh.vertices)[closest_vertices_ids]
    closest_vertices_pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(closest_vertices)
    )
    closest_vertices_pcd.colors = o3d.utility.Vector3dVector(
        [[1, 0, 1] for _ in range(len(closest_vertices))]
    )
    o3d.visualization.draw([actual_mesh, closest_vertices_pcd])
