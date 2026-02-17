"""
MVMP: 3D Multi-View MediaPipe
Facial landmark detection for 3D meshes
"""

from .core.facemarker import Facemarker, FacemarkerResult

__version__ = "0.2.5"
__all__ = ["Facemarker", "FacemarkerResult"]
