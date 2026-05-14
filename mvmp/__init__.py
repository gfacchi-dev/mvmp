"""
MVMP: 3D Multi-View MediaPipe
Facial landmark detection for 3D meshes
"""

from .core.facemarker import Facemarker, FacemarkerResult

__version__ = "1.3.2"
__all__ = ["Facemarker", "FacemarkerResult"]
