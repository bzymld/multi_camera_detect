"""
双摄像头目标检测与测距系统 - 核心模块
"""

__version__ = '1.0.0'
__author__ = 'Multi Camera Detection System'

from .config_loader import ConfigLoader
from .camera_stream import CameraStream, MultiCameraManager
from .detector import ObjectDetector
from .monocular_distance import MonocularDistance
from .coordinate_transform import CoordinateTransform
from .utils import FPSCounter, ResultLogger

__all__ = [
    'ConfigLoader',
    'CameraStream',
    'MultiCameraManager',
    'ObjectDetector',
    'MonocularDistance',
    'CoordinateTransform',
    'FPSCounter',
    'ResultLogger'
]