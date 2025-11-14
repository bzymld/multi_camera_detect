"""
配置加载模块
负责读取和解析摄像头配置文件
"""
import json
import numpy as np
from typing import Dict, List, Any


class ConfigLoader:
    """配置文件加载器"""
    
    def __init__(self, config_path: str):
        """
        初始化配置加载器
        
        参数:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = None
        self.cameras = []
        
    def load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        返回:
            配置字典
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            print(f"配置文件加载成功: {self.config_path}")
            return self.config
        except Exception as e:
            print(f"配置文件加载失败: {e}")
            raise
    
    def parse_cameras(self) -> List[Dict[str, Any]]:
        """
        解析摄像头配置
        
        返回:
            摄像头配置列表
        """
        if self.config is None:
            raise ValueError("请先加载配置文件")
        
        self.cameras = []
        for cam_config in self.config.get('cameras', []):
            camera_info = {
                'camera_id': cam_config.get('camera_id'),
                'name': cam_config.get('name'),
                'rtsp_url': cam_config.get('rtsp_url'),
                'camera_matrix': np.array(cam_config.get('camera_matrix')),
                'dist_coeffs': np.array(cam_config.get('dist_coeffs')),
                'height_mm': cam_config.get('height_mm', 2000),
                'tilt_angle': cam_config.get('tilt_angle', 20),
                'pole_offset_mm': np.array(cam_config.get('pole_offset_mm', [0, 0, 0])),
                'undistort_alpha': cam_config.get('undistort_alpha', 0.0)
            }
            self.cameras.append(camera_info)
        
        print(f"解析到 {len(self.cameras)} 个摄像头配置")
        return self.cameras
    
    def get_detection_config(self) -> Dict[str, Any]:
        """
        获取检测配置
        
        返回:
            检测配置字典
        """
        if self.config is None:
            raise ValueError("请先加载配置文件")
        
        return {
            'model_path': self.config.get('yolo_model', 'yolov8s.pt'),
            'confidence_threshold': self.config.get('detection', {}).get('confidence_threshold', 0.5)
        }
    
    def get_tracker_config(self) -> Dict[str, Any]:
        """
        获取跟踪器配置
        
        返回:
            跟踪器配置字典
        """
        if self.config is None:
            raise ValueError("请先加载配置文件")
        
        return self.config.get('tracker', {
            'max_disappeared': 30,
            'feature_threshold': 0.6,
            'position_threshold': 3.0
        })
    
    def get_camera_by_id(self, camera_id: int) -> Dict[str, Any]:
        """
        根据ID获取摄像头配置
        
        参数:
            camera_id: 摄像头ID
            
        返回:
            摄像头配置字典
        """
        for camera in self.cameras:
            if camera['camera_id'] == camera_id:
                return camera
        raise ValueError(f"未找到ID为 {camera_id} 的摄像头配置")