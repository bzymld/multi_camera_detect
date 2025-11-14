"""
坐标转换模块
将相机坐标系转换到统一的地面坐标系
"""
import numpy as np
from typing import Tuple, List


class CoordinateTransform:
    """坐标转换类"""
    
    def __init__(self, camera_height: float, tilt_angle: float, 
                 pole_offset: np.ndarray = None):
        """
        初始化坐标转换器
        
        参数:
            camera_height: 相机高度（毫米）
            tilt_angle: 相机俯仰角（度）
            pole_offset: 相机相对于柱子的偏移 [x, y, z]（毫米）
        """
        self.camera_height = camera_height / 1000.0  # 转换为米
        self.tilt_angle = tilt_angle
        self.tilt_rad = np.deg2rad(tilt_angle)
        
        if pole_offset is None:
            self.pole_offset = np.array([0.0, 0.0, 0.0])
        else:
            self.pole_offset = np.array(pole_offset) / 1000.0  # 转换为米
        
        # 构建旋转矩阵（绕X轴旋转，因为是俯仰角）
        self.rotation_matrix = self._create_rotation_matrix()
    
    def _create_rotation_matrix(self) -> np.ndarray:
        """
        创建从相机坐标系到地面坐标系的旋转矩阵
        
        返回:
            3x3旋转矩阵
        """
        # 相机坐标系：Z轴向前，Y轴向下，X轴向右
        # 地面坐标系：X轴水平向右，Y轴水平向前，Z轴向上
        
        # 绕X轴旋转（俯仰角）
        cos_t = np.cos(self.tilt_rad)
        sin_t = np.sin(self.tilt_rad)
        
        # 旋转矩阵
        R_tilt = np.array([
            [1, 0, 0],
            [0, cos_t, -sin_t],
            [0, sin_t, cos_t]
        ])
        
        # 坐标系转换矩阵（相机到世界）
        # 相机：X右，Y下，Z前
        # 世界：X右，Y前，Z上
        R_cam_to_world = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ])
        
        # 组合旋转
        R = R_cam_to_world @ R_tilt
        
        return R
    
    def camera_to_ground(self, camera_point: np.ndarray) -> Tuple[float, float]:
        """
        将相机坐标系的点转换到地面坐标系
        
        参数:
            camera_point: 相机坐标系下的3D点 [x, y, z]（米）
            
        返回:
            地面坐标系下的2D点 (x, y)（米）
        """
        # 应用旋转
        rotated_point = self.rotation_matrix @ camera_point
        
        # 平移到地面坐标系（相机位置的偏移）
        # 相机在地面上的投影点作为原点
        world_x = rotated_point[0] + self.pole_offset[0]
        world_y = rotated_point[1] + self.pole_offset[1]
        world_z = rotated_point[2] + self.camera_height + self.pole_offset[2]
        
        # 投影到地面（z=0平面）
        # 这里假设目标已经在地面上，所以直接返回x和y
        ground_x = world_x
        ground_y = world_y
        
        return ground_x, ground_y
    
    def image_point_to_ground(self, image_point: Tuple[float, float], 
                             distance: float, 
                             camera_matrix: np.ndarray) -> Tuple[float, float]:
        """
        将图像点和距离转换到地面坐标
        
        参数:
            image_point: 图像坐标 (x, y)
            distance: 目标距离（米）
            camera_matrix: 相机内参矩阵
            
        返回:
            地面坐标 (x, y)（米）
        """
        px, py = image_point
        
        # 提取相机参数
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        
        # 计算归一化相机坐标
        x_norm = (px - cx) / fx
        y_norm = (py - cy) / fy
        
        # 使用已经计算好的距离（这是地面水平距离）
        ground_distance = distance
        
        # 计算地面X坐标（横向偏移）
        # 使用相似三角形：ground_x / ground_distance = x_norm / 1
        ground_x = x_norm * ground_distance
        
        # Y坐标就是地面距离（向前的距离）
        ground_y = ground_distance
        
        # 添加相机偏移
        ground_x += self.pole_offset[0]
        ground_y += self.pole_offset[1]
        
        return ground_x, ground_y
    
    def batch_transform_detections(self, detections: List[dict], 
                                   distances: List[float],
                                   camera_matrix: np.ndarray) -> List[Tuple[float, float]]:
        """
        批量转换检测结果到地面坐标
        
        参数:
            detections: 检测结果列表
            distances: 距离列表
            camera_matrix: 相机内参矩阵
            
        返回:
            地面坐标列表 [(x, y), ...]
        """
        ground_coords = []
        
        for det, dist in zip(detections, distances):
            bottom_center = det['bottom_center']
            ground_coord = self.image_point_to_ground(bottom_center, dist, camera_matrix)
            ground_coords.append(ground_coord)
        
        return ground_coords
    
    @staticmethod
    def merge_detections_from_cameras(detections_cam0: List[dict], 
                                      coords_cam0: List[Tuple[float, float]],
                                      detections_cam1: List[dict],
                                      coords_cam1: List[Tuple[float, float]],
                                      distance_threshold: float = 0.5) -> List[dict]:
        """
        合并来自不同相机的检测结果（基于地面坐标）
        
        参数:
            detections_cam0: 相机0的检测结果
            coords_cam0: 相机0的地面坐标
            detections_cam1: 相机1的检测结果
            coords_cam1: 相机1的地面坐标
            distance_threshold: 距离阈值（米），小于此值认为是同一目标
            
        返回:
            合并后的检测结果列表
        """
        merged_detections = []
        used_indices_cam1 = set()
        
        # 遍历相机0的检测
        for i, (det0, coord0) in enumerate(zip(detections_cam0, coords_cam0)):
            best_match_idx = -1
            best_match_dist = float('inf')
            
            # 在相机1中查找匹配
            for j, (det1, coord1) in enumerate(zip(detections_cam1, coords_cam1)):
                if j in used_indices_cam1:
                    continue
                
                # 检查类别是否相同
                if det0['class_id'] != det1['class_id']:
                    continue
                
                # 计算地面坐标距离
                dist = np.sqrt((coord0[0] - coord1[0])**2 + (coord0[1] - coord1[1])**2)
                
                if dist < distance_threshold and dist < best_match_dist:
                    best_match_idx = j
                    best_match_dist = dist
            
            # 如果找到匹配，合并；否则单独添加
            if best_match_idx >= 0:
                used_indices_cam1.add(best_match_idx)
                coord1 = coords_cam1[best_match_idx]
                
                # 取两个坐标的平均值
                merged_coord = (
                    (coord0[0] + coord1[0]) / 2,
                    (coord0[1] + coord1[1]) / 2
                )
                
                merged_det = det0.copy()
                merged_det['ground_coord'] = merged_coord
                merged_det['camera_id'] = 'merged'
                merged_detections.append(merged_det)
            else:
                merged_det = det0.copy()
                merged_det['ground_coord'] = coord0
                merged_det['camera_id'] = 0
                merged_detections.append(merged_det)
        
        # 添加相机1中未匹配的检测
        for j, (det1, coord1) in enumerate(zip(detections_cam1, coords_cam1)):
            if j not in used_indices_cam1:
                merged_det = det1.copy()
                merged_det['ground_coord'] = coord1
                merged_det['camera_id'] = 1
                merged_detections.append(merged_det)
        
        return merged_detections