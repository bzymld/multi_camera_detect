"""
单目测距模块
基于相机标定参数和几何关系计算目标距离
"""
import numpy as np
from typing import Tuple, List


class MonocularDistance:
    """单目测距类"""
    
    # 参考高度（毫米）
    PERSON_HEIGHT = 1700  # 行人平均高度
    CAR_HEIGHT = 1500     # 汽车平均高度
    TRUCK_HEIGHT = 2500   # 卡车平均高度
    BUS_HEIGHT = 3000     # 公交车平均高度
    
    def __init__(self, camera_matrix: np.ndarray, camera_height: float, tilt_angle: float):
        """
        初始化单目测距器
        
        参数:
            camera_matrix: 相机内参矩阵
            camera_height: 相机高度（毫米）
            tilt_angle: 相机俯仰角（度）
        """
        self.camera_matrix = camera_matrix
        self.camera_height = camera_height
        self.tilt_angle = tilt_angle
        self.tilt_rad = np.deg2rad(tilt_angle)
        
        # 提取焦距
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]
    
    def calculate_distance_by_height(self, bbox_height: float, object_height: float) -> float:
        """
        基于实际高度计算距离
        
        参数:
            bbox_height: 边界框高度（像素）
            object_height: 目标实际高度（毫米）
            
        返回:
            距离（米）
        """
        if bbox_height <= 0:
            return 0.0
        
        # 使用相似三角形原理: distance = (real_height * focal_length) / pixel_height
        distance_mm = (object_height * self.fy) / bbox_height
        distance_m = distance_mm / 1000.0
        
        return distance_m
    
    # def calculate_distance_by_ground_point(self, image_point: Tuple[float, float]) -> float:
    #     """
    #     基于地面点计算距离（考虑相机俯仰角）
    #
    #     参数:
    #         image_point: 图像坐标 (x, y)，通常是目标底部中心点
    #
    #     返回:
    #         距离（米）
    #     """
    #     px, py = image_point
    #
    #     # 计算像素点相对于图像中心的偏移
    #     dx = px - self.cx
    #     dy = py - self.cy
    #
    #     # 计算视线与水平面的夹角
    #     # 注意：相机坐标系Y轴向下，所以dy为正表示目标在图像下方
    #     theta_pixel = np.arctan(dy / self.fy)
    #
    #     # 考虑相机俯仰角：相机向下倾斜，所以要加上俯仰角
    #     # 最终角度 = 像素角度 + 俯仰角
    #     theta = theta_pixel + self.tilt_rad
    #
    #     # 如果视线朝上（theta < 0），无法确定距离
    #     if theta <= 0:
    #         return 0.0
    #
    #     # 计算地面距离（水平距离）
    #     # distance = height / tan(theta)
    #     distance_mm = self.camera_height / np.tan(theta)
    #     distance_m = distance_mm / 1000.0
    #
    #     return distance_m

    def calculate_distance_by_ground_point(self, image_point: Tuple[float, float]) -> float:
        """
        基于地面点计算距离（考虑相机俯仰角）

        参数:
            image_point: 图像坐标 (x, y)，通常是目标底部中心点

        返回:
            距离（米）
        """
        px, py = image_point

        # 计算像素点相对于图像中心的偏移
        dx = px - self.cx
        dy = py - self.cy

        # 计算相机坐标系中的射线方向
        ray_direction = np.array([dx / self.fx, dy / self.fy, 1.0])  # 归一化射线

        # 使用俯仰角调整射线方向
        # 在这里我们可以应用俯仰角来调整射线在z轴上的分量
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(self.tilt_rad), -np.sin(self.tilt_rad)],
            [0, np.sin(self.tilt_rad), np.cos(self.tilt_rad)]
        ])

        ray_direction = np.dot(rotation_matrix, ray_direction)  # 变换射线方向

        # 计算与地面的交点（沿z轴）
        if ray_direction[2] <= 0:
            return 0.0  # 射线与地面没有交点

        distance_mm = self.camera_height / ray_direction[2]  # 沿z轴的距离
        distance_m = distance_mm / 1000.0

        return distance_m
    
    def calculate_distance_hybrid(self, bbox_height: float, bottom_center: Tuple[float, float],
                                  object_height: float) -> float:
        """
        混合方法计算距离（结合高度和地面点）
        
        参数:
            bbox_height: 边界框高度（像素）
            bottom_center: 底部中心点坐标 (x, y)
            object_height: 目标实际高度（毫米）
            
        返回:
            距离（米）
        """
        # 方法1：基于高度
        dist1 = self.calculate_distance_by_height(bbox_height, object_height)
        
        # 方法2：基于地面点
        dist2 = self.calculate_distance_by_ground_point(bottom_center)
        
        # 取平均或选择更可靠的方法
        # 这里我们优先使用地面点方法，因为它考虑了相机俯仰角
        if dist2 > 0:
            return dist2
        else:
            return dist1
    
    def get_object_height_by_class(self, class_id: int) -> float:
        """
        根据类别ID获取目标参考高度
        
        参数:
            class_id: 类别ID
            
        返回:
            目标高度（毫米）
        """
        if class_id == 0:  # person
            return self.PERSON_HEIGHT
        elif class_id == 2:  # car
            return self.CAR_HEIGHT
        elif class_id == 5:  # bus
            return self.BUS_HEIGHT
        elif class_id == 7:  # truck
            return self.TRUCK_HEIGHT
        else:
            return self.PERSON_HEIGHT  # 默认值
    
    def calculate_distances_for_detections(self, detections: List[dict]) -> List[float]:
        """
        为检测结果列表计算距离
        
        参数:
            detections: 检测结果列表
            
        返回:
            距离列表（米）
        """
        distances = []
        
        for det in detections:
            bbox = det['bbox']
            bottom_center = det['bottom_center']
            class_id = det['class_id']
            
            # 计算边界框高度
            bbox_height = bbox[3] - bbox[1]
            
            # 获取目标参考高度
            object_height = self.get_object_height_by_class(class_id)
            
            # 使用地面点方法计算距离（更准确）
            distance = self.calculate_distance_by_ground_point(bottom_center)
            
            # 如果地面点方法失败，使用高度方法
            if distance <= 0:
                distance = self.calculate_distance_by_height(bbox_height, object_height)
            
            distances.append(distance)
        
        return distances
    
    def pixel_to_camera_ray(self, image_point: Tuple[float, float]) -> np.ndarray:
        """
        将像素坐标转换为相机坐标系下的射线方向
        
        参数:
            image_point: 图像坐标 (x, y)
            
        返回:
            归一化的射线方向向量 [x, y, z]
        """
        px, py = image_point
        
        # 转换到归一化相机坐标
        x = (px - self.cx) / self.fx
        y = (py - self.cy) / self.fy
        z = 1.0
        
        # 归一化
        ray = np.array([x, y, z])
        ray = ray / np.linalg.norm(ray)
        
        return ray
    
    def estimate_3d_position(self, image_point: Tuple[float, float], distance: float) -> np.ndarray:
        """
        估计目标的3D位置（相机坐标系）
        
        参数:
            image_point: 图像坐标 (x, y)
            distance: 距离（米）
            
        返回:
            3D坐标 [x, y, z]（米）
        """
        ray = self.pixel_to_camera_ray(image_point)
        
        # 计算3D坐标
        # 注意：这里的距离是沿着射线的距离
        position_3d = ray * distance
        
        return position_3d