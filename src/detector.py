"""
YOLOv8检测模块
负责行人和汽车的目标检测
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict


class ObjectDetector:
    """目标检测器类"""
    
    # COCO数据集类别ID
    PERSON_CLASS_ID = 0  # 行人
    CAR_CLASS_ID = 2     # 汽车
    TRUCK_CLASS_ID = 7   # 卡车
    BUS_CLASS_ID = 5     # 公交车
    
    # 我们关注的类别
    TARGET_CLASSES = [PERSON_CLASS_ID, CAR_CLASS_ID, TRUCK_CLASS_ID, BUS_CLASS_ID]
    
    # 类别名称映射
    CLASS_NAMES = {
        0: 'person',
        2: 'car',
        5: 'bus',
        7: 'truck'
    }
    
    def __init__(self, model_path: str = 'yolov8s.pt', confidence_threshold: float = 0.5):
        """
        初始化检测器
        
        参数:
            model_path: YOLOv8模型路径
            confidence_threshold: 置信度阈值
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        
    def load_model(self) -> bool:
        """
        加载YOLOv8模型
        
        返回:
            是否加载成功
        """
        try:
            self.model = YOLO(self.model_path)
            print(f"YOLOv8模型加载成功: {self.model_path}")
            return True
        except Exception as e:
            print(f"YOLOv8模型加载失败: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        检测图像中的目标
        
        参数:
            frame: 输入图像
            
        返回:
            检测结果列表，每个元素包含:
            - class_id: 类别ID
            - class_name: 类别名称
            - confidence: 置信度
            - bbox: 边界框 [x1, y1, x2, y2]
            - center: 中心点 [cx, cy]
            - bottom_center: 底部中心点 [cx, by]
        """
        if self.model is None:
            raise ValueError("模型未加载，请先调用load_model方法")
        
        # 执行检测
        results = self.model(frame, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # 只保留目标类别且置信度高于阈值的检测
                if class_id in self.TARGET_CLASSES and confidence >= self.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # 计算中心点和底部中心点
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    by = y2  # 底部Y坐标
                    
                    detection = {
                        'class_id': class_id,
                        'class_name': self.CLASS_NAMES.get(class_id, 'unknown'),
                        'confidence': confidence,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'center': [float(cx), float(cy)],
                        'bottom_center': [float(cx), float(by)]
                    }
                    detections.append(detection)
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict], 
                       distances: List[float] = None, 
                       ground_coords: List[Tuple[float, float]] = None) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        参数:
            frame: 输入图像
            detections: 检测结果列表
            distances: 距离列表（可选）
            ground_coords: 地面坐标列表（可选）
            
        返回:
            绘制了检测框的图像
        """
        img = frame.copy()
        
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # 设置颜色
            if det['class_id'] == self.PERSON_CLASS_ID:
                color = (0, 255, 0)  # 绿色
            else:
                color = (255, 0, 0)  # 蓝色
            
            # 绘制边界框
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # 绘制底部中心点
            cx, by = det['bottom_center']
            cv2.circle(img, (int(cx), int(by)), 5, (0, 0, 255), -1)
            
            # 准备标签文本
            label = f"{class_name}: {confidence:.2f}"
            
            # 添加距离信息
            if distances is not None and idx < len(distances):
                distance = distances[idx]
                label += f" {distance:.2f}m"
            
            # 添加地面坐标信息
            if ground_coords is not None and idx < len(ground_coords):
                gx, gy = ground_coords[idx]
                label += f" ({gx:.1f}, {gy:.1f})"
            
            # 绘制标签
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_y = max(int(y1) - 10, label_size[1])
            cv2.rectangle(img, (int(x1), label_y - label_size[1] - 5), 
                         (int(x1) + label_size[0], label_y + 5), color, -1)
            cv2.putText(img, label, (int(x1), label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return img
    
    @staticmethod
    def filter_by_class(detections: List[Dict], class_ids: List[int]) -> List[Dict]:
        """
        根据类别ID过滤检测结果
        
        参数:
            detections: 检测结果列表
            class_ids: 要保留的类别ID列表
            
        返回:
            过滤后的检测结果列表
        """
        return [det for det in detections if det['class_id'] in class_ids]
    
    @staticmethod
    def get_person_detections(detections: List[Dict]) -> List[Dict]:
        """
        获取行人检测结果
        
        参数:
            detections: 检测结果列表
            
        返回:
            行人检测结果列表
        """
        return [det for det in detections if det['class_id'] == ObjectDetector.PERSON_CLASS_ID]
    
    @staticmethod
    def get_vehicle_detections(detections: List[Dict]) -> List[Dict]:
        """
        获取车辆检测结果
        
        参数:
            detections: 检测结果列表
            
        返回:
            车辆检测结果列表
        """
        vehicle_ids = [ObjectDetector.CAR_CLASS_ID, ObjectDetector.TRUCK_CLASS_ID, 
                      ObjectDetector.BUS_CLASS_ID]
        return [det for det in detections if det['class_id'] in vehicle_ids]
