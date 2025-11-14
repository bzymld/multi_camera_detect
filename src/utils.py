"""
工具函数模块
提供各种辅助功能
"""
import cv2
import numpy as np
import time
from typing import List, Tuple, Dict
import json


class FPSCounter:
    """帧率计数器"""
    
    def __init__(self, window_size: int = 30):
        """
        初始化FPS计数器
        
        参数:
            window_size: 计算平均FPS的窗口大小
        """
        self.window_size = window_size
        self.frame_times = []
        self.last_time = time.time()
    
    def update(self) -> float:
        """
        更新并返回当前FPS
        
        返回:
            当前FPS
        """
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.last_time = current_time
        
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
        
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            return fps
        return 0
    
    def get_fps(self) -> float:
        """
        获取当前FPS（不更新）
        
        返回:
            当前FPS
        """
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            return fps
        return 0


class ResultLogger:
    """结果日志记录器"""
    
    def __init__(self, log_file: str = None):
        """
        初始化日志记录器
        
        参数:
            log_file: 日志文件路径（可选）
        """
        self.log_file = log_file
        self.results_buffer = []
    
    def log_detection(self, camera_id: int, timestamp: float, 
                     detections: List[Dict], distances: List[float],
                     ground_coords: List[Tuple[float, float]]):
        """
        记录检测结果
        
        参数:
            camera_id: 摄像头ID
            timestamp: 时间戳
            detections: 检测结果列表
            distances: 距离列表
            ground_coords: 地面坐标列表
        """
        result = {
            'camera_id': camera_id,
            'timestamp': timestamp,
            'objects': []
        }
        
        for det, dist, coord in zip(detections, distances, ground_coords):
            obj = {
                'class_name': det['class_name'],
                'confidence': det['confidence'],
                'bbox': det['bbox'],
                'distance': dist,
                'ground_coord': {
                    'x': coord[0],
                    'y': coord[1]
                }
            }
            result['objects'].append(obj)
        
        self.results_buffer.append(result)
        
        # 打印到控制台
        print(f"\n[摄像头 {camera_id}] 时间: {timestamp:.2f}")
        for obj in result['objects']:
            print(f"  - {obj['class_name']}: 距离={obj['distance']:.2f}m, "
                  f"坐标=({obj['ground_coord']['x']:.2f}, {obj['ground_coord']['y']:.2f})")
    
    def save_to_file(self):
        """保存日志到文件"""
        if self.log_file and len(self.results_buffer) > 0:
            try:
                with open(self.log_file, 'w', encoding='utf-8') as f:
                    json.dump(self.results_buffer, f, indent=2)
                print(f"日志已保存到: {self.log_file}")
            except Exception as e:
                print(f"保存日志失败: {e}")
    
    def clear_buffer(self):
        """清空缓冲区"""
        self.results_buffer.clear()


def draw_info_panel(frame: np.ndarray, camera_id: int, fps: float, 
                    detection_count: int) -> np.ndarray:
    """
    在图像上绘制信息面板
    
    参数:
        frame: 输入图像
        camera_id: 摄像头ID
        fps: 帧率
        detection_count: 检测数量
        
    返回:
        绘制了信息面板的图像
    """
    img = frame.copy()
    h, w = img.shape[:2]
    
    # 绘制半透明背景
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    
    # 绘制文本信息
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"Camera: {camera_id}", (20, 35), font, 0.6, (255, 255, 255), 2)
    cv2.putText(img, f"FPS: {fps:.1f}", (20, 60), font, 0.6, (255, 255, 255), 2)
    cv2.putText(img, f"Objects: {detection_count}", (20, 85), font, 0.6, (255, 255, 255), 2)
    
    return img


def resize_frame(frame: np.ndarray, max_width: int = 1280, max_height: int = 720) -> np.ndarray:
    """
    调整图像大小以适应显示
    
    参数:
        frame: 输入图像
        max_width: 最大宽度
        max_height: 最大高度
        
    返回:
        调整后的图像
    """
    h, w = frame.shape[:2]
    
    # 计算缩放比例
    scale = min(max_width / w, max_height / h)
    
    if scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized
    
    return frame


def concat_frames_horizontal(frames: List[np.ndarray]) -> np.ndarray:
    """
    水平拼接多个图像
    
    参数:
        frames: 图像列表
        
    返回:
        拼接后的图像
    """
    if len(frames) == 0:
        return None
    
    if len(frames) == 1:
        return frames[0]
    
    # 确保所有图像高度相同
    max_height = max(f.shape[0] for f in frames)
    resized_frames = []
    
    for frame in frames:
        if frame.shape[0] != max_height:
            scale = max_height / frame.shape[0]
            new_w = int(frame.shape[1] * scale)
            resized = cv2.resize(frame, (new_w, max_height), interpolation=cv2.INTER_AREA)
            resized_frames.append(resized)
        else:
            resized_frames.append(frame)
    
    # 水平拼接
    concatenated = np.hstack(resized_frames)
    
    return concatenated



def create_bird_eye_view(ground_coords: List[Tuple[float, float]], 
                        class_ids: List[int],
                        view_size: Tuple[int, int] = (900, 900),
                        scale: float = 15.0,
                        max_distance: float = 30.0) -> np.ndarray:
    """
    创建鸟瞰图显示目标位置（原点居中，支持四路摄像头东南西北方向）
    
    参数:
        ground_coords: 地面坐标列表 [(x, y), ...]
        class_ids: 类别ID列表
        view_size: 视图大小 (height, width)
        scale: 缩放比例（像素/米）
        max_distance: 最大显示距离（米）
        
    返回:
        鸟瞰图图像
    """
    h, w = view_size
    bird_view = np.ones((h, w, 3), dtype=np.uint8) * 250
    
    # 原点在中心
    center_x = w // 2
    center_y = h // 2
    
    # 绘制背景区域（四个象限用不同颜色）
    # 东北（右上，X+Y+）- 浅蓝
    cv2.rectangle(bird_view, (center_x, 0), (w, center_y), (255, 250, 240), -1)
    # 东南（右下，X+Y-）- 浅绿
    cv2.rectangle(bird_view, (center_x, center_y), (w, h), (240, 255, 245), -1)
    # 西南（左下，X-Y-）- 浅黄
    cv2.rectangle(bird_view, (0, center_y), (center_x, h), (240, 250, 255), -1)
    # 西北（左上，X-Y+）- 浅粉
    cv2.rectangle(bird_view, (0, 0), (center_x, center_y), (250, 240, 255), -1)
    
    # 绘制同心圆（距离圈）
    for radius_m in [5, 10, 15, 20, 25, 30]:
        if radius_m <= max_distance:
            radius_px = int(radius_m * scale)
            color = (200, 200, 200) if radius_m % 10 == 0 else (220, 220, 220)
            thickness = 2 if radius_m % 10 == 0 else 1
            cv2.circle(bird_view, (center_x, center_y), radius_px, color, thickness)
            
            # 标注距离（在东方向）
            if radius_m % 10 == 0:
                label = f"{radius_m}m"
                cv2.putText(bird_view, label, (center_x + radius_px + 3, center_y - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    
    # 绘制网格线（每5米一条）
    for i in range(-30, 31, 5):
        if abs(i) <= max_distance:
            # 垂直线（X方向）
            x_pos = int(center_x + i * scale)
            if 0 <= x_pos < w:
                line_color = (180, 180, 180) if i == 0 else (230, 230, 230)
                thickness = 2 if i == 0 else 1
                cv2.line(bird_view, (x_pos, 0), (x_pos, h), line_color, thickness)
            
            # 水平线（Y方向）
            y_pos = int(center_y - i * scale)  # Y轴向上为正
            if 0 <= y_pos < h:
                line_color = (180, 180, 180) if i == 0 else (230, 230, 230)
                thickness = 2 if i == 0 else 1
                cv2.line(bird_view, (0, y_pos), (w, y_pos), line_color, thickness)
    
    # 绘制方向指示箭头
    arrow_len = 50
    arrow_color = (80, 80, 80)
    arrow_thickness = 3
    
    # 北（上，Y正方向）
    cv2.arrowedLine(bird_view, (center_x, center_y), (center_x, center_y - arrow_len),
                    arrow_color, arrow_thickness, tipLength=0.25)
    cv2.putText(bird_view, "NORTH", (center_x - 30, center_y - arrow_len - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # 东（右，X正方向）
    cv2.arrowedLine(bird_view, (center_x, center_y), (center_x + arrow_len, center_y),
                    arrow_color, arrow_thickness, tipLength=0.25)
    cv2.putText(bird_view, "EAST", (center_x + arrow_len + 10, center_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # 南（下，Y负方向）
    cv2.arrowedLine(bird_view, (center_x, center_y), (center_x, center_y + arrow_len),
                    arrow_color, arrow_thickness, tipLength=0.25)
    cv2.putText(bird_view, "SOUTH", (center_x - 30, center_y + arrow_len + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # 西（左，X负方向）
    cv2.arrowedLine(bird_view, (center_x, center_y), (center_x - arrow_len, center_y),
                    arrow_color, arrow_thickness, tipLength=0.25)
    cv2.putText(bird_view, "WEST", (center_x - arrow_len - 50, center_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # 绘制原点（中心点）
    cv2.circle(bird_view, (center_x, center_y), 18, (0, 0, 200), -1)
    cv2.circle(bird_view, (center_x, center_y), 18, (255, 255, 255), 2)
    cv2.circle(bird_view, (center_x, center_y), 8, (255, 255, 0), -1)
    cv2.putText(bird_view, "O", (center_x - 6, center_y + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # 统计目标数量（按象限）
    person_count = 0
    vehicle_count = 0
    quadrant_counts = {
        'NE': 0,  # 东北 (X+, Y+)
        'SE': 0,  # 东南 (X+, Y-)
        'SW': 0,  # 西南 (X-, Y-)
        'NW': 0   # 西北 (X-, Y+)
    }
    
    # 绘制检测目标
    for coord, class_id in zip(ground_coords, class_ids):
        gx, gy = coord
        
        # 转换到图像坐标（原点在中心）
        px = int(center_x + gx * scale)  # X: 左负右正
        py = int(center_y - gy * scale)  # Y: 下负上正（图像Y轴向下）
        
        # 检查是否在视图范围内
        if 0 <= px < w and 0 <= py < h:
            # 根据类别选择颜色和大小
            if class_id == 0:  # person
                color = (0, 200, 0)
                radius = 12
                person_count += 1
                label = "P"
            else:  # vehicle
                color = (200, 0, 0)
                radius = 16
                vehicle_count += 1
                label = "V"
            
            # 确定象限
            if gx >= 0 and gy >= 0:
                quadrant_counts['NE'] += 1
            elif gx >= 0 and gy < 0:
                quadrant_counts['SE'] += 1
            elif gx < 0 and gy < 0:
                quadrant_counts['SW'] += 1
            else:  # gx < 0 and gy >= 0
                quadrant_counts['NW'] += 1
            
            # 绘制目标
            cv2.circle(bird_view, (px, py), radius, color, -1)
            cv2.circle(bird_view, (px, py), radius, (255, 255, 255), 2)
            
            # 显示标签
            cv2.putText(bird_view, label, (px - 6, py + 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # 显示坐标和距离
            dist = np.sqrt(gx**2 + gy**2)
            coord_text = f"({gx:.1f},{gy:.1f})"
            dist_text = f"{dist:.1f}m"
            
            # 根据位置决定文字显示位置（避免遮挡）
            if px > center_x:  # 右半边
                text_x = px + radius + 5
            else:  # 左半边
                text_x = px - radius - 65
            
            if py < center_y:  # 上半边
                text_y = py
            else:  # 下半边
                text_y = py + 10
            
            cv2.putText(bird_view, coord_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            cv2.putText(bird_view, dist_text, (text_x, text_y + 13),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 150), 1)
            
            # 绘制到中心的虚线
            line_color = (180, 180, 180)
            dx = px - center_x
            dy = py - center_y
            line_dist = np.sqrt(dx*dx + dy*dy)
            if line_dist > 20:  # 只为距离较远的目标画线
                steps = int(line_dist / 10)
                for i in range(0, steps, 2):
                    t1 = i / steps
                    t2 = min((i + 1) / steps, 1.0)
                    p1 = (int(center_x + dx * t1), int(center_y + dy * t1))
                    p2 = (int(center_x + dx * t2), int(center_y + dy * t2))
                    cv2.line(bird_view, p1, p2, line_color, 1)
    
    # 添加信息面板
    info_height = 140
    cv2.rectangle(bird_view, (0, 0), (w, info_height), (255, 255, 255), -1)
    cv2.line(bird_view, (0, info_height), (w, info_height), (150, 150, 150), 2)
    
    # 标题
    cv2.putText(bird_view, "Bird's Eye View - 360 Degree Coverage", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # 总体统计
    cv2.putText(bird_view, f"Total: {person_count + vehicle_count} objects", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.putText(bird_view, f"Persons: {person_count}", (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2)
    cv2.putText(bird_view, f"Vehicles: {vehicle_count}", (10, 104),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 0, 0), 2)
    
    # 象限统计（四路摄像头方向）
    quad_x = 250
    cv2.putText(bird_view, "Quadrant Distribution:", (quad_x, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.putText(bird_view, f"NE(+,+): {quadrant_counts['NE']:2d}  |  SE(+,-): {quadrant_counts['SE']:2d}",
                (quad_x, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(bird_view, f"NW(-,+): {quadrant_counts['NW']:2d}  |  SW(-,-): {quadrant_counts['SW']:2d}",
                (quad_x, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    
    # 图例
    legend_x = 600
    cv2.circle(bird_view, (legend_x, 72), 9, (0, 200, 0), -1)
    cv2.circle(bird_view, (legend_x, 72), 9, (255, 255, 255), 2)
    cv2.putText(bird_view, "Person", (legend_x + 18, 77),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv2.circle(bird_view, (legend_x, 97), 9, (200, 0, 0), -1)
    cv2.circle(bird_view, (legend_x, 97), 9, (255, 255, 255), 2)
    cv2.putText(bird_view, "Vehicle", (legend_x + 18, 102),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # 范围说明
    cv2.putText(bird_view, f"Max Range: {max_distance}m", (legend_x + 130, 77),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(bird_view, "Origin: Center (0,0)", (legend_x + 130, 97),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    
    # 坐标系说明
    cv2.putText(bird_view, "Coordinate System: X(West- / East+), Y(South- / North+)", (10, 128),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)
    
    return bird_view