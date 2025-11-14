"""
多摄像头目标检测与测距系统 - 主程序（支持1-N路）
"""
import cv2
import numpy as np
import sys
import os
import time
from pathlib import Path
from typing import List, Dict

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config_loader import ConfigLoader
from camera_stream import CameraStream, MultiCameraManager
from detector import ObjectDetector
from monocular_distance import MonocularDistance
from coordinate_transform import CoordinateTransform
from utils import FPSCounter, ResultLogger, draw_info_panel, create_bird_eye_view


class MultiCameraSystem:
    """多摄像头检测系统（支持1-N路）"""
    
    def __init__(self, config_path: str, camera_indices: List[int] = None):
        """
        初始化系统
        
        参数:
            config_path: 配置文件路径
            camera_indices: 要使用的摄像头索引列表，None表示使用全部
        """
        self.config_path = config_path
        
        # 加载配置
        print("正在加载配置...")
        self.config_loader = ConfigLoader(config_path)
        self.config_loader.load_config()
        all_cameras = self.config_loader.parse_cameras()
        self.detection_config = self.config_loader.get_detection_config()
        
        # 选择要使用的摄像头
        if camera_indices is None:
            self.cameras_config = all_cameras
            print(f"使用所有摄像头: {len(all_cameras)} 个")
        else:
            self.cameras_config = [all_cameras[i] for i in camera_indices if i < len(all_cameras)]
            print(f"使用指定摄像头: {len(self.cameras_config)} 个")
        
        for cam in self.cameras_config:
            print(f"  - {cam['name']} (ID: {cam['camera_id']})")
        
        # 初始化组件
        self.camera_manager = MultiCameraManager()
        self.detector = None
        self.distance_calculators = {}
        self.coord_transforms = {}
        self.fps_counters = {}
        self.result_logger = ResultLogger()
        
        # 性能优化参数
        self.detect_every_n_frames = 2  # 每N帧检测一次
        self.frame_counter = 0
        self.last_detections = {}  # 缓存检测结果
        
        # 显示参数（根据摄像头数量自动调整）
        num_cameras = len(self.cameras_config)
        if num_cameras == 1:
            self.display_width = 800
        elif num_cameras == 2:
            self.display_width = 640
        else:
            self.display_width = 480  # 3-4路使用更小的宽度
        
        # 运行标志
        self.is_running = False
    
    def initialize(self) -> bool:
        """
        初始化所有组件
        
        返回:
            是否初始化成功
        """
        try:
            # 初始化检测器
            print("\n正在加载YOLOv8模型...")
            model_path = self.detection_config['model_path']
            
            # 性能建议
            num_cameras = len(self.cameras_config)
            if num_cameras > 2 and 'yolov8s' in model_path:
                print(f"提示：处理{num_cameras}路视频流，强烈建议使用yolov8n.pt")
            
            self.detector = ObjectDetector(
                model_path=model_path,
                confidence_threshold=self.detection_config['confidence_threshold']
            )
            if not self.detector.load_model():
                return False
            
            # 初始化每个摄像头
            print("\n正在初始化摄像头...")
            for cam_config in self.cameras_config:
                camera_id = cam_config['camera_id']
                
                # 创建摄像头流
                camera_stream = CameraStream(
                    camera_id=camera_id,
                    rtsp_url=cam_config['rtsp_url'],
                    camera_matrix=cam_config['camera_matrix'],
                    dist_coeffs=cam_config['dist_coeffs'],
                    undistort_alpha=cam_config['undistort_alpha'],
                    reconnect_interval=3,
                    max_queue_size=2
                )
                self.camera_manager.add_camera(camera_id, camera_stream)
                
                # 创建测距器
                self.distance_calculators[camera_id] = MonocularDistance(
                    camera_matrix=cam_config['camera_matrix'],
                    camera_height=cam_config['height_mm'],
                    tilt_angle=cam_config['tilt_angle']
                )
                
                # 创建坐标转换器
                self.coord_transforms[camera_id] = CoordinateTransform(
                    camera_height=cam_config['height_mm'],
                    tilt_angle=cam_config['tilt_angle'],
                    pole_offset=cam_config['pole_offset_mm']
                )
                
                # 创建FPS计数器
                self.fps_counters[camera_id] = FPSCounter()
                
                print(f"摄像头 {camera_id} ({cam_config['name']}) 初始化完成")
                print(f"  - 高度: {cam_config['height_mm']}mm, 俯仰角: {cam_config['tilt_angle']}度")
            
            return True
            
        except Exception as e:
            print(f"初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_camera_frame(self, camera_id: int, frame: np.ndarray, do_detection: bool = True):
        """
        处理单个摄像头的帧
        
        参数:
            camera_id: 摄像头ID
            frame: 图像帧
            do_detection: 是否执行检测
            
        返回:
            (处理后的帧, 检测结果, 距离, 地面坐标)
        """
        # 如果不执行检测，使用缓存的结果
        if not do_detection and camera_id in self.last_detections:
            cached = self.last_detections[camera_id]
            detections = cached['detections']
            distances = cached['distances']
            ground_coords = cached['ground_coords']
        else:
            # 目标检测
            detections = self.detector.detect(frame)
            
            if len(detections) == 0:
                self.last_detections[camera_id] = {
                    'detections': [],
                    'distances': [],
                    'ground_coords': []
                }
                return frame, [], [], []
            
            # 计算距离
            distance_calc = self.distance_calculators[camera_id]
            distances = distance_calc.calculate_distances_for_detections(detections)
            
            # 转换到地面坐标系
            coord_transform = self.coord_transforms[camera_id]
            camera_config = self.config_loader.get_camera_by_id(camera_id)
            ground_coords = coord_transform.batch_transform_detections(
                detections, distances, camera_config['camera_matrix']
            )
            
            # 缓存检测结果
            self.last_detections[camera_id] = {
                'detections': detections,
                'distances': distances,
                'ground_coords': ground_coords
            }
        
        # 绘制检测结果
        result_frame = self.detector.draw_detections(frame, detections, distances, ground_coords)
        
        return result_frame, detections, distances, ground_coords
    
    def run(self):
        """运行系统主循环"""
        # 启动所有摄像头
        print("\n正在启动摄像头...")
        self.camera_manager.start_all()
        time.sleep(2)  # 等待摄像头稳定
        
        self.is_running = True
        num_cameras = len(self.cameras_config)
        print(f"\n系统运行中... 处理 {num_cameras} 路视频流")
        print("控制说明:")
        print("  - 按 'q' 退出")
        print("  - 按 's' 保存截图")
        print("  - 按 'd' 切换跳帧模式")
        print("  - 按 'b' 切换BEV显示\n")
        
        show_bev = True
        
        try:
            while self.is_running:
                self.frame_counter += 1
                
                # 判断是否执行检测（跳帧优化）
                do_detection = (self.frame_counter % self.detect_every_n_frames == 0)
                
                # 读取所有摄像头的帧
                frames_dict = self.camera_manager.read_all()
                
                display_frames = []
                all_ground_coords = []
                all_class_ids = []
                active_cameras = 0
                
                # 处理每个摄像头的帧
                for camera_id in sorted(frames_dict.keys()):
                    ret, frame = frames_dict[camera_id]
                    
                    if not ret or frame is None:
                        # 创建等待画面
                        wait_frame = np.zeros((360, 480, 3), dtype=np.uint8)
                        cv2.putText(wait_frame, f"Camera {camera_id}", (150, 150),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(wait_frame, "Connecting...", (150, 200),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        display_frames.append(wait_frame)
                        continue
                    
                    active_cameras += 1
                    
                    # 处理帧
                    result_frame, detections, distances, ground_coords = \
                        self.process_camera_frame(camera_id, frame, do_detection)
                    
                    # 统一调整画面大小
                    h, w = result_frame.shape[:2]
                    target_height = int(h * self.display_width / w)
                    result_frame = cv2.resize(result_frame, (self.display_width, target_height))
                    
                    # 更新FPS
                    fps = self.fps_counters[camera_id].update()
                    
                    # 绘制信息面板
                    result_frame = draw_info_panel(result_frame, camera_id, fps, len(detections))
                    
                    # 记录结果
                    if do_detection and len(detections) > 0:
                        self.result_logger.log_detection(
                            camera_id, time.time(), detections, distances, ground_coords
                        )
                        
                        # 收集地面坐标用于鸟瞰图
                        all_ground_coords.extend(ground_coords)
                        all_class_ids.extend([d['class_id'] for d in detections])
                    
                    display_frames.append(result_frame)
                
                # 显示结果
                if len(display_frames) > 0:
                    # 根据摄像头数量决定布局
                    if num_cameras == 1:
                        concat_frame = display_frames[0]
                    elif num_cameras == 2:
                        # 水平拼接
                        max_h = max(f.shape[0] for f in display_frames)
                        resized = []
                        for f in display_frames:
                            if f.shape[0] != max_h:
                                ratio = max_h / f.shape[0]
                                new_w = int(f.shape[1] * ratio)
                                f = cv2.resize(f, (new_w, max_h))
                            resized.append(f)
                        concat_frame = np.hstack(resized)
                    elif num_cameras <= 4:
                        # 2x2网格布局
                        max_h = max(f.shape[0] for f in display_frames)
                        max_w = max(f.shape[1] for f in display_frames)
                        
                        # 统一大小
                        resized = []
                        for f in display_frames:
                            if f.shape[0] != max_h or f.shape[1] != max_w:
                                f = cv2.resize(f, (max_w, max_h))
                            resized.append(f)
                        
                        # 补齐到4个
                        while len(resized) < 4:
                            blank = np.zeros((max_h, max_w, 3), dtype=np.uint8)
                            resized.append(blank)
                        
                        # 2x2拼接
                        top_row = np.hstack(resized[:2])
                        bottom_row = np.hstack(resized[2:4])
                        concat_frame = np.vstack([top_row, bottom_row])
                    else:
                        # 更多路数，简单水平拼接（可以改进为更复杂的网格）
                        max_h = max(f.shape[0] for f in display_frames)
                        resized = []
                        for f in display_frames:
                            if f.shape[0] != max_h:
                                ratio = max_h / f.shape[0]
                                new_w = int(f.shape[1] * ratio)
                                f = cv2.resize(f, (new_w, max_h))
                            resized.append(f)
                        concat_frame = np.hstack(resized)
                    
                    # 在画面上显示激活摄像头数
                    status_text = f"Active: {active_cameras}/{num_cameras}"
                    cv2.putText(concat_frame, status_text, (concat_frame.shape[1] - 200, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('Multi Camera Detection', concat_frame)
                    
                    # 显示鸟瞰图
                    if show_bev and len(all_ground_coords) > 0:
                        bird_view = create_bird_eye_view(all_ground_coords, all_class_ids)
                        cv2.imshow('Bird Eye View', bird_view)
                    elif show_bev and do_detection:
                        bird_view = create_bird_eye_view([], [])
                        cv2.imshow('Bird Eye View', bird_view)
                
                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n用户请求退出...")
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    if len(display_frames) > 0:
                        cv2.imwrite(f'capture_multi_{timestamp}.jpg', concat_frame)
                        print(f"已保存截图: capture_multi_{timestamp}.jpg")
                elif key == ord('d'):
                    self.detect_every_n_frames = 1 if self.detect_every_n_frames > 1 else 3
                    print(f"跳帧模式: 每{self.detect_every_n_frames}帧检测一次")
                elif key == ord('b'):
                    show_bev = not show_bev
                    if not show_bev:
                        cv2.destroyWindow('Bird Eye View')
                    print(f"BEV显示: {'开启' if show_bev else '关闭'}")
        
        except KeyboardInterrupt:
            print("\n检测到中断信号...")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("\n正在清理资源...")
        self.is_running = False
        
        # 停止所有摄像头
        self.camera_manager.stop_all()
        
        # 关闭所有窗口
        cv2.destroyAllWindows()
        
        print("系统已关闭")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='多摄像头目标检测与测距系统')
    parser.add_argument('--config', type=str,
                       default='config/dual_camera_config_backup_20251110_145700.json',
                       help='配置文件路径')
    parser.add_argument('--cameras', type=str, default=None,
                       help='指定使用的摄像头索引，用逗号分隔，如: 0,1 或 0,1,2,3')
    
    args = parser.parse_args()
    
    # 解析摄像头索引
    camera_indices = None
    if args.cameras:
        try:
            camera_indices = [int(x.strip()) for x in args.cameras.split(',')]
            print(f"指定使用摄像头索引: {camera_indices}")
        except:
            print("警告: 摄像头索引解析失败，将使用所有摄像头")
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        return
    
    print("=" * 60)
    print("多摄像头目标检测与测距系统")
    print("=" * 60)
    
    # 创建系统实例
    system = MultiCameraSystem(args.config, camera_indices)
    
    # 初始化系统
    if not system.initialize():
        print("系统初始化失败")
        return
    
    print("\n系统初始化成功!")
    
    # 运行系统
    system.run()


if __name__ == '__main__':
    main()