"""
单摄像头目标检测与测距系统 - 主程序
"""
import cv2
import numpy as np
import sys
import os
import time
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config_loader import ConfigLoader
from camera_stream import CameraStream
from detector import ObjectDetector
from monocular_distance import MonocularDistance
from coordinate_transform import CoordinateTransform
from utils import FPSCounter, ResultLogger, draw_info_panel, create_bird_eye_view


class SingleCameraSystem:
    """单摄像头检测系统"""
    
    def __init__(self, config_path: str, camera_index: int = 0):
        """
        初始化系统
        
        参数:
            config_path: 配置文件路径
            camera_index: 使用配置中的第几个摄像头（0或1）
        """
        self.config_path = config_path
        self.camera_index = camera_index
        
        # 加载配置
        print("正在加载配置...")
        self.config_loader = ConfigLoader(config_path)
        self.config_loader.load_config()
        self.cameras_config = self.config_loader.parse_cameras()
        self.detection_config = self.config_loader.get_detection_config()
        
        # 选择要使用的摄像头
        if camera_index >= len(self.cameras_config):
            raise ValueError(f"摄像头索引 {camera_index} 超出范围，配置中只有 {len(self.cameras_config)} 个摄像头")
        
        self.camera_config = self.cameras_config[camera_index]
        print(f"使用摄像头: {self.camera_config['name']} (ID: {self.camera_config['camera_id']})")
        
        # 初始化组件
        self.camera_stream = None
        self.detector = None
        self.distance_calculator = None
        self.coord_transform = None
        self.fps_counter = FPSCounter()
        self.result_logger = ResultLogger()
        
        # 性能优化参数
        self.detect_every_n_frames = 2  # 每N帧检测一次
        self.frame_counter = 0
        self.last_detection = {
            'detections': [],
            'distances': [],
            'ground_coords': []
        }
        self.display_width = 800  # 单路可以显示大一点
        
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
            if 'yolov8s' in model_path:
                print("提示：在笔记本上运行可能较慢，建议使用yolov8n.pt以提升速度")
            
            self.detector = ObjectDetector(
                model_path=model_path,
                confidence_threshold=self.detection_config['confidence_threshold']
            )
            if not self.detector.load_model():
                return False
            
            # 初始化摄像头
            print("\n正在初始化摄像头...")
            self.camera_stream = CameraStream(
                camera_id=self.camera_config['camera_id'],
                rtsp_url=self.camera_config['rtsp_url'],
                camera_matrix=self.camera_config['camera_matrix'],
                dist_coeffs=self.camera_config['dist_coeffs'],
                undistort_alpha=self.camera_config['undistort_alpha'],
                reconnect_interval=3,  # 3秒重连间隔
                max_queue_size=2  # 小队列降低延迟
            )
            
            # 创建测距器
            self.distance_calculator = MonocularDistance(
                camera_matrix=self.camera_config['camera_matrix'],
                camera_height=self.camera_config['height_mm'],
                tilt_angle=self.camera_config['tilt_angle']
            )
            
            # 创建坐标转换器
            self.coord_transform = CoordinateTransform(
                camera_height=self.camera_config['height_mm'],
                tilt_angle=self.camera_config['tilt_angle'],
                pole_offset=self.camera_config['pole_offset_mm']
            )
            
            print(f"摄像头初始化完成")
            print(f"  - 相机高度: {self.camera_config['height_mm']}mm")
            print(f"  - 俯仰角: {self.camera_config['tilt_angle']}度")
            print(f"  - RTSP地址: {self.camera_config['rtsp_url']}")
            
            return True
            
        except Exception as e:
            print(f"初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_frame(self, frame: np.ndarray, do_detection: bool = True):
        """
        处理单帧图像
        
        参数:
            frame: 图像帧
            do_detection: 是否执行检测
            
        返回:
            (处理后的帧, 检测结果, 距离, 地面坐标)
        """
        # 如果不执行检测，使用缓存的结果
        if not do_detection:
            detections = self.last_detection['detections']
            distances = self.last_detection['distances']
            ground_coords = self.last_detection['ground_coords']
        else:
            # 目标检测
            detections = self.detector.detect(frame)
            
            if len(detections) == 0:
                self.last_detection = {
                    'detections': [],
                    'distances': [],
                    'ground_coords': []
                }
                return frame, [], [], []
            
            # 计算距离
            distances = self.distance_calculator.calculate_distances_for_detections(detections)
            
            # 转换到地面坐标系
            ground_coords = self.coord_transform.batch_transform_detections(
                detections, distances, self.camera_config['camera_matrix']
            )
            
            # 缓存检测结果
            self.last_detection = {
                'detections': detections,
                'distances': distances,
                'ground_coords': ground_coords
            }
        
        # 绘制检测结果
        result_frame = self.detector.draw_detections(frame, detections, distances, ground_coords)
        
        return result_frame, detections, distances, ground_coords
    
    def run(self):
        """运行系统主循环"""
        # 连接并启动摄像头
        print("\n正在连接摄像头...")
        if not self.camera_stream.connect():
            print("摄像头连接失败")
            return
        
        self.camera_stream.start()
        time.sleep(2)  # 等待摄像头稳定
        
        self.is_running = True
        print("\n系统运行中...")
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
                
                # 读取帧
                ret, frame = self.camera_stream.read()
                
                if not ret or frame is None:
                    # 显示等待重连画面
                    wait_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(wait_frame, "Waiting for camera...", (150, 240),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imshow('Single Camera Detection', wait_frame)
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('q'):
                        break
                    continue
                
                # 处理帧
                result_frame, detections, distances, ground_coords = \
                    self.process_frame(frame, do_detection)
                
                # 统一调整画面大小
                h, w = result_frame.shape[:2]
                target_height = int(h * self.display_width / w)
                result_frame = cv2.resize(result_frame, (self.display_width, target_height))
                
                # 更新FPS
                fps = self.fps_counter.update()
                
                # 绘制信息面板
                result_frame = draw_info_panel(
                    result_frame, 
                    self.camera_config['camera_id'], 
                    fps, 
                    len(detections)
                )
                
                # 添加距离调试信息
                if len(detections) > 0 and len(distances) > 0:
                    debug_text = f"First: {distances[0]:.2f}m"
                    cv2.putText(result_frame, debug_text, (10, result_frame.shape[0] - 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # 记录结果（仅在实际检测时）
                if do_detection and len(detections) > 0:
                    self.result_logger.log_detection(
                        self.camera_config['camera_id'], 
                        time.time(), 
                        detections, 
                        distances, 
                        ground_coords
                    )
                
                # 显示主窗口
                cv2.imshow('Single Camera Detection', result_frame)
                
                # 显示鸟瞰图
                if show_bev and len(ground_coords) > 0:
                    class_ids = [d['class_id'] for d in detections]
                    bird_view = create_bird_eye_view(ground_coords, class_ids)
                    cv2.imshow('Bird Eye View', bird_view)
                elif show_bev and do_detection:
                    # 显示空的鸟瞰图
                    bird_view = create_bird_eye_view([], [])
                    cv2.imshow('Bird Eye View', bird_view)
                
                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n用户请求退出...")
                    break
                elif key == ord('s'):
                    # 保存截图
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f'capture_cam{self.camera_config["camera_id"]}_{timestamp}.jpg'
                    cv2.imwrite(filename, result_frame)
                    print(f"已保存截图: {filename}")
                elif key == ord('d'):
                    # 切换跳帧模式
                    self.detect_every_n_frames = 1 if self.detect_every_n_frames > 1 else 3
                    print(f"跳帧模式: 每{self.detect_every_n_frames}帧检测一次")
                elif key == ord('b'):
                    # 切换BEV显示
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
        
        # 停止摄像头
        if self.camera_stream:
            self.camera_stream.stop()
        
        # 关闭所有窗口
        cv2.destroyAllWindows()
        
        # 保存日志
        # self.result_logger.save_to_file()
        
        print("系统已关闭")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='单摄像头目标检测与测距系统')
    parser.add_argument('--config', type=str, 
                       default='config/dual_camera_config_backup_20251110_145700.json',
                       help='配置文件路径')
    parser.add_argument('--camera', type=int, default=0,
                       help='使用第几个摄像头 (0或1)')
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        return
    
    print("=" * 60)
    print("单摄像头目标检测与测距系统")
    print("=" * 60)
    
    # 创建系统实例
    system = SingleCameraSystem(args.config, args.camera)
    
    # 初始化系统
    if not system.initialize():
        print("系统初始化失败")
        return
    
    print("\n系统初始化成功!")
    
    # 运行系统
    system.run()


if __name__ == '__main__':
    main()