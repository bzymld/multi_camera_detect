"""
摄像头流处理模块
负责多线程实时拉取RTSP视频流
"""
import cv2
import threading
import queue
import time
from typing import Optional, Tuple
import numpy as np


class CameraStream:
    """摄像头流处理类"""
    
    def __init__(self, camera_id: int, rtsp_url: str, camera_matrix: np.ndarray, 
                 dist_coeffs: np.ndarray, undistort_alpha: float = 0.0,
                 reconnect_interval: int = 3, max_queue_size: int = 2):
        """
        初始化摄像头流
        
        参数:
            camera_id: 摄像头ID
            rtsp_url: RTSP流地址
            camera_matrix: 相机内参矩阵
            dist_coeffs: 畸变系数
            undistort_alpha: 去畸变参数
            reconnect_interval: 断连重连间隔（秒）
            max_queue_size: 队列最大长度（减小以降低延迟）
        """
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.undistort_alpha = undistort_alpha
        self.reconnect_interval = reconnect_interval
        
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=max_queue_size)  # 减小队列降低延迟
        self.is_running = False
        self.thread = None
        
        # 断连统计
        self.reconnect_count = 0
        self.last_reconnect_time = 0
        self.consecutive_failures = 0
        
        # 获取视频尺寸用于去畸变映射
        self.newcameramtx = None
        self.roi = None
        self.mapx = None
        self.mapy = None
        
    def connect(self) -> bool:
        """
        连接到RTSP流
        
        返回:
            连接是否成功
        """
        try:
            # 释放旧连接
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            # 使用优化的参数连接RTSP
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            
            # 设置缓冲区大小（减少延迟）
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # 设置超时（毫秒）
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            
            if not self.cap.isOpened():
                print(f"[摄像头 {self.camera_id}] 连接失败: {self.rtsp_url}")
                return False
            
            # 读取一帧以获取图像尺寸
            ret, frame = self.cap.read()
            if ret:
                h, w = frame.shape[:2]
                # 计算新的相机矩阵和去畸变映射
                self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(
                    self.camera_matrix, self.dist_coeffs, (w, h), self.undistort_alpha, (w, h)
                )
                self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                    self.camera_matrix, self.dist_coeffs, None, 
                    self.newcameramtx, (w, h), cv2.CV_32FC1
                )
                print(f"[摄像头 {self.camera_id}] 连接成功 ({w}x{h})")
                self.consecutive_failures = 0  # 重置失败计数
                return True
            else:
                print(f"[摄像头 {self.camera_id}] 无法读取帧")
                return False
                
        except Exception as e:
            print(f"[摄像头 {self.camera_id}] 连接异常: {e}")
            self.consecutive_failures += 1
            return False
    
    def start(self):
        """启动视频流读取线程"""
        if self.is_running:
            print(f"摄像头 {self.camera_id} 已经在运行")
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._read_stream, daemon=True)
        self.thread.start()
        print(f"摄像头 {self.camera_id} 流处理线程已启动")
    
    def _read_stream(self):
        """读取视频流的线程函数"""
        frame_skip_counter = 0
        
        while self.is_running:
            try:
                # 检查是否需要重连
                if self.cap is None or not self.cap.isOpened():
                    current_time = time.time()
                    if current_time - self.last_reconnect_time >= self.reconnect_interval:
                        print(f"[摄像头 {self.camera_id}] 连接断开，尝试重连...")
                        self.last_reconnect_time = current_time
                        self.reconnect_count += 1
                        
                        if self.connect():
                            print(f"[摄像头 {self.camera_id}] 重连成功 (第{self.reconnect_count}次)")
                        else:
                            time.sleep(1)
                            continue
                    else:
                        time.sleep(0.5)
                        continue
                
                # 读取帧
                ret, frame = self.cap.read()
                
                if not ret:
                    print(f"[摄像头 {self.camera_id}] 读取帧失败，连接可能断开")
                    self.consecutive_failures += 1
                    
                    # 连续失败多次后强制重连
                    if self.consecutive_failures >= 5:
                        print(f"[摄像头 {self.camera_id}] 连续失败{self.consecutive_failures}次，强制重连")
                        if self.cap is not None:
                            self.cap.release()
                            self.cap = None
                        self.consecutive_failures = 0
                    
                    time.sleep(0.1)
                    continue
                
                # 重置失败计数
                self.consecutive_failures = 0
                
                # 跳帧优化（每2帧取1帧，降低延迟）
                frame_skip_counter += 1
                if frame_skip_counter % 2 != 0:
                    continue
                
                # 检测是否进行去畸变
                if self.mapx is None or self.mapy is None:
                    print(f"[摄像头 {self.camera_id}] 去畸变映射未正确初始化")
                # 去畸变
                if self.mapx is not None and self.mapy is not None:
                    frame = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)
                
                # 清空队列保持最新帧（降低延迟）
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
                
                # 放入新帧
                try:
                    self.frame_queue.put(frame, timeout=0.1)
                except queue.Full:
                    pass  # 队列满时丢弃帧
                
            except Exception as e:
                print(f"[摄像头 {self.camera_id}] 读取流异常: {e}")
                time.sleep(1)
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        读取一帧图像
        
        返回:
            (是否成功, 图像帧)
        """
        try:
            frame = self.frame_queue.get(timeout=1.0)
            return True, frame
        except queue.Empty:
            return False, None
    
    def stop(self):
        """停止视频流读取"""
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
        print(f"摄像头 {self.camera_id} 流处理已停止")
    
    def is_connected(self) -> bool:
        """
        检查是否已连接
        
        返回:
            是否已连接
        """
        return self.cap is not None and self.cap.isOpened()


class MultiCameraManager:
    """多摄像头管理器"""
    
    def __init__(self):
        """初始化多摄像头管理器"""
        self.cameras = {}
    
    def add_camera(self, camera_id: int, camera_stream: CameraStream):
        """
        添加摄像头
        
        参数:
            camera_id: 摄像头ID
            camera_stream: 摄像头流对象
        """
        self.cameras[camera_id] = camera_stream
    
    def start_all(self):
        """启动所有摄像头"""
        for camera_id, camera in self.cameras.items():
            if camera.connect():
                camera.start()
    
    def stop_all(self):
        """停止所有摄像头"""
        for camera in self.cameras.values():
            camera.stop()
    
    def get_camera(self, camera_id: int) -> Optional[CameraStream]:
        """
        获取指定摄像头
        
        参数:
            camera_id: 摄像头ID
            
        返回:
            摄像头流对象
        """
        return self.cameras.get(camera_id)
    
    def read_all(self) -> dict:
        """
        读取所有摄像头的当前帧
        
        返回:
            {camera_id: (ret, frame)}
        """
        frames = {}
        for camera_id, camera in self.cameras.items():
            ret, frame = camera.read()
            frames[camera_id] = (ret, frame)
        return frames