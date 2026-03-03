"""推理模块

支持图片/视频/摄像头实时检测，输出检测框、置信度，保存检测结果，
支持 CPU/GPU 自动切换，包含异常处理。
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from ultralytics import YOLO


class Detector:
    """检测器类"""
    
    def __init__(self, model_path: str):
        """
        初始化检测器
        
        Args:
            model_path: 模型路径
        """
        self.device = self._get_device()
        self.model = self._load_model(model_path)
        self.output_dir = Path('runs/detect')
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def _get_device(self) -> str:
        """
        获取可用设备
        
        Returns:
            设备字符串，如 'cuda' 或 'cpu'
        """
        if torch.cuda.is_available():
            print("CUDA 可用，使用 GPU 推理")
            return 'cuda'
        else:
            print("CUDA 不可用，使用 CPU 推理")
            return 'cpu'
    
    def _load_model(self, model_path: str) -> YOLO:
        """
        加载模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            YOLO 模型实例
        """
        try:
            model = YOLO(model_path)
            model.to(self.device)
            print(f"模型加载成功: {model_path}")
            return model
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def detect_image(self, 
                    image_path: str, 
                    conf_threshold: float = 0.25, 
                    iou_threshold: float = 0.45, 
                    save: bool = True, 
                    show: bool = False) -> Dict:
        """
        检测图片
        
        Args:
            image_path: 图片路径
            conf_threshold: 置信度阈值，默认 0.25
            iou_threshold: IOU 阈值，默认 0.45
            save: 是否保存结果，默认 True
            show: 是否显示结果，默认 False
            
        Returns:
            检测结果字典
        """
        try:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"无法读取图片: {image_path}")
            
            # 检测
            results = self.model(image, 
                               conf=conf_threshold, 
                               iou=iou_threshold, 
                               device=self.device)
            
            # 处理结果
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy().item()
                    cls = int(box.cls[0].cpu().numpy().item())
                    label = result.names[cls]
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_id': cls,
                        'class_name': label
                    })
            
            # 保存结果
            output_path = None
            if save:
                output_path = self.output_dir / f"detect_{Path(image_path).name}"
                results[0].save(str(output_path))
                print(f"检测结果已保存到: {output_path}")
            
            # 显示结果
            if show:
                annotated_image = results[0].plot()
                cv2.imshow('Detection Result', annotated_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            return {
                'image_path': image_path,
                'detections': detections,
                'output_path': str(output_path) if save else None
            }
            
        except Exception as e:
            print(f"图片检测过程中出现错误: {e}")
            raise
    
    def detect_video(self, 
                    video_path: str, 
                    conf_threshold: float = 0.25, 
                    iou_threshold: float = 0.45, 
                    save: bool = True, 
                    show: bool = False) -> Dict:
        """
        检测视频
        
        Args:
            video_path: 视频路径
            conf_threshold: 置信度阈值，默认 0.25
            iou_threshold: IOU 阈值，默认 0.45
            save: 是否保存结果，默认 True
            show: 是否显示结果，默认 False
            
        Returns:
            检测结果字典
        """
        try:
            # 打开视频
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise FileNotFoundError(f"无法打开视频: {video_path}")
            
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 准备输出视频
            output_path = None
            out = None
            if save:
                output_path = self.output_dir / f"detect_{Path(video_path).name}"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            frame_count = 0
            detections = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 检测
                result = self.model(frame, 
                                   conf=conf_threshold, 
                                   iou=iou_threshold, 
                                   device=self.device)[0]
                
                # 处理结果
                frame_detections = []
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy().item()
                    cls = int(box.cls[0].cpu().numpy().item())
                    label = result.names[cls]
                    
                    frame_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_id': cls,
                        'class_name': label
                    })
                
                detections.append(frame_detections)
                
                # 获取标注后的帧
                annotated_frame = result.plot()
                
                # 保存结果
                if save and out is not None:
                    out.write(annotated_frame)
                
                # 显示结果
                if show:
                    cv2.imshow('Detection Result', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
                print(f"处理帧: {frame_count}/{total_frames}", end='\r')
            
            # 释放资源
            cap.release()
            if save and out is not None:
                out.release()
                print(f"\n检测结果已保存到: {output_path}")
            cv2.destroyAllWindows()
            
            return {
                'video_path': video_path,
                'detections': detections,
                'output_path': str(output_path) if save else None,
                'frame_count': frame_count
            }
            
        except Exception as e:
            print(f"视频检测过程中出现错误: {e}")
            raise
    
    def detect_webcam(self, 
                     cam_id: int = 0, 
                     conf_threshold: float = 0.25, 
                     iou_threshold: float = 0.45, 
                     save: bool = False, 
                     show: bool = True) -> Dict:
        """
        检测摄像头
        
        Args:
            cam_id: 摄像头ID，默认 0
            conf_threshold: 置信度阈值，默认 0.25
            iou_threshold: IOU 阈值，默认 0.45
            save: 是否保存结果，默认 False
            show: 是否显示结果，默认 True
            
        Returns:
            检测结果字典
        """
        try:
            # 打开摄像头
            cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened():
                raise Exception(f"无法打开摄像头: {cam_id}")
            
            # 获取摄像头信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 准备输出视频
            output_path = None
            out = None
            if save:
                output_path = self.output_dir / f"detect_webcam.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            frame_count = 0
            detections = []
            
            print("按 'q' 退出摄像头检测")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 检测
                result = self.model(frame, 
                                   conf=conf_threshold, 
                                   iou=iou_threshold, 
                                   device=self.device)[0]
                
                # 处理结果
                frame_detections = []
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy().item()
                    cls = int(box.cls[0].cpu().numpy().item())
                    label = result.names[cls]
                    
                    frame_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_id': cls,
                        'class_name': label
                    })
                
                detections.append(frame_detections)
                
                # 获取标注后的帧
                annotated_frame = result.plot()
                
                # 保存结果
                if save and out is not None:
                    out.write(annotated_frame)
                
                # 显示结果
                if show:
                    cv2.imshow('Webcam Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
            
            # 释放资源
            cap.release()
            if save and out is not None:
                out.release()
                print(f"检测结果已保存到: {output_path}")
            cv2.destroyAllWindows()
            
            return {
                'cam_id': cam_id,
                'detections': detections,
                'output_path': str(output_path) if save else None,
                'frame_count': frame_count
            }
            
        except Exception as e:
            print(f"摄像头检测过程中出现错误: {e}")
            raise
    
    def batch_detect(self, 
                    image_paths: List[str], 
                    conf_threshold: float = 0.25, 
                    iou_threshold: float = 0.45, 
                    save: bool = True) -> List[Dict]:
        """
        批量检测图片
        
        Args:
            image_paths: 图片路径列表
            conf_threshold: 置信度阈值，默认 0.25
            iou_threshold: IOU 阈值，默认 0.45
            save: 是否保存结果，默认 True
            
        Returns:
            检测结果列表
        """
        results = []
        for image_path in image_paths:
            result = self.detect_image(
                image_path=image_path,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                save=save,
                show=False
            )
            results.append(result)
        return results


def main():
    """主函数"""
    # 示例：加载模型
    # detector = Detector('models/best_face_detection.pt')
    
    # 示例：检测图片
    # result = detector.detect_image('test.jpg', save=True, show=True)
    # print(f"检测到 {len(result['detections'])} 个目标")
    
    # 示例：检测视频
    # result = detector.detect_video('test.mp4', save=True, show=True)
    # print(f"处理了 {result['frame_count']} 帧")
    
    # 示例：检测摄像头
    # result = detector.detect_webcam(cam_id=0, save=False, show=True)
    # print(f"处理了 {result['frame_count']} 帧")


if __name__ == '__main__':
    main()