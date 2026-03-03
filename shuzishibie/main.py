"""主入口模块

提供人脸检测、车牌识别、自定义数据集训练的完整运行示例，
包含命令行参数解析和示例代码。
"""

import argparse
import os
from pathlib import Path
from data_loader import DataLoader
from trainer import Trainer
from detector import Detector
from utils import create_directory


def parse_args():
    """
    解析命令行参数
    
    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(description='YOLOv8 多场景目标检测')
    
    # 基本参数
    parser.add_argument('--task', type=str, choices=['face', 'license', 'train', 'detect'], 
                      default='detect', help='任务类型')
    parser.add_argument('--source', type=str, default='', help='输入源（图片/视频/摄像头）')
    parser.add_argument('--model', type=str, default='', help='模型路径')
    
    # 训练参数
    parser.add_argument('--data', type=str, default='', help='数据配置文件路径')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批量大小')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='学习率')
    parser.add_argument('--img-size', type=int, default=640, help='输入图片尺寸')
    
    # 推理参数
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU阈值')
    parser.add_argument('--save', action='store_true', default=True, help='保存结果')
    parser.add_argument('--show', action='store_true', default=False, help='显示结果')
    
    return parser.parse_args()

def face_detection(args):
    """
    人脸检测
    
    Args:
        args: 命令行参数
    """
    print("人脸检测模式")
    
    # 默认模型
    if not args.model:
        args.model = 'yolov8n-face.pt'  # 人脸检测预训练模型
    
    # 确保模型存在
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"模型文件不存在: {args.model}")
        print("正在使用默认的 YOLOv8n 模型进行人脸检测...")
        args.model = 'yolov8n.pt'
    
    # 初始化检测器
    detector = Detector(args.model)
    
    # 处理不同输入源
    if not args.source:
        # 使用摄像头
        print("使用摄像头进行人脸检测...")
        detector.detect_webcam(cam_id=0, 
                              conf_threshold=args.conf, 
                              iou_threshold=args.iou, 
                              save=args.save, 
                              show=args.show)
    else:
        # 检查输入源类型
        source_path = Path(args.source)
        if source_path.is_file():
            # 检查文件扩展名
            ext = source_path.suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                # 图片检测
                print(f"检测图片: {args.source}")
                detector.detect_image(args.source, 
                                     conf_threshold=args.conf, 
                                     iou_threshold=args.iou, 
                                     save=args.save, 
                                     show=args.show)
            elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
                # 视频检测
                print(f"检测视频: {args.source}")
                detector.detect_video(args.source, 
                                     conf_threshold=args.conf, 
                                     iou_threshold=args.iou, 
                                     save=args.save, 
                                     show=args.show)
            else:
                print(f"不支持的文件类型: {ext}")
        else:
            print(f"输入源不存在: {args.source}")

def license_plate_detection(args):
    """
    车牌识别
    
    Args:
        args: 命令行参数
    """
    print("车牌识别模式")
    
    # 默认模型
    if not args.model:
        args.model = 'yolov8n.pt'  # 通用目标检测模型
    
    # 初始化检测器
    detector = Detector(args.model)
    
    # 处理不同输入源
    if not args.source:
        # 使用摄像头
        print("使用摄像头进行车牌识别...")
        detector.detect_webcam(cam_id=0, 
                              conf_threshold=args.conf, 
                              iou_threshold=args.iou, 
                              save=args.save, 
                              show=args.show)
    else:
        # 检查输入源类型
        source_path = Path(args.source)
        if source_path.is_file():
            # 检查文件扩展名
            ext = source_path.suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                # 图片检测
                print(f"检测图片: {args.source}")
                detector.detect_image(args.source, 
                                     conf_threshold=args.conf, 
                                     iou_threshold=args.iou, 
                                     save=args.save, 
                                     show=args.show)
            elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
                # 视频检测
                print(f"检测视频: {args.source}")
                detector.detect_video(args.source, 
                                     conf_threshold=args.conf, 
                                     iou_threshold=args.iou, 
                                     save=args.save, 
                                     show=args.show)
            else:
                print(f"不支持的文件类型: {ext}")
        else:
            print(f"输入源不存在: {args.source}")

def custom_training(args):
    """
    自定义数据集训练
    
    Args:
        args: 命令行参数
    """
    print("自定义数据集训练模式")
    
    # 检查数据配置文件
    if not args.data:
        print("错误: 请指定数据配置文件路径")
        return
    
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"数据配置文件不存在: {args.data}")
        return
    
    # 初始化训练器
    trainer = Trainer()
    
    # 开始训练
    print(f"开始训练，数据配置: {args.data}")
    print(f"训练轮数: {args.epochs}")
    print(f"批量大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"输入图片尺寸: {args.img_size}")
    
    # 训练
    try:
        result = trainer.train(
            data=args.data,
            model=args.model if args.model else 'yolov8n.pt',
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            img_size=args.img_size,
            name='custom_training'
        )
        print("训练完成！")
        if result.get('best_model_path'):
            print(f"最佳模型保存路径: {result['best_model_path']}")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")

def general_detection(args):
    """
    通用目标检测
    
    Args:
        args: 命令行参数
    """
    print("通用目标检测模式")
    
    # 默认模型
    if not args.model:
        args.model = 'yolov8n.pt'  # 默认使用 YOLOv8n
    
    # 初始化检测器
    detector = Detector(args.model)
    
    # 处理不同输入源
    if not args.source:
        # 使用摄像头
        print("使用摄像头进行目标检测...")
        detector.detect_webcam(cam_id=0, 
                              conf_threshold=args.conf, 
                              iou_threshold=args.iou, 
                              save=args.save, 
                              show=args.show)
    else:
        # 检查输入源类型
        source_path = Path(args.source)
        if source_path.is_file():
            # 检查文件扩展名
            ext = source_path.suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                # 图片检测
                print(f"检测图片: {args.source}")
                detector.detect_image(args.source, 
                                     conf_threshold=args.conf, 
                                     iou_threshold=args.iou, 
                                     save=args.save, 
                                     show=args.show)
            elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
                # 视频检测
                print(f"检测视频: {args.source}")
                detector.detect_video(args.source, 
                                     conf_threshold=args.conf, 
                                     iou_threshold=args.iou, 
                                     save=args.save, 
                                     show=args.show)
            else:
                print(f"不支持的文件类型: {ext}")
        else:
            print(f"输入源不存在: {args.source}")

def main():
    """
    主函数
    """
    # 解析参数
    args = parse_args()
    
    # 创建必要的目录
    create_directory('models')
    create_directory('datasets')
    create_directory('runs/train')
    create_directory('runs/detect')
    
    # 根据任务类型执行不同的操作
    if args.task == 'face':
        face_detection(args)
    elif args.task == 'license':
        license_plate_detection(args)
    elif args.task == 'train':
        custom_training(args)
    elif args.task == 'detect':
        general_detection(args)
    else:
        print(f"未知任务类型: {args.task}")


if __name__ == '__main__':
    main()