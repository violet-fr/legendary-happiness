"""训练模块

封装 YOLOv8 训练函数，支持自定义 epoch、batch_size、学习率，
保存最佳模型，训练过程可视化（损失曲线），支持 CPU/GPU 自动切换。
"""

import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List, Any
from ultralytics import YOLO


class Trainer:
    """训练器类"""
    
    def __init__(self):
        """初始化训练器"""
        self.device = self._get_device()
        self.models_dir = Path('models')
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
    
    def _get_device(self) -> str:
        """
        获取可用设备
        
        Returns:
            设备字符串，如 'cuda' 或 'cpu'
        """
        if torch.cuda.is_available():
            print("CUDA 可用，使用 GPU 训练")
            return 'cuda'
        else:
            print("CUDA 不可用，使用 CPU 训练")
            return 'cpu'
    
    def train(self, 
              data: str, 
              model: str = 'yolov8n.pt', 
              epochs: int = 100, 
              batch_size: int = 16, 
              learning_rate: float = 0.01, 
              img_size: int = 640, 
              project: str = 'runs/train', 
              name: str = 'exp', 
              save_best: bool = True, 
              patience: int = 10, 
              **kwargs) -> Dict:
        """
        训练 YOLOv8 模型
        
        Args:
            data: 数据配置文件路径
            model: 模型路径或名称，默认 'yolov8n.pt'
            epochs: 训练轮数，默认 100
            batch_size: 批量大小，默认 16
            learning_rate: 学习率，默认 0.01
            img_size: 输入图片尺寸，默认 640
            project: 项目目录，默认 'runs/train'
            name: 实验名称，默认 'exp'
            save_best: 是否保存最佳模型，默认 True
            patience: 早停 patience，默认 10
            **kwargs: 其他训练参数
            
        Returns:
            训练结果字典
        """
        try:
            # 加载模型
            yolo_model = YOLO(model)
            
            # 训练
            results = yolo_model.train(
                data=data,
                epochs=epochs,
                batch=batch_size,
                lr0=learning_rate,
                imgsz=img_size,
                project=project,
                name=name,
                save_dir=str(self.results_dir / name),
                save=True,
                save_period=1,
                patience=patience,
                device=self.device,
                **kwargs
            )
            
            # 保存最佳模型到 models 目录
            if save_best:
                best_model_path = Path(project) / name / 'weights' / 'best.pt'
                if best_model_path.exists():
                    import shutil
                    dest_path = self.models_dir / f"best_{name}.pt"
                    shutil.copy2(best_model_path, dest_path)
                    print(f"最佳模型已保存到: {dest_path}")
            
            # 生成训练曲线
            self._generate_training_plots(Path(project) / name)
            
            return {
                'model': model,
                'data': data,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'results': results,
                'best_model_path': str(dest_path) if save_best and best_model_path.exists() else None
            }
            
        except Exception as e:
            print(f"训练过程中出现错误: {e}")
            raise
    
    def _generate_training_plots(self, exp_dir: Path) -> None:
        """
        生成训练曲线
        
        Args:
            exp_dir: 实验目录
        """
        # 读取训练日志
        results_file = exp_dir / 'results.csv'
        if not results_file.exists():
            print(f"训练日志文件不存在: {results_file}")
            return
        
        # 读取数据
        df = pd.read_csv(results_file)
        
        # 创建保存目录
        plots_dir = exp_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # 损失曲线
        plt.figure(figsize=(12, 8))
        
        # 训练损失
        if 'train/box_loss' in df.columns:
            plt.subplot(2, 2, 1)
            plt.plot(df['epoch'], df['train/box_loss'], label='box_loss')
            plt.plot(df['epoch'], df['train/cls_loss'], label='cls_loss')
            if 'train/dfl_loss' in df.columns:
                plt.plot(df['epoch'], df['train/dfl_loss'], label='dfl_loss')
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
        
        # 验证损失
        if 'val/box_loss' in df.columns:
            plt.subplot(2, 2, 2)
            plt.plot(df['epoch'], df['val/box_loss'], label='box_loss')
            plt.plot(df['epoch'], df['val/cls_loss'], label='cls_loss')
            if 'val/dfl_loss' in df.columns:
                plt.plot(df['epoch'], df['val/dfl_loss'], label='dfl_loss')
            plt.title('Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
        
        # mAP
        if 'metrics/mAP50-95(B)' in df.columns:
            plt.subplot(2, 2, 3)
            plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50')
            plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95')
            plt.title('mAP')
            plt.xlabel('Epoch')
            plt.ylabel('mAP')
            plt.legend()
        
        # PR 曲线
        if 'metrics/precision(B)' in df.columns:
            plt.subplot(2, 2, 4)
            plt.plot(df['epoch'], df['metrics/precision(B)'], label='precision')
            plt.plot(df['epoch'], df['metrics/recall(B)'], label='recall')
            plt.title('Precision-Recall')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'training_curves.png')
        plt.close()
        
        print(f"训练曲线已保存到: {plots_dir / 'training_curves.png'}")
    
    def resume_training(self, 
                       model_path: str, 
                       data: str, 
                       epochs: int = 100, 
                       **kwargs) -> Dict:
        """
        恢复训练
        
        Args:
            model_path: 模型路径
            data: 数据配置文件路径
            epochs: 训练轮数，默认 100
            **kwargs: 其他训练参数
            
        Returns:
            训练结果字典
        """
        try:
            # 加载模型
            yolo_model = YOLO(model_path)
            
            # 恢复训练
            results = yolo_model.train(
                data=data,
                epochs=epochs,
                resume=True,
                device=self.device,
                **kwargs
            )
            
            return {
                'model': model_path,
                'data': data,
                'epochs': epochs,
                'results': results
            }
            
        except Exception as e:
            print(f"恢复训练过程中出现错误: {e}")
            raise
    
    def validate(self, 
                 model_path: str, 
                 data: str, 
                 img_size: int = 640, 
                 **kwargs) -> Dict:
        """
        验证模型
        
        Args:
            model_path: 模型路径
            data: 数据配置文件路径
            img_size: 输入图片尺寸，默认 640
            **kwargs: 其他验证参数
            
        Returns:
            验证结果字典
        """
        try:
            # 加载模型
            yolo_model = YOLO(model_path)
            
            # 验证
            results = yolo_model.val(
                data=data,
                imgsz=img_size,
                device=self.device,
                **kwargs
            )
            
            return {
                'model': model_path,
                'data': data,
                'results': results
            }
            
        except Exception as e:
            print(f"验证过程中出现错误: {e}")
            raise
    
    def export_model(self, 
                     model_path: str, 
                     format: str = 'onnx', 
                     **kwargs) -> str:
        """
        导出模型
        
        Args:
            model_path: 模型路径
            format: 导出格式，默认 'onnx'
            **kwargs: 其他导出参数
            
        Returns:
            导出模型路径
        """
        try:
            # 加载模型
            yolo_model = YOLO(model_path)
            
            # 导出
            export_path = yolo_model.export(
                format=format,
                device=self.device,
                **kwargs
            )
            
            print(f"模型已导出到: {export_path}")
            return export_path
            
        except Exception as e:
            print(f"导出过程中出现错误: {e}")
            raise


def main():
    """主函数"""
    trainer = Trainer()
    
    # 示例：训练人脸检测模型
    print("训练人脸检测模型...")
    # trainer.train(
    #     data='datasets/wider_face/data.yaml',
    #     model='yolov8n.pt',
    #     epochs=50,
    #     batch_size=8,
    #     learning_rate=0.001,
    #     name='face_detection'
    # )
    
    # 示例：训练车牌识别模型
    print("训练车牌识别模型...")
    # trainer.train(
    #     data='datasets/ccpd/data.yaml',
    #     model='yolov8n.pt',
    #     epochs=50,
    #     batch_size=8,
    #     learning_rate=0.001,
    #     name='license_plate_detection'
    # )


if __name__ == '__main__':
    main()