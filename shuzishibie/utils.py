"""工具模块

提供绘制检测框、生成训练损失曲线、计算并绘制混淆矩阵等辅助功能。
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def draw_bboxes(image: np.ndarray, 
                bboxes: List[Dict], 
                class_names: Optional[List[str]] = None) -> np.ndarray:
    """
    绘制检测框
    
    Args:
        image: 输入图片
        bboxes: 检测框列表，每个元素包含 'bbox', 'confidence', 'class_id', 'class_name'
        class_names: 类别名称列表
        
    Returns:
        绘制了检测框的图片
    """
    # 复制图片以避免修改原始图片
    result = image.copy()
    
    # 颜色映射
    colors = [
        (0, 0, 255),    # 红色
        (0, 255, 0),    # 绿色
        (255, 0, 0),    # 蓝色
        (0, 255, 255),  # 黄色
        (255, 0, 255),  # 洋红色
        (255, 255, 0),  # 青色
        (128, 0, 0),    # 深红色
        (0, 128, 0),    # 深绿色
        (0, 0, 128),    # 深蓝色
        (128, 128, 0)   # 深黄色
    ]
    
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox['bbox']
        confidence = bbox['confidence']
        class_id = bbox['class_id']
        class_name = bbox.get('class_name', f'Class {class_id}')
        
        # 选择颜色
        color = colors[class_id % len(colors)]
        
        # 绘制边界框
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(result, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return result

def plot_training_curves(results_file: str, save_path: Optional[str] = None) -> None:
    """
    绘制训练曲线
    
    Args:
        results_file: 训练结果CSV文件路径
        save_path: 保存路径，默认 None
    """
    import pandas as pd
    
    # 读取数据
    df = pd.read_csv(results_file)
    
    # 创建画布
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
    
    if save_path:
        save_dir = Path(save_path).parent
        save_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path)
        print(f"训练曲线已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_confusion_matrix(y_true: List[int], 
                         y_pred: List[int], 
                         class_names: List[str], 
                         save_path: Optional[str] = None) -> None:
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        save_path: 保存路径，默认 None
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 归一化
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 创建画布
    plt.figure(figsize=(10, 8))
    
    # 绘制混淆矩阵
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    if save_path:
        save_dir = Path(save_path).parent
        save_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path)
        print(f"混淆矩阵已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()

def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    """
    计算评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        
    Returns:
        评估指标字典
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    
    return metrics

def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    调整图片大小
    
    Args:
        image: 输入图片
        target_size: 目标大小 (width, height)
        
    Returns:
        调整大小后的图片
    """
    return cv2.resize(image, target_size)

def save_image(image: np.ndarray, save_path: str) -> None:
    """
    保存图片
    
    Args:
        image: 输入图片
        save_path: 保存路径
    """
    save_dir = Path(save_path).parent
    save_dir.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(save_path, image)
    print(f"图片已保存到: {save_path}")

def load_image(image_path: str) -> np.ndarray:
    """
    加载图片
    
    Args:
        image_path: 图片路径
        
    Returns:
        加载的图片
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    # 转换为 RGB 格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_file_list(directory: str, extensions: List[str] = None) -> List[str]:
    """
    获取目录下的文件列表
    
    Args:
        directory: 目录路径
        extensions: 文件扩展名列表，默认 None
        
    Returns:
        文件路径列表
    """
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            if extensions:
                if any(file.lower().endswith(ext) for ext in extensions):
                    file_list.append(os.path.join(root, file))
            else:
                file_list.append(os.path.join(root, file))
    return file_list

def create_directory(directory: str) -> None:
    """
    创建目录
    
    Args:
        directory: 目录路径
    """
    Path(directory).mkdir(exist_ok=True, parents=True)

def print_metrics(metrics: Dict) -> None:
    """
    打印评估指标
    
    Args:
        metrics: 评估指标字典
    """
    print("评估指标:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


# 导入必要的模块
import os


def main():
    """主函数"""
    # 示例：绘制检测框
    # image = cv2.imread('test.jpg')
    # bboxes = [
    #     {'bbox': [100, 100, 200, 200], 'confidence': 0.95, 'class_id': 0, 'class_name': 'face'},
    #     {'bbox': [300, 150, 400, 250], 'confidence': 0.90, 'class_id': 1, 'class_name': 'license_plate'}
    # ]
    # result = draw_bboxes(image, bboxes)
    # save_image(result, 'output.jpg')
    
    # 示例：绘制训练曲线
    # plot_training_curves('runs/train/exp/results.csv', 'training_curves.png')
    
    # 示例：绘制混淆矩阵
    # y_true = [0, 1, 0, 1, 0]
    # y_pred = [0, 1, 1, 1, 0]
    # class_names = ['face', 'license_plate']
    # plot_confusion_matrix(y_true, y_pred, class_names, 'confusion_matrix.png')


if __name__ == '__main__':
    main()