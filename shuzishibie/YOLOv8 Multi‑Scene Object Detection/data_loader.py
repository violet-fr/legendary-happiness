"""数据加载模块

支持VOC/COCO格式数据集加载，提供人脸/车牌公开数据集下载指引，
自定义数据集标注说明，以及数据预处理和标注格式转换功能。
"""

import os
import yaml
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class DataLoader:
    """数据加载器类"""
    
    def __init__(self):
        """初始化数据加载器"""
        self.datasets_dir = Path('datasets')
        self.datasets_dir.mkdir(exist_ok=True)
    
    def load_dataset(self, data_config: str) -> Dict:
        """
        加载数据集配置
        
        Args:
            data_config: 数据配置文件路径或配置字典
            
        Returns:
            数据集配置字典
        """
        if isinstance(data_config, str):
            with open(data_config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            config = data_config
        
        return config
    
    def prepare_face_dataset(self, dataset_name: str = 'wider_face') -> Path:
        """
        准备人脸数据集
        
        Args:
            dataset_name: 数据集名称，支持 'wider_face' 或 'coco'
            
        Returns:
            数据集路径
        """
        dataset_path = self.datasets_dir / dataset_name
        dataset_path.mkdir(exist_ok=True)
        
        if dataset_name == 'wider_face':
            print("WIDER FACE 数据集下载指引:")
            print("1. 访问 http://shuoyang1213.me/WIDERFACE/")
            print("2. 下载 WIDER_train.zip, WIDER_val.zip, WIDER_test.zip")
            print("3. 下载 face_annotations.zip")
            print("4. 将所有文件解压到", dataset_path)
        elif dataset_name == 'coco':
            print("COCO 数据集下载指引:")
            print("1. 访问 https://cocodataset.org/")
            print("2. 下载 2017 Train/Val 数据集")
            print("3. 解压到", dataset_path)
        
        return dataset_path
    
    def prepare_license_plate_dataset(self, dataset_name: str = 'ccpd') -> Path:
        """
        准备车牌数据集
        
        Args:
            dataset_name: 数据集名称，支持 'ccpd' 或 'openalpr'
            
        Returns:
            数据集路径
        """
        dataset_path = self.datasets_dir / dataset_name
        dataset_path.mkdir(exist_ok=True)
        
        if dataset_name == 'ccpd':
            print("CCPD 数据集下载指引:")
            print("1. 访问 https://github.com/detectRecog/CCPD")
            print("2. 按照README中的链接下载数据集")
            print("3. 解压到", dataset_path)
        elif dataset_name == 'openalpr':
            print("OpenALPR 数据集下载指引:")
            print("1. 访问 https://github.com/openalpr/openalpr")
            print("2. 按照README中的链接下载数据集")
            print("3. 解压到", dataset_path)
        
        return dataset_path
    
    def convert_voc_to_yolo(self, voc_dir: str, output_dir: str) -> None:
        """
        将VOC格式转换为YOLO格式
        
        Args:
            voc_dir: VOC数据集目录
            output_dir: YOLO格式输出目录
        """
        import xml.etree.ElementTree as ET
        
        voc_dir = Path(voc_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 类别映射
        classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
                   'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
                   'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
                   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
                   'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
                   'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
                   'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
                   'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
                   'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
                   'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 
                   'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
                   'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
                   'hair drier', 'toothbrush']
        
        class_to_id = {cls: i for i, cls in enumerate(classes)}
        
        # 处理标注文件
        ann_dir = voc_dir / 'Annotations'
        img_dir = voc_dir / 'JPEGImages'
        
        for ann_file in ann_dir.glob('*.xml'):
            tree = ET.parse(ann_file)
            root = tree.getroot()
            
            img_width = int(root.find('size/width').text)
            img_height = int(root.find('size/height').text)
            
            output_file = output_dir / f"{ann_file.stem}.txt"
            with open(output_file, 'w') as f:
                for obj in root.findall('object'):
                    cls = obj.find('name').text
                    if cls not in class_to_id:
                        continue
                    
                    cls_id = class_to_id[cls]
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    
                    # 转换为YOLO格式 (x_center, y_center, width, height) 归一化
                    x_center = (xmin + xmax) / 2 / img_width
                    y_center = (ymin + ymax) / 2 / img_height
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height
                    
                    f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")
        
        print(f"VOC格式转换为YOLO格式完成，输出目录: {output_dir}")
    
    def create_data_yaml(self, dataset_path: str, classes: List[str], output_path: str) -> None:
        """
        创建YOLO训练所需的data.yaml文件
        
        Args:
            dataset_path: 数据集路径
            classes: 类别列表
            output_path: 输出yaml文件路径
        """
        dataset_path = Path(dataset_path)
        
        data = {
            'train': str(dataset_path / 'train' / 'images'),
            'val': str(dataset_path / 'val' / 'images'),
            'test': str(dataset_path / 'test' / 'images') if (dataset_path / 'test' / 'images').exists() else None,
            'nc': len(classes),
            'names': classes
        }
        
        # 移除None值
        data = {k: v for k, v in data.items() if v is not None}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        
        print(f"data.yaml文件创建完成: {output_path}")
    
    def download_dataset(self, url: str, save_path: str) -> Path:
        """
        下载数据集
        
        Args:
            url: 数据集下载链接
            save_path: 保存路径
            
        Returns:
            下载后的文件路径
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        
        print(f"开始下载: {url}")
        print(f"保存到: {save_path}")
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress = (downloaded / total_size) * 100
                    print(f"下载进度: {progress:.2f}%", end='\r')
        
        print("\n下载完成!")
        return save_path
    
    def extract_dataset(self, archive_path: str, extract_path: str) -> Path:
        """
        解压数据集
        
        Args:
            archive_path: 压缩文件路径
            extract_path: 解压路径
            
        Returns:
            解压后的目录路径
        """
        archive_path = Path(archive_path)
        extract_path = Path(extract_path)
        extract_path.mkdir(exist_ok=True, parents=True)
        
        print(f"开始解压: {archive_path}")
        print(f"解压到: {extract_path}")
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        elif archive_path.suffix in ['.tar', '.tar.gz', '.tgz']:
            with tarfile.open(archive_path, 'r') as tar_ref:
                tar_ref.extractall(extract_path)
        
        print("解压完成!")
        return extract_path


def main():
    """主函数"""
    data_loader = DataLoader()
    
    # 示例：准备人脸数据集
    face_path = data_loader.prepare_face_dataset('wider_face')
    print(f"人脸数据集路径: {face_path}")
    
    # 示例：准备车牌数据集
    license_path = data_loader.prepare_license_plate_dataset('ccpd')
    print(f"车牌数据集路径: {license_path}")
    
    # 示例：创建data.yaml
    classes = ['face']
    data_loader.create_data_yaml('datasets/wider_face', classes, 'datasets/wider_face/data.yaml')


if __name__ == '__main__':
    main()