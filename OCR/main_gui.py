import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QTextEdit, QSlider, QComboBox,
    QCheckBox, QGroupBox, QGridLayout, QMessageBox, QSplitter
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from preprocessor import ImagePreprocessor
from ocr_engine import OCREngine
from postprocessor import PostProcessor

class OCRGUI(QMainWindow):
    """
    OCR系统的PyQt5 GUI界面
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能文档OCR识别系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化模块
        self.preprocessor = ImagePreprocessor()
        self.ocr_engine = OCREngine()
        self.postprocessor = PostProcessor()
        
        # 存储变量
        self.image = None
        self.processed_image = None
        self.ocr_results = []
        self.recognized_text = ""
        
        # 创建主布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # 创建顶部文件上传区域
        self.create_file_upload_section()
        
        # 创建中间处理和结果显示区域
        self.create_processing_section()
        
        # 创建底部导出区域
        self.create_export_section()
        
    def create_file_upload_section(self):
        """
        创建文件上传区域
        """
        upload_layout = QHBoxLayout()
        
        self.upload_button = QPushButton("上传图片")
        self.upload_button.clicked.connect(self.upload_image)
        
        self.file_label = QLabel("未选择文件")
        
        upload_layout.addWidget(self.upload_button)
        upload_layout.addWidget(self.file_label, 1)
        
        self.main_layout.addLayout(upload_layout)
    
    def create_processing_section(self):
        """
        创建处理和结果显示区域
        """
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧图像显示区域
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # 原始图像
        self.original_image_label = QLabel("原始图像")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setFixedHeight(300)
        
        # 处理后图像
        self.processed_image_label = QLabel("处理后图像")
        self.processed_image_label.setAlignment(Qt.AlignCenter)
        self.processed_image_label.setFixedHeight(300)
        
        left_layout.addWidget(self.original_image_label)
        left_layout.addWidget(self.processed_image_label)
        
        # 右侧文本显示区域
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        self.text_label = QLabel("识别结果")
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(False)
        
        right_layout.addWidget(self.text_label)
        right_layout.addWidget(self.text_edit)
        
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([600, 600])
        
        # 处理参数设置
        params_group = QGroupBox("处理参数")
        params_layout = QGridLayout()
        
        # 语言选择
        params_layout.addWidget(QLabel("语言选择:"), 0, 0)
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["中文", "英文", "日文"])
        params_layout.addWidget(self.lang_combo, 0, 1)
        
        # 置信度阈值
        params_layout.addWidget(QLabel("置信度阈值:"), 1, 0)
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(60)
        self.confidence_label = QLabel("0.6")
        self.confidence_slider.valueChanged.connect(lambda val: self.confidence_label.setText(f"{val/100:.2f}"))
        params_layout.addWidget(self.confidence_slider, 1, 1)
        params_layout.addWidget(self.confidence_label, 1, 2)
        
        # 预处理选项
        params_layout.addWidget(QLabel("预处理选项:"), 2, 0)
        
        self.denoise_check = QCheckBox("去噪")
        self.denoise_check.setChecked(True)
        params_layout.addWidget(self.denoise_check, 2, 1)
        
        self.binarize_check = QCheckBox("二值化")
        self.binarize_check.setChecked(True)
        params_layout.addWidget(self.binarize_check, 2, 2)
        
        self.correct_skew_check = QCheckBox("倾斜校正")
        self.correct_skew_check.setChecked(True)
        params_layout.addWidget(self.correct_skew_check, 3, 1)
        
        self.enhance_contrast_check = QCheckBox("对比度增强")
        self.enhance_contrast_check.setChecked(True)
        params_layout.addWidget(self.enhance_contrast_check, 3, 2)
        
        self.optimize_resolution_check = QCheckBox("分辨率优化")
        self.optimize_resolution_check.setChecked(True)
        params_layout.addWidget(self.optimize_resolution_check, 4, 1)
        
        # 处理按钮
        self.process_button = QPushButton("开始处理")
        self.process_button.clicked.connect(self.process_image)
        params_layout.addWidget(self.process_button, 5, 0, 1, 3)
        
        params_group.setLayout(params_layout)
        
        self.main_layout.addWidget(splitter)
        self.main_layout.addWidget(params_group)
    
    def create_export_section(self):
        """
        创建导出区域
        """
        export_layout = QHBoxLayout()
        
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["txt", "docx", "xlsx"])
        
        self.export_button = QPushButton("导出文件")
        self.export_button.clicked.connect(self.export_result)
        
        export_layout.addWidget(QLabel("导出格式:"))
        export_layout.addWidget(self.export_format_combo)
        export_layout.addWidget(self.export_button)
        export_layout.addStretch(1)
        
        self.main_layout.addLayout(export_layout)
    
    def upload_image(self):
        """
        上传图片文件
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片文件", "", "Image Files (*.jpg *.jpeg *.png)", options=options
        )
        
        if file_path:
            self.file_label.setText(file_path)
            # 读取图片
            self.image = cv2.imread(file_path)
            # 显示原始图像
            self.display_image(self.image, self.original_image_label)
    
    def display_image(self, image, label):
        """
        在标签中显示图像
        """
        if image is not None:
            # 转换颜色空间
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 调整大小以适应标签
            h, w, _ = image.shape
            label_width = label.width()
            label_height = label.height()
            
            if w > label_width or h > label_height:
                scale = min(label_width / w, label_height / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = cv2.resize(image, (new_w, new_h))
            
            # 创建QImage
            h, w, _ = image.shape
            q_image = QImage(image.data, w, h, 3 * w, QImage.Format_RGB888)
            
            # 创建QPixmap并显示
            pixmap = QPixmap.fromImage(q_image)
            label.setPixmap(pixmap)
    
    def process_image(self):
        """
        处理图像并执行OCR识别
        """
        if self.image is None:
            QMessageBox.warning(self, "警告", "请先上传图片")
            return
        
        try:
            # 获取参数
            lang = self.lang_combo.currentText()
            # PaddleOCR 支持的语言代码，日文应使用 ja
            lang_map = {"中文": "ch", "英文": "en", "日文": "ja"}
            
            confidence_threshold = self.confidence_slider.value() / 100
            
            # 更新OCR引擎语言
            self.ocr_engine = OCREngine(lang=lang_map[lang])
            
            # 执行预处理
            self.processed_image = self.preprocessor.process(
                self.image,
                denoise=self.denoise_check.isChecked(),
                binarize=self.binarize_check.isChecked(),
                correct_skew=self.correct_skew_check.isChecked(),
                enhance_contrast=self.enhance_contrast_check.isChecked(),
                optimize_resolution=self.optimize_resolution_check.isChecked()
            )
            
            # 显示处理后图像
            self.display_image(self.processed_image, self.processed_image_label)
            
            # 执行OCR识别
            self.ocr_results = self.ocr_engine.recognize(self.processed_image, confidence_threshold)
            
            # 后处理
            self.recognized_text = self.postprocessor.process(self.ocr_results)
            
            # 显示识别结果
            self.text_edit.setPlainText(self.recognized_text)
            
            # 显示检测框
            bbox_image = self.ocr_engine.draw_bboxes(self.image, self.ocr_results)
            self.display_image(bbox_image, self.original_image_label)
            
            QMessageBox.information(self, "成功", "OCR识别完成")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理失败: {str(e)}")
    
    def export_result(self):
        """
        导出识别结果
        """
        if not self.recognized_text:
            QMessageBox.warning(self, "警告", "没有识别结果可导出")
            return
        
        # 获取导出格式
        format = self.export_format_combo.currentText()
        
        # 选择保存路径
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存文件", f"ocr_result.{format}", f"{format.upper()} Files (*.{format})", options=options
        )
        
        if file_path:
            # 执行导出
            success = self.postprocessor.export(self.text_edit.toPlainText(), file_path, format=format)
            
            if success:
                QMessageBox.information(self, "成功", f"文件已导出到: {file_path}")
            else:
                QMessageBox.critical(self, "错误", "导出失败")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OCRGUI()
    window.show()
    sys.exit(app.exec_())
