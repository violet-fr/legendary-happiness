import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

# 支持高DPI
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

from preprocessor import ImagePreprocessor
from ocr_engine import OCREngine
from postprocessor import PostProcessor

class OCRGUI(tk.Tk):
    """
    OCR系统的Tkinter GUI界面
    """
    
    def __init__(self):
        super().__init__()
        self.title("智能文档OCR识别系统")
        # 调整初始窗口大小，确保能够完全显示
        self.geometry("1000x700")
        # 允许窗口大小调整
        self.resizable(True, True)
        # 添加最大化按钮
        self.state('normal')
        
        # 初始化模块
        self.preprocessor = ImagePreprocessor()
        self.ocr_engine = OCREngine(use_vl_model=True, use_vl_service=False)
        self.postprocessor = PostProcessor()
        
        # 存储变量
        self.image = None
        self.processed_image = None
        self.ocr_results = []
        self.recognized_text = ""
        
        # 创建主布局
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
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
        upload_frame = tk.Frame(self.main_frame)
        upload_frame.pack(fill=tk.X, pady=5)
        
        self.upload_button = tk.Button(upload_frame, text="上传图片", command=self.upload_image)
        self.upload_button.pack(side=tk.LEFT, padx=5)
        
        self.file_label = tk.Label(upload_frame, text="未选择文件", anchor=tk.W)
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    def create_processing_section(self):
        """
        创建处理和结果显示区域
        """
        # 创建主分割器
        main_paned = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 左侧图像显示区域
        left_frame = tk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # 图像显示区域布局
        image_grid = tk.Frame(left_frame)
        image_grid.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 原始图像
        original_frame = tk.LabelFrame(image_grid, text="原始图像")
        original_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=5, pady=5)
        
        self.original_image_label = tk.Label(original_frame, text="请上传图片")
        self.original_image_label.pack(fill=tk.BOTH, expand=True)
        
        # 处理后图像
        processed_frame = tk.LabelFrame(image_grid, text="处理后图像")
        processed_frame.grid(row=1, column=0, sticky=tk.NSEW, padx=5, pady=5)
        
        self.processed_image_label = tk.Label(processed_frame, text="处理后图像将显示在这里")
        self.processed_image_label.pack(fill=tk.BOTH, expand=True)
        
        # 设置网格权重
        image_grid.grid_rowconfigure(0, weight=1)
        image_grid.grid_rowconfigure(1, weight=1)
        image_grid.grid_columnconfigure(0, weight=1)
        
        # 右侧文本显示区域
        right_frame = tk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)
        
        text_frame = tk.LabelFrame(right_frame, text="识别结果")
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 添加文本滚动条
        text_scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.text_edit = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=text_scroll.set)
        self.text_edit.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        text_scroll.config(command=self.text_edit.yview)
        
        # 处理参数设置
        params_frame = tk.LabelFrame(self.main_frame, text="处理参数")
        params_frame.pack(fill=tk.X, pady=5)
        
        # 创建参数网格
        params_grid = tk.Frame(params_frame)
        params_grid.pack(padx=10, pady=10)
        
        # 语言选择
        tk.Label(params_grid, text="语言选择:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.lang_var = tk.StringVar(value="中文")
        lang_options = ["中文", "英文", "日文"]
        lang_menu = ttk.Combobox(params_grid, textvariable=self.lang_var, values=lang_options, state="readonly")
        lang_menu.grid(row=0, column=1, padx=5, pady=5)
        
        # 置信度阈值
        tk.Label(params_grid, text="置信度阈值:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.confidence_var = tk.DoubleVar(value=0.6)
        confidence_scale = ttk.Scale(params_grid, from_=0, to=1, orient=tk.HORIZONTAL, variable=self.confidence_var)
        confidence_scale.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        self.confidence_label = tk.Label(params_grid, text="0.60")
        self.confidence_label.grid(row=1, column=2, padx=5, pady=5)
        confidence_scale.bind("<Motion>", lambda e: self.confidence_label.config(text=f"{self.confidence_var.get():.2f}"))
        
        # 预处理选项
        tk.Label(params_grid, text="预处理选项:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.denoise_var = tk.BooleanVar(value=True)
        tk.Checkbutton(params_grid, text="去噪", variable=self.denoise_var).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        self.binarize_var = tk.BooleanVar(value=True)
        tk.Checkbutton(params_grid, text="二值化", variable=self.binarize_var).grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        
        self.correct_skew_var = tk.BooleanVar(value=True)
        tk.Checkbutton(params_grid, text="倾斜校正", variable=self.correct_skew_var).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        self.enhance_contrast_var = tk.BooleanVar(value=True)
        tk.Checkbutton(params_grid, text="对比度增强", variable=self.enhance_contrast_var).grid(row=3, column=2, sticky=tk.W, padx=5, pady=5)
        
        self.optimize_resolution_var = tk.BooleanVar(value=True)
        tk.Checkbutton(params_grid, text="分辨率优化", variable=self.optimize_resolution_var).grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        
        # OCR引擎选项
        self.use_vl_service_var = tk.BooleanVar(value=False)
        tk.Checkbutton(params_grid, text="使用PaddleOCR-VL服务", variable=self.use_vl_service_var).grid(row=4, column=2, sticky=tk.W, padx=5, pady=5)
        
        # 高级预处理参数
        tk.Label(params_grid, text="去噪方法:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.denoise_method_var = tk.StringVar(value="gaussian")
        denoise_options = ["gaussian", "median"]
        denoise_menu = ttk.Combobox(params_grid, textvariable=self.denoise_method_var, values=denoise_options, state="readonly", width=10)
        denoise_menu.grid(row=5, column=1, padx=5, pady=5)
        
        tk.Label(params_grid, text="模糊核大小:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        self.blur_kernel_var = tk.IntVar(value=3)
        blur_kernel_scale = ttk.Scale(params_grid, from_=1, to=9, orient=tk.HORIZONTAL, variable=self.blur_kernel_var, command=lambda val: self.blur_kernel_label.config(text=str(int(float(val)) if int(float(val)) % 2 != 0 else int(float(val)) + 1)))
        blur_kernel_scale.grid(row=6, column=1, padx=5, pady=5, sticky=tk.EW)
        self.blur_kernel_label = tk.Label(params_grid, text="3")
        self.blur_kernel_label.grid(row=6, column=2, padx=5, pady=5)
        
        tk.Label(params_grid, text="二值化块大小:").grid(row=7, column=0, sticky=tk.W, padx=5, pady=5)
        self.binary_block_var = tk.IntVar(value=11)
        binary_block_scale = ttk.Scale(params_grid, from_=3, to=31, orient=tk.HORIZONTAL, variable=self.binary_block_var, command=lambda val: self.binary_block_label.config(text=str(int(float(val)) if int(float(val)) % 2 != 0 else int(float(val)) + 1)))
        binary_block_scale.grid(row=7, column=1, padx=5, pady=5, sticky=tk.EW)
        self.binary_block_label = tk.Label(params_grid, text="11")
        self.binary_block_label.grid(row=7, column=2, padx=5, pady=5)
        
        tk.Label(params_grid, text="二值化常数:").grid(row=8, column=0, sticky=tk.W, padx=5, pady=5)
        self.binary_c_var = tk.IntVar(value=2)
        binary_c_scale = ttk.Scale(params_grid, from_=0, to=10, orient=tk.HORIZONTAL, variable=self.binary_c_var, command=lambda val: self.binary_c_label.config(text=str(int(float(val)))))
        binary_c_scale.grid(row=8, column=1, padx=5, pady=5, sticky=tk.EW)
        self.binary_c_label = tk.Label(params_grid, text="2")
        self.binary_c_label.grid(row=8, column=2, padx=5, pady=5)
        
        # 处理按钮
        self.process_button = tk.Button(params_grid, text="开始处理", command=self.process_image, width=20)
        self.process_button.grid(row=9, column=0, columnspan=3, pady=10)
    
    def create_export_section(self):
        """
        创建导出区域
        """
        export_frame = tk.Frame(self.main_frame)
        export_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(export_frame, text="导出格式:").pack(side=tk.LEFT, padx=5)
        
        self.export_format_var = tk.StringVar(value="txt")
        export_options = ["txt", "docx", "xlsx"]
        export_menu = ttk.Combobox(export_frame, textvariable=self.export_format_var, values=export_options, state="readonly", width=10)
        export_menu.pack(side=tk.LEFT, padx=5)
        
        self.export_button = tk.Button(export_frame, text="导出文件", command=self.export_result)
        self.export_button.pack(side=tk.LEFT, padx=5)
    
    def upload_image(self):
        """
        上传图片文件
        """
        file_path = filedialog.askopenfilename(
            title="选择图片文件",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            self.file_label.config(text=file_path)
            # 读取图片（处理中文路径）
            try:
                # 使用PIL读取图片，再转换为numpy数组
                pil_image = Image.open(file_path)
                self.image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                # 显示原始图像
                self.display_image(self.image, self.original_image_label)
            except Exception as e:
                messagebox.showerror("错误", f"读取图片失败: {str(e)}")
                self.image = None
    
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
            
            # 固定图像显示尺寸，避免每次处理后放大
            fixed_width = 400
            fixed_height = 300
            
            # 调整大小以适应固定尺寸
            h, w, _ = image.shape
            if w > fixed_width or h > fixed_height:
                scale = min(fixed_width / w, fixed_height / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = cv2.resize(image, (new_w, new_h))
            
            # 创建PIL图像
            pil_image = Image.fromarray(image)
            # 创建PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # 保存引用，防止被垃圾回收
            label.image = photo
            # 显示图像
            label.config(image=photo, text="")
    
    def process_image(self):
        """
        处理图像并执行OCR识别
        """
        if self.image is None:
            messagebox.showwarning("警告", "请先上传图片")
            return
        
        try:
            # 获取参数
            lang = self.lang_var.get()
            # PaddleOCR 支持的语言代码，日文应使用 ja
            lang_map = {"中文": "ch", "英文": "en", "日文": "ja"}
            
            confidence_threshold = self.confidence_var.get()
            
            # 更新OCR引擎语言
            self.ocr_engine = OCREngine(lang=lang_map[lang], use_vl_model=True, use_vl_service=self.use_vl_service_var.get())
            
            # 执行预处理
            # 确保核大小为奇数
            blur_kernel = self.blur_kernel_var.get()
            if blur_kernel % 2 == 0:
                blur_kernel += 1
            
            binary_block_size = self.binary_block_var.get()
            if binary_block_size % 2 == 0:
                binary_block_size += 1
            
            self.processed_image = self.preprocessor.process(
                self.image,
                denoise=self.denoise_var.get(),
                binarize=self.binarize_var.get(),
                correct_skew=self.correct_skew_var.get(),
                enhance_contrast=self.enhance_contrast_var.get(),
                optimize_resolution=self.optimize_resolution_var.get(),
                denoise_method=self.denoise_method_var.get(),
                blur_kernel=blur_kernel,
                binary_block_size=binary_block_size,
                binary_c=self.binary_c_var.get()
            )
            
            # 显示处理后图像
            self.display_image(self.processed_image, self.processed_image_label)
            
            # 执行OCR识别
            self.ocr_results = self.ocr_engine.recognize(self.processed_image, confidence_threshold)
            
            # 后处理
            self.recognized_text = self.postprocessor.process(self.ocr_results)
            
            # 显示识别结果
            self.text_edit.delete(1.0, tk.END)
            self.text_edit.insert(tk.END, self.recognized_text)
            
            # 显示检测框
            bbox_image = self.ocr_engine.draw_bboxes(self.image, self.ocr_results)
            self.display_image(bbox_image, self.original_image_label)
            
            messagebox.showinfo("成功", "OCR识别完成")
            
        except Exception as e:
            messagebox.showerror("错误", f"处理失败: {str(e)}")
    
    def export_result(self):
        """
        导出识别结果
        """
        if not self.recognized_text:
            messagebox.showwarning("警告", "没有识别结果可导出")
            return
        
        # 获取导出格式
        format = self.export_format_var.get()
        
        # 选择保存路径
        file_path = filedialog.asksaveasfilename(
            title="保存文件",
            defaultextension=f".{format}",
            filetypes=[(f"{format.upper()} Files", f"*.{format}")]
        )
        
        if file_path:
            # 执行导出
            success = self.postprocessor.export(self.text_edit.get(1.0, tk.END), file_path, format=format)
            
            if success:
                messagebox.showinfo("成功", f"文件已导出到: {file_path}")
            else:
                messagebox.showerror("错误", "导出失败")

if __name__ == "__main__":
    app = OCRGUI()
    app.mainloop()
