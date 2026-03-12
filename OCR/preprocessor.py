import cv2
import numpy as np
from PIL import Image

class ImagePreprocessor:
    """
    图像预处理类，用于在OCR识别前对图像进行增强处理
    """
    
    def __init__(self):
        """
        初始化图像预处理类
        """
        pass
    
    def process(self, image, denoise=True, binarize=True, correct_skew=True, enhance_contrast=True, optimize_resolution=True, denoise_method='gaussian', blur_kernel=3, binary_block_size=11, binary_c=2):
        """
        执行完整的图像预处理流程
        
        Args:
            image: 输入图像（numpy array或PIL Image）
            denoise: 是否执行去噪处理
            binarize: 是否执行二值化处理
            correct_skew: 是否执行倾斜校正
            enhance_contrast: 是否执行对比度增强
            optimize_resolution: 是否执行分辨率优化
            denoise_method: 去噪方法，'gaussian'或'median'
            blur_kernel: 模糊核大小，奇数
            binary_block_size: 二值化块大小，奇数
            binary_c: 二值化常数
            
        Returns:
            处理后的图像（numpy array）
        """
        try:
            # 转换为numpy array
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 分辨率优化
            if optimize_resolution:
                gray = self._optimize_resolution(gray)
            
            # 对比度增强
            if enhance_contrast:
                gray = self._enhance_contrast(gray)
            
            # 去噪处理
            if denoise:
                gray = self._denoise(gray, method=denoise_method, kernel=blur_kernel)
            
            # 二值化处理
            if binarize:
                gray = self._binarize(gray, block_size=binary_block_size, c=binary_c)
            
            # 倾斜校正
            if correct_skew:
                gray = self._correct_skew(gray)
            
            return gray
        except Exception as e:
            print(f"预处理失败: {str(e)}")
            return image
    
    def _denoise(self, image, method='gaussian', kernel=3):
        """
        去噪处理
        
        Args:
            image: 灰度图像
            method: 去噪方法，'gaussian'或'median'
            kernel: 模糊核大小，奇数
            
        Returns:
            去噪后的图像
        """
        if method == 'gaussian':
            return cv2.GaussianBlur(image, (kernel, kernel), 0)
        elif method == 'median':
            return cv2.medianBlur(image, kernel)
        return image
    
    def _binarize(self, image, block_size=11, c=2):
        """
        二值化处理
        
        使用自适应阈值二值化，适用于光照不均的文档
        
        Args:
            image: 灰度图像
            block_size: 二值化块大小，奇数
            c: 二值化常数
            
        Returns:
            二值化后的图像
        """
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, c
        )
    
    def _correct_skew(self, image):
        """
        倾斜校正
        
        使用霍夫变换检测文本行角度，然后旋转图像
        
        Args:
            image: 二值化图像
            
        Returns:
            校正后的图像
        """
        # 检测边缘
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # 霍夫变换检测直线
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is not None:
            angles = []
            for line in lines:
                rho, theta = line[0]
                angle = theta * 180 / np.pi - 90
                if abs(angle) < 45:  # 只考虑小角度倾斜
                    angles.append(angle)
            
            if angles:
                # 计算平均角度
                mean_angle = np.mean(angles)
                
                # 获取图像尺寸
                h, w = image.shape
                center = (w // 2, h // 2)
                
                # 旋转矩阵
                M = cv2.getRotationMatrix2D(center, mean_angle, 1.0)
                
                # 执行旋转
                rotated = cv2.warpAffine(
                    image, M, (w, h), flags=cv2.INTER_CUBIC, 
                    borderMode=cv2.BORDER_REPLICATE
                )
                
                return rotated
        
        return image
    
    def _enhance_contrast(self, image):
        """
        对比度增强
        
        使用直方图均衡化提升文字清晰度
        
        Args:
            image: 灰度图像
            
        Returns:
            增强对比度后的图像
        """
        return cv2.equalizeHist(image)
    
    def _optimize_resolution(self, image):
        """
        分辨率优化
        
        如果图片宽度小于800px，自动进行无损放大
        
        Args:
            image: 输入图像
            
        Returns:
            优化分辨率后的图像
        """
        h, w = image.shape
        if w < 800:
            scale_factor = 800 / w
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        return image