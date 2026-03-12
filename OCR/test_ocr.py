import cv2
import numpy as np
from ocr_engine import OCREngine
from preprocessor import ImagePreprocessor

# 读取测试图像
image = cv2.imread('test2.jpg')
print(f"测试图像形状: {image.shape}")

# 初始化图像预处理器
preprocessor = ImagePreprocessor()

# 预处理图像 - 针对手写体进行优化
processed_image = preprocessor.process(
    image,
    denoise=True,  # 启用轻度去噪，减少噪声干扰
    binarize=False,  # 暂时禁用二值化，先增强对比度
    correct_skew=True,
    enhance_contrast=True,
    optimize_resolution=True,
    denoise_method='gaussian',
    blur_kernel=3,
    binary_block_size=11,
    binary_c=2
)

# 进一步增强对比度和亮度
alpha = 4.0  # 更高的对比度增益，使手写体更明显
beta = 100   # 更高的亮度增益，使手写体更清晰
processed_image = cv2.convertScaleAbs(processed_image, alpha=alpha, beta=beta)

# 添加适度的锐化处理，增强手写体的笔画
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # 适度的锐化
processed_image = cv2.filter2D(processed_image, -1, kernel)

# 应用高斯模糊，减少噪声
processed_image = cv2.GaussianBlur(processed_image, (3, 3), 0)

# 应用全局阈值处理，使手写体更清晰
_, processed_image = cv2.threshold(processed_image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 调整图像大小，提高分辨率
processed_image = cv2.resize(processed_image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)  # 适度的分辨率

# 保存增强后的图像
cv2.imwrite('enhanced_test.png', processed_image)
print("增强后的图像已保存到 enhanced_test.png")

# 保存预处理后的图像
cv2.imwrite('processed_test.png', processed_image)
print("预处理后的图像已保存到 processed_test.png")

# 初始化OCR引擎（使用中文，优先使用PaddleOCR，尝试手写体识别）
ocr_engine = OCREngine(use_paddleocr=True, lang='ch', use_vl_model=True, use_vl_service=False)

# 执行OCR识别
results = ocr_engine.recognize(processed_image, confidence_threshold=0.3)

# 打印识别结果
print("识别结果:")
for result in results:
    print(f"文本: {result['text']}")
    print(f"置信度: {result['confidence']}")
    print(f"坐标: {result['bbox']}")
    print()

# 绘制检测框
bbox_image = ocr_engine.draw_bboxes(image, results)

# 保存结果图像
cv2.imwrite('result_test.jpg', bbox_image)
print("结果图像已保存到 result_test.jpg")

# 计算识别准确率
target_text = "思想的好"
recognized_text = ''.join([result['text'] for result in results])  # 不添加空格
print(f"目标文本: {target_text}")
print(f"识别文本: {recognized_text}")

# 检查是否成功识别
if target_text in recognized_text:
    print("✓ 成功识别目标文本！")
else:
    print("✗ 未能识别目标文本，需要进一步优化。")

# 检查显示框是否完全包含所有文字
def check_bboxes():
    # 读取结果图像
    result_image = cv2.imread('result_test.jpg')
    if result_image is None:
        print("✗ 无法读取result_test.jpg图像")
        return False
    
    # 转换为灰度图
    gray = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
    
    # 二值化
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 统计文字区域和显示框数量
    text_areas = 0
    bbox_areas = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # 过滤小面积
            text_areas += 1
    
    # 检查结果图像中是否有绿色显示框
    # 绿色的HSV范围
    hsv = cv2.cvtColor(result_image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 查找绿色显示框的轮廓
    bbox_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in bbox_contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # 过滤小面积
            bbox_areas += 1
    
    print(f"检测到的文字区域数量: {text_areas}")
    print(f"检测到的显示框数量: {bbox_areas}")
    
    # 检查显示框是否完全包含所有文字
    if text_areas > 0 and bbox_areas > 0:
        print("✓ 显示框检测成功")
        return True
    else:
        print("✗ 显示框检测失败，可能未完全包含文字")
        return False

# 运行显示框检查
check_bboxes()
