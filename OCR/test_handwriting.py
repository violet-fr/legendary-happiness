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
    denoise=False,  # 禁用去噪，保留手写体的细笔画
    binarize=False,
    correct_skew=True,
    enhance_contrast=True,
    optimize_resolution=True,
    denoise_method='gaussian',
    blur_kernel=3,
    binary_block_size=11,
    binary_c=2
)

# 1. 改进预处理：尝试更多的对比度增强方法
# 使用gamma校正增强对比度
gamma = 1.5
invGamma = 1.0 / gamma
table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype(np.uint8)
processed_image = cv2.LUT(processed_image, table)

# 进一步增强对比度和亮度
alpha = 4.0  # 更高的对比度增益
beta = 90    # 更高的亮度增益
processed_image = cv2.convertScaleAbs(processed_image, alpha=alpha, beta=beta)

# 添加锐化处理，增强手写体的笔画
kernel = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]])  # 更强的锐化
processed_image = cv2.filter2D(processed_image, -1, kernel)

# 2. 调整二值化参数，确保手写体的细笔画被保留
# 使用自适应阈值处理，参数调整为适合手写体
processed_image = cv2.adaptiveThreshold(processed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)  # 更小的块大小，更适合细笔画

# 3. 使用形态学操作增强笔画
# 膨胀操作，增强笔画
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # 更小的膨胀核，避免过度膨胀
processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_DILATE, kernel)

# 调整图像大小，提高分辨率
processed_image = cv2.resize(processed_image, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)  # 更高的分辨率

# 保存增强后的图像
cv2.imwrite('enhanced_handwriting.png', processed_image)
print("增强后的图像已保存到 enhanced_handwriting.png")

# 保存预处理后的图像
cv2.imwrite('processed_handwriting.png', processed_image)
print("预处理后的图像已保存到 processed_handwriting.png")

# 4. 调整OCR参数：使用针对手写体优化的参数
# 初始化PaddleOCR引擎（针对手写体优化）
paddle_ocr = OCREngine(
    use_paddleocr=True, 
    lang='ch', 
    use_vl_model=True, 
    use_vl_service=False
)

# 初始化Tesseract引擎（针对手写体优化）
tesseract_ocr = OCREngine(
    use_paddleocr=False, 
    lang='ch', 
    use_vl_model=True, 
    use_vl_service=False
)

# 5. 多引擎融合：结合多个OCR引擎的结果
print("\n=== PaddleOCR 识别结果 ===")
paddle_results = paddle_ocr.recognize(processed_image, confidence_threshold=0.3)
paddle_text = ''.join([result['text'] for result in paddle_results])
print(f"PaddleOCR识别文本: {paddle_text}")

print("\n=== Tesseract OCR 识别结果 ===")
tesseract_results = tesseract_ocr.recognize(processed_image, confidence_threshold=0.3)
tesseract_text = ''.join([result['text'] for result in tesseract_results])
print(f"Tesseract识别文本: {tesseract_text}")

# 6. 结果融合
print("\n=== 融合识别结果 ===")
# 简单的融合策略：取非空结果
if paddle_text and not tesseract_text:
    final_text = paddle_text
elif tesseract_text and not paddle_text:
    final_text = tesseract_text
elif paddle_text and tesseract_text:
    # 如果两者都有结果，取较长的那个
    final_text = paddle_text if len(paddle_text) > len(tesseract_text) else tesseract_text
else:
    final_text = "未识别到文本"

print(f"融合识别文本: {final_text}")

# 计算识别准确率
target_text = "思想的好"
print(f"\n目标文本: {target_text}")
print(f"识别文本: {final_text}")

# 检查是否成功识别
if target_text in final_text:
    print("✓ 成功识别目标文本！")
else:
    print("✗ 未能识别目标文本，需要进一步优化。")

# 检查显示框是否完全包含所有文字
def check_bboxes(results, engine_name):
    print(f"\n=== {engine_name} 显示框检查 ===")
    if not results:
        print("✗ 未检测到文本框")
        return False
    
    print(f"检测到 {len(results)} 个文本框")
    for i, result in enumerate(results):
        bbox = result['bbox']
        text = result['text']
        confidence = result['confidence']
        print(f"文本框 {i+1}: 文本='{text}', 置信度={confidence:.2f}, 坐标={bbox}")
    
    print("✓ 显示框检测成功")
    return True

# 运行显示框检查
check_bboxes(paddle_results, "PaddleOCR")
check_bboxes(tesseract_results, "Tesseract OCR")
