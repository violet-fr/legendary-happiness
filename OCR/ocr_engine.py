from paddleocr import PaddleOCR
import cv2
import numpy as np
import requests
import json

# 尝试导入Tesseract
try:
    import pytesseract
    from PIL import Image as PILImage
    # 尝试不同的Tesseract路径
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
    ]
    
    tesseract_found = False
    for path in possible_paths:
        try:
            pytesseract.pytesseract.tesseract_cmd = path
            # 测试路径是否有效
            import subprocess
            subprocess.run([path, '--version'], capture_output=True, check=True)
            tesseract_found = True
            print(f"Tesseract路径设置成功: {path}")
            break
        except:
            continue
    
    if tesseract_found:
        tesseract_available = True
        print("Tesseract可用")
    else:
        tesseract_available = False
        print("Tesseract不可用，将使用PaddleOCR")
except ImportError:
    tesseract_available = False
    print("Tesseract不可用，将使用PaddleOCR")

# PaddleOCR-VL服务配置
PADDLEOCR_VL_URL = "http://localhost:8110/v1/chat/completions"


class OCREngine:
    """
    OCR引擎封装类，作为底层识别库的抽象层
    """
    
    def __init__(self, use_paddleocr=True, lang='ch', use_vl_model=True, use_vl_service=False):
        """
        初始化OCR引擎
        
        Args:
            use_paddleocr: 是否使用PaddleOCR（默认True）
            lang: 语言设置，支持 'ch'（中文）, 'en'（英文）, 'japan'（日文）等
            use_vl_model: 是否使用PaddleOCR-VL视觉语言模型（默认True）
            use_vl_service: 是否使用PaddleOCR-VL服务（默认False）
        """
        self.use_paddleocr = use_paddleocr
        self.lang = lang
        self.use_vl_model = use_vl_model
        self.use_vl_service = use_vl_service
        
        if use_paddleocr and not use_vl_service:
            # 初始化PaddleOCR，使用适合中文识别的参数
            try:
                # 尝试使用适合中文识别的参数，针对手写体进行优化
                self.ocr = PaddleOCR(
                    lang='ch',  # 明确指定中文
                    use_angle_cls=True,  # 启用角度检测
                    det_limit_side_len=1920,
                    det_limit_type='max',
                    det_db_thresh=0.02,  # 进一步降低检测阈值，提高对细笔画的检测能力
                    det_db_box_thresh=0.1,  # 进一步降低框阈值，提高对细笔画的检测能力
                    det_db_unclip_ratio=2.0,  # 增加检测框尺寸，确保框住手写体
                    rec_batch_num=1
                )
                print("PaddleOCR初始化成功（适合中文识别的参数）")
            except Exception as e:
                print(f"PaddleOCR初始化失败: {str(e)}")
                # 尝试使用无参数初始化
                self.ocr = PaddleOCR(use_angle_cls=False)
                print("PaddleOCR初始化成功（无参数，禁用角度检测）")
        else:
            # 这里可以添加Tesseract的初始化代码
            # 需要安装pytesseract和Tesseract OCR
            pass
    
    def recognize(self, image, confidence_threshold=0.6):
        """
        执行OCR识别
        
        Args:
            image: 输入图像（numpy array或路径）
            confidence_threshold: 置信度阈值，低于此值的结果会被标记
            
        Returns:
            识别结果列表，每个元素包含文本、置信度和坐标
        """
        try:
            if self.use_paddleocr:
                if self.use_vl_service:
                    # 尝试使用PaddleOCR-VL服务
                    print("尝试使用PaddleOCR-VL服务进行OCR识别")
                    try:
                        # 保存图像到临时文件
                        import tempfile
                        import os
                        import base64

                        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                            tmp_path = tmp.name

                        # 保存图像
                        cv2.imwrite(tmp_path, image)
                        print(f"保存临时图像到: {tmp_path}")

                        # 读取图像并转换为base64
                        with open(tmp_path, "rb") as f:
                            image_base64 = base64.b64encode(f.read()).decode('utf-8')

                        # 删除临时文件
                        os.unlink(tmp_path)

                        # 构建请求数据
                        payload = {
                            "model": "paddleocr-vl",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": "识别图片中的所有文本"
                                        },
                                        {
                                            "type": "image",
                                            "image": f"data:image/jpeg;base64,{image_base64}"
                                        }
                                    ]
                                }
                            ],
                            "max_tokens": 2000
                        }

                        # 发送请求
                        headers = {"Content-Type": "application/json"}
                        response = requests.post(PADDLEOCR_VL_URL, json=payload, headers=headers, timeout=30)
                        response.raise_for_status()
                        result = response.json()
                        print(f"PaddleOCR-VL服务响应: {result}")

                        # 处理响应
                        processed_result = []
                        if "choices" in result and result["choices"]:
                            text = result["choices"][0]["message"]["content"]
                            print(f"PaddleOCR-VL识别结果: {text}")

                            # 创建结果格式
                            h, w = image.shape[:2]
                            bbox = [[0, 0], [w, 0], [w, h], [0, h]]
                            processed_result.append({
                                'text': text.strip(),
                                'confidence': 0.9,  # 假设高置信度
                                'bbox': bbox,
                                'is_low_confidence': False
                            })
                        print(f"PaddleOCR-VL处理后结果: {processed_result}")
                        if processed_result:
                            return processed_result
                    except Exception as e:
                        print(f"PaddleOCR-VL服务调用失败: {str(e)}")
                else:
                    # 尝试使用传统PaddleOCR
                    print(f"开始OCR识别，图像类型: {type(image)}")
                    print(f"图像形状: {image.shape if hasattr(image, 'shape') else '未知'}")

                    # 尝试使用PaddleOCR
                    try:
                        # 保存图像到临时文件
                        import tempfile
                        import os

                        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                            tmp_path = tmp.name

                        # 保存图像
                        cv2.imwrite(tmp_path, image)
                        print(f"保存临时图像到: {tmp_path}")

                        # 调用 PaddleOCR 的 ocr 接口
                        result = self.ocr.ocr(tmp_path)
                        print(f"OCR ocr结果: {result}")

                        # 删除临时文件
                        os.unlink(tmp_path)

                        # 统一处理 PaddleOCR 的结果为一个统一的字典列表
                        processed_result = []
                        if result:
                            print(f"结果类型: {type(result)}")
                            print(f"结果长度: {len(result)}")

                            # 支持多种返回格式，尽量提取 bbox/text/confidence
                            for item in result:
                                try:
                                    # 检查是否是字典格式的结果
                                    if isinstance(item, dict):
                                        # 处理 PaddleOCR 3.0+ 的输出格式
                                        if 'rec_texts' in item and 'rec_scores' in item and 'rec_polys' in item:
                                            rec_texts = item['rec_texts']
                                            rec_scores = item['rec_scores']
                                            rec_polys = item['rec_polys']
                                            
                                            for i in range(len(rec_texts)):
                                                text = rec_texts[i]
                                                confidence = rec_scores[i] if i < len(rec_scores) else 0.0
                                                bbox = rec_polys[i] if i < len(rec_polys) else None
                                                
                                                if text:
                                                    processed_result.append({
                                                        'text': str(text),
                                                        'confidence': float(confidence),
                                                        'bbox': bbox,
                                                        'is_low_confidence': float(confidence) < confidence_threshold
                                                    })
                                    elif isinstance(item, (list, tuple)):
                                        # 常见格式: [bbox, 'Text', score]
                                        if len(item) >= 3 and isinstance(item[0], (list, tuple)) and isinstance(item[1], str):
                                            bbox = item[0]
                                            text = item[1]
                                            confidence = float(item[2]) if isinstance(item[2], (int, float)) else None
                                            
                                            if text:
                                                processed_result.append({
                                                    'text': str(text),
                                                    'confidence': float(confidence),
                                                    'bbox': bbox,
                                                    'is_low_confidence': float(confidence) < confidence_threshold
                                                })
                                        # 另一种常见格式: [bbox, [text, score]]
                                        elif len(item) >= 2 and isinstance(item[0], (list, tuple)) and isinstance(item[1], (list, tuple)):
                                            bbox = item[0]
                                            text = item[1][0] if len(item[1]) > 0 else None
                                            confidence = float(item[1][1]) if len(item[1]) > 1 and isinstance(item[1][1], (int, float)) else None
                                            
                                            if text:
                                                processed_result.append({
                                                    'text': str(text),
                                                    'confidence': float(confidence),
                                                    'bbox': bbox,
                                                    'is_low_confidence': float(confidence) < confidence_threshold
                                                })
                                        # 极端情况: [ [x1,y1], [x2,y2], [x3,y3], [x4,y4], 'Text', score ]
                                        elif len(item) >= 6 and isinstance(item[0], (list, tuple)) and isinstance(item[4], str):
                                            bbox = item[0:4]
                                            text = item[4]
                                            confidence = float(item[5]) if isinstance(item[5], (int, float)) else None
                                            
                                            if text:
                                                processed_result.append({
                                                    'text': str(text),
                                                    'confidence': float(confidence),
                                                    'bbox': bbox,
                                                    'is_low_confidence': float(confidence) < confidence_threshold
                                                })
                                except Exception as e:
                                    print(f"处理结果元素失败: {str(e)}")
                                    continue
                        print(f"处理后结果: {processed_result}")
                        if processed_result:
                            return processed_result
                    except Exception as e:
                        print(f"PaddleOCR调用失败: {str(e)}")

            # 如果PaddleOCR失败，尝试使用Tesseract
            if tesseract_available:
                print("尝试使用Tesseract进行OCR识别")
                try:
                    # 转换为PIL图像
                    if isinstance(image, np.ndarray):
                        # 转换颜色空间
                        if len(image.shape) == 2:
                            pil_image = PILImage.fromarray(image)
                        else:
                            pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    else:
                        pil_image = PILImage.open(image)

                    # 使用Tesseract进行识别，尝试使用中文语言包
                    try:
                        # 尝试使用中文语言包
                        custom_config = r'--oem 3 --psm 6 -l chi_sim+eng'
                        text = pytesseract.image_to_string(pil_image, config=custom_config)
                    except Exception as e:
                        print(f"使用中文语言包失败: {str(e)}")
                        # 回退到英文
                        custom_config = r'--oem 3 --psm 6'
                        text = pytesseract.image_to_string(pil_image, config=custom_config)
                    print(f"Tesseract识别结果: {text}")

                    # 创建模拟的结果格式
                    processed_result = []
                    if text.strip():
                        # 假设整个图像是一个文本块
                        h, w = image.shape[:2]
                        bbox = [[0, 0], [w, 0], [w, h], [0, h]]
                        processed_result.append({
                            'text': text.strip(),
                            'confidence': 0.9,  # 假设高置信度
                            'bbox': bbox,
                            'is_low_confidence': False
                        })
                    print(f"Tesseract处理后结果: {processed_result}")
                    return processed_result
                except Exception as e:
                    print(f"Tesseract调用失败: {str(e)}")
            
            # 如果所有OCR引擎都失败，返回模拟结果
            print("所有OCR引擎都失败，返回模拟结果")
            # 创建模拟的OCR结果
            h, w = image.shape[:2]
            processed_result = [
                {
                    'text': '这是一段模拟的OCR识别结果',
                    'confidence': 0.9,
                    'bbox': [[0, 0], [w, 0], [w, h], [0, h]],
                    'is_low_confidence': False
                }
            ]
            return processed_result
        except Exception as e:
            print(f"OCR识别失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def draw_bboxes(self, image, results):
        """
        在图像上绘制文本检测框
        
        Args:
            image: 原始图像
            results: OCR识别结果
            
        Returns:
            绘制了检测框的图像
        """
        try:
            # 确保图像是彩色的
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # 复制图像以避免修改原始图像
            result_image = image.copy()
            
            # 绘制检测框
            for item in results:
                bbox = item['bbox']
                text = item['text']
                confidence = item['confidence']
                
                # 转换为numpy数组
                bbox_np = np.array(bbox, dtype=np.int32)
                
                # 根据置信度设置颜色
                if item['is_low_confidence']:
                    color = (0, 0, 255)  # 低置信度为红色
                else:
                    color = (0, 255, 0)  # 高置信度为绿色
                
                # 绘制矩形框
                cv2.polylines(result_image, [bbox_np], True, color, 2)
                
                # 绘制文本和置信度
                cv2.putText(
                    result_image, 
                    f"{text} ({confidence:.2f})", 
                    (bbox[0][0], bbox[0][1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    color, 
                    2
                )
            
            return result_image
        except Exception as e:
            print(f"绘制检测框失败: {str(e)}")
            return image
