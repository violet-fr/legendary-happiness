import re
import pandas as pd
from docx import Document

class PostProcessor:
    """
    后处理与结构化类，用于处理OCR识别结果
    """
    
    def __init__(self):
        """
        初始化后处理类
        """
        # 常见OCR错误映射
        self.error_map = {
            'O': '0',
            'o': '0',
            'l': '1',
            'I': '1',
            'B': '8',
            'S': '5',
            'Z': '2',
            'z': '2',
            '\u3000': ' '  # 全角空格转半角
        }
    
    def process(self, ocr_results):
        """
        执行完整的后处理流程
        
        Args:
            ocr_results: OCR识别结果列表
            
        Returns:
            处理后的文本
        """
        try:
            # 文本纠错
            corrected_results = self._correct_text(ocr_results)
            
            # 排版还原
            structured_text = self._reconstruct_layout(corrected_results)
            
            return structured_text
        except Exception as e:
            print(f"后处理失败: {str(e)}")
            return ""
    
    def _correct_text(self, ocr_results):
        """
        文本纠错
        
        基于简单的词典匹配或正则规则，修正常见的OCR错误
        
        Args:
            ocr_results: OCR识别结果列表
            
        Returns:
            纠错后的结果列表
        """
        corrected_results = []
        
        for item in ocr_results:
            text = item['text']
            
            # 应用错误映射
            corrected_text = text
            for wrong, correct in self.error_map.items():
                corrected_text = corrected_text.replace(wrong, correct)
            
            # 正则处理：修正数字中的常见错误
            # 例如：将"123O"修正为"1230"
            corrected_text = re.sub(r'(\d+)O(\d+)', r'\g<1>0\2', corrected_text)
            corrected_text = re.sub(r'(\d+)l(\d+)', r'\g<1>1\2', corrected_text)
            
            # 更新结果
            corrected_item = item.copy()
            corrected_item['text'] = corrected_text
            corrected_results.append(corrected_item)
        
        return corrected_results
    
    def _reconstruct_layout(self, ocr_results):
        """
        排版还原
        
        根据坐标信息，尝试还原文本的段落结构和换行符
        
        Args:
            ocr_results: 纠错后的OCR结果列表
            
        Returns:
            结构化的文本
        """
        if not ocr_results:
            return ""
        
        # 按y坐标排序（从上到下）
        sorted_results = sorted(ocr_results, key=lambda x: x['bbox'][0][1])
        
        structured_text = []
        current_line = []
        current_y = None
        line_threshold = 15  # 行间距阈值
        
        for item in sorted_results:
            text = item['text']
            bbox = item['bbox']
            y = bbox[0][1]  # 文本框顶部y坐标
            
            # 检查是否是新行
            if current_y is None or abs(y - current_y) > line_threshold:
                # 保存当前行
                if current_line:
                    structured_text.append(' '.join(current_line))
                    current_line = []
                current_y = y
            
            current_line.append(text)
        
        # 添加最后一行
        if current_line:
            structured_text.append(' '.join(current_line))
        
        return '\n'.join(structured_text)
    
    def export(self, text, output_path, format='txt'):
        """
        导出功能
        
        支持将识别结果导出为 .txt, .docx 和 .xlsx
        
        Args:
            text: 处理后的文本
            output_path: 输出文件路径
            format: 导出格式，支持 'txt', 'docx', 'xlsx'
            
        Returns:
            bool: 导出是否成功
        """
        try:
            if format == 'txt':
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
            
            elif format == 'docx':
                doc = Document()
                # 按行添加段落
                for line in text.split('\n'):
                    doc.add_paragraph(line)
                doc.save(output_path)
            
            elif format == 'xlsx':
                # 将文本按行分割，创建DataFrame
                lines = text.split('\n')
                df = pd.DataFrame({'文本': lines})
                df.to_excel(output_path, index=False)
            
            else:
                print(f"不支持的导出格式: {format}")
                return False
            
            return True
        except Exception as e:
            print(f"导出失败: {str(e)}")
            return False