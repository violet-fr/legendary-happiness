import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

from preprocessor import ImagePreprocessor
from ocr_engine import OCREngine
from postprocessor import PostProcessor

# 页面配置
st.set_page_config(
    page_title="智能文档OCR识别系统",
    page_icon="📄",
    layout="wide"
)

# 标题
st.title("智能文档OCR识别系统")

# 侧边栏参数设置
st.sidebar.header("参数设置")

# 语言选择
lang = st.sidebar.selectbox(
    "语言选择",
    options=["中文", "英文", "日文"],
    index=0
)

# 映射语言到PaddleOCR的语言代码
lang_map = {
    "中文": "ch",
    "英文": "en",
    # 使用 PaddleOCR 支持的日文语言代码，避免使用非标准名称
    "日文": "ja"
}

# 置信度阈值
confidence_threshold = st.sidebar.slider(
    "识别置信度阈值",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.05
)

# 预处理参数
st.sidebar.subheader("预处理参数")
denoise = st.sidebar.checkbox("去噪", value=True)
binarize = st.sidebar.checkbox("二值化", value=True)
correct_skew = st.sidebar.checkbox("倾斜校正", value=True)
enhance_contrast = st.sidebar.checkbox("对比度增强", value=True)
optimize_resolution = st.sidebar.checkbox("分辨率优化", value=True)

# 文件上传
st.header("文件上传")
uploaded_file = st.file_uploader(
    "上传图片或PDF文件",
    type=["jpg", "jpeg", "png", "pdf"]
)

# 主处理逻辑
if uploaded_file is not None:
    try:
        # 读取上传的文件
        if uploaded_file.type == "application/pdf":
            # 这里可以添加PDF处理逻辑
            st.warning("PDF处理功能暂未实现，请上传图片文件")
        else:
            # 读取图片
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # 初始化各个模块
            preprocessor = ImagePreprocessor()
            ocr_engine = OCREngine(lang=lang_map[lang])
            postprocessor = PostProcessor()
            
            # 图像预处理
            st.subheader("图像处理")
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="原始图像", use_column_width=True)
            
            # 执行预处理
            processed_image = preprocessor.process(
                image_np,
                denoise=denoise,
                binarize=binarize,
                correct_skew=correct_skew,
                enhance_contrast=enhance_contrast,
                optimize_resolution=optimize_resolution
            )
            
            with col2:
                st.image(processed_image, caption="预处理后图像", use_column_width=True)
            
            # OCR识别
            st.subheader("OCR识别")
            ocr_results = ocr_engine.recognize(processed_image, confidence_threshold)
            
            # 绘制检测框
            bbox_image = ocr_engine.draw_bboxes(image_np, ocr_results)
            
            # 后处理
            structured_text = postprocessor.process(ocr_results)
            
            # 显示结果
            col3, col4 = st.columns(2)
            
            with col3:
                st.image(bbox_image, caption="识别结果（带检测框）", use_column_width=True)
            
            with col4:
                st.subheader("识别文本")
                # 可编辑的文本框
                edited_text = st.text_area("识别结果（可编辑）", structured_text, height=400)
                
                # 复制按钮
                if st.button("复制文本"):
                    st.write("文本已复制到剪贴板")
                
                # 导出选项
                st.subheader("导出选项")
                export_format = st.selectbox(
                    "选择导出格式",
                    options=["txt", "docx", "xlsx"],
                    index=0
                )
                
                if st.button("导出文件"):
                    with tempfile.NamedTemporaryFile(suffix=f".{export_format}", delete=False) as tmp:
                        output_path = tmp.name
                    
                    success = postprocessor.export(edited_text, output_path, format=export_format)
                    
                    if success:
                        # 读取文件并提供下载
                        with open(output_path, "rb") as f:
                            st.download_button(
                                label=f"下载{export_format.upper()}文件",
                                data=f,
                                file_name=f"ocr_result.{export_format}",
                                mime=f"application/{export_format}"
                            )
                        # 删除临时文件
                        os.unlink(output_path)
                    else:
                        st.error("导出失败")
    
    except Exception as e:
        st.error(f"处理失败: {str(e)}")

# 关于部分
st.sidebar.header("关于")
st.sidebar.info(
    "智能文档OCR识别系统\n"
    "基于PaddleOCR和OpenCV\n"
    "支持中文、英文、日文识别\n"
    "提供图像预处理、OCR识别、后处理和导出功能"
)
