# 智能文档OCR识别系统

一个功能强大的中文OCR识别系统，支持多种图像预处理方法、多引擎OCR识别以及多种格式导出。

## 项目架构

本项目采用模块化设计，主要包含以下核心模块：

```
┌─────────────────────────────────────────────────────────────┐
│                      GUI层 (Tkinter)                        │
│                   main_tkinter.py / main.py                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      业务逻辑层                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │   预处理模块     │  │   OCR引擎模块    │  │  后处理模块  │  │
│  │  preprocessor   │  │   ocr_engine    │  │ postprocessor│ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      底层引擎层                              │
│    PaddleOCR    │    Tesseract    │    PaddleVL服务        │
└─────────────────────────────────────────────────────────────┘
```

## 特点

- **多引擎支持**: 集成PaddleOCR和Tesseract双引擎，支持引擎自动切换
- **图像预处理**: 提供去噪、二值化、倾斜校正、对比度增强、分辨率优化等多种预处理方法
- **文本后处理**: 支持文本纠错、排版还原功能
- **可视化界面**: 简洁易用的Tkinter图形界面，支持图像预览和参数调节
- **多格式导出**: 支持导出为TXT、DOCX、XLSX格式
- **参数灵活配置**: 支持通过界面调整OCR参数和预处理参数

## 目录结构

```
OCR/
├── main_tkinter.py          # Tkinter GUI主程序
├── main.py                  # Streamlit Web界面主程序
├── main_gui.py              # GUI备选方案
├── ocr_engine.py            # OCR引擎封装模块
├── preprocessor.py          # 图像预处理模块
├── postprocessor.py         # 文本后处理模块
├── optimize_ocr.py           # OCR参数优化脚本
├── test_ocr.py              # OCR测试脚本
├── test_handwriting.py      # 手写体OCR测试脚本
├── requirements.txt         # 项目依赖
├── test.jpg                 # 测试图片1（印刷体）
├── test2.jpg                # 测试图片2（手写体）
├── result_test.jpg          # OCR结果可视化图片
├── processed_test.jpg       # 预处理后的图片
└── enhanced_test.jpg        # 增强后的图片
```

## 引擎类型及名称

| 引擎 | 类型 | 说明 |
|------|------|------|
| **PaddleOCR** | 深度学习OCR | 百度开源的中文OCR引擎，支持多种语言 |
| **Tesseract** | 传统OCR | Google开源的OCR引擎，支持多语言 |
| **PaddleOCR-VL** | 视觉语言模型 | 支持图像+文本的多模态识别（需独立服务） |

### PaddleOCR模型信息

- **检测模型**: DB (Differentiable Binarization)
- **识别模型**: CRNN (Convolutional Recurrent Neural Network)
- **语言支持**: 中文(ch)、英文(en)、日文(ja)等

### Tesseract模型信息

- **语言包**: chi_sim (简体中文), eng (英文)
- **引擎模式**: OEM 3 (默认)
- **页面分割模式**: PSM 6 (自动分块)

## 安装依赖

```bash
pip install -r requirements.txt
```

### 核心依赖说明

| 依赖包 | 版本要求 | 说明 |
|--------|----------|------|
| opencv-python | >=4.5.5 | 图像处理 |
| Pillow | >=9.0.0 | 图像读取 |
| numpy | >=1.20.0 | 数值计算 |
| paddleocr | >=2.6.1.3 | PaddleOCR引擎 |
| paddlepaddle | >=2.3.0 | PaddlePaddle深度学习框架 |
| pytesseract | >=0.3.8 | Tesseract Python接口 |
| streamlit | >=1.10.0 | Web界面框架 |
| pandas | >=1.3.0 | 数据处理 |
| python-docx | >=0.8.11 | Word文档生成 |

### Tesseract安装（可选）

如需使用Tesseract引擎，需额外安装：

1. 下载Tesseract安装包: https://github.com/UB-Mannheim/tesseract/wiki
2. 安装后配置路径到系统环境变量
3. 下载中文语言包: https://github.com/tesseract-ocr/tessdata

## 使用方法

### 启动GUI界面

```bash
python main_tkinter.py
```

### 启动Web界面（可选）

```bash
python main.py
```

## 参数配置说明

### OCR引擎参数 (ocr_engine.py)

```python
PaddleOCR(
    lang='ch',                    # 语言：ch(中文), en(英文), ja(日文)
    use_angle_cls=True,          # 启用角度检测
    det_limit_side_len=1920,     # 检测边长限制
    det_limit_type='max',        # 限制类型
    det_db_thresh=0.02,           # 检测阈值（越低越灵敏）
    det_db_box_thresh=0.1,       # 框选阈值
    det_db_unclip_ratio=2.0,     # 扩展比例（影响框大小）
    rec_batch_num=1               # 识别批次大小
)
```

### 预处理参数 (preprocessor.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| denoise | True | 是否去噪 |
| binarize | True | 是否二值化 |
| correct_skew | True | 是否倾斜校正 |
| enhance_contrast | True | 是否对比度增强 |
| optimize_resolution | True | 是否分辨率优化 |
| denoise_method | gaussian | 去噪方法：gaussian/median |
| blur_kernel | 3 | 模糊核大小（奇数） |
| binary_block_size | 11 | 二值化块大小 |
| binary_c | 2 | 二值化常数 |

## 开发说明

### 模块职责

- **ocr_engine.py**: OCR引擎抽象层，统一接口调用PaddleOCR/Tesseract
- **preprocessor.py**: 图像预处理流水线
- **postprocessor.py**: 文本后处理（纠错、格式化、导出）
- **main_tkinter.py**: Tkinter桌面应用入口

### 扩展开发

如需添加新的OCR引擎，只需在 `ocr_engine.py` 中扩展 `OCREngine` 类即可。

## 许可证

本项目仅供学习和研究使用。
