import cv2
import numpy as np
import subprocess
import os

# 运行次数计数器
run_count = 0
max_runs = 20

# 初始化参数
param_set = 0
param_sets = [
    # 方案1：高对比度和亮度，高分辨率
    {"alpha": 2.5, "beta": 70, "fx": 2.0, "fy": 2.0},
    # 方案2：适中对比度和亮度，适中分辨率
    {"alpha": 2.0, "beta": 50, "fx": 1.5, "fy": 1.5},
    # 方案3：低对比度和亮度，低分辨率
    {"alpha": 1.5, "beta": 30, "fx": 1.0, "fy": 1.0},
    # 方案4：高对比度，低亮度
    {"alpha": 3.0, "beta": 20, "fx": 1.5, "fy": 1.5},
    # 方案5：低对比度，高亮度
    {"alpha": 1.0, "beta": 80, "fx": 1.5, "fy": 1.5},
    # 方案6：极高对比度，适中亮度
    {"alpha": 4.0, "beta": 50, "fx": 1.5, "fy": 1.5},
    # 方案7：适中对比度，极高亮度
    {"alpha": 2.0, "beta": 100, "fx": 1.5, "fy": 1.5},
    # 方案8：高对比度和亮度，极高分辨率
    {"alpha": 2.5, "beta": 70, "fx": 3.0, "fy": 3.0},
]

# 循环运行测试
while run_count < max_runs:
    run_count += 1
    print(f"\n=== 运行次数: {run_count}/{max_runs} ===")
    
    # 获取当前参数集
    current_params = param_sets[param_set % len(param_sets)]
    param_set += 1
    
    print(f"使用参数: alpha={current_params['alpha']}, beta={current_params['beta']}, fx={current_params['fx']}, fy={current_params['fy']}")
    
    # 修改test_ocr.py文件中的参数
    with open('test_ocr.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 更新参数
    content = content.replace('alpha = 2.5  # 对比度增益', f"alpha = {current_params['alpha']}  # 对比度增益")
    content = content.replace('beta = 70    # 亮度增益', f"beta = {current_params['beta']}    # 亮度增益")
    content = content.replace('processed_image = cv2.resize(processed_image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)', 
                             f'processed_image = cv2.resize(processed_image, None, fx={current_params["fx"]}, fy={current_params["fy"]}, interpolation=cv2.INTER_CUBIC)')
    
    # 写回文件
    with open('test_ocr.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # 删除之前的结果文件
    result_file = 'd:\\TARE\\OCR\\result_test.jpg'
    if os.path.exists(result_file):
        os.remove(result_file)
        print("已删除之前的result_test.jpg文件")
    
    # 运行测试脚本
    result = subprocess.run(['python', 'test_ocr.py'], capture_output=True, text=True, cwd='d:\\TARE\\OCR')
    
    # 打印输出
    print(result.stdout)
    if result.stderr:
        print("错误输出:")
        print(result.stderr)
    
    # 检查是否成功识别
    if "✓ 成功识别目标文本！" in result.stdout:
        print("\n🎉 成功识别目标文本！")
        break
    
    # 检查是否需要调整PaddleOCR参数
    if run_count % 4 == 0:
        print("\n调整PaddleOCR参数...")
        # 修改ocr_engine.py文件中的参数
        with open('ocr_engine.py', 'r', encoding='utf-8') as f:
            ocr_content = f.read()
        
        # 降低检测阈值
        current_thresh = 0.1 - (run_count // 4) * 0.02
        current_box_thresh = 0.4 - (run_count // 4) * 0.05
        
        if current_thresh < 0.02:
            current_thresh = 0.02
        if current_box_thresh < 0.2:
            current_box_thresh = 0.2
        
        # 查找并替换参数
        import re
        ocr_content = re.sub(r'det_db_thresh=0\.\d+', f'det_db_thresh={current_thresh}', ocr_content)
        ocr_content = re.sub(r'det_db_box_thresh=0\.\d+', f'det_db_box_thresh={current_box_thresh}', ocr_content)
        
        with open('ocr_engine.py', 'w', encoding='utf-8') as f:
            f.write(ocr_content)
        
        print(f"更新PaddleOCR参数: det_db_thresh={current_thresh}, det_db_box_thresh={current_box_thresh}")

# 总结
if run_count >= max_runs:
    print(f"\n⚠️  已运行{max_runs}次，未能成功识别目标文本。")
    print("\n优化建议:")
    print("1. 尝试调整图像预处理参数，特别是对比度和亮度")
    print("2. 考虑使用不同的图像增强方法，如直方图均衡化")
    print("3. 调整PaddleOCR的检测和识别参数")
    print("4. 检查测试图像的质量，确保文本清晰可见")
    print("5. 尝试使用不同的OCR引擎或模型")
else:
    print(f"\n✅ 在第{run_count}次运行中成功识别目标文本！")
    print("\n成功参数:")
    print(f"- 对比度增益: {current_params['alpha']}")
    print(f"- 亮度增益: {current_params['beta']}")
    print(f"- 分辨率缩放: {current_params['fx']}x")
    print("\n建议将这些参数应用到实际应用中。")
