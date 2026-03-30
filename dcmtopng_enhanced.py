import pydicom
import numpy as np
import cv2
from pathlib import Path
from skimage import exposure

def dicom_to_png_enhanced(dicom_path, png_path, apply_clahe=True, use_optimal_window=True):
    """
    改进的DICOM到PNG转换函数，提供更好的图像质量
    
    参数:
    - dicom_path: DICOM文件路径
    - png_path: 输出PNG文件路径
    - apply_clahe: 是否应用CLAHE对比度增强
    - use_optimal_window: 是否使用优化的窗宽窗位
    """
    try:
        ds = pydicom.dcmread(dicom_path)
        img = ds.pixel_array.astype(np.float32)
        
        # 处理像素数据的缩放和偏移
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            slope = float(ds.RescaleSlope)
            intercept = float(ds.RescaleIntercept)
            img = img * slope + intercept
            print(f"应用了RescaleSlope: {slope}, RescaleIntercept: {intercept}")
        
        # 获取窗宽窗位
        wc, ww = get_optimal_window_params(ds, img, use_optimal_window)
        
        # 应用窗宽窗位
        img_windowed = apply_windowing(img, wc, ww)
        
        # 转换为8位图像
        img_8bit = (img_windowed * 255.0).astype(np.uint8)
        
        # 应用CLAHE对比度增强（可选）
        if apply_clahe:
            img_8bit = apply_clahe_enhancement(img_8bit)
        
        # 处理图像方向
        img_final = handle_image_orientation(img_8bit, ds)
        
        # 保存图像
        cv2.imwrite(str(png_path), img_final)
        print(f"已保存为: {png_path} (窗位: {wc:.1f}, 窗宽: {ww:.1f})")
        
        return True
        
    except Exception as e:
        print(f"处理文件 {dicom_path} 时出错: {e}")
        return False

def get_optimal_window_params(ds, img, use_optimal=True):
    """获取最佳的窗宽窗位参数"""
    wc = None
    ww = None
    
    # 首先尝试从DICOM标签获取
    if 'WindowCenter' in ds and 'WindowWidth' in ds:
        wc = ds.WindowCenter
        ww = ds.WindowWidth
        
        # 处理多值情况
        if isinstance(wc, pydicom.multival.MultiValue):
            wc = float(wc[0])
        else:
            wc = float(wc)
            
        if isinstance(ww, pydicom.multival.MultiValue):
            ww = float(ww[0])
        else:
            ww = float(ww)
            
        print(f"使用DICOM标签中的窗位/窗宽: {wc}/{ww}")
    
    # 如果没有窗宽窗位或选择使用优化参数
    if wc is None or ww is None or use_optimal:
        if use_optimal:
            # 使用更智能的方法计算窗宽窗位
            wc_opt, ww_opt = calculate_optimal_window(img)
            if wc is None or ww is None:
                wc, ww = wc_opt, ww_opt
                print(f"使用优化计算的窗位/窗宽: {wc:.1f}/{ww:.1f}")
            else:
                # 对于X光图像，优先使用DICOM原始参数，除非它们明显不合理
                original_ratio = ww / (np.max(img) - np.min(img))
                if original_ratio < 0.1:  # 原始窗宽太窄，可能不合适
                    wc, ww = wc_opt, ww_opt
                    print(f"原始窗宽过窄，使用优化计算的窗位/窗宽: {wc:.1f}/{ww:.1f}")
                else:
                    print(f"保持DICOM原始窗位/窗宽: {wc}/{ww}")
        else:
            # 使用基本统计方法，但更保守
            wc = np.percentile(img, 50)  # 使用中位数而不是均值
            ww = np.percentile(img, 90) - np.percentile(img, 10)  # 使用80%范围
            print(f"使用统计计算的窗位/窗宽: {wc:.1f}/{ww:.1f}")
    
    return wc, ww

def calculate_optimal_window(img):
    """计算最佳窗宽窗位 - 针对X光图像优化"""
    # 对于X光图像，使用更保守的方法
    
    # 方法1：使用更大的百分位范围，避免过度限制动态范围
    p5, p95 = np.percentile(img, [5, 95])
    
    # 方法2：如果动态范围很大，使用全范围的一定比例
    full_range = np.max(img) - np.min(img)
    img_mean = np.mean(img)
    
    # 选择更保守的窗宽窗位
    if full_range > 2000:  # 大动态范围，可能是原始DICOM数据
        ww = full_range * 0.8  # 使用80%的动态范围
        wc = img_mean
        print(f"使用大动态范围策略: 全范围={full_range:.1f}")
    else:
        # 使用百分位方法，但范围更大
        ww = p95 - p5
        wc = np.percentile(img, 50)  # 使用中位数
        print(f"使用百分位策略: P5-P95范围")
    
    return wc, ww

def apply_windowing(img, wc, ww):
    """应用窗宽窗位"""
    min_val = wc - ww / 2
    max_val = wc + ww / 2
    
    # 应用窗宽窗位
    img_windowed = np.clip(img, min_val, max_val)
    img_windowed = (img_windowed - min_val) / (max_val - min_val)
    img_windowed = np.clip(img_windowed, 0, 1)
    
    return img_windowed

def apply_clahe_enhancement(img_8bit):
    """应用CLAHE对比度限制自适应直方图均衡化 - 针对X光图像优化"""
    # 对X光图像使用更温和的CLAHE参数
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))  # 降低clipLimit，增大tileGridSize
    img_enhanced = clahe.apply(img_8bit)
    
    # 可选：与原图混合，进一步减少过度增强
    alpha = 0.7  # 增强图像的权重
    img_blended = cv2.addWeighted(img_enhanced, alpha, img_8bit, 1-alpha, 0)
    
    return img_blended

def handle_image_orientation(img, ds):
    """处理图像方向"""
    # 检查图像方向信息
    if hasattr(ds, 'ImageOrientationPatient'):
        # 这里可以根据需要实现图像方向校正
        pass
    
    # 检查PhotometricInterpretation
    if hasattr(ds, 'PhotometricInterpretation'):
        if ds.PhotometricInterpretation == 'MONOCHROME1':
            # MONOCHROME1表示较小的值应该显示为更亮
            img = 255 - img
            print("应用了MONOCHROME1反转")
    
    return img

def dicom_to_png_conservative(dicom_path, png_path):
    """
    保守的DICOM到PNG转换函数，专门针对X光图像优化
    优先使用DICOM原始参数，只做必要的处理
    """
    try:
        ds = pydicom.dcmread(dicom_path)
        img = ds.pixel_array.astype(np.float32)
        
        # 处理像素数据的缩放和偏移
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            slope = float(ds.RescaleSlope)
            intercept = float(ds.RescaleIntercept)
            img = img * slope + intercept
            print(f"应用了RescaleSlope: {slope}, RescaleIntercept: {intercept}")
        
        # 优先使用DICOM原始窗宽窗位
        wc = None
        ww = None
        if 'WindowCenter' in ds and 'WindowWidth' in ds:
            wc = ds.WindowCenter
            ww = ds.WindowWidth
            
            if isinstance(wc, pydicom.multival.MultiValue):
                wc = float(wc[0])
            else:
                wc = float(wc)
                
            if isinstance(ww, pydicom.multival.MultiValue):
                ww = float(ww[0])
            else:
                ww = float(ww)
                
            print(f"使用DICOM原始窗位/窗宽: {wc}/{ww}")
        else:
            # 如果没有原始参数，使用保守的统计方法
            wc = np.percentile(img, 50)
            ww = np.percentile(img, 85) - np.percentile(img, 15)  # 使用70%范围
            print(f"使用保守统计计算的窗位/窗宽: {wc:.1f}/{ww:.1f}")
        
        # 应用窗宽窗位
        img_windowed = apply_windowing(img, wc, ww)
        
        # 转换为8位图像
        img_8bit = (img_windowed * 255.0).astype(np.uint8)
        
        # 只应用轻微的对比度调整
        img_8bit = cv2.convertScaleAbs(img_8bit, alpha=1.1, beta=5)
        
        # 处理图像方向
        img_final = handle_image_orientation(img_8bit, ds)
        
        # 保存图像
        cv2.imwrite(str(png_path), img_final)
        print(f"已保存为: {png_path} (保守版本，窗位: {wc:.1f}, 窗宽: {ww:.1f})")
        
        return True
        
    except Exception as e:
        print(f"处理文件 {dicom_path} 时出错: {e}")
        return False

def batch_convert_with_options():
    """批量转换，提供多种选项"""
    dcm_dir = Path(".")
    dcm_files = list(dcm_dir.glob("*.dcm"))
    
    if not dcm_files:
        print("未找到dcm文件")
        return
    
    print(f"找到 {len(dcm_files)} 个DICOM文件")
    
    # 创建输出文件夹
    output_dirs = {
        'standard': dcm_dir / "output_standard",
        'optimized': dcm_dir / "output_optimized", 
        'clahe': dcm_dir / "output_clahe",
        'conservative': dcm_dir / "output_conservative"  # 新增保守版本
    }
    
    for dir_name, dir_path in output_dirs.items():
        dir_path.mkdir(exist_ok=True)
        print(f"创建输出文件夹: {dir_path}")
    
    # 创建不同版本的输出
    for dcm_file in dcm_files:
        base_name = dcm_file.stem
        print(f"\n处理文件: {dcm_file.name}")
        
        # 1. 标准版本 - 使用DICOM原始窗宽窗位
        png_file_std = output_dirs['standard'] / f"{base_name}.png"
        print("  -> 生成标准版本...")
        dicom_to_png_enhanced(dcm_file, png_file_std, apply_clahe=False, use_optimal_window=False)
        
        # 2. 优化窗宽窗位版本 - 使用智能计算的窗宽窗位
        png_file_opt = output_dirs['optimized'] / f"{base_name}.png"
        print("  -> 生成优化窗宽窗位版本...")
        dicom_to_png_enhanced(dcm_file, png_file_opt, apply_clahe=False, use_optimal_window=True)
        
        # 3. CLAHE增强版本 - 使用优化窗宽窗位 + CLAHE对比度增强
        png_file_clahe = output_dirs['clahe'] / f"{base_name}.png"
        print("  -> 生成CLAHE增强版本...")
        dicom_to_png_enhanced(dcm_file, png_file_clahe, apply_clahe=True, use_optimal_window=True)
        
        # 4. 保守版本 - 使用原始DICOM参数但应用温和增强
        png_file_conservative = output_dirs['conservative'] / f"{base_name}.png"
        print("  -> 生成保守版本...")
        dicom_to_png_conservative(dcm_file, png_file_conservative)

if __name__ == "__main__":
    print("=== 改进的DICOM转PNG工具 ===")
    print("将生成四个版本，分别保存在不同文件夹中：")
    print("1. output_standard/     - 标准版本（使用DICOM原始窗宽窗位）")
    print("2. output_optimized/    - 优化窗宽窗位版本（智能计算窗宽窗位）")
    print("3. output_clahe/        - CLAHE增强版本（优化窗宽窗位 + 对比度增强）")
    print("4. output_conservative/ - 保守版本（优先原始参数 + 轻微调整）")
    print()
    print("各版本区别说明：")
    print("- 标准版本：使用DICOM文件中的原始WindowCenter/WindowWidth参数")
    print("- 优化版本：使用统计学方法计算更适合的窗宽窗位，突出重要细节")
    print("- CLAHE版本：在优化版本基础上添加自适应直方图均衡化，增强局部对比度")
    print("- 保守版本：针对X光图像优化，避免过曝，保持细节")
    print("="*60)
    
    batch_convert_with_options()
    
    print("\n" + "="*60)
    print("转换完成！请检查以下文件夹中的结果：")
    print("- output_standard/      标准版本")
    print("- output_optimized/     优化版本") 
    print("- output_clahe/         增强版本")
    print("- output_conservative/  保守版本（推荐用于X光图像）")
    print("建议使用图像查看器同时打开四个版本进行对比。")
