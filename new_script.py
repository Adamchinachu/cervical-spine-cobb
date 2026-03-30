import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import math
import pandas as pd
import os
import pydicom
import time
# 你要识别的类别
vertebrae_classes = ['C2', 'C3', 'C4', 'C5', 'C6', 'C7']
needed_check = {'C2', 'C3', 'C4', 'C5', 'C6'}
from pyd.NumpyReshape import _x5y7z
c2c7_model = _x5y7z() 
def dicom_to_png_standard(dicom_path, png_path):
    """标准的DICOM到PNG转换函数"""
    try:
        ds = pydicom.dcmread(dicom_path)
        img = ds.pixel_array.astype(np.float32)
        
        # 处理像素数据的缩放和偏移
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            slope = float(ds.RescaleSlope)
            intercept = float(ds.RescaleIntercept)
            img = img * slope + intercept
        
        # 获取窗宽窗位
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
        
        # 如果没有窗宽窗位，使用基本统计方法
        if wc is None or ww is None:
            wc = np.percentile(img, 50)
            ww = np.percentile(img, 90) - np.percentile(img, 10)
        
        # 应用窗宽窗位
        min_val = wc - ww / 2
        max_val = wc + ww / 2
        img_windowed = np.clip(img, min_val, max_val)
        img_windowed = (img_windowed - min_val) / (max_val - min_val)
        img_windowed = np.clip(img_windowed, 0, 1)
        
        # 转换为8位图像
        img_8bit = (img_windowed * 255.0).astype(np.uint8)
        
        # 处理PhotometricInterpretation
        if hasattr(ds, 'PhotometricInterpretation'):
            if ds.PhotometricInterpretation == 'MONOCHROME1':
                img_8bit = 255 - img_8bit
        
        # 保存图像
        cv2.imwrite(str(png_path), img_8bit)
        return True
        
    except Exception as e:
        print(f"DICOM转换出错: {e}")
        return False

def judge_left_right_by_model(img, c3c4c5_combined_box, leftright_model):
    """
    使用leftright.pt模型判断左右方向
    如果C3C4C5组合的框在识别框的左侧则是左，反之是右
    """
    try:
        # 使用leftright模型进行检测
        results = leftright_model(img, conf=0.3, iou=0.5)
        result = results[0]
        
        if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:
            print("leftright模型未检测到任何框")
            return 'skip'
        
        # 获取检测框的坐标和置信度 (原图尺寸)
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        confidences = result.boxes.conf.cpu().numpy()  # 置信度
        
        if len(boxes) > 0:
            # 找到置信度最高的检测框
            max_conf_idx = np.argmax(confidences)
            max_confidence = confidences[max_conf_idx]
            
            print(f"最高置信度: {max_confidence:.3f}")
            
            # 如果置信度低于0.6，判断出错
            if max_confidence < 0.3:
                print(f"置信度过低 ({max_confidence:.3f} < 0.6)，跳过此图")
                return 'skip'
            
            detection_box = boxes[max_conf_idx]  # x1, y1, x2, y2
            detection_center_x = (detection_box[0] + detection_box[2]) / 2
            
            # C3C4C5组合框的中心x坐标
            c3c4c5_center_x = (c3c4c5_combined_box[0] + c3c4c5_combined_box[2]) / 2
            
            print(f"检测框中心x: {detection_center_x}, C3C4C5组合框中心x: {c3c4c5_center_x}")
            
            # 如果C3C4C5组合框在检测框左侧，则是左
            if c3c4c5_center_x < detection_center_x:
                return 'left'
            else:
                return 'right'
        else:
            print("未获取到有效的检测框")
            return 'skip'
            
    except Exception as e:
        print(f"leftright模型判断出错: {e}")
        return 'skip'

def get_combined_box_c3c4c5(masks, classes, names, img_shape):
    """
    获取C3、C4、C5的组合边界框
    """
    h_img, w_img = img_shape[:2]
    combined_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    
    for mask, cls_id in zip(masks, classes):
        class_name = names[cls_id]
        if class_name in ['C3', 'C4', 'C5']:
            mask_resized = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
            combined_mask = np.logical_or(combined_mask, mask_resized > 0.5)
    
    # 找到组合mask的边界框
    y_indices, x_indices = np.where(combined_mask)
    if len(y_indices) == 0:
        return None
    
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    return (x_min, y_min, x_max, y_max)  # x1, y1, x2, y2

def find_main_edge_cross_centroid(hull_points, class_name):
    """
    C2：只在质心下方选x差值最大的边；
    C7：只在质心上方选横跨质心x且x差值最大的边；
    其它：只在质心下方选横跨质心x且x差值最大的边。
    """
    # 计算质心
    M = cv2.moments(hull_points)
    if M['m00'] != 0:
        centroid_x = M['m10'] / M['m00']
        centroid_y = M['m01'] / M['m00']
    else:
        centroid_x = np.mean(hull_points[:, 0])
        centroid_y = np.mean(hull_points[:, 1])

    max_xlen = 0
    best_edge = None
    n_hull = len(hull_points)
    for j in range(n_hull):
        p1 = hull_points[j]
        p2 = hull_points[(j + 1) % n_hull]
        mid_y = (p1[1] + p2[1]) / 2
        edge_xlen = abs(p2[0] - p1[0])
        if class_name == 'C2':
            # 只要在质心下方，且两个端点都在质心下方，x差值最大即可
            if (
                mid_y > centroid_y and
                p1[1] > centroid_y and
                p2[1] > centroid_y and
                edge_xlen > max_xlen
            ):
                max_xlen = edge_xlen
                best_edge = (tuple(p1), tuple(p2))
        elif class_name == 'C8':
            # 质心上方且横跨质心x
            if mid_y < centroid_y and (p1[0] - centroid_x) * (p2[0] - centroid_x) < 0 and edge_xlen > max_xlen:
                max_xlen = edge_xlen
                best_edge = (tuple(p1), tuple(p2))
        else:
            # 质心下方且横跨质心x
            if mid_y > centroid_y and (p1[0] - centroid_x) * (p2[0] - centroid_x) < 0 and edge_xlen > max_xlen:
                max_xlen = edge_xlen
                best_edge = (tuple(p1), tuple(p2))
    return best_edge

# ================== 角度计算相关函数 =====================
def normalize_vector(v):
    x, y = v
    if x > 0:
        x = -x
    return (x, y)

def vector_from(p1, p2):
    return normalize_vector((p2[0] - p1[0], p2[1] - p1[1]))

def angle_between(v1, v2):
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    if mag1 == 0 or mag2 == 0:
        return 0.0
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cos_angle = dot / (mag1 * mag2)
    cos_angle = min(1.0, max(-1.0, cos_angle))
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    if (v1[0] * v2[1] - v1[1] * v2[0] > 0):
        return angle_deg
    else:
        return -angle_deg

# ========== 文件夹批量处理并输出Excel =============
def process_folder(img_folder, c2c7_model_path, leftright_model_path, output_excel):
    global c2c7_model
    from pathlib import Path
    img_folder = Path(img_folder)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.dcm', '.DCM']
    image_files = []
    for ext in image_extensions:
        image_files.extend(img_folder.glob(f"*{ext}"))
        image_files.extend(img_folder.glob(f"*{ext.upper()}"))
    image_files = list(set(image_files))  # 去重，防止同名文件重复
    
    if not image_files:
        print("未找到图片")
        return
    
    # 加载模型
    
    leftright_model = YOLO(leftright_model_path)
    
    angle_keys = ["C2-3", "C3-4", "C4-5", "C5-6", "C6-7", "C2-7"]
    all_results = {}
    save_dir = Path(img_folder / "results")
    save_dir.mkdir(exist_ok=True)
    
    print(f"找到 {len(image_files)} 张图片，开始处理...")
    
    for img_file in image_files:
        print(f"处理图片: {img_file}")
        try:
            # 检查是否为DICOM文件，如果是则先转换
            if img_file.suffix.lower() in ['.dcm']:
                png_file = save_dir / f"{img_file.stem}.png"
                if dicom_to_png_standard(img_file, png_file):
                    img_file = png_file  # 使用转换后的PNG文件
                    print(f"DICOM文件已转换为: {png_file}")
                else:
                    print(f"DICOM转换失败，跳过: {img_file}")
                    continue
            
            img = cv2.imread(str(img_file))
            h_img, w_img = img.shape[:2]
            
            # 使用c2_c7模型进行检测
            results = c2c7_model.predict(str(img_file), conf=0.5, iou=0.7)
            result = results[0]
            names = result.names
            print(f"检测到类别: {names}")
            
            if not hasattr(result, 'masks') or result.masks is None:
                print("未检测到分割掩码")
                all_results[img_file.name] = {k: "null" for k in angle_keys}
                # 保存可视化图片（无点）
                cv2.imwrite(str(save_dir / img_file.name), img)
                continue
            
            masks = result.masks.data.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            # 检查是否检测到所有必需的椎体
            detected_classes = {names[cls_id] for cls_id in classes}
            if not needed_check.issubset(detected_classes):
                print(f"未检测到全部必需椎体 {needed_check}，当前检测到: {detected_classes}，跳过")
                all_results[img_file.name] = {k: "null" for k in angle_keys}
                # 保存可视化图片（无点）
                cv2.imwrite(str(save_dir / img_file.name), img)
                continue
            
            # 获取C3C4C5的组合边界框
            c3c4c5_box = get_combined_box_c3c4c5(masks, classes, names, img.shape)
            if c3c4c5_box is None:
                print("无法获取C3C4C5组合框")
                all_results[img_file.name] = {k: "null" for k in angle_keys}
                cv2.imwrite(str(save_dir / img_file.name), img)
                continue
            
            # 使用leftright模型判断方向
            direction = judge_left_right_by_model(img, c3c4c5_box, leftright_model)
            print(f"模型判断方向: {direction}")
            
            # 如果leftright模型置信度过低，跳过此图
            if direction == 'skip':
                print("leftright模型置信度过低，跳过此图")
                all_results[img_file.name] = {k: "null" for k in angle_keys}
                cv2.imwrite(str(save_dir / img_file.name), img)
                continue
            
            vertebrae_points = {}
            for mask, cls_id in zip(masks, classes):
                class_name = names[cls_id]
                if class_name not in vertebrae_classes:
                    continue
                
                mask_resized = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
                mask_uint8 = (mask_resized > 0.5).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if not contours:
                    continue
                
                cnt = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(cnt)
                hull_points = hull.reshape(-1, 2)
                best_edge = find_main_edge_cross_centroid(hull_points, class_name)
                
                if best_edge is not None:
                    p1, p2 = best_edge
                    # 按方向区分_1/_2
                    if direction == 'left':
                        if p1[0] < p2[0]:
                            vertebrae_points[f'{class_name}_1'] = tuple(map(int, p1))
                            vertebrae_points[f'{class_name}_2'] = tuple(map(int, p2))
                        else:
                            vertebrae_points[f'{class_name}_1'] = tuple(map(int, p2))
                            vertebrae_points[f'{class_name}_2'] = tuple(map(int, p1))
                    elif direction == 'right':
                        if p1[0] > p2[0]:
                            vertebrae_points[f'{class_name}_1'] = tuple(map(int, p1))
                            vertebrae_points[f'{class_name}_2'] = tuple(map(int, p2))
                        else:
                            vertebrae_points[f'{class_name}_1'] = tuple(map(int, p2))
                            vertebrae_points[f'{class_name}_2'] = tuple(map(int, p1))
                    else:
                        vertebrae_points[f'{class_name}_1'] = tuple(map(int, p1))
                        vertebrae_points[f'{class_name}_2'] = tuple(map(int, p2))
            
            # 可视化并保存图片
            img_vis = img.copy()
            for vertebra in range(2, 8):
                pt1 = vertebrae_points.get(f'C{vertebra}_1')
                pt2 = vertebrae_points.get(f'C{vertebra}_2')
                # 画点和标注
                for idx, pt in enumerate([pt1, pt2], 1):
                    if pt is not None:
                        cv2.circle(img_vis, pt, 2, (0, 0, 255), -1)
                        cv2.putText(img_vis, f'C{vertebra}_{idx}', (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                # 画线
                if pt1 is not None and pt2 is not None:
                    cv2.line(img_vis, pt1, pt2, (255, 0, 0), 2)
            
            # 绘制C3C4C5组合框用于调试
            if c3c4c5_box:
                cv2.rectangle(img_vis, (c3c4c5_box[0], c3c4c5_box[1]), (c3c4c5_box[2], c3c4c5_box[3]), (255, 255, 0), 2)
                cv2.putText(img_vis, 'C3C4C5 Combined', (c3c4c5_box[0], c3c4c5_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imwrite(str(save_dir / img_file.name), img_vis)
            
            # 角度计算与结果保存
            # 每个角度单独判断，能算的就算，不能算的为null
            pairs = {
                "C2-3": ([vertebrae_points.get('C2_1'), vertebrae_points.get('C2_2')], [vertebrae_points.get('C3_1'), vertebrae_points.get('C3_2')]),
                "C3-4": ([vertebrae_points.get('C3_1'), vertebrae_points.get('C3_2')], [vertebrae_points.get('C4_1'), vertebrae_points.get('C4_2')]),
                "C4-5": ([vertebrae_points.get('C4_1'), vertebrae_points.get('C4_2')], [vertebrae_points.get('C5_1'), vertebrae_points.get('C5_2')]),
                "C5-6": ([vertebrae_points.get('C5_1'), vertebrae_points.get('C5_2')], [vertebrae_points.get('C6_1'), vertebrae_points.get('C6_2')]),
                "C6-7": ([vertebrae_points.get('C6_1'), vertebrae_points.get('C6_2')], [vertebrae_points.get('C7_1'), vertebrae_points.get('C7_2')]),
                "C2-7": ([vertebrae_points.get('C2_1'), vertebrae_points.get('C2_2')], [vertebrae_points.get('C7_1'), vertebrae_points.get('C7_2')]),
            }
            
            angles = {}
            for junction, pair in pairs.items():
                if None in pair[0] or None in pair[1]:
                    angles[junction] = "null"
                else:
                    v1 = vector_from(pair[0][0], pair[0][1])
                    v2 = vector_from(pair[1][0], pair[1][1])
                    angle = angle_between(v1, v2)
                    angles[junction] = angle
            
            all_results[img_file.name] = angles
            
        except Exception as e:
            print(f"{img_file.name} 处理出错: {e}")
            all_results[img_file.name] = {k: "null" for k in angle_keys}
            continue
    
    # 汇总写入Excel
    if not all_results:
        print("没有有效结果，未写入Excel。")
        return
    
    # 创建Excel文件夹
    excel_dir = Path("Excel")
    excel_dir.mkdir(exist_ok=True)
    
    # 构建完整的Excel文件路径
    excel_path = excel_dir / output_excel
    
    df = pd.DataFrame.from_dict(all_results, orient='index')
    df.index.name = 'Image'
    df.reset_index(inplace=True)
    df.to_excel(str(excel_path), index=False)
    print(f"已写入Excel: {excel_path}")

def main(img_path, c2c7_model_path, leftright_model_path, show_image=True):
    # 加载模型
    global c2c7_model
    leftright_model = YOLO(leftright_model_path)
    
    img = cv2.imread(img_path)
    h_img, w_img = img.shape[:2]
    
    # 使用c2_c7模型进行检测
    results = c2c7_model.predict(img_path, conf=0.5, iou=0.7)
    result = results[0]
    names = result.names
    
    if not hasattr(result, 'masks') or result.masks is None:
        print('未检测到分割掩码')
        return
    
    masks = result.masks.data.cpu().numpy()  # (N, H, W)
    classes = result.boxes.cls.cpu().numpy().astype(int)
    
    # 检查是否检测到所有必需的椎体
    detected_classes = {names[cls_id] for cls_id in classes}
    print(f"检测到的类别: {detected_classes}")
    
    if not needed_check.issubset(detected_classes):
        print(f"未检测到全部必需椎体 {needed_check}，跳过处理")
        return
    
    # 获取C3C4C5的组合边界框
    c3c4c5_box = get_combined_box_c3c4c5(masks, classes, names, img.shape)
    if c3c4c5_box is None:
        print("无法获取C3C4C5组合框")
        return
    
    print(f"C3C4C5组合框: {c3c4c5_box}")
    
    # 使用leftright模型判断方向
    direction = judge_left_right_by_model(img, c3c4c5_box, leftright_model)
    print(f"模型判断方向: {direction}")
    
    # 如果leftright模型置信度过低，跳过处理
    if direction == 'skip':
        print("leftright模型置信度过低，跳过处理")
        return
    
    # 记录每个椎体的主边
    vertebrae_points = {}
    for mask, cls_id in zip(masks, classes):
        confidence = result.boxes.conf[cls_id].cpu().numpy()
        class_name = names[cls_id]
        print(f"检测到 {class_name}，置信度: {confidence:.2f}")
        
        if class_name not in vertebrae_classes:
            continue
        
        mask_resized = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        mask_uint8 = (mask_resized > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        hull_points = hull.reshape(-1, 2)
        best_edge = find_main_edge_cross_centroid(hull_points, class_name)
        
        #绘制凸包
        cv2.polylines(img, [hull_points], isClosed=True, color=(0, 255, 0), thickness=1)
        #绘制质心
        M = cv2.moments(hull_points)
        if M['m00'] != 0:   
            centroid_x = int(M['m10'] / M['m00'])
            centroid_y = int(M['m01'] / M['m00'])
            cv2.circle(img, (centroid_x, centroid_y), 3, (255, 0, 0), -1)
            cv2.putText(img, f'{class_name} centroid', (centroid_x + 5, centroid_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        
        if best_edge is not None:
            p1, p2 = best_edge
            # 按方向区分_1/_2
            if direction == 'left':
                if p1[0] < p2[0]:
                    vertebrae_points[f'{class_name}_1'] = tuple(map(int, p1))
                    vertebrae_points[f'{class_name}_2'] = tuple(map(int, p2))
                else:
                    vertebrae_points[f'{class_name}_1'] = tuple(map(int, p2))
                    vertebrae_points[f'{class_name}_2'] = tuple(map(int, p1))
            elif direction == 'right':
                if p1[0] > p2[0]:
                    vertebrae_points[f'{class_name}_1'] = tuple(map(int, p1))
                    vertebrae_points[f'{class_name}_2'] = tuple(map(int, p2))
                else:
                    vertebrae_points[f'{class_name}_1'] = tuple(map(int, p2))
                    vertebrae_points[f'{class_name}_2'] = tuple(map(int, p1))
            else:
                vertebrae_points[f'{class_name}_1'] = tuple(map(int, p1))
                vertebrae_points[f'{class_name}_2'] = tuple(map(int, p2))
    
    # 打印所有点
    for name in [f'C{i}_{j}' for i in range(2,8) for j in (1,2)]:
        print(f"{name}: {vertebrae_points.get(name)}")

    # 在原图上绘制点和标注
    img_vis = img.copy()
    for name in [f'C{i}_{j}' for i in range(2,8) for j in (1,2)]:
        pt = vertebrae_points.get(name)
        if pt is not None:
            cv2.circle(img_vis, pt, 2, (0,0,255), -1)
            
            cv2.putText(img_vis, name, (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    
    # 绘制C3C4C5组合框用于调试
    if c3c4c5_box:
        cv2.rectangle(img_vis, (c3c4c5_box[0], c3c4c5_box[1]), (c3c4c5_box[2], c3c4c5_box[3]), (255, 255, 0), 2)
        cv2.putText(img_vis, 'C3C4C5 Combined', (c3c4c5_box[0], c3c4c5_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # 根据参数决定是否显示图像窗口
    if show_image:
        #保存为临时图片。使用系统自带软件打开
        temp_image_path = "temp_result.png"
        cv2.imwrite(temp_image_path, img_vis)
        os.startfile(temp_image_path)   

    # ========== 角度计算与写入Excel =============
    # 只在所有点都存在时才计算，否则写入null
    required_points = [f'C{i}_{j}' for i in range(2,8) for j in (1,2)]
    angle_keys = ["C2-3", "C3-4", "C4-5", "C5-6", "C6-7", "C2-7"]
    if all(k in vertebrae_points for k in required_points):
        pairs = {
            "C2-3": ([vertebrae_points['C2_1'], vertebrae_points['C2_2']], [vertebrae_points['C3_1'], vertebrae_points['C3_2']]),
            "C3-4": ([vertebrae_points['C3_1'], vertebrae_points['C3_2']], [vertebrae_points['C4_1'], vertebrae_points['C4_2']]),
            "C4-5": ([vertebrae_points['C4_1'], vertebrae_points['C4_2']], [vertebrae_points['C5_1'], vertebrae_points['C5_2']]),
            "C5-6": ([vertebrae_points['C5_1'], vertebrae_points['C5_2']], [vertebrae_points['C6_1'], vertebrae_points['C6_2']]),
            "C6-7": ([vertebrae_points['C6_1'], vertebrae_points['C6_2']], [vertebrae_points['C7_1'], vertebrae_points['C7_2']]),
            "C2-7": ([vertebrae_points['C2_1'], vertebrae_points['C2_2']], [vertebrae_points['C7_1'], vertebrae_points['C7_2']]),
        }
        angles = {}
        for junction, (points1, points2) in pairs.items():
            v1 = vector_from(points1[0], points1[1])
            v2 = vector_from(points2[0], points2[1])
            angle = angle_between(v1, v2)
            angles[junction] = angle
        print("角度计算结果：")
        for k, v in angles.items():
            print(f"{k}: {v:.2f}")
    else:
        print("有点未检测到，写入null。")
        angles = {k: "null"  for k in angle_keys}
    
    # 写入Excel（无论是否识别成功都写一行）
    # 创建Excel文件夹
    excel_dir = Path("Excel")
    excel_dir.mkdir(exist_ok=True)
    
    # 构建完整的Excel文件路径
    excel_path = excel_dir / 'angles_new.xlsx'
    
    df = pd.DataFrame([angles])
    df.to_excel(str(excel_path), index=False)
    print(f"角度已写入 {excel_path}")

if __name__ == "__main__":
    c2c7_model_path = "te.pt"
    leftright_model_path = "leftright.pt"
    # 单张图片处理
    img_path = 'D:\\NewSystemData\\0.AllCode\\PythonCode\\jizhuiw8924\\Dicom\\output_standard\\18.png'  # 可修改
    #main(img_path, c2c7_model_path, leftright_model_path)
    #exit(0)
    
    # 文件夹批量处理
    img_folder = "D:\\NewSystemData\\0.AllCode\\PythonCode\\jizhuiw8924\\png\\png"  # 修改为你的图片文件夹
    output_excel = "new_png_compressed_resultsC7.xlsx"  # 输出Excel路径
    process_folder(img_folder, c2c7_model_path, leftright_model_path, output_excel)
