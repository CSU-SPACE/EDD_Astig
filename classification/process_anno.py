import os
import json
import pandas as pd
import math
import re
import numpy as np
import cv2
import argparse
import sys

# --- Part 1: 几何计算辅助函数 (无变动) ---


def euclidean_distance(p1, p2):
    """计算两点间的欧氏距离。"""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_angle_with_vertical(start_point, end_point):
    """计算向量与垂直向下方向的夹角（度）。"""
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    magnitude = math.sqrt(dx**2 + dy**2)
    if magnitude == 0:
        return 0.0
    cos_theta = max(-1.0, min(1.0, dy / magnitude))
    return math.degrees(math.acos(cos_theta))


def calculate_angle_between_lines(p1, p2, p3, p4):
    """计算两条线段之间的锐角（度）。"""
    dx1, dy1 = p2[0] - p1[0], p2[1] - p1[1]
    dx2, dy2 = p4[0] - p3[0], p4[1] - p3[1]
    angle1_rad, angle2_rad = math.atan2(dy1, dx1), math.atan2(dy2, dx2)
    angle_diff_deg = abs(math.degrees(angle1_rad - angle2_rad))
    angle_diff_deg = min(angle_diff_deg, 360 - angle_diff_deg)
    return min(angle_diff_deg, 180 - angle_diff_deg)


def fit_circle_to_points(points):
    """根据一组二维点拟合最小外接圆，返回圆心。"""
    if not points or len(points) < 3:
        return None
    contour = np.array(points, dtype=np.float32)
    (x, y), _ = cv2.minEnclosingCircle(contour)
    return (float(x), float(y))


# --- Part 2: 主处理函数 (已更新) ---


def generate_comprehensive_report(table_path, annotations_folder, output_path):
    """
    读取表格和统一的标注文件夹，计算各项指标，并生成一份综合报告。
    """
    # 1. 读取基础表格
    try:
        df = (
            pd.read_excel(table_path)
            if table_path.lower().endswith((".xlsx", ".xls"))
            else pd.read_csv(table_path)
        )
    except FileNotFoundError:
        print(f"错误: 基础表格文件 '{table_path}' 未找到。")
        return

    id_column_name = df.columns[0]
    df[id_column_name] = df[id_column_name].astype(str)
    df.set_index(id_column_name, inplace=True)
    print(f"使用 '{id_column_name}' 列作为ID进行匹配。")

    # 2. 初始化所有新列
    new_columns = [
        "左眼睫毛下垂角度",
        "右眼睫毛下垂角度",
        "下睑缘相对距离",
        "外眦点相对距离",
        "角膜中心相对距离",
        "下睑缘与内眦连线夹角",
        "角膜中心与内眦连线夹角",
    ]
    for col in new_columns:
        df[col] = np.nan

    # 3. 在单个文件夹中处理所有标注文件
    print("\n--- 开始处理标注数据 ---")
    if not os.path.isdir(annotations_folder):
        print(f"错误: 标注文件夹 '{annotations_folder}' 不存在。")
        return

    file_list = [f for f in os.listdir(annotations_folder) if f.endswith(".json")]
    total_files = len(file_list)
    print(f"在文件夹中找到 {total_files} 个JSON文件，开始处理...")

    for i, filename in enumerate(file_list):
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split(".")

        # 文件名必须是 'ID.类型' 格式，例如 '1.1' 或 '123.4'
        if len(parts) != 2:
            continue

        file_id, file_type = parts[0], parts[1]

        if file_id not in df.index:
            continue

        sys.stdout.write(f"\r正在处理: {i+1}/{total_files} ({filename})")
        sys.stdout.flush()

        try:
            full_path = os.path.join(annotations_folder, filename)
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # --- 根据文件类型进行分发处理 ---

            # A. 处理睫毛文件
            if file_type in ["4", "5"]:
                points = {
                    s.get("label"): s.get("points")[0]
                    for s in data.get("shapes", [])
                    if s.get("points")
                }
                if "start_eyelash" in points and "end_eyelash" in points:
                    angle = calculate_angle_with_vertical(
                        points["start_eyelash"], points["end_eyelash"]
                    )
                    col = "左眼睫毛下垂角度" if file_type == "4" else "右眼睫毛下垂角度"
                    df.loc[file_id, col] = round(angle, 2)

            # B. 处理综合特征文件
            elif file_type == "1":
                shapes = data.get("shapes", [])
                points = {
                    s["label"]: s["points"][0]
                    for s in shapes
                    if s.get("shape_type") == "point" and s.get("points")
                }
                cornea_polygons = [
                    s["points"]
                    for s in shapes
                    if s.get("label") == "cornea" and s.get("shape_type") == "polygon"
                ]

                required_points = [
                    "left_inner",
                    "right_inner",
                    "left_lower",
                    "right_lower",
                    "left_outer",
                    "right_outer",
                ]
                if not all(p in points for p in required_points):
                    continue

                base_dist = euclidean_distance(
                    points["left_inner"], points["right_inner"]
                )
                if base_dist == 0:
                    continue

                if len(cornea_polygons) == 2:
                    # 注意：这里假设X坐标越大是左眼（图像坐标系），如果反了需要调整 > 为 <
                    center_x1 = np.mean([p[0] for p in cornea_polygons[0]])
                    center_x2 = np.mean([p[0] for p in cornea_polygons[1]])

                    left_poly = (
                        cornea_polygons[0]
                        if center_x1 > center_x2
                        else cornea_polygons[1]
                    )
                    right_poly = (
                        cornea_polygons[1]
                        if center_x1 > center_x2
                        else cornea_polygons[0]
                    )

                    points["left_center"] = fit_circle_to_points(left_poly)
                    points["right_center"] = fit_circle_to_points(right_poly)

                if "left_center" in points and "right_center" in points:
                    center_dist = euclidean_distance(
                        points["left_center"], points["right_center"]
                    )
                    df.loc[file_id, "角膜中心相对距离"] = round(
                        center_dist / base_dist, 4
                    )
                    angle_center = calculate_angle_between_lines(
                        points["left_center"],
                        points["right_center"],
                        points["left_inner"],
                        points["right_inner"],
                    )
                    df.loc[file_id, "角膜中心与内眦连线夹角"] = round(angle_center, 2)

                lower_dist = euclidean_distance(
                    points["left_lower"], points["right_lower"]
                )
                outer_dist = euclidean_distance(
                    points["left_outer"], points["right_outer"]
                )
                angle_lower = calculate_angle_between_lines(
                    points["left_lower"],
                    points["right_lower"],
                    points["left_inner"],
                    points["right_inner"],
                )

                df.loc[file_id, "下睑缘相对距离"] = round(lower_dist / base_dist, 4)
                df.loc[file_id, "外眦点相对距离"] = round(outer_dist / base_dist, 4)
                df.loc[file_id, "下睑缘与内眦连线夹角"] = round(angle_lower, 2)

        except Exception as e:
            print(f"\n处理文件 {filename} 时发生错误: {e}")

    # 4. 保存最终报告
    print("\n\n--- 所有处理完成，正在保存最终报告 ---")
    df.reset_index(inplace=True)
    try:
        if output_path.lower().endswith(".csv"):
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
        else:
            df.to_excel(output_path, index=False)
        print(f"成功！报告已保存至: '{output_path}'")
    except Exception as e:
        print(f"保存文件失败: {e}")


# --- Part 3: 脚本执行入口 (已更新) ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从统一的眼部标注数据文件夹生成综合分析报告。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-t", "--table", required=True, help="输入的原始表格文件路径 (CSV or Excel)"
    )
    parser.add_argument(
        "-i", "--input", required=True, help="包含所有JSON标注文件的【统一】文件夹路径"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="输出报告的文件路径 (e.g., report.xlsx)"
    )

    args = parser.parse_args()

    generate_comprehensive_report(
        table_path=args.table, annotations_folder=args.input, output_path=args.output
    )
