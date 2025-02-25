import os
import pandas as pd
from odf.opendocument import OpenDocumentSpreadsheet
from odf.table import Table, TableRow, TableCell
from odf.text import P

def rename_images_and_create_ods(image_folder, output_ods_path, prefix="frame_", start_idx=7200, digits=4):
    """
    將資料夾內的圖片重新命名並生成 ODS 文件。
    
    Args:
        image_folder (str): 圖片資料夾路徑。
        output_ods_path (str): ODS 文件輸出路徑。
        prefix (str): 新檔名的前綴。
        start_idx (int): 起始序號。
        digits (int): 序號的位數，例如 4 代表 `0000` 格式。
    """
    # 確保資料夾存在
    if not os.path.exists(image_folder):
        print("圖片資料夾不存在，請檢查路徑！")
        return

    # 獲取圖片檔案清單並排序
    image_files = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if not image_files:
        print("資料夾內沒有找到圖片檔案！")
        return

    # 生成 ODS 文件
    ods = OpenDocumentSpreadsheet()
    table = Table(name="Annotations")

    # 添加表頭
    header = TableRow()
    for column_name in ["time", "action"]:
        cell = TableCell()
        cell.addElement(P(text=column_name))
        header.addElement(cell)
    table.addElement(header)

    # 依序重新命名圖片並填充 ODS 表格
    for idx, image_file in enumerate(image_files):
        new_name = f"{prefix}{str(start_idx + idx).zfill(digits)}.png"  # 新檔名
        old_path = os.path.join(image_folder, image_file)
        new_path = os.path.join(image_folder, new_name)

        # 重命名圖片
        os.rename(old_path, new_path)

        # 添加到 ODS
        row = TableRow()
        for value in [new_name, ""]:  # "action" 欄位初始為空
            cell = TableCell()
            cell.addElement(P(text=value))
            row.addElement(cell)
        table.addElement(row)

    # 將表格添加到文檔
    ods.spreadsheet.addElement(table)

    # 儲存 ODS 檔案
    ods.save(output_ods_path)
    print(f"圖片已重新命名，ODS 文件保存至：{output_ods_path}")

# 設定參數
image_folder = "/home/yuchu/pig/test_frames/"  # 圖片資料夾路徑
output_ods_path = "/home/yuchu/pig/output.ods"  # 輸出的 ODS 文件路徑
prefix = "frame_"  # 檔名前綴
start_idx = 7200  # 起始序號
digits = 4  # 序號位數，例如 0000

# 執行函式
rename_images_and_create_ods(image_folder, output_ods_path, prefix, start_idx, digits)
