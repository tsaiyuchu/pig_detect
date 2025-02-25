import pandas as pd

def count_actions_in_ods(ods_path):
    """
    統計 ODS 檔案中每個動作的次數。
    
    Args:
        ods_path (str): ODS 檔案路徑。
    
    Returns:
        dict: 動作名稱對應的數量。
    """
    try:
        # 使用 pandas 讀取 ODS 檔案
        df = pd.read_excel(ods_path, engine='odf')
        
        # 確認檔案包含 'action' 欄位
        if 'action' not in df.columns:
            print("ODS 檔案中未找到 'action' 欄位！")
            return {}

        # 統計每個動作的數量
        action_counts = df['action'].value_counts().to_dict()
        return action_counts

    except Exception as e:
        print(f"讀取或統計失敗: {e}")
        return {}

# 使用範例
ods_path = "/home/yuchu/pig/dataset/train.ods"  # 替換為你的檔案路徑
action_counts = count_actions_in_ods(ods_path)

if action_counts:
    print("動作統計結果:")
    for action, count in action_counts.items():
        print(f"{action}: {count}")
else:
    print("無法統計動作數量，請檢查檔案格式或路徑。")
