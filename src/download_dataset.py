import vitaldb
import pandas as pd
import os
import logging
from tqdm import tqdm
from config import TRACK_NAMES, DATA_DIR

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_data() -> None:
    # 尋找同時包含這些欄位的有效 Case ID
    caseids = vitaldb.find_cases(TRACK_NAMES)
    logging.info(f"總共找到 {len(caseids)} 個符合條件的病例，準備開始批次下載...")

    # 建立一個資料夾來存放所有病例的 CSV 檔
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 開始迴圈下載全部資料
    for caseid in tqdm(caseids, desc="下載與處理進度"):
        
        file_path = os.path.join(DATA_DIR, f"case_{caseid}.csv")
        
        # 防呆機制：斷點續傳
        if os.path.exists(file_path):
            continue
            
        try:
            # 下載單一病例並轉換為 DataFrame，1Hz 採樣
            vf = vitaldb.VitalFile(caseid, TRACK_NAMES)
            df = vf.to_pandas(TRACK_NAMES, 1/1)
            
            # 清除斷訊空值
            df = df.ffill().dropna()
            
            # 確保清理後這個病例還有資料
            if not df.empty:
                # 新增一個 Time_Step 欄位
                df.insert(0, 'Time_Step', range(len(df)))
                
                # 將這個 Episode 存成獨立的檔案
                df.to_csv(file_path, index=False)
                
        except Exception as e:
            # 紀錄錯誤
            logging.error(f"病例 {caseid} 處理失敗: {e}")
            continue

    logging.info("🎉 全部病例下載與清洗完成！")

if __name__ == "__main__":
    download_data()