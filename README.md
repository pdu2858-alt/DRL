# 深度強化學習自動麻醉深度控制 (Automated Anesthesia Control via Deep Reinforcement Learning)

這是一個基於深度強化學習（Deep Reinforcement Learning, DRL）的醫療應用專案，旨在訓練一個 AI 代理人（Agent）來自動控制病患在手術過程中的麻醉深度。

本專案利用 VitalDB 開放醫療資料庫的真實手術數據，首先訓練一個「虛擬病患模型」來模擬人體對麻醉藥物的反應，接著在這個虛擬環境中使用 Soft Actor-Critic (SAC) 演算法訓練強化學習代理人，使其學會根據病患當前的腦電雙頻指數（BIS）自動調整異丙酚（Propofol）的給藥速率，以將麻醉深度精準維持在安全的目標區間。

## 系統架構 (System Architecture)

```text
[ 階段一：資料工程 Phase 1 ]
  (VitalDB 雲端資料庫) 
           ↓ API 批次下載
[ 1. 離線經驗池 (Offline Experience Buffer) ] ── (src/download_dataset.py)
  - 特徵萃取：BIS, SEF, Propofol Rate
  - 時間對齊與插值 (1 Hz)
  - 產出：3300+ 獨立病患 CSV 檔
           ↓
[ 階段二：虛擬環境建構 Phase 2 ]
[ 2. 病患動力學模型 (Environment Dynamics Model) ] ── (src/train_env_model.py)
  - 監督式學習 (MLP 神經網路)
  - 特徵標準化 (StandardScaler)
  - 輸入 (S_t, A_t) -> 預測下一秒狀態 (S_t+1)
  - 產出：virtual_patient_model.pth
           ↓
[ 階段三：強化學習代理人 Phase 3 ]
[ 3. 模擬手術室環境介面 (Gymnasium Wrapper) ] ── (src/patient_gym_env.py)
  - 定義狀態空間與連續動作空間
  - 設計「胡蘿蔔與大棒」獎勵機制 (Reward Shaping)
           ↓ 🔄 互動試錯 (Interaction Loop)
[ 4. 決策代理人訓練 (Agent Optimization) ] ── (src/train_agent.py)
  - 演算法：Soft Actor-Critic (SAC)
  - 離線策略學習 (150,000 Time Steps)
  - 產出：sac_anesthesia_agent.zip
           ↓
[ 階段四：成效驗證 Phase 4 ]
[ 5. 效能評估與視覺化 (Evaluation & Visualization) ] ── (src/evaluate_agent.py)
  - 影子測試 (Shadow Testing)
  - 麻醉深度軌跡與給藥速率對比
  - 產出：rl_result_demo.png
```

## 專案架構與檔案說明

專案主要包含以下幾個階段與對應的程式碼檔案：

### 1. 資料收集與前處理
*   **`download_dataset.py`**: 使用 `vitaldb` 套件從公開資料庫下載包含特定特徵（BIS/BIS, BIS/SEF, Orchestra/PPF20_RATE）的真實病例資料，以 1Hz 進行重採樣，清理缺失值後存入 `vitaldb_rl_dataset/` 目錄中，建立用於訓練的離線數據集。
*   **`final.py`**: 用於測試與驗證 VitalDB API 單筆資料下載與基本資料清理的雛形腳本。

### 2. 虛擬病患模型訓練 (Environment Dynamics Model)
*   **`train_env_model.py`**: 由於無法直接在真實病患身上進行強化學習試錯，此腳本利用收集到的真實手術數據，訓練一個多層感知機（MLP）作為環境動力學模型（Dynamics Model）。
    *   **輸入 (Input)**: 狀態 $S_t$ (當下 BIS, SEF) 與 動作 $A_t$ (Propofol 給藥速率)
    *   **輸出 (Output)**: 下一秒的狀態 $S_{t+1}$ (預測的下一秒 BIS, SEF)
    *   訓練完成的模型會被儲存為 `virtual_patient_model.pth`，同時會儲存資料標準化的 Scaler (`scaler_X.pkl`, `scaler_Y.pkl`) 供後續環境使用。

### 3. 強化學習環境建置
*   **`patient_gym_env.py`**: 將上述訓練好的「虛擬病患模型」封裝成標準的 OpenAI Gymnasium 環境 (`AnesthesiaEnv`)。
    *   **Observation Space (狀態空間)**: `[BIS, SEF]`
    *   **Action Space (動作空間)**: 異丙酚給藥速率 (0 到 200 mL/hr)
    *   **Reward Function (獎勵函數)**: 設計了精細的獎勵機制。目標是將 BIS 維持在 40~60 之間（給予高獎勵），若大於 60（太清醒）或小於 40（麻醉太深）則給予不同程度的懲罰。特別加入了防擺爛機制，嚴懲病患極度清醒時仍不給藥的行為。

### 4. 代理人訓練與評估
*   **`train_agent.py`**: 使用 Stable Baselines3 函式庫中的 **SAC (Soft Actor-Critic)** 演算法，在我們自建的 `AnesthesiaEnv` 虛擬環境中進行訓練，讓 Agent 從與虛擬病患的互動中學習最佳給藥策略。訓練後的模型存為 `sac_anesthesia_agent.zip`。
*   **`evaluate_agent.py`**: 載入訓練好的 SAC Agent，進行一場模擬手術的推論測試。將 Agent 控制下的 BIS 變化趨勢以及對應的 Propofol 給藥速率繪製成圖表 (`rl_result_demo.png`)，以視覺化驗證 Agent 將 BIS 穩定控制在 40~60 安全區間的能力。

## 執行流程建議

1.  **下載資料**: 執行 `python download_dataset.py` 建立資料集。
2.  **訓練環境模型**: 執行 `python train_env_model.py` 訓練虛擬病患。
3.  **訓練 RL Agent**: 執行 `python train_agent.py` 在虛擬環境中訓練控制策略。
4.  **成效評估**: 執行 `python evaluate_agent.py` 產出視覺化成果圖。



## 技術棧

*   Python
*   PyTorch (建立與訓練虛擬病患神經網路)
*   Gymnasium (建構強化學習環境)
*   Stable Baselines3 (SAC 強化學習演算法)
*   VitalDB (醫療開源數據獲取)
*   Pandas, NumPy, Scikit-Learn (資料處理與特徵工程)
*   Matplotlib (結果視覺化)
