## Why

目前的專案是一系列獨立的 Python 腳本，包含了資料處理、虛擬病患模型訓練、強化學習環境建置與代理人評估。這些腳本雖然能夠運作，但在超參數管理、程式碼結構以及型別安全上較為薄弱，不利於後續擴充、維護以及加入更複雜的模型特徵。為了讓專案能夠像一個標準的 AI 產品/研究庫，我們需要引入設定檔管理、模組化目錄結構以及標準日誌。

## What Changes

- 將散落的硬編碼設定（如目標 BIS 上下限、模型路徑、訓練輪數等）集中到單一設定檔模組。
- 將核心邏輯（環境建置、模型定義、訓練腳本）移至 `src/` 目錄。
- 將訓練好的模型檔案集中至 `models/` 目錄。
- 刪除無用的開發測試腳本 `final.py`。
- 加入 `logging` 日誌系統，取代大量的 `print()`。
- 為所有關鍵函式與類別加上型別提示 (Type Hinting)。
- 建立 `requirements.txt` 以管理套件依賴。

## Capabilities

### New Capabilities
- `project-structure`: 定義與建立標準的 Python 專案目錄與檔案結構。
- `configuration-management`: 統一集中管理所有的訓練與環境超參數。

### Modified Capabilities
<!-- No existing capabilities to modify since openspec is just initialized -->

## Impact

- **Affected Code**: `download_dataset.py`, `train_env_model.py`, `patient_gym_env.py`, `train_agent.py`, `evaluate_agent.py` 皆會被修改與搬移。
- **Dependencies**: 會加入標準函式庫 `logging` 與 `typing` 的使用，並明確列出外部套件於 `requirements.txt`。
- **Systems**: 訓練模型與讀取資料的路徑會隨目錄結構改變。
