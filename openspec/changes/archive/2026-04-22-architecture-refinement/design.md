## Context

目前的強化學習自動麻醉深度控制專案為個人學術/研究專案的初期階段，所有的流程（資料收集、環境訓練、強化學習代理人訓練與驗證）皆由獨立的腳本完成。隨著專案複雜度提高，設定（例如：模型存放路徑、目標 BIS 區間、訓練步數等）四處散落在不同的檔案中，導致維護困難。此外，使用大量的 `print()` 不利於記錄長時間的訓練過程。因此，有必要進行架構重構與標準化精進。

## Goals / Non-Goals

**Goals:**
- 將硬編碼的參數提取到集中管理的設定檔 (`config.py`) 中。
- 建立標準的 Python 專案目錄結構 (`src/`, `models/`, `data/`)。
- 將標準輸出日誌替換為 Python 內建的 `logging` 模組。
- 在所有主要的函式與類別宣告中加上 Type Hinting (型別提示)。
- 建立 `requirements.txt` 以管理依賴套件。

**Non-Goals:**
- 改變現有核心邏輯：我們不會修改神經網路的模型架構、環境的獎勵函數公式或是強化學習演算法的邏輯，僅進行架構與工程面的重構。
- 增加新的模型特徵。

## Decisions

- **Configuration Management**: 選擇使用原生的 `src/config.py` 而不是 `yaml` 或 `json`。因為專案相對簡單且完全基於 Python，使用 `config.py` 可以方便導入變數且支援註解，同時避免了引入額外解析器 (如 `PyYAML`) 的麻煩。
- **Directory Structure**: 
  - `src/`: 放置所有執行的 Python 源代碼。
  - `models/`: 放置 `.pth` 模型檔、`.zip` RL 代理人檔與 `.pkl` 正規化工具。
  - `data/`: `vitaldb_rl_dataset/` 將保持在其原本的位址，或者移至 `data/` 下集中管理，為了簡單起見，我們將資料夾預期保留在原位但將相關路徑變數設定在 config 中。
- **Logging**: 使用 Python 內建的 `logging`，設定基本的 `StreamHandler` 將日誌輸出至終端機，並保留未來輸出至 `.log` 檔案的能力。

## Risks / Trade-offs

- **[Risk] Path Issues**: 移動檔案後，可能會發生找不到模型或資料夾的錯誤。
  - **Mitigation**: 所有的路徑必須使用 `config.py` 中的變數，並結合 `os.path` 確保跨平台相容性與相對路徑正確性。
- **[Risk] Module Import Errors**: 將腳本移入 `src/` 後，互相引用的方式需要調整。
  - **Mitigation**: 調整 `import` 敘述，確保在同一個 `src/` 目錄下可以直接互相引用。
