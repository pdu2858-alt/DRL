## ADDED Requirements

### Requirement: Standard Project Directory Structure
The project MUST have a standardized directory structure to separate source code, models, and data.

#### Scenario: Script execution after restructuring
- **WHEN** 源代碼檔案從 `src/` 目錄執行時
- **THEN** 程式能夠正確讀取或寫入位於 `models/` 目錄下的模型檔案（`.pth`, `.zip`, `.pkl` 等），以及 `vitaldb_rl_dataset/` 下的資料。

### Requirement: Useless test scripts are removed
The useless testing scripts MUST be removed from the core project directory.

#### Scenario: Clean core repository
- **WHEN** 檢查專案根目錄或 `src/` 目錄時
- **THEN** 原本作為單筆測試用的 `final.py` 不應存在。

### Requirement: Dependency Management
The project MUST provide a requirements.txt file listing all dependencies.

#### Scenario: Install dependencies
- **WHEN** 使用者執行 `pip install -r requirements.txt` 時
- **THEN** 所有的必要套件 (vitaldb, pandas, torch, gymnasium, stable_baselines3, scikit-learn, matplotlib) 都能正確被安裝。
