# configuration-management Specification

## Purpose
TBD - created by archiving change architecture-refinement. Update Purpose after archive.
## Requirements
### Requirement: Centralized Configuration
All hyperparameters, file paths, and constants MUST be centralized in a single configuration module.

#### Scenario: Modifying hyperparameters
- **WHEN** 開發者想要改變 SAC 訓練步數或目標 BIS 安全範圍時
- **THEN** 開發者只需要修改 `src/config.py` 中的對應變數，而不需要去尋找並修改訓練腳本或環境腳本。

### Requirement: Standard Logging System
All program output messages MUST use Python's standard logging system instead of print statements.

#### Scenario: Viewing training progress
- **WHEN** 執行訓練腳本 `train_env_model.py` 或 `train_agent.py` 時
- **THEN** 終端機顯示的訊息必須帶有適當的日誌層級（如 INFO, WARNING, ERROR）與時間戳記。

### Requirement: Static Type Hinting
All custom functions and class methods MUST include Python static type hinting.

#### Scenario: Code maintenance
- **WHEN** 開發者閱讀或修改 `patient_gym_env.py` 等腳本時
- **THEN** 可以清楚地從函式簽名（Signature）中得知輸入參數與返回值的資料型態。

