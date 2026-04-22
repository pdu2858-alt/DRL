import os

# 目錄路徑設定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "vitaldb_rl_dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 模型檔案路徑
VIRTUAL_PATIENT_MODEL_PATH = os.path.join(MODELS_DIR, "virtual_patient_model.pth")
SCALER_X_PATH = os.path.join(MODELS_DIR, "scaler_X.pkl")
SCALER_Y_PATH = os.path.join(MODELS_DIR, "scaler_Y.pkl")
AGENT_MODEL_PATH = os.path.join(MODELS_DIR, "sac_anesthesia_agent")

# VitalDB 下載與特徵設定
TRACK_NAMES = [
    'BIS/BIS',              # State 
    'BIS/SEF',              # State 
    'Orchestra/PPF20_RATE'  # Action 
]

# 環境與訓練超參數
MAX_FILES_FOR_TRAIN = 200     # 訓練環境模型時最多使用的檔案數
ENV_TRAIN_BATCH_SIZE = 1024
ENV_TRAIN_EPOCHS = 20
ENV_TRAIN_LR = 0.001

# 強化學習 Agent 超參數
AGENT_TOTAL_TIMESTEPS = 150000
EPISODE_MAX_STEPS = 300

# 獎勵函數 (Reward Function) 設定
BIS_TARGET_MIN = 40
BIS_TARGET_MAX = 60
BIS_PUNISH_AWAKE_THRESHOLD = 80
PROP_RATE_PUNISH_THRESHOLD = 30
