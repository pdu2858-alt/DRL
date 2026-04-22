import os
import glob
import logging
from typing import Tuple
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

from config import DATA_DIR, VIRTUAL_PATIENT_MODEL_PATH, SCALER_X_PATH, SCALER_Y_PATH
from config import MAX_FILES_FOR_TRAIN, ENV_TRAIN_BATCH_SIZE, ENV_TRAIN_EPOCHS, ENV_TRAIN_LR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VitalDynamicsDataset(Dataset):
    def __init__(self, data_dir: str = DATA_DIR, max_files: int = MAX_FILES_FOR_TRAIN):
        super().__init__()
        logging.info("正在載入與處理資料...")
        all_files = glob.glob(os.path.join(data_dir, "*.csv"))[:max_files]
        
        X_list, Y_list = [], []
        
        for file in all_files:
            df = pd.read_csv(file)
            if len(df) < 10: continue
            
            states = df[['BIS/BIS', 'BIS/SEF']].values
            actions = df[['Orchestra/PPF20_RATE']].values
            
            current_states = states[:-1]
            current_actions = actions[:-1]
            next_states = states[1:]
            
            X_concat = np.hstack((current_states, current_actions))
            
            X_list.append(X_concat)
            Y_list.append(next_states)
            
        self.X = np.vstack(X_list)
        self.Y = np.vstack(Y_list)
        
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        
        self.X_scaled = self.scaler_X.fit_transform(self.X)
        self.Y_scaled = self.scaler_Y.fit_transform(self.Y)
        
        # 儲存 Scaler 於 models 資料夾
        os.makedirs(os.path.dirname(SCALER_X_PATH), exist_ok=True)
        joblib.dump(self.scaler_X, SCALER_X_PATH)
        joblib.dump(self.scaler_Y, SCALER_Y_PATH)
        
        logging.info(f"資料集準備完成！總共 {len(self.X_scaled)} 筆訓練樣本 (S_t, A_t -> S_t+1)")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.FloatTensor(self.X_scaled[idx]), 
                torch.FloatTensor(self.Y_scaled[idx]))

class EnvDynamicsModel(nn.Module):
    def __init__(self):
        super(EnvDynamicsModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def train_model() -> None:
    device = torch.device('cpu')
    logging.info(f"目前使用的運算裝置: {device}")
    
    dataset = VitalDynamicsDataset() 
    dataloader = DataLoader(dataset, batch_size=ENV_TRAIN_BATCH_SIZE, shuffle=True)
    
    model = EnvDynamicsModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=ENV_TRAIN_LR)
    
    logging.info("\n開始訓練環境動力學模型...")
    for epoch in range(ENV_TRAIN_EPOCHS):
        model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch [{epoch+1}/{ENV_TRAIN_EPOCHS}], MSE Loss: {avg_loss:.6f}")
        
    os.makedirs(os.path.dirname(VIRTUAL_PATIENT_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), VIRTUAL_PATIENT_MODEL_PATH)
    logging.info(f"🎉 模型訓練完成並已儲存為 {VIRTUAL_PATIENT_MODEL_PATH}")

if __name__ == "__main__":
    train_model()