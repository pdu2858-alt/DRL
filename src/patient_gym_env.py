import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
import joblib
from typing import Tuple, Dict, Any, Optional
from train_env_model import EnvDynamicsModel
from config import VIRTUAL_PATIENT_MODEL_PATH, SCALER_X_PATH, SCALER_Y_PATH
from config import EPISODE_MAX_STEPS, BIS_TARGET_MIN, BIS_TARGET_MAX, BIS_PUNISH_AWAKE_THRESHOLD, PROP_RATE_PUNISH_THRESHOLD

class AnesthesiaEnv(gym.Env):
    def __init__(self, 
                 model_path: str = VIRTUAL_PATIENT_MODEL_PATH, 
                 scaler_x_path: str = SCALER_X_PATH, 
                 scaler_y_path: str = SCALER_Y_PATH):
        super(AnesthesiaEnv, self).__init__()
        
        self.device = torch.device('cpu')
        self.model = EnvDynamicsModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.scaler_X = joblib.load(scaler_x_path)
        self.scaler_Y = joblib.load(scaler_y_path)
        
        self.action_space = spaces.Box(low=0, high=200, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        
        self.state: Optional[np.ndarray] = None
        self.steps: int = 0
        self.max_steps: int = EPISODE_MAX_STEPS

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.state = np.array([90.0, 25.0], dtype=np.float32)
        self.steps = 0
        return self.state, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.state is None:
            raise ValueError("Environment must be reset before calling step()")
            
        input_data = np.hstack((self.state, action)).reshape(1, -1)
        input_scaled = self.scaler_X.transform(input_data)
        input_tensor = torch.FloatTensor(input_scaled).to(self.device)
        
        with torch.no_grad():
            pred_scaled = self.model(input_tensor).cpu().numpy()
        
        next_state = self.scaler_Y.inverse_transform(pred_scaled).flatten()
        
        bis_val = next_state[0]
        
        if BIS_TARGET_MIN <= bis_val <= BIS_TARGET_MAX:
            reward = 10.0 
        elif bis_val > BIS_TARGET_MAX:
            reward = - (bis_val - BIS_TARGET_MAX) * 0.5 
            if bis_val > BIS_PUNISH_AWAKE_THRESHOLD and action[0] < PROP_RATE_PUNISH_THRESHOLD:
                reward -= 10.0 
        else:
            reward = - (BIS_TARGET_MIN - bis_val) * 0.8
            
        self.state = next_state
        self.steps += 1
        
        terminated = self.steps >= self.max_steps
        truncated = False
        
        return self.state, float(reward), terminated, truncated, {}