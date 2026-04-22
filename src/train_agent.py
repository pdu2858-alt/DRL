import logging
from stable_baselines3 import SAC
from patient_gym_env import AnesthesiaEnv
from config import AGENT_TOTAL_TIMESTEPS, AGENT_MODEL_PATH
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_sac_agent() -> None:
    env = AnesthesiaEnv()
    
    model = SAC("MlpPolicy", env, verbose=1, device="cpu")
    
    logging.info("開始訓練強化學習代理人...")
    model.learn(total_timesteps=AGENT_TOTAL_TIMESTEPS)
    
    os.makedirs(os.path.dirname(AGENT_MODEL_PATH), exist_ok=True)
    model.save(AGENT_MODEL_PATH)
    logging.info(f"🎉 Agent 訓練完成並儲存至 {AGENT_MODEL_PATH}")

if __name__ == "__main__":
    train_sac_agent()