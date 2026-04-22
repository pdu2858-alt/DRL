import logging
import os
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from patient_gym_env import AnesthesiaEnv
import numpy as np
import matplotlib
from config import AGENT_MODEL_PATH, BIS_TARGET_MAX, BIS_TARGET_MIN, BASE_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
matplotlib.rcParams['axes.unicode_minus'] = False

def evaluate() -> None:
    env = AnesthesiaEnv()
    try:
        model = SAC.load(AGENT_MODEL_PATH, device="cpu")
    except Exception as e:
        logging.error(f"Failed to load agent model from {AGENT_MODEL_PATH}: {e}")
        return

    bis_history = []
    action_history = []

    obs, _ = env.reset()
    bis_history.append(obs[0])

    logging.info("正在進行模擬測試...")
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        bis_history.append(obs[0])
        action_history.append(action[0])
        
        done = terminated or truncated

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    time_steps = np.arange(len(bis_history))

    ax1.plot(time_steps, bis_history, color='blue', linewidth=2, label='Agent 控制的 BIS 值')
    ax1.axhline(y=BIS_TARGET_MAX, color='red', linestyle='--', alpha=0.5, label=f'安全上限 ({BIS_TARGET_MAX})')
    ax1.axhline(y=BIS_TARGET_MIN, color='red', linestyle='--', alpha=0.5, label=f'安全下限 ({BIS_TARGET_MIN})')
    ax1.fill_between(time_steps, BIS_TARGET_MIN, BIS_TARGET_MAX, color='green', alpha=0.1, label='理想標靶區間')
    ax1.set_title('深度強化學習 Agent 麻醉深度控制表現', fontsize=14)
    ax1.set_ylabel('BIS (麻醉深度)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2.plot(time_steps[:-1], action_history, color='orange', linewidth=2, label='Propofol 給藥速率')
    ax2.set_xlabel('時間 (秒)', fontsize=12)
    ax2.set_ylabel('給藥速率 (mL/hr)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(BASE_DIR, 'rl_result_demo.png')
    plt.savefig(output_path, dpi=300)
    logging.info(f"🎉 測試完成！請去資料夾尋找 '{output_path}'，這就是你的期中報告亮點！")

if __name__ == "__main__":
    evaluate()