# https://stable-baselines3.readthedocs.io/en/master/
import os
import stable_baselines3
from gymnasium.wrappers import RecordEpisodeStatistics
from callback import TqdmProgressBarCallback, SaveInfoCallback  
from stable_baselines3.common.callbacks import CallbackList

from env import HEV

start_time = 0
step_size = 1 # 1초단위
config = None
stop_time = 1800
total_timesteps = int(stop_time)
episodes = 1
callbacks = None

## set models
train_cycle = os.path.join("test_cycle", "wltp_1Hz.csv")  # Urban 사이클 파일 경로
env = HEV(start_time=start_time, step_size=step_size, config=config, profile_name=train_cycle)
model = stable_baselines3.SAC("MlpPolicy", env)

## callback setting
progress_callback = TqdmProgressBarCallback(total_timesteps=episodes * total_timesteps)
save_path = os.path.join("episode_data", "episode_info_sac.json")
info_callback = SaveInfoCallback(log_path = save_path)
callback = CallbackList([progress_callback, info_callback])

## do learn
model.learn(total_timesteps=episodes * total_timesteps, callback=callback)
model.save("trained_model")