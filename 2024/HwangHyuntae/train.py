# https://stable-baselines3.readthedocs.io/en/master/
import stable_baselines3
from gymnasium.wrappers import RecordEpisodeStatistics
from callback import TqdmProgressBarCallback, SaveInfoCallback  
from stable_baselines3.common.callbacks import CallbackList

from env import HEV

start_time = 0
step_size = 1
config = None
stop_time = 1800
total_timesteps = int(stop_time)
episodes = 1
callbacks = None


env = HEV(start_time=start_time, step_size=step_size, config=config,)
model = stable_baselines3.SAC("MlpPolicy", env)

progress_callback = TqdmProgressBarCallback(total_timesteps=episodes * total_timesteps)
info_callback = SaveInfoCallback(log_path="episode_info.json")
callback = CallbackList([progress_callback, info_callback])

model.learn(total_timesteps=episodes * total_timesteps, callback=callback)
model.save("trained_model")