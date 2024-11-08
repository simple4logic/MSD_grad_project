from stable_baselines3 import PPO, SAC

from env import HEV

start_time = 0
step_size = 1
config = None
stop_time = 1800
total_timesteps = int(stop_time)
episodes = 100
callbacks = None


env = HEV(start_time=start_time, step_size=step_size, config=config,)
model = SAC("MlpPolicy", env)
model.learn(total_timesteps = episodes*total_timesteps, callback = callbacks)