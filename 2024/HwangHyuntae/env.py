import numpy as np
import pandas as pd
import gymnasium as gym
import wandb

profile_name = 'wltp_1Hz.csv'

class HEV(gym.Env):
    def __init__(self, start_time=0, step_size=1, config=None) -> None:
        super(HEV, self).__init__()
    ###################################################    
    ######### 이 변수명들은 바꾸지 말아주세요 ############
        self.start_time = start_time
        self.step_size = step_size
        self.config = config
        self.vehicle_speed_profile = np.array(pd.read_csv(profile_name))[:,0]
        self.soc_init = 70/100
        self.time_profile = np.arange(self.vehicle_speed_profile.shape[0])
        self.stop_time = self.vehicle_speed_profile.shape[0] - 1
        self.state_init = np.array([self.soc_init, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        self.soc_base = 70/100
        self.actsize = 3
        self.obssize = len(self.state)

        self.time = self.start_time
        self.state = self.state_init
        self.reward = None
        self.done = False
    ##################################################
    ##################################################
    #########필요한 차량 제원 정보들 추가##############
        self.engine_map

        self.motor_map



    ######### state 계산에 필요한 차량 로직들 ##############
    def Engine(self,):
        return

    def Battery(self,):
        return


    def step(self,action):
        



        is_done = lambda time: time >= self.stop_time
        self.time += self.step_size
        done = is_done(self.time)
        return self.state, self.reward, self.done,


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.state = self.state_init
        self.time = self.start_time
        return self.state,