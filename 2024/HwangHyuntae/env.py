import numpy as np
import pandas as pd
import gymnasium as gym
from HEV_simul import update_vehicle_states
# import wandb

profile_name = 'wltp_1Hz.csv' # wltp cycle (value fixed)

class HEV(gym.Env):
    def __init__(self, start_time=0, step_size=1, config=None) -> None:
        super(HEV, self).__init__()
    ######### 이 변수명들은 바꾸지 말아주세요 ############
        self.start_time = start_time
        self.step_size = step_size
        self.config = config
        self.vehicle_speed_profile = np.array(pd.read_csv(profile_name))[:,0]
        self.soc_init = 70/100
        self.time_profile = np.arange(self.vehicle_speed_profile.shape[0]) ## 1800s steps
        self.stop_time = self.vehicle_speed_profile.shape[0] - 1

        # number of observable states
        # we use soc, fuel_dot
        self.state_init = np.array([self.soc_init, 0], dtype=np.float32)
        self.soc_base = 70/100
        self.actsize = 3 # number of possible acctions
        self.obssize = len(self.state)

        self.time = self.start_time
        self.state = self.state_init
        self.reward = None
        self.done = False

    #########필요한 차량 제원 정보들 추가##############
        self.engine_map     # = effmap (in w_eng, T_eng -> out eff)

        self.motor_map      # = effmap (in w_mot, T_mot -> out eff)


    ######### state 계산에 필요한 차량 로직들 ##############
    # notation - PSI in the paper
    # take w_eng and T_eng as input
    # return m_fuel rate
    def Engine(self, w, T):
        LHV = 44 * 10^6 #LHV of gasoline, 44MJ/kg(while diesel is 42.5 MJ/kg)
        eff = 0.25 # usually 25% for gasoline
        m_dot = (w * T) / (eff * LHV) 
        return m_dot

    #battery model : agm80ah
    class Battery:
        def V_oc(self, SoC_t):
            V_max = 12.85 # @ 100%
            V_min = 11.65 # @ 0&
            current_Voc = V_min + (V_max - V_min) * SoC_t
            return current_Voc
        
        def R_0(self, SoC_t):
            soc_values = [0, 20, 40, 60, 80, 100]  # SoC in percentage
            resistance_values = [10, 8, 6, 5, 4, 3]  # Internal resistance in mΩ
            resistance = np.interp(SoC_t, soc_values, resistance_values)
            return resistance



    # action을 넣어주면 해당 action에 따라 1 time step만큼 모델을 돌림
    # 그 이후 다음 step의 state를 리턴해줌
    def step(self,action):
        # 액션 정의: [T_eng, T_bsg, T_brk]
        T_eng, T_bsg, T_brk = action

        # State unpacking
        ## two goal : SoC, fuel_dot
        SoC, fuel_dot = self.state
        current_vel = self.vehicle_speed_profile[int(self.time)]

        # modeling을 통과시켜서 state update
        SoC, fuel_dot = update_vehicle_states(T_eng, T_bsg, T_brk, SoC, current_vel)

        # reward definition
        # 새로운 state에 대해서 reward를 계산
        soc_reward = - (abs(self.soc_base - SoC)) ** 2 # 멀어질 수록 더 -가 커짐 // 해보고 exp도 가능
        fuel_reward = -fuel_dot # 클수록 안좋음

        new_state = np.array([SoC, fuel_dot], dtype=np.float32)
        self.state = new_state

        reward = soc_reward + fuel_reward

        #--------------------------------- closing phase ------------------------ #
        is_done = lambda time: time >= self.stop_time
        self.time += self.step_size
        done = is_done(self.time)
        return self.state, reward, done,


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.state = self.state_init
        self.time = self.start_time
        return self.state,