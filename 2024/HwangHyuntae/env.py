import numpy as np
import pandas as pd
import gymnasium as gym
# from HEV_simul import update_vehicle_states
# import wandb

profile_name = 'wltp_1Hz.csv' # wltp cycle (value fixed)

class MQ4:
    Mass = 1500         # Vehicle mass (kg)
    wheel_R = 0.3       # Wheel radius (m)
    gravity = 9.81      # Gravitational constant (m/s^2)
    drag_coeff = 0.3    # Aerodynamic drag coefficient
    rho_air = 1.225     # Air density (kg/m^3)
    Area_front = 2.5    # Frontal area (m^2)
    roll_coeff = 0.01   # Rolling resistance coefficient
    tau_belt = 1        # belt ratio
    tau_fdr = 1         # final drive ratio
    C_nom = 50000.0     # ??
    # I_bias = 1          # constant current bias
    # alpha = 1           # road grade (slope of the road)  ## time variant, depends on the road condition
    w_stall = 500       # minimum engine speed not to stall
    w_idle = 80         # speed without giving any power


class HEV(gym.Env):
    def __init__(self, start_time=0, step_size=1, config=None) -> None:
        super(HEV, self).__init__()
    #------------------------- step parameter /*DO NOT CHANGE*/ ------------------------ #
        self.start_time = start_time
        self.step_size = step_size
        self.config = config
        self.vehicle_speed_profile = np.array(pd.read_csv(profile_name))[:,0]
        self.soc_init = 70/100
        self.time_profile = np.arange(self.vehicle_speed_profile.shape[0]) ## 1800s steps
        self.stop_time = self.vehicle_speed_profile.shape[0] - 1

        # number of observable states
        # we use soc, fuel_dot, prev_v_veh
        self.state_init = np.array([self.soc_init, 0, 0], dtype=np.float32)
        self.soc_base = 70/100
        self.actsize = 3 # number of possible acctions
        self.obssize = len(self.state)

        self.time = self.start_time
        self.state = self.state_init
        self.reward = None
        self.done = False

    #------------------------- car specification ------------------------ #
        self.car_config = MQ4
        self.engine_map     # = effmap (in w_eng, T_eng -> out eff)
        self.motor_map      # = effmap (in w_mot, T_mot -> out eff)


    #------------------------- car modeling functions and logics ------------------------ #
    #battery model : agm80ah
    def V_oc(SoC_t):
        V_max = 12.85 # @ 100%
        V_min = 11.65 # @ 0&
        current_Voc = V_min + (V_max - V_min) * SoC_t
        return current_Voc
    
    def R_0(SoC_t):
        soc_values = [0, 20, 40, 60, 80, 100]  # SoC in percentage
        resistance_values = [10, 8, 6, 5, 4, 3]  # Internal resistance in mΩ
        resistance = np.interp(SoC_t, soc_values, resistance_values)
        return resistance
    
    # take w_eng and T_eng as input
    # return fuel consumption rate
    def engine_modeling(w, T):
        LHV = 44 * 10^6 #LHV of gasoline, 44MJ/kg(while diesel is 42.5 MJ/kg)
        eff = 0.25 # usually 25% for gasoline
        m_dot = (w * T) / (eff * LHV) 
        return m_dot
    
    # get gear ratio using gear number
    def gear_ratio(n_gear):
        tau_gear = {
            1: 4.651,
            2: 2.831,
            3: 1.842,
            4: 1.386,
            5: 1.000,
            6: 0.772
        }
        return tau_gear[n_gear]
    
    # return slip angular speed
    # take gear number, w_eng, T_eng
    def slip_speed(n_gear, w_eng, T_eng):
        ## TODO : slip speed 계산
        omega_slip = 0 # temp value
        return omega_slip
    
    # efficiency of motor(bsg)
    def eta_motor(w, T):
        ## TODO put eff map here
        eff = 0.9
        return eff

    ## efficiency of transmission
    def eta_transmission(n_gear, T_trans, w_trans):
        ## TODO put eff map here
        eff = 0.9
        return eff
    
    ## vehicle modeling
    # Define a function to update the vehicle states based on control inputs
    def update_vehicle_states(self, T_eng, T_bsg, SoC, v_veh):
        car = self.car_config
        # random init
        n_g = 1 # gear number from where # it is also time variant
        stop = 0
        #-------------------------------------------------
        # Powertrain equations
        # 1. Engine model
        m_fuel_dot = self.engine_modeling(w_eng, T_eng)

        # 2. BSG model
        w_bsg = car.tau_belt * w_eng

        if (T_bsg < 0):
            P_bsg = T_bsg * w_bsg * self.eta_motor(w_bsg, T_bsg)
        elif (T_bsg > 0):
            P_bsg = T_bsg * w_bsg / self.eta_motor(w_bsg, T_bsg)
        else:
            print("wrong value : T_bsg is 0")


        # 3. Battery Model (need function V_oc, R_0 from pack supplier)
        I_t = (self.V_oc(SoC) - np.sqrt(self.V_oc(SoC)**2 - 4 * self.R_0(SoC) * P_bsg)) / (2 * self.R_0(SoC))
        SoC -= (self.time_step / car.C_nom) * (I_t + car.I_a)


        # 4. Torque Converter Model
        T_pt = T_bsg + T_eng # from the figure 2 block diagram
        T_tc = T_pt
        w_p = w_tc + self.slip_speed(n_g, w_eng, T_eng)
        if(w_p > car.w_stall):
            w_eng = w_p
        elif(w_p >= 0) and (w_p < car.w_stall) :
            w_eng = car.w_idle
            
            if(stop):
                w_eng = 0
        else:
            print("wrong value : w_p is below 0")

        # 5. Transmission Model
        w_out = v_veh / car.wheel_R
        w_trans = car.tau_fdr * w_out

        w_tc = self.gear_ratio(n_g) * car.tau_fdr * v_veh / car.wheel_R
        T_trans = self.gear_ratio(n_g) * T_tc
        T_out = car.tau_fdr * T_trans
        if(T_trans >= 0):
            T_out *= self.eta_transmission(n_g, T_trans, w_trans)
        else:
            T_out /= self.eta_transmission(n_g, T_trans, w_trans)

        return SoC, m_fuel_dot
    
    # Define a function to calculate the required torque from acceleration
    def req_T_calculation(self, v_veh_t, v_veh_t_next, time_step, n_gear=1):
        # 가속도 계산
        a_veh = (v_veh_t_next - v_veh_t) / time_step

        car = self.car_config
        
        # 필요 토크 계산
        T_req = (car.Mass * a_veh * car.wheel_R) / (self.gear_ratio(n_gear) * car.tau_fdr)
        return T_req

    #------------------------- Step function -------------------------- #
    # action을 넣어주면 해당 action에 따라 1 time step만큼 모델을 돌림
    # t-state를 받아서 t+1 state를 반환
    def step(self,action):
        '''
        이전 state에서 속도를 가져오면 prev_vel로 일단 이번 step을 뛰고
        curr_vel로 바뀌어야 함. 그래서 t_req 계산에 prev->curr 로 바뀌도록 하는 토크임
        '''
        #----------------------------- state unpacking phase ------------------------ #
        # get state values from t state
        ## two goal : SoC, fuel_dot
        SoC_t0, fuel_dot_t0, prev_v_veh = self.state
        current_v_veh = self.vehicle_speed_profile[int(self.time)] # 이 속도로 바뀌어야만 함

        #----------------------------- action unpacking phase ------------------------ #
        # get Torque value from action (input)
        ## 현재 시점에 필요한 토크 prev->curr로 바뀌기 위해 필요한 토크
        T_req = self.req_T_calculation(prev_v_veh, current_v_veh, self.step_size, n_gear=1) ##TODO gear 단수를 state에?
        T_eng, T_bsg = action
        T_brk = T_req - (T_eng + T_bsg)

        #----------------------------- modeling phase ------------------------ #
        # make state values of t+1 state
        # prev_v로 이번 step을 뛰어야함
        SoC_t1, fuel_dot_t1 = self.update_vehicle_states(T_eng, T_bsg, SoC_t0, prev_v_veh)

        #----------------------------- reward definition phase ------------------------ #
        # 새로운 state에 대해서 reward를 계산
        soc_reward = - (abs(self.soc_base - SoC_t1)) ** 2 # 멀어질 수록 더 -가 커짐 // 해보고 exp도 가능
        fuel_reward = -fuel_dot_t1 # 클수록 안좋음

        #----------------------------- state update phase ------------------------ #
        new_state = np.array([SoC_t1, fuel_dot_t1, current_v_veh], dtype=np.float32)
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
        return self.state