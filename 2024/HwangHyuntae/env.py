import numpy as np
import pandas as pd
import gymnasium as gym
# import wandb

profile_name = 'wltp_1Hz.csv' # wltp cycle (value fixed)

## 23년식 가솔린 터보 1.6 하이브리드 2WD / 5인승
class MQ4: ## car_config
    Mass = 1775         # Vehicle mass (kg)
    wheel_R = 0.3       # Wheel radius (m)
    gravity = 9.81      # Gravitational constant (m/s^2)
    drag_coeff = 0.35   # Aerodynamic drag coefficient
    rho_air = 1.225     # Air density (kg/m^3)
    Area_front = 2.7829 # Frontal area (m^2)
    roll_coeff = 0.015  # Rolling resistance coefficient
    tau_belt = 1        # belt ratio
    tau_fdr = 3.5       # final drive ratio
    C_nom = 50000.0     # capacity of the battery (Ah)
    I_aux = 0.0         # auxiliary current (A) // assume none
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
        self.state_init = np.array([self.soc_init, 0, 0, 0], dtype=np.float64)
        self.soc_base = 70/100
        self.actsize = 3 # number of possible acctions

        self.state = self.state_init
        self.obssize = len(self.state)

        self.time = self.start_time
        self.reward = None
        self.done = False

    #------------------------- space limitation ------------------------ #
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0]),  # [soc 최소값, fuel_dot 최소값, prev_v_veh 최소값]
            high=np.array([1, np.inf, np.inf, np.inf]),  # [soc 최대값, fuel_dot 최대값, prev_v_veh 최대값]
            dtype=np.float64
        )

        self.action_space = gym.spaces.Box(
            low=np.array([0.0]), 
            high=np.array([1.0]), 
            dtype=np.float64
        )
    #------------------------- car specification ------------------------ #
        self.car_config = MQ4
        # self.engine_map     # = effmap (in w_eng, T_eng -> out eff)
        # self.motor_map      # = effmap (in w_mot, T_mot -> out eff)


    #------------------------- car modeling functions and logics ------------------------ #
    #battery model : agm80ah
    def V_oc(self, SoC_t):
        V_max = 12.85 # @ 100%
        V_min = 11.65 # @ 0%
        current_Voc = V_min + (V_max - V_min) * SoC_t
        return current_Voc
    
    def R_0(self, SoC_t):
        soc_values = [0, 0.2, 0.4, 0.6, 0.8, 1]  # SoC in percentage
        resistance_values = [10, 8, 6, 5, 4, 3]  # Internal resistance in mΩ
        resistance = np.interp(SoC_t, soc_values, resistance_values)
        return resistance
    
    # take w_eng and T_eng as input
    # return fuel consumption rate
    # TODO BSFG map을 써서 적용
    # g/(kWh) -> km/L 연비 계산 가능
    # reward 
    def engine_modeling(self, w, T):
        LHV = 44e6 #LHV of gasoline, 44MJ/kg(while diesel is 42.5 MJ/kg)
        eff = 0.25 # usually 25% for gasoline
        m_dot = (w * T) / (eff * LHV) 
        return m_dot
    
    def gear_number(self, v_veh):
        if v_veh < 15:
            return 1
        elif v_veh < 30:
            return 2
        elif v_veh < 50:
            return 3
        elif v_veh < 70:
            return 4
        elif v_veh < 90:
            return 5
        elif v_veh < 110:
            return 6
        elif v_veh < 140:
            return 7
        else:
            return 8

    # get gear ratio using gear number
    ## TODO - 더 정확한 referecne 필요. + 8단 -> 6단으로 수정 필요
    def gear_ratio(self, n_gear):
        tau_gear = {
            1: 4.808,
            2: 2.901,
            3: 1.864,
            4: 1.424,
            5: 1.219,
            6: 1.000,
            7: 0.799,
            8: 0.648
        }
        return tau_gear[n_gear]
    
    # return slip angular speed
    # take gear number, w_eng, T_eng
    # engine <-> transmission slip speed
    def slip_speed(self, n_gear, w_eng, T_eng):
        ## TODO : slip speed 계산
        omega_slip = 0 # temp value
        return omega_slip
    
    # efficiency of motor(bsg)
    def eta_motor(self, w, T):
        ## TODO put eff map here
        eff = 0.9
        return eff

    ## efficiency of transmission
    def eta_transmission(self, n_gear, T_trans, w_trans):
        ## TODO put eff map here
        eff = 0.9
        return eff
    
    ## vehicle modeling
    # Define a function to update the vehicle states based on control inputs
    def update_vehicle_states(self, T_eng, T_bsg, SoC, v_veh, w_eng, stop=False):
        car = self.car_config # load config
        n_g = self.gear_number(v_veh) # it is also time variant # TODO gear ratio 제약조건 -> 매초 1단 밖에 안바뀜
        
        m_fuel_dot = self.engine_modeling(w_eng, T_eng)

        w_out = v_veh / car.wheel_R
        w_tc = self.gear_ratio(n_g) * car.tau_fdr * w_out

        ############################ w_eng calculation ################################
        # w_eng -> 이전 시점(t-1)에서 계산한 다음 시점의 w_eng, 즉 t 시점 w_eng
        # 현재 t 시점에서는 다음 시점 (t+1) next_w_eng를 계산해서 state로 리턴
        next_w_eng = w_tc + self.slip_speed(n_g, w_eng, T_eng)

        # w_eng이 stall angular speed보다 작으면 차량이 정지할 수도 있음
        # w_eng를 최소한의 속도로 유지
        if(next_w_eng >= 0) and (next_w_eng < car.w_stall) :
            next_w_eng = car.w_idle
            if(stop):
                next_w_eng = 0
        elif (next_w_eng < 0):
            print("wrong value : w_p is below 0")
        else:
            pass # when next_w_eng >= car.w_stall
        ###################################################################################

        w_trans = car.tau_fdr * w_out
        w_bsg = car.tau_belt * w_eng

        if (T_bsg < 0):
            # 회생 제동 (충전)
            P_bsg = T_bsg * w_bsg * self.eta_motor(w_bsg, T_bsg)
        else: 
            # (T_bsg >= 0): # 가속
            P_bsg = T_bsg * w_bsg / self.eta_motor(w_bsg, T_bsg)
        # eta_motor != 0 이라는 전제

        # 3. Battery Model (need function V_oc, R_0 from pack supplier)
        root = self.V_oc(SoC)**2 - 4 * self.R_0(SoC) * P_bsg # verify root >= 0
        I_t = (self.V_oc(SoC) - np.sqrt(root if root >= 0 else 0)) / (2 * self.R_0(SoC))
        SoC -= (self.step_size / car.C_nom) * (I_t + car.I_aux)

        # 4. Torque Converter Model
        T_pt = T_bsg + T_eng # from the figure 2 block diagram
        T_tc = T_pt

        T_trans = self.gear_ratio(n_g) * T_tc
        T_out = car.tau_fdr * T_trans
        if(T_trans >= 0):
            T_out *= self.eta_transmission(n_g, T_trans, w_trans)
        else:
            T_out /= self.eta_transmission(n_g, T_trans, w_trans)

        return float(SoC), float(m_fuel_dot), next_w_eng
    

    # Define a function to calculate the required torque from acceleration
    def req_T_calculation(self, v_veh_t, v_veh_t_next, step_size):
        n_gear = self.gear_number(v_veh_t)

        # 가속도 계산
        a_veh = (v_veh_t_next - v_veh_t) / step_size
        car = self.car_config
        g = car.gravity

        F_drag = 0.5 * car.drag_coeff * car.rho_air * car.Area_front * (v_veh_t**2)
        F_roll = car.Mass * g * car.roll_coeff * np.cos(0) # assume slope = 0
        F_grade = car.Mass * g * np.sin(0) # assume slope = 0
        F_resist = F_drag + F_roll + F_grade
        
        # 필요 토크 계산
        T_wheel = car.Mass * a_veh * car.wheel_R + F_resist * car.wheel_R
        T_req = T_wheel / (self.gear_ratio(n_gear) * car.tau_fdr)

        ## 속도가 줄었을 때 -> engine torque 줄이고, 나머지 minus torque는 motor로 줄이는 식으로
        ## motor 

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
        SoC_t0, fuel_dot_t0, prev_v_veh, prev_w_eng = self.state
        current_v_veh = self.vehicle_speed_profile[int(self.time)] # 이 속도로 바뀌어야만 함

        #-------------------- --------- action unpacking phase ------------------------ #
        # get Torque value from action (input)
        ## 현재 시점에 필요한 토크 prev->curr로 바뀌기 위해 필요한 토크
        T_req = self.req_T_calculation(prev_v_veh, current_v_veh, self.step_size) ##TODO gear 단수를 state에?
        
        ## max torque에 대한 비율
        ## TODO 넣어준 action에 대한 함수로 작성
        ## eng max torque -> const
        ## motor max torque -> variable

        ## eng, bsg -> action 대한 함수
        # T_eng, T_bsg = action ## 범위 -1 ~ 1
        ratio = action
        T_eng = T_req * ratio
        T_bsg = T_req * (1 - ratio)

        ## T_brk -> 제거
        ## T_req = Teng + Tbsg

        ## (T_eng + T_bsg) => T_out
        ## T_out(모터 + ) - T_brk  / ~ => a_veh

        #----------------------------- modeling phase ------------------------ #
        # make state values of t+1 state
        # prev_v_veh == 현재 시간 차량의 속도
        # current_v_veh == 다음 시간 차량의 속도
        SoC_t1, fuel_dot_t1, next_w_eng = self.update_vehicle_states(T_eng, T_bsg, SoC_t0, prev_v_veh, prev_w_eng)

        #----------------------------- reward definition phase ------------------------ #
        # 새로운 state에 대해서 reward를 계산
        soc_reward = - (abs(self.soc_base - SoC_t1)) ** 2 # 멀어질 수록 더 -가 커짐 // 해보고 exp도 가능
        fuel_reward = -fuel_dot_t1 # 클수록 안좋음

        #----------------------------- state update phase ------------------------ #
        new_state = np.array([SoC_t1, fuel_dot_t1, current_v_veh, next_w_eng], dtype=np.float64)
        self.state = new_state

        reward = soc_reward + fuel_reward


        #--------------------------------- for debugging ------------------------ #
        info = {
            "time": self.time,  # Current time
            "SoC": float(self.state[0]),  # Initial battery state of charge
            "fuel_dot": float(self.state[1]),  # Initial fuel consumption rate
            "current_speed": float(self.state[2]),  # Initial vehicle speed
            "current_angular_speed": float(self.state[3]),  # Initial vehicle speed
            "reward": float(reward),
        }

        #--------------------------------- closing phase ------------------------ #
        is_done = lambda time: time >= self.stop_time
        self.time += self.step_size
        done = is_done(self.time)
        return self.state, reward, done, False, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.state = self.state_init
        self.time = self.start_time
        
        info = {} #none for reset

        return self.state, info