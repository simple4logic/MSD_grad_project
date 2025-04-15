import math
import os
from typing import Tuple
import numpy as np
import pandas as pd
import gymnasium as gym
from scipy.interpolate import RegularGridInterpolator
# import wandb

profile_name = 'wltp_1Hz.csv' # wltp cycle (value fixed)

## 23년식 가솔린 터보 1.6 하이브리드 2WD / 5인승
## https://www.kiamedia.com/us/en/models/sorento-hev/2023/specifications
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
    battery_cap = 1.5   # capacity of the battery (kWh)
    I_aux = 0.0         # auxiliary current (A) // assume none
    # I_bias = 1          # constant current bias
    # alpha = 1           # road grade (slope of the road)  ## time variant, depends on the road condition
    w_stall = 52.36     # minimum engine speed not to stall // rad/s
    w_idle = 8.38       # speed without giving any power // rad/s


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
        # we use soc, fuel_dot, prev_v_veh, prev_w_eng
        self.state_init = np.array([self.soc_init, 0, 0, 0], dtype=np.float64)
        self.soc_base = 70/100
        self.actsize = 3 # number of possible actions

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
        SoC_t = np.clip(SoC_t, 0.0, 1.0)
        if SoC_t <= 0.3:
            # 0.0 ~ 0.3 → 15V → 28V
            cell_voltage = 15.0 + (SoC_t / 0.3) * (28.0 - 15.0)
        else:
            # 0.3 ~ 1.0 → 28V → 36V
            cell_voltage = 28.0 + ((SoC_t - 0.3) / 0.7) * (36.0 - 28.0)
        
        return cell_voltage * 9 # 9 cells in series
    
    def R_0(self, SoC_t):
        internal_resistance = 0.009194 / (SoC_t + 0.999865) + 0.000001 # 1 cell
        total_resistance = 9 * internal_resistance # total 9 cell
        return total_resistance
    
    # take w_eng and T_eng as input
    # return fuel consumption rate
    # TODO BSFG map을 써서 적용
    # reward 
    # g/(kWh) -> km/L 연비 계산 가능
    def engine_modeling(self, w_eng, T):
        # LHV = 44e6 #LHV of gasoline, 44MJ/kg(while diesel is 42.5 MJ/kg)
        # eff = 0.25 # usually 25% for gasoline
        # m_dot = engine_power / (eff * LHV)

        filename = os.path.join(".", "vehicle_map", "BSFC_SNU.csv")
        df = pd.read_csv(filename, index_col=0)
        torque_grid = df.index.values.astype(float)
        rpm_grid = np.array([float(col) for col in df.columns])
        eff_map = df.values.astype(float)
        
        eff_interpolator = RegularGridInterpolator(
            (torque_grid, rpm_grid),
            eff_map,
            bounds_error=False,
            fill_value=None
        )
        
        rpm_eng = w_eng * 60 / (2 * np.pi)
        bsfc_point = np.array([T, rpm_eng])
        BSFC = eff_interpolator(bsfc_point)[0]

        engine_power = w_eng * T
        m_dot = (engine_power * BSFC) / (3.6e9)
        return m_dot
    
    def gear_number(self, v_veh): # 후진 고려 X
        if v_veh < 6: # ~25 km/h
            return 1
        elif v_veh < 12: # ~50 km/h
            return 2
        elif v_veh < 18: # ~75 km/h
            return 3
        elif v_veh < 25: # ~100 km/h
            return 4
        elif v_veh < 33: # ~120 km/h
            return 5
        else: # bigger than 33 (130 km/h ~)
            return 6

    # get gear ratio using gear number
    def gear_ratio(self, n_gear):
        tau_gear = {
            1: 4.639,
            2: 2.826,
            3: 1.841,
            4: 1.386,
            5: 1.000,
            6: 0.772,
        }
        return tau_gear[n_gear]
    
    # return slip angular speed
    # take gear number, w_eng, T_eng
    # engine <-> transmission slip speed
    ## 상용 데이터 일단 사용하는 방식으로
    def slip_speed(self, n_gear, w_eng, T_eng):
        ## TODO : slip speed 계산
        omega_slip = 0 # temp value
        return omega_slip
    
    # efficiency of motor(bsg)
    def eta_motor(self, w_eng, T):
        filename = os.path.join(".", "vehicle_map", "Eff_P2_SNU.csv")
        df = pd.read_csv(filename, index_col=0)
        torque_grid = df.index.values.astype(float)
        rpm_grid = np.array([float(col) for col in df.columns])
        eff_map = df.values.astype(float)
        
        eff_interpolator = RegularGridInterpolator(
            (torque_grid, rpm_grid),
            eff_map,
            bounds_error=False,
            fill_value=None
        )
        
        rpm_eng = w_eng * 60 / (2 * np.pi)
        point = np.array([T, rpm_eng])
        eff = eff_interpolator(point)[0]
        return eff/100

    ## efficiency of transmission
    def eta_transmission(self, n_gear, T_trans, w_trans):
        ## TODO put eff map here
        eff = 0.9
        return eff
    
    ## take rpm and return maximum available torque
    def get_engine_max_torque(self, w_eng) -> float:
        rpm_eng = w_eng * 60 / (2 * np.pi)  # rad/s -> rpm

        A = 2000  # 최대값 위치 (1450과 3500 사이의 임의 값)
        T_eng_max = 36 * (rpm_eng / A) * math.exp(1 - rpm_eng / A) # Kg·m
        T_eng_max = 9.81 * T_eng_max # N·m

        return T_eng_max
    
    # 전기모터 최대 토크 함수 (입력: rpm, scalar)
    def get_motor_max_torque(self, w_eng) -> float:
        rpm_eng = w_eng * 60 / (2 * np.pi)  # rad/s -> rpm

        T_motor_max = 280.0  # 최대 전기모터 토크 (Nm)
        threshold = 2000.0   # rpm threshold
        decay_factor = 1000.0 # 지수 감소율
        if rpm_eng <= threshold:
            return T_motor_max
        else:
            return T_motor_max * math.exp(-(rpm_eng - threshold) / decay_factor)

    # 회생제동 토크 모델 (입력: rpm, scalar)
    def get_motor_max_break(self, w_eng) -> float:
        rpm_eng = w_eng * 60 / (2 * np.pi)  # rad/s -> rpm

        T_max_regen = 280.0    # 최대 회생제동 토크 (양수값; 실제 토크는 음수)
        threshold = 2000.0    # 일정 rpm 이하에서는 최대 회생토크 유지
        decay_factor = 1000.0 # 지수 감소율
        if rpm_eng <= threshold:
            return -T_max_regen
        else:
            return -T_max_regen * math.exp(-(rpm_eng - threshold) / decay_factor)

    ## vehicle modeling
    # Define a function to update the vehicle states based on control inputs
    def update_vehicle_states(self, n_g, T_eng : float, T_bsg, T_brk, SoC, v_veh, w_eng, stop=False):
        car = self.car_config # load config

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
        # C_nom = Ah이기 때문에 분자도 A * hour 로 통일시켜줌
        battey_voltage = 270
        DIFF = ((self.step_size / 3600) * battey_voltage * (I_t + car.I_aux)) / (car.battery_cap * 1000)
        SoC -= DIFF # Wh / Wh -> %
        # print("---------------------------------------------------------------------")
        # print("T_bsg : ", T_bsg, "w_bsg : ", w_bsg)
        # print("P_bsg : ", P_bsg)
        # print(f"current : {I_t}, resistance : {self.R_0(SoC)}, voltage : {self.V_oc(SoC)}")
        # print("SoC : ", SoC, "DIFF : ", DIFF)
        # print("")

        # 4. Torque Converter Model
        T_pt = T_bsg + T_eng - T_brk # from the figure 2 block diagram
        T_tc = T_pt

        T_trans = self.gear_ratio(n_g) * T_tc
        T_out = car.tau_fdr * T_trans
        if(T_trans >= 0):
            T_out *= self.eta_transmission(n_g, T_trans, w_trans)
        else:
            T_out /= self.eta_transmission(n_g, T_trans, w_trans)

        return float(SoC), float(m_fuel_dot), next_w_eng
    

    # Define a function to calculate the required torque from acceleration
    def req_T_calculation(self, v_veh_t, v_veh_t_next, n_gear, step_size):
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

        return T_req
    

    # Define a function to split the power between the engine and the BSG
    def power_split_HCU(self, ratio, SoC, T_req, w_eng):
        #** T_req = T_eng(pos) + T_bsg(neg, pos) - T_brk(pos) **#

        # 1. Clip the ratio to the realistic range [0, 1]
        real_ratio = max(min(ratio, 1.0), 0.0)
        
        # 2. Compute maximum engine torque from current w_eng(rad/s)
        T_max_eng = self.get_engine_max_torque(w_eng)   # positive
        T_max_bsg = self.get_motor_max_torque(w_eng)    # positive
        T_max_regen = self.get_motor_max_break(w_eng)   # minus

        # 3. T_eng 계산 (Soc 상태 고려)
        if SoC < 0.2:
            T_eng = T_max_eng # 일부로 max로 넣어서, soc를 회생제동 시킴 -> 충전
        else:
            T_eng = T_max_eng * real_ratio


        ## case 1 : 제동 상황
        if(T_req < 0):
            # motor regen 제동만으로 충분한 경우 (T_max_regen 크기 > required)
            if SoC >= 1.0: # 배터리 완충
                T_bsg = 0 # 회생제동 절대 X
                T_brk = T_eng - T_req
            else: # 배터리 충전 가능
                if(T_max_regen < (T_req - T_eng)):
                    T_bsg = T_req - T_eng
                    T_brk = 0
                # motor regen 제동으로는 부족한 경우
                else:
                    T_bsg = T_max_regen
                    T_brk = (T_eng + T_bsg) - T_req

        ## case 2. 가속 상황
        else: # T_req >= 0
            T_brk = 0 # brake 사용할 필요 X

            # 4. Compute BSG torque requirement
            T_bsg = T_req - T_eng

            if(T_bsg > 0): # 배터리 소모
                if(T_bsg > T_max_bsg): # BSG 토크가 낼 수 있는 최대치 넘어가는 경우
                    T_bsg = T_max_bsg
                    T_eng = T_req - T_bsg # BSG를 max 만큼 끌어쓰고, 나머지는 T_eng으로 다시 채움 (T_req를 bsg max + eng max로 못할 수는 없음)
                #else : do nothing 
            else: # T_bsg < 0, 회생제동, 충전
                if SoC >= 1.0: # 배터리 완충 (충전 불가)
                   T_bsg = 0
                   T_eng = T_req
                else:  # 충전 가능
                    if(T_bsg < T_max_regen): # BSG 충전 역토크의 최대치를 넘어가는 경우
                        T_bsg = T_max_regen
                        T_brk = T_bsg + T_eng - T_req
                    #else : do nothing


        # 5. Enforce engine on/off switching constraints:
        ## TODO 엔진토크가 0보다 커지면 engine ON / 2.5초내로 engine의 on off를 바꾸는건 불가능함 (있으면 좋은 것)
        ## engine on off 마다 reward 를 줄수도 있긴 한데 좀 까다로울 수도 있음
        ## engine을 키면 한N(~3)초 정도는 다시 토크를 0 으로 설정하는건 불가능하게 제약 필요 **

        return T_eng, T_bsg, T_brk


    #------------------------- Step function -------------------------- #
    # action을 넣어주면 해당 action에 따라 1 time step만큼 모델을 돌림
    # t-state를 받아서 t+1 state를 반환
    def step(self,action):
        '''
        이번 step은 prev_v_veh로 뛰고 다음 step은 curr_v_veh로 뛰고 싶은 상황
        따라서 T_req 토크를 통해서 속도가 prev->curr 로 바뀌도록 함
        process : gear 계산 -> T_req 계산 -> HCU(동력분배) -> 차량 state update -> reward 계산
        '''
        #----------------------------- 0. state unpacking phase ------------------------ #
        # get state values from t state
        # 이번 스텝에서 뛸 속도 prev_v, 다음 스텝에서 뛸 curr_v를 goal로 전달
        SoC_t0, fuel_dot_t0, prev_v_veh, prev_w_eng = self.state
        current_v_veh = self.vehicle_speed_profile[int(self.time)] / 3.6 # 목표 속도 (km/h -> m/s)

        #-------------------------- 1. request Torque calculation --------------------- #
        ## 현재 시점에 필요한 토크 prev->curr로 바뀌기 위해 필요한 토크 계산
        # 먼저 현재 step에서 뛸 속도로부터 기어를 결정 (0.2초당 1번 기어 변경 가능 -> 제약 조건 따로 X)
        n_gear = self.gear_number(prev_v_veh)

        # 기어 결정 이후 request torque 계산
        T_req = self.req_T_calculation(prev_v_veh, current_v_veh, n_gear, self.step_size)

        #-------------------------------- 2. HCU phase -------------------------------- #
        ## HCU 동력분배 - action과 state를 받아서 T_eng, T_bsg 결정
        # 여기서 만들어진 T_eng, T_bsg는 반드시 내야한다
        ratio = action # action unpack (확장 가능)

        # T_req = T_eng(pos) + T_bsg(neg, pos) - T_brk(pos)
        T_eng, T_bsg, T_brk = self.power_split_HCU(ratio, SoC_t0, T_req, prev_w_eng)
        if (type(T_eng) == np.ndarray):
            T_eng = T_eng[0]
        if (type(T_bsg) == np.ndarray):
            T_bsg = T_bsg[0]
        #-------------------------------- 3. state update phase -------------------------------- #
        ## t+1 state의 값들을 계산
        # prev_v_veh == 현재 시간 차량의 속도, current_v_veh == 다음 시간 차량의 속도
        SoC_t1, fuel_dot_t1, next_w_eng = self.update_vehicle_states(n_gear, T_eng, T_bsg, T_brk, SoC_t0, prev_v_veh, prev_w_eng)

        #----------------------------- reward definition phase ------------------------ #
        # 새로운 state에 대해서 reward를 계산
        soc_reward = - ((abs(self.soc_base - SoC_t1))**2)*100 # 멀어질 수록 더 -가 커짐
        fuel_reward = - fuel_dot_t1*100 # 클수록 안좋음

        #----------------------------- state update phase ------------------------ #
        new_state = np.array([SoC_t1, fuel_dot_t1, current_v_veh, next_w_eng], dtype=np.float64)
        self.state = new_state

        reward = soc_reward + fuel_reward # 원하는 target에 따라서 tuning 할 수 있음 reward 에 배치

        #--------------------------------- for debugging ------------------------ #
        info = {
            "time"                  : self.time,            # Current time
            "ratio"                 : ratio,               # Current gear
            "T_req"                 : float(T_req),         # Requested torque
            "T_eng"                 : float(T_eng),         # Engine torque
            "T_bsg"                 : float(T_bsg),         # BSG torque
            "SoC"                   : float(SoC_t1),    # 이번 스텝 시작 시의 SoC (%)
            "prev_w_eng"            : float(prev_w_eng),    # 이번 스텝에서 사용한 w_eng
            "T_eng_max"             : float(self.get_engine_max_torque(prev_w_eng)),    # 이번 스텝에서의 max torque
            "soc_reward"            : float(soc_reward),
            "fuel_reward"           : float(fuel_reward),
            "total_reward"          : float(reward),
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
    

if __name__ == "__main__":
    env = HEV()
    # print(env.eta_motor(209, 100))
    print(env.engine_modeling(209, 100))
