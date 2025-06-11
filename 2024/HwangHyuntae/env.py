import math
import os
from typing import Tuple
import numpy as np
import pandas as pd
import gymnasium as gym
from scipy.interpolate import RegularGridInterpolator
# import wandb

# 23년식 가솔린 터보 1.6 하이브리드 2WD / 5인승
# https://www.kiamedia.com/us/en/models/sorento-hev/2023/specifications

# (mass, battery_cap, wheel_R, Area_front)
# Guide Auto Web, “2023 Kia Sorento HEV LX Specifications,” The Car Guide. [Online]. Available: https://www.guideautoweb.com/en/makes/kia/sorento/2023/specifications/hev-lx/. [Accessed: May 22, 2025].


class MQ4:  # car_config
    gravity = 9.81      # Gravitational constant (m/s^2)
    drag_coeff = 0.32   # Aerodynamic drag coefficient
    # -> Kia Corporation, *2022 Sorento Specifications Sheet*, Jan. 2022. [Online]. Available: https://www.kia.com/content/dam/kwcms/sg/en/pdf/Brochure/Brochure_Specs/SorentoBrochureSpecsSheet_Jan2022.pdf
    rho_air = 1.204     # Air density (kg/m^3)
    roll_coeff = 0.015  # Rolling resistance coefficient
    # -> T. D. Gillespie, *Fundamentals of Vehicle Dynamics*, Warrendale, PA: Society of Automotive Engineers, 1992, p. 117. ISBN: 1-56091-199-9.
    I_aux = 0.0         # auxiliary current (A) // assume none!

    # need Actual values
    Mass = 1869         # Vehicle mass (kg) *checked!
    wheel_R = 0.36865   # Wheel radius (m)   P235/65R17  *checked!
    Area_front = 3.2205  # Frontal area, H*W = 1.695 * 1.9 (m^2) *checked!
    tau_belt = 2        # belt ratio btw bsg and engine *checked!
    # S. J. Boyd, "Hybrid Electric Vehicle Control Strategy Based on Power Loss Calculations," Master of Science Thesis, Virginia Polytechnic Institute and State University, Blacksburg, VA, USA, 2006.
    eta_belt = 0.95     # belt efficiency motorshaft -> crankshaft // assumed!
    tau_fdr = 3.510     # final drive ratio *checked!
    # -> Kia America, “2023 Sorento HEV Specifications,” Kia Media. [Online]. Available: https://www.kiamedia.com/us/en/models/sorento-hev/2023/specifications. [Accessed: May 22, 2025]
    battery_cap = 1.5   # capacity of the battery (kWh) *checked!
    w_stall = 40.8      # minimum engine speed not to stall // rad/s
    # speed without giving any power // rad/s (ref - 800 RPM) *checked!
    w_idle = 83.775804
    # M. A. Fard, M. Yousefi, and G. Aggarwal, “Sustainable Hybrid Vehicle Idle Speed Control Using PID and Fuzzy Logic,” in Proc. IEEE Smart World Congress 2023: IoT for Sustainable Smart Cities, Portsmouth, United Kingdom, 2023, pp. 1–5.

# **wheel diameter = P235/65R17 -> (235*0.65*2+17*25.4) = 737.3mm, wheel radius = 737.3mm /1000 /2 = 0.36865m


class HEV(gym.Env):
    def __init__(self, start_time=0, step_size=1, config=None, profile_name='wltp.csv') -> None:
        super(HEV, self).__init__()
    # ------------------------- step parameter /*DO NOT CHANGE*/ ------------------------ #
        self.start_time = start_time
        self.step_size = step_size
        self.config = config
        self.vehicle_speed_profile = np.array(pd.read_csv(profile_name))[:, 0]
        self.soc_init = 70/100
        self.time_profile = np.arange(
            self.vehicle_speed_profile.shape[0])  # 1800s steps
        self.stop_time = self.vehicle_speed_profile.shape[0] - 1

        # number of observable states
        # we use soc, fuel_dot, prev_v_veh, prev_w_eng
        self.state_init = np.array([self.soc_init, 0, 0, 0], dtype=np.float64)
        self.soc_base = 70/100
        self.actsize = 3  # number of possible actions

        self.state = self.state_init
        self.obssize = len(self.state)

        self.time = self.start_time
        self.reward = None
        self.done = False

        # for torque slew rate limit
        self.prev_T_eng = 0.0  # prev step engine torque

    # ------------------------- space limitation ------------------------ #
        self.observation_space = gym.spaces.Box(
            # [soc 최소값, fuel_dot 최소값, prev_v_veh 최소값]
            low=np.array([0, 0, 0, 0]),
            # [soc 최대값, fuel_dot 최대값, prev_v_veh 최대값]
            high=np.array([1, np.inf, np.inf, np.inf]),
            dtype=np.float64
        )

        self.action_space = gym.spaces.Box(
            low=np.array([0.0]),
            high=np.array([1.0]),
            dtype=np.float64
        )
    # ------------------------- car specification ------------------------ #
        self.car_config = MQ4
        bsfc_filename = os.path.join(".", "vehicle_map", "BSFC_SNU.csv")
        self.engine_map = pd.read_csv(bsfc_filename, index_col=0)
        eff_filename = os.path.join(".", "vehicle_map", "Eff_P2_SNU.csv")
        self.motor_map = pd.read_csv(eff_filename, index_col=0)

    # ------------------------- car modeling functions and logics ------------------------ #

    def V_oc(self, SoC_t):
        SoC_t = np.clip(SoC_t, 0.0, 1.0)
        if SoC_t <= 0.3:
            # 0.0 ~ 0.3 → 15V → 28V
            cell_voltage = 15.0 + (SoC_t / 0.3) * (28.0 - 15.0)
        else:
            # 0.3 ~ 1.0 → 28V → 36V
            cell_voltage = 28.0 + ((SoC_t - 0.3) / 0.7) * (36.0 - 28.0)

        return cell_voltage * 9  # 9 cells in series

    def R_0(self, SoC_t):
        internal_resistance = 0.009194 / \
            (SoC_t + 0.999865) + 0.000001  # 1 cell
        total_resistance = 9 * internal_resistance  # total 9 cell
        return total_resistance

    # take w_eng and T_eng as input
    # return fuel consumption rate
    def engine_modeling(self, w_eng, T):
        df = self.engine_map
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

    def gear_number(self, v_veh):  # 후진 고려 X
        if v_veh < 6:  # ~25 km/h
            return 1
        elif v_veh < 12:  # ~50 km/h
            return 2
        elif v_veh < 18:  # ~75 km/h
            return 3
        elif v_veh < 25:  # ~100 km/h
            return 4
        elif v_veh < 33:  # ~120 km/h
            return 5
        else:  # bigger than 33 (130 km/h ~)
            return 6

    # get gear ratio using gear number
    def gear_ratio(self, n_gear):
        # https://www.kiamedia.com/us/en/models/sorento-hev/2023/specifications
        tau_gear = {
            1: 4.639,
            2: 2.826,
            3: 1.841,
            4: 1.386,
            5: 1.000,
            6: 0.772,
        }
        return tau_gear[n_gear]

    def get_engine_ramp_rate(self, v_veh: float) -> float:
        """
        현재 속도(v_veh, m/s)에 따른 엔진 토크 램프율(N·m/s) 반환.
        WLTP 0→100 kph (0→27.78 m/s) 가속 3개 구간 기반.
        (0 -> 100 km/h 데이터를 기반으로 램프율 계산)
        """
        if v_veh < 16.667:  # 0 ~ 60 km/h
            return 50.5
        elif v_veh < 22.222:  # 60 ~ 80 km/h
            return 160.9
        elif v_veh < 27.778:  # 80 ~ 100 km/h
            return 124.3
        # 그 이상: 마지막 값 유지... 보통은 더 감소하는 것 같기도
        else:  # TODO -> 거동이 올바른지 다시 확인
            return 124.3  # N·m/s

    # return slip angular speed
    # take gear number, w_eng, T_eng
    # engine <-> transmission slip speed
    # 상용 데이터 일단 사용하는 방식으로
    def slip_speed(self, n_gear, w_eng, T_eng):
        # TODO : slip speed 계산
        omega_slip = 0  # temp value
        return omega_slip

    # efficiency of motor(bsg)
    def eta_motor(self, w_eng, T):
        df = self.motor_map
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

    # efficiency of transmission
    def eta_transmission(self, n_gear, T_trans, w_trans):
        # put eff map here to make result better
        eff = 0.95
        return eff

    # take rpm and return maximum available torque
    def get_engine_max_torque(self, w_eng) -> float:
        rpm_eng = w_eng * 60 / (2 * np.pi)  # rad/s -> rpm

        A = 2000  # 최대값 위치 (1450과 3500 사이의 임의 값)
        T_eng_max = 36 * (rpm_eng / A) * math.exp(1 - rpm_eng / A)  # Kg·m
        T_eng_max = 9.81 * T_eng_max  # N·m

        return T_eng_max

    # 전기모터 최대 토크 함수 (입력: rpm, scalar)
    def get_motor_max_torque(self, w_eng) -> float:
        rpm_eng = w_eng * 60 / (2 * np.pi)  # rad/s -> rpm

        T_motor_max = 280.0  # 최대 전기모터 토크 (Nm)
        threshold = 2000.0   # rpm threshold
        decay_factor = 1000.0  # 지수 감소율
        if rpm_eng <= threshold:
            return T_motor_max
        else:
            return T_motor_max * math.exp(-(rpm_eng - threshold) / decay_factor)

    # 회생제동 토크 모델 (입력: rpm, scalar)
    def get_motor_max_break(self, w_eng) -> float:
        rpm_eng = w_eng * 60 / (2 * np.pi)  # rad/s -> rpm

        T_max_regen = 280.0    # 최대 회생제동 토크 (양수값; 실제 토크는 음수)
        threshold = 2000.0    # 일정 rpm 이하에서는 최대 회생토크 유지
        decay_factor = 1000.0  # 지수 감소율
        if rpm_eng <= threshold:
            return -T_max_regen
        else:
            return -T_max_regen * math.exp(-(rpm_eng - threshold) / decay_factor)

    # vehicle modeling
    # Define a function to update the vehicle states based on control inputs
    def update_vehicle_states(self, n_g, T_eng: float, T_bsg, T_brk, SoC, v_veh, w_eng, stop=False):
        car = self.car_config  # load config

        m_fuel_dot = self.engine_modeling(w_eng, T_eng)

        w_out = v_veh / car.wheel_R
        w_tc = self.gear_ratio(n_g) * car.tau_fdr * w_out

        ############################ w_eng calculation ################################
        # w_eng -> 이전 시점(t-1)에서 계산한 다음 시점의 w_eng, 즉 t 시점 w_eng
        # 현재 t 시점에서는 다음 시점 (t+1) next_w_eng를 계산해서 state로 리턴
        next_w_eng = w_tc + self.slip_speed(n_g, w_eng, T_eng)

        # w_eng이 stall angular speed보다 작으면 차량이 정지할 수도 있음
        # w_eng를 최소한의 속도로 유지
        if (next_w_eng >= 0) and (next_w_eng < car.w_stall):
            next_w_eng = car.w_idle
            if (stop):
                next_w_eng = 0
        elif (next_w_eng < 0):
            print("wrong value : w_p is below 0")
        else:
            pass  # when next_w_eng >= car.w_stall
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

        # 3. Battery Model
        root = self.V_oc(SoC)**2 - 4 * self.R_0(SoC) * \
            P_bsg  # verify root >= 0
        I_t = (self.V_oc(SoC) - np.sqrt(root if root >= 0 else 0)) / \
            (2 * self.R_0(SoC))
        # C_nom = Ah이기 때문에 분자도 A * hour 로 통일시켜줌
        battey_voltage = 270  # Voltage
        DIFF = ((self.step_size / 3600) * battey_voltage *
                (I_t + car.I_aux)) / (car.battery_cap * 1000)
        SoC -= DIFF  # Wh / Wh -> %

        # # 4. Torque Converter Model
        # T_pt = T_bsg + T_eng - T_brk # from the figure 2 block diagram
        # T_tc = T_pt

        # T_trans = self.gear_ratio(n_g) * T_tc
        # T_out = car.tau_fdr * T_trans
        # if(T_trans >= 0):
        #     T_out *= self.eta_transmission(n_g, T_trans, w_trans)
        # else:
        #     T_out /= self.eta_transmission(n_g, T_trans, w_trans)

        return float(SoC), float(m_fuel_dot), next_w_eng

    # Define a function to calculate the required torque from acceleration

    def req_T_calculation(self, v_veh_t, v_veh_t_next, n_gear, step_size):
        # 가속도 계산
        a_veh = (v_veh_t_next - v_veh_t) / step_size
        car = self.car_config
        g = car.gravity

        F_drag = 0.5 * car.drag_coeff * \
            car.rho_air * car.Area_front * (v_veh_t**2)
        F_roll = car.Mass * g * car.roll_coeff * np.cos(0)  # assume slope = 0
        F_grade = car.Mass * g * np.sin(0)  # assume slope = 0
        F_resist = F_drag + F_roll + F_grade

        # 필요 토크 계산
        T_wheel = car.Mass * a_veh * car.wheel_R + F_resist * car.wheel_R
        T_req = T_wheel / (self.gear_ratio(n_gear) * car.tau_fdr)

        return T_req

    # Define a function to split the power between the engine and the BSG
    # T_req_wheel : 최종적으로 바퀴에 걸려야하는 토크
    # n_gear : 현재 속도에 대응하는 기어
    # w_eng : 현재 엔진 속도 (rad/s)
    # v_veh : 현재 속도 (m/s)

    def power_split_HCU(self, ratio, SoC, T_req_wheel, w_eng, v_veh, n_gear):
        # ** T_req = T_eng(pos) + T_bsg(neg, pos) - T_brk(pos) **#
        # T_req_wheel = T_req_crank * (gear ratio * tau_fdr) * eff

        car = self.car_config
        tau_total = self.gear_ratio(n_gear) * self.car_config.tau_fdr
        eta_trans = 0.95  # TODO -> transmission eff 함수의 값 : n_gear, w_bsg, T_bsg에 따라서 변동하는 값

        # 1. calculate T_req_crank from T_req_wheel
        # 최종적으로 바퀴에 걸려야하는 토크 -> 크랭크 축이 내야하는 토크
        if T_req_wheel >= 0:
            T_req_eng_shaft = T_req_wheel / (tau_total * eta_trans)
        else:
            T_req_eng_shaft = T_req_wheel * eta_trans / tau_total

        # 2. Compute maximum motor torque from current w_eng(rad/s)
        w_bsg = car.tau_belt * w_eng
        T_max_bsg_motor_shaft = self.get_motor_max_torque(w_bsg)   # 모터축, positive
        T_max_regen_motor_shaft = self.get_motor_max_break(w_bsg)  # 모터축, negative

        # 3. convert max torque from motor shaft to crankshaft
        # 모터축 -> 엔진축(크랭크축)으로 변환. 나중에 합치기 위해서
        def to_eng_shaft(T_m):
            return T_m * car.tau_belt * car.eta_belt if T_m >= 0 \
                else T_m / (car.tau_belt * car.eta_belt)

        # 4. get all max torque @ crankshaft
        T_max_bsg_eng_shaft = to_eng_shaft(T_max_bsg_motor_shaft)   # 크랭크축, positive
        T_max_regen_eng_shaft = to_eng_shaft(T_max_regen_motor_shaft)  # 크랭크축, negative(regen)
        T_max_eng = self.get_engine_max_torque(w_eng)   # 크랭크축, positive

        # 5. Clip the ratio to the realistic range [0, 1]
        clipped_ratio = np.clip(ratio, 0, 1)
        T_eng = T_max_eng * clipped_ratio  # eng torque 결정

        # 6. T_eng 계산 (Soc 상태 고려)
        if SoC < 0.2:
            T_eng = T_max_eng  # 만약 SoC가 20% 이하라면 action 무시하고 엔진을 최대 토크로 사용

        # 7. Torque slew rate limit
        # T_eng -> 이전 토크 대비 delta_T 만큼만 바뀔 수 있음
        delta_T = self.get_engine_ramp_rate(v_veh) * self.step_size  # N·m/s
        T_eng = float(np.clip(
            T_eng,
            self.prev_T_eng - delta_T,
            self.prev_T_eng + delta_T))
        self.prev_T_eng = T_eng  # 현재 토크 저장 (다음 step에 이전 토크로 사용 예정)

        # ------------------------- torque split cases -------------------------- #
        # T_bsg = T_req_crank - T_eng
        T_brk = 0  # brake torque 초기화

        # case 1 : 제동 상황
        if (T_req_eng_shaft < 0):
            # motor regen 제동만으로 충분한 경우 (T_max_regen 크기 > required)
            if SoC >= 1.0:  # 배터리 완충
                T_bsg = 0  # 회생제동 절대 X
                T_brk = T_eng - T_req_eng_shaft
            else:  # 배터리 충전 가능
                # 제동으로 해결이 가능한 경우
                if (np.abs(T_max_regen_eng_shaft) > np.abs(T_req_eng_shaft - T_eng)):
                    T_bsg = T_req_eng_shaft - T_eng
                    T_brk = 0
                # motor regen 제동으로는 부족한 경우
                else:
                    T_bsg = T_max_regen_eng_shaft
                    T_brk = (T_eng + T_bsg) - T_req_eng_shaft

        # case 2. 가속 상황
        else:  # T_req >= 0
            T_brk = 0  # brake 사용할 필요 X

            # 4. Compute BSG torque requirement
            T_bsg = T_req_eng_shaft - T_eng

            if (T_bsg > 0):  # 배터리 소모
                if (T_bsg > T_max_bsg_eng_shaft):  # BSG 토크가 낼 수 있는 최대치 넘어가는 경우
                    T_bsg = T_max_bsg_eng_shaft
                    # BSG를 max 만큼 끌어쓰고, 나머지는 T_eng으로 다시 채움 (T_req를 bsg max + eng max로 못할 수는 없음)
                    T_eng = T_req_eng_shaft - T_bsg
                # else : do nothing
                # -> T_bsg = T_req_crank - T_eng

            else:  # T_bsg < 0, 회생제동, 충전
                if SoC >= 1.0:  # 배터리 완충 (충전 불가)
                    T_bsg = 0
                    T_eng = T_req_eng_shaft
                else:  # 충전 가능
                    if (np.abs(T_bsg) > np.abs(T_max_regen_eng_shaft)):  # BSG 충전 역토크의 최대치를 넘어가는 경우
                        T_bsg = T_max_regen_eng_shaft
                        T_brk = (T_bsg + T_eng) - T_req_eng_shaft

        def from_eng_shaft_to_motor(T_crank):
            tau = self.car_config.tau_belt
            eta = self.car_config.eta_belt
            return (T_crank / (tau*eta) if T_crank >= 0
                    else T_crank * (tau*eta))

        T_bsg_motor_shaft = from_eng_shaft_to_motor(T_bsg)  # 크랭크축 -> 모터축으로 변환

        return T_eng, T_bsg_motor_shaft, T_brk

    # ------------------------- Step function -------------------------- #
    # action을 넣어주면 해당 action에 따라 1 time step만큼 모델을 돌림
    # t-state를 받아서 t+1 state를 반환
    def step(self, action):
        '''
        이번 step은 prev_v_veh로 뛰고 다음 step은 curr_v_veh로 뛰고 싶은 상황
        따라서 T_req 토크를 통해서 속도가 prev->curr 로 바뀌도록 함
        process : gear 계산 -> T_req 계산 -> HCU(동력분배) -> 차량 state update -> reward 계산
        '''
        # ----------------------------- 0. state unpacking phase ------------------------ #
        # get state values from t state
        # 이번 스텝에서 뛸 속도 prev_v, 다음 스텝에서 뛸 curr_v를 goal로 전달
        SoC_t0, fuel_dot_t0, prev_v_veh, prev_w_eng = self.state
        current_v_veh = self.vehicle_speed_profile[int(
            self.time)] / 3.6  # 목표 속도 (km/h -> m/s)

        # -------------------------- 1. request Torque calculation --------------------- #
        # 현재 시점에 필요한 토크 prev->curr로 바뀌기 위해 필요한 토크 계산
        # 먼저 현재 step에서 뛸 속도로부터 기어를 결정 (0.2초당 1번 기어 변경 가능 -> 제약 조건 따로 X)
        n_gear = self.gear_number(prev_v_veh)

        # 기어 결정 이후 request torque 계산
        # T_req = "최종적으로 바퀴에 걸려야하는 토크"
        T_req = self.req_T_calculation(
            prev_v_veh, current_v_veh, n_gear, self.step_size)

        # -------------------------------- 2. HCU phase -------------------------------- #
        # HCU 동력분배 - action과 state를 받아서 T_eng, T_bsg 결정
        # 여기서 만들어진 T_eng, T_bsg는 반드시 내야한다
        ratio = action  # action unpack (확장 가능)

        # T_req = T_eng(pos) + T_bsg(neg, pos) - T_brk(pos)
        # 모든 토크는 본인의 축 기준!!
        # T_eng -> crankshaft, T_bsg -> motor shaft
        T_eng, T_bsg, T_brk = self.power_split_HCU(
            ratio, SoC_t0, T_req, prev_w_eng, prev_v_veh, n_gear)

        # to fit dimension
        if (type(T_eng) == np.ndarray):
            T_eng = T_eng[0]
        if (type(T_bsg) == np.ndarray):
            T_bsg = T_bsg[0]
        # -------------------------------- 3. state update phase -------------------------------- #
        # t+1 state의 값들을 계산
        # prev_v_veh == 현재 시간 차량의 속도, current_v_veh == 다음 시간 차량의 속도
        SoC_t1, fuel_dot_t1, next_w_eng = self.update_vehicle_states(
            n_gear, T_eng, T_bsg, T_brk, SoC_t0, prev_v_veh, prev_w_eng)

        # ----------------------------- reward definition phase ------------------------ #
        # 새로운 state에 대해서 reward를 계산
        soc_reward = - ((abs(self.soc_base - SoC_t1))**2)*10  # 멀어질 수록 더 -가 커짐
        fuel_reward = - fuel_dot_t1*100  # 클수록 안좋음
        # 원하는 target에 따라서 tuning 할 수 있음 reward 에 배치
        total_reward = 1 + soc_reward + fuel_reward

        # ----------------------------- state update phase ------------------------ #
        new_state = np.array(
            [SoC_t1, fuel_dot_t1, current_v_veh, next_w_eng], dtype=np.float64)
        self.state = new_state

        # --------------------------------- for debugging ------------------------ #
        info = {
            "time": self.time,            # Current time
            "ratio": ratio,                # Current gear
            "T_req": float(T_req),         # Requested torque
            "T_eng": float(T_eng),         # Engine torque
            "T_bsg": float(T_bsg),         # BSG torque
            "SoC": float(SoC_t1),        # 이번 스텝 시작 시의 SoC (%)
            "prev_w_eng": float(prev_w_eng),    # 이번 스텝에서 사용한 w_eng
            # 이번 스텝에서의 max torque
            "T_eng_max": float(self.get_engine_max_torque(prev_w_eng)),
            "soc_reward": float(soc_reward),
            "fuel_reward": float(fuel_reward),
            "total_reward": float(total_reward),
        }

        # --------------------------------- closing phase ------------------------ #
        def is_done(time): return time >= self.stop_time
        self.time += self.step_size
        done = is_done(self.time)
        return self.state, total_reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = self.state_init
        self.time = self.start_time

        info = {}  # none for reset

        return self.state, info


if __name__ == "__main__":
    env = HEV()
    # print(env.eta_motor(209, 100))
    # print(env.engine_modeling(209, 100))
