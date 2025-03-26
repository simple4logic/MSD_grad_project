import numpy as np
import matplotlib.pyplot as plt
import math

# 엔진 최대 토크 함수 (입력: w_eng [rad/s])
def get_engine_max_torque_scalar(w_eng):
    # 입력 w_eng는 rad/s 단위
    rpm_eng = w_eng * 60 / (2 * math.pi)  # rad/s -> rpm
    A = 2000.0  # 최대 토크 피크 rpm
    # Kg·m 단위로 토크 계산 후 N·m로 변환
    T_eng_max = 36 * (rpm_eng / A) * math.exp(1 - rpm_eng / A)
    T_eng_max = 9.81 * T_eng_max
    return T_eng_max

# 전기모터 최대 토크 함수 (입력: rpm, scalar)
def get_motor_max_torque_scalar(rpm):
    T_motor_max = 280.0  # 최대 전기모터 토크 (Nm)
    threshold = 2000.0   # rpm threshold
    if rpm <= threshold:
        return T_motor_max
    else:
        return T_motor_max * math.exp(-(rpm - threshold) / 1000.0)

# 회생제동 토크 모델 (입력: rpm, scalar)
def get_motor_max_break_scalar(rpm):
    T_max_regen = 280.0    # 최대 회생제동 토크 (양수값; 실제 토크는 음수)
    threshold = 2000.0    # 일정 rpm 이하에서는 최대 회생토크 유지
    decay_factor = 1000.0 # 지수 감소율
    if rpm <= threshold:
        return -T_max_regen
    else:
        return -T_max_regen * math.exp(-(rpm - threshold) / decay_factor)

# 플롯용 범위 설정
# motor 함수는 rpm을 입력받으므로, rpm_range를 0 ~ 6000 rpm으로 설정
rpm_range = np.linspace(0, 6000, 600)
# 엔진 함수는 rad/s를 입력받으므로, rpm을 rad/s로 변환
w_eng_range = rpm_range * (2 * math.pi) / 60

# 각 rpm에 대한 토크 계산 (scalar 함수 호출)
engine_torque = np.array([get_engine_max_torque_scalar(w) for w in w_eng_range])
motor_torque = np.array([get_motor_max_torque_scalar(rpm) for rpm in rpm_range])
regen_torque = np.array([get_motor_max_break_scalar(rpm) for rpm in rpm_range])

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(rpm_range, engine_torque, label="Engine Max Torque", linewidth=2)
plt.plot(rpm_range, motor_torque, label="Motor Max Torque", linewidth=2)
plt.plot(rpm_range, regen_torque, label="Regenerative Braking Torque", linestyle="--", linewidth=2)
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.8)  # 토크 0 기준선
plt.xlabel("RPM (Positive: Forward Rotation)")
plt.ylabel("Torque (Nm)")
plt.title("Combined Torque Curves: Engine, Motor, and Regenerative Braking")
plt.legend()
plt.grid(True)
plt.show()
