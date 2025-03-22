'''
Visualize the torque and power of the engine as a function of RPM.
The engine torque is given as a function of RPM as follows:
- 0 ~ 1450 rpm: linearly increasing from 0 to 36 kg·m
- 1450 ~ 3500 rpm: constant at 36 kg·m
- 3500 ~ 6000 rpm: linearly decreasing from 36 to 28.65 kg·m
'''

import numpy as np
import matplotlib.pyplot as plt
import math

def torque_rpm(x):
    if x < 0:
        raise ValueError("x는 양수여야 합니다.")
    A = 2000  # 최대값 위치 (1450과 3500 사이의 임의 값)
    return 36 * (x / A) * math.exp(1 - x / A) * 9.81

# RPM 범위 생성
rpm_values = np.linspace(0, 8000, 600)
torque_values = np.array([torque_rpm(rpm) for rpm in rpm_values])

# 토크 단위 변환: kg·m -> N·m (1 kg·m ≈ 9.81 N·m)
torque_Nm = torque_values * 9.81

# 각속도 계산: ω = 2π * rpm / 60 (rad/s)
angular_velocity = (2 * np.pi * rpm_values) / 60

# 파워 계산: P = T (N·m) * ω (rad/s), 단위: Watt
power_watts = torque_Nm * angular_velocity

# Watt -> kW 변환
power_kW = power_watts  / 735.5

# plot 생성: 왼쪽 y축은 토크 (kg·m), 오른쪽 y축은 파워 (kW)
fig, ax1 = plt.subplots(figsize=(8, 4))

color_torque = 'tab:blue'
ax1.set_xlabel('Engine RPM')
ax1.set_ylabel('Torque (kg·m)', color=color_torque)
ax1.plot(rpm_values, torque_values, color=color_torque, label="Torque (kg·m)")
ax1.tick_params(axis='y', labelcolor=color_torque)
ax1.grid(True)

ax2 = ax1.twinx()  # 두 번째 y축 생성
color_power = 'tab:red'
ax2.set_ylabel('Power (ps)', color=color_power)
ax2.plot(rpm_values, power_kW, color=color_power, label="Power (ps)")
ax2.tick_params(axis='y', labelcolor=color_power)

plt.title("Engine Torque and Power vs RPM")
fig.tight_layout()
plt.show()
