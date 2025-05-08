import numpy as np
import matplotlib.pyplot as plt

# Original functions
def V_oc(SoC_t):
    V_max = 12.85 # @ 100%
    V_min = 11.65 # @ 0%
    current_Voc = V_min + (V_max - V_min) * SoC_t
    return current_Voc

def R_0(SoC_t):
    soc_values = [0, 0.2, 0.4, 0.6, 0.8, 1]  # SoC in percentage
    resistance_values = [10, 8, 6, 5, 4, 3]  # Internal resistance in mΩ
    resistance = np.interp(SoC_t, soc_values, resistance_values)
    return resistance / 1000 # return in ohm

# New scalar versions
def V_oc_new(SoC_t):
    SoC_t = np.clip(SoC_t, 0.0, 1.0)
    if SoC_t <= 0.3:
        cell_voltage = 15.0 + (SoC_t / 0.3) * (28.0 - 15.0)
    else:
        cell_voltage = 28.0 + ((SoC_t - 0.3) / 0.7) * (36.0 - 28.0)
    return cell_voltage * 9

def R_0_new(SoC_t):
    internal_resistance = 0.009194 / (SoC_t + 0.999865) + 0.000001
    return 9 * internal_resistance

# Vectorize new functions
V_oc_new_vec = np.vectorize(V_oc_new)
R_0_new_vec = np.vectorize(R_0_new)

# Data
SoC = np.linspace(0, 1, 100)
volt_old = V_oc(SoC)
res_old = R_0(SoC)
volt_new = V_oc_new_vec(SoC)
res_new = R_0_new_vec(SoC)

# 2x2 plot
# fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# axes[0,0].plot(SoC, volt_old)
# axes[0,0].set_title('V_oc (original)')
# axes[0,0].set_ylabel('Voltage (V)')
# axes[0,0].grid(True)

# axes[0,1].plot(SoC, res_old)
# axes[0,1].set_title('R_0 (original)')
# axes[0,1].grid(True)

# axes[1,0].plot(SoC, volt_new)
# axes[1,0].set_title('V_oc_new')
# axes[1,0].set_xlabel('SoC')
# axes[1,0].set_ylabel('Voltage (V)')
# axes[1,0].grid(True)

# axes[1,1].plot(SoC, res_new)
# axes[1,1].set_title('R_0_new')
# axes[1,1].set_xlabel('SoC')
# axes[1,1].grid(True)

## -------------------------------------------
# 2x1 plot
fig, axes = plt.subplots(2, 1, figsize=(8, 8))

axes[0].plot(SoC, volt_new)
axes[0].set_title('V_oc_new')
axes[0].set_xlabel('SoC')
axes[0].set_ylabel('Voltage (V)')
axes[0].grid(True)

axes[1].plot(SoC, res_new)
axes[1].set_title('R_0_new')
axes[1].set_xlabel('SoC')
axes[1].grid(True)

plt.tight_layout()
plt.show()
