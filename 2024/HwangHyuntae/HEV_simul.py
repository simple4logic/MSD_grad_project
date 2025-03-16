## Model based on MQ4

import numpy as np
import math as m

# Define constants
time_step = 1  # Time step in seconds

# Initialize variables
## init by myself (OK)
SoC = 0.5  # Initial battery state of charge
v_veh = 0.0  # Initial vehicle velocity (m/s)
# v_lim = 20.0  # Speed limit at current segment (m/s)
# v_prime_lim = 30.0  # Upcoming speed limit (m/s)
# t_s = 0  # Time to the start of next green light
# t_e = 0  # Time to the end of next green light
# d_tfc = 1000  # Distance to the upcoming traffic light (m)
# d_prime_lim = 500  # Distance to the segment with new speed limit (m)
# d_rem = 10000  # Remaining distance of the trip (m)

w_eng = 0 # 0 at the very first stage
w_tc = 0

# Vehicle and powertrain model parameters
R_w = 0.3  # Wheel radius (m)

## not sure params
tau_belt = 1            # belt ratio
tau_fdr = 1             # final drive ratio
C_nom = 50000.0         # ??
I_a = 1                 # constant current bias
alpha = 1               # road grade (slope of the road)  ## time variant, depends on the road condition
w_stall = 500           # minimum engine speed not to stall
w_idle = 80             # speed without giving any power


# low-frequency quasi-static nonlinear maps
# notation - PSI in the paper
# take w_eng and T_eng as input
# return m_fuel rate
def engine_modeling(w, T):
    LHV = 44 * 10^6 #LHV of gasoline, 44MJ/kg(while diesel is 42.5 MJ/kg)
    eff = 0.25 # usually 25% for gasoline
    m_dot = (w * T) / (eff * LHV) 
    return m_dot

# quasi-static efficiency map
# take w_bsg and T_bsg as input
# return efficiency
# need efficiency map of the motor inside
def eta_function(w, T):
    efficiency = 0
    return efficiency

## both Voc and R_0 can be obatained from the pack supplier!!
# return cell open circuit voltage
def V_oc(SoC_t):
    V_max = 12.85 # @ 100%
    V_min = 11.65 # @ 0&
    current_Voc = V_min + (V_max - V_min) * SoC_t
    return current_Voc

# return internal resistance (mΩ)
# agm80ah model
def R_0(SoC_t):
    soc_values = [0, 20, 40, 60, 80, 100]  # SoC in percentage
    resistance_values = [10, 8, 6, 5, 4, 3]  # Internal resistance in mΩ
    resistance = np.interp(SoC_t, soc_values, resistance_values)
    return resistance

#-------------------------
# return slip angular speed
# take gear number, w_eng, T_eng
def slip_speed(n_gear, w_eng, T_eng):
    omega_slip = 0 # temp value
    return omega_slip

# return tau_gear(=gear ratio)
# is a function is gear number
# gear number is btw 1 ~ 6
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


## efficiency of transmission
def eta_transmission(n_gear, T_trans, w_trans):
    return


# Define a function to update the vehicle states based on control inputs
def update_vehicle_states(T_eng, T_bsg, T_brk, SoC, v_veh):
    # random init
    n_g = 1 # gear number from where # it is also time variant
    stop = 0
    #----------------------------------------

    # Powertrain equations
    # 1. Engine model
    m_fuel_dot = engine_modeling(w_eng, T_eng)

    # 2. BSG model
    w_bsg = tau_belt * w_eng

    if (T_bsg < 0):
        P_bsg = T_bsg * w_bsg * eta_function(w_bsg, T_bsg)
    elif (T_bsg > 0):
        P_bsg = T_bsg * w_bsg / eta_function(w_bsg, T_bsg)
    else:
        print("wrong value : T_bsg is 0")


    # 3. Battery Model (need function V_oc, R_0 from pack supplier)
    I_t = (V_oc(SoC) - np.sqrt(V_oc(SoC)**2 - 4 * R_0(SoC) * P_bsg)) / (2 * R_0(SoC))
    SoC -= (time_step / C_nom) * (I_t + I_a)


    # 4. Torque Converter Model
    T_pt = T_bsg + T_eng # from the figure 2 block diagram
    T_tc = T_pt
    w_p = w_tc + slip_speed(n_g, w_eng, T_eng)
    if(w_p > w_stall):
        w_eng = w_p
    elif(w_p >= 0) and (w_p < w_stall) :
        w_eng = w_idle
        
        if(stop):
            w_eng = 0
    else:
        print("wrong value : w_p is below 0")

    # 5. Transmission Model
    w_out = v_veh / R_w
    w_trans = tau_fdr * w_out

    w_tc = gear_ratio(n_g) * tau_fdr * v_veh / R_w
    T_trans = gear_ratio(n_g) * T_tc
    T_out = tau_fdr * T_trans
    if(T_trans >= 0):
        T_out *= eta_transmission(n_g, T_trans, w_trans)
    else:
        T_out /= eta_transmission(n_g, T_trans, w_trans)

    # 6. Vehicle Longitudinal Dynamics Model
    # a_veh = ((T_out - T_brk) / (M * R_w)) \
    #         - (0.5 * C_d * rho_a * A_f * v_veh**2) / M \
    #         - g * C_r * m.cos(alpha) * v_veh \
    #         - g * m.sin(alpha)

    # Update velocity and position
    # v_veh = max(0, v_veh + a_veh * time_step)  # Ensure velocity is non-negative
    # remain distance
    # d_rem = max(0, d_rem - v_veh * time_step)  # Ensure distance is non-negative

    return SoC, m_fuel_dot


## WILL NOT BE USED
# Calculate traffic light and road segment dynamics
def update_traffic_states(v_veh, d_tfc, d_prime_lim, t_s, t_e):
    # Update distances to traffic light and speed limit change
    d_tfc = max(0, d_tfc - v_veh * time_step)
    d_prime_lim = max(0, d_prime_lim - v_veh * time_step)

    # Traffic light timing update (assumes a fixed period for simplicity)
    light_cycle = 60  # Total cycle time for the traffic light (s)
    t_s = (t_s - time_step) % light_cycle
    t_e = (t_e - time_step) % light_cycle

    return d_tfc, d_prime_lim, t_s, t_e

# Define the main simulation loop
## will be replaced by ecosim or sumo
def simulate_trip(T_eng, T_bsg, T_brk, steps=100):
    global SoC, v_veh, v_lim, v_prime_lim, t_s, t_e, d_tfc, d_prime_lim, d_rem

    for step in range(steps):
        # Update vehicle states with current control inputs
        SoC, v_veh, d_rem = update_vehicle_states(T_eng, T_bsg, T_brk, SoC, v_veh, d_rem, v_lim)

        # Update traffic states
        d_tfc, d_prime_lim, t_s, t_e = update_traffic_states(v_veh, d_tfc, d_prime_lim, t_s, t_e)
        ## here, will get data from some files and be loaded here

        # Print the states for debugging (optional)
        print(f"Step {step}: SoC={SoC:.2f}, v_veh={v_veh:.2f} m/s, d_rem={d_rem:.2f} m, d_tfc={d_tfc:.2f} m")

    # Return final states
    return SoC, v_veh, v_lim, v_prime_lim, t_s, t_e, d_tfc, d_prime_lim, d_rem

#----------------------------------------------------------------------------------------------------------------

## model responding to the traffic lught
def calculate_traffic_light_mode(v_veh, b_max, d_tfc, v_lim, a_max, t_s, t_e):
    """
    Calculate the current traffic light control mode (m_tfc) based on vehicle state and traffic light parameters.
    
    Parameters:
    v_veh (float): Vehicle speed (m/s)
    b_max (float): Maximum allowable deceleration (m/s^2)
    d_tfc (float): Distance to the upcoming traffic light (m)
    v_lim (float): Speed limit (m/s)
    a_max (float): Maximum allowable acceleration (m/s^2)
    t_s (float): Time to the start of the next green light (s)
    t_e (float): Time to the end of the current green light (s)

    Returns:
    int: Traffic light control mode (0, 1, 2, or 3)
    """
    # Calculate the critical braking distance
    d_critical = (v_veh ** 2) / (2 * b_max)
    
    # Calculate maximum distance vehicle can travel during green light phase
    d_max = sum([min(v_lim, v_veh + i * a_max) for i in range(int(t_e) + 1)])

    # Determine the traffic light control mode
    if d_tfc > d_critical:
        return 0  # Default state (m_tfc = 0)
    elif d_tfc <= d_critical and t_s > 0:
        return 3  # Prepare to stop (m_tfc = 3)
    elif d_tfc <= d_critical and d_tfc <= d_max and t_s == 0:
        return 1  # Go state (m_tfc = 1)
    elif d_tfc <= d_critical and d_tfc > d_max and t_s == 0:
        return 2  # Stop state (m_tfc = 2)
    else:
        return 0  # Fallback to default state if conditions are not met


# Example control inputs
T_eng = 50  # Engine torque (Nm)
T_bsg = 20  # Motor torque (Nm)
T_brk = 10  # Brake torque (Nm)

# Run the simulation for 100 steps
final_states = simulate_trip(T_eng, T_bsg, T_brk, steps=100)
print(f"Final States: {final_states}")
