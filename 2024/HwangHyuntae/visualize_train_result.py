import json
import os
import matplotlib.pyplot as plt

# JSON 파일 경로
json_name = "episode_info_sac.json"
# json_name = "last_loop.json"
save_path = os.path.join("episode_data", json_name)

# JSON 파일 읽기
with open(save_path, "r") as f:
    data = json.load(f)

last_episode_start = max(i for i, entry in enumerate(data) if entry["time"] == 0)
data = data[last_episode_start:]

# 각 키에 해당하는 값을 리스트로 추출합니다.
times = [entry["time"] for entry in data]
T_req = [entry["T_req"] for entry in data]
T_eng = [entry["T_eng"] for entry in data]
T_bsg = [entry["T_bsg"] for entry in data]
action = [entry["ratio"] for entry in data]
prev_w_eng = [entry["prev_w_eng"] for entry in data]
T_eng_max = [entry["T_eng_max"] for entry in data]

soc_values = [entry["SoC"] for entry in data]
soc_reward = [entry["soc_reward"] for entry in data]
fuel_reward = [entry["fuel_reward"] for entry in data]
total_reward = [entry["total_reward"] for entry in data]
# fuel_dot_values = [entry["fuel_dot"] for entry in data]


fig, ax1 = plt.subplots(figsize=(10, 5))

# 왼쪽 y축: T_eng와 T_bsg 플롯
ax1.plot(times, T_eng, label="T_eng", color="blue")
ax1.plot(times, T_bsg, label="T_bsg", color="purple")
# plt.plot(times, T_req, label="T_req", color="green")
# plt.plot(times, T_eng_max, label="T_eng_max", color="red")
# plt.plot(times, prev_w_eng, label="prev_w_eng", color="green")
# plt.plot(times, soc_reward, label="soc_reward", color="blue")
# plt.plot(times, fuel_reward, label="fuel_reward", color="brown")
# plt.plot(times, total_reward, label="total_reward", color="skyblue")
# plt.plot(times, fuel_dot_values, label="Fuel Dot", color="red")
ax1.set_xlabel("Time")
ax1.set_ylabel("Engine/BSG Torque", color="black")
ax1.tick_params(axis="y", labelcolor="black")

# 오른쪽 y축: SoC 플롯
ax2 = ax1.twinx()
# ax2.plot(times, action, label="action", color="pink")
ax2.plot(times, soc_values, label="SoC", color="red")
ax2.set_ylabel("State of Charge (SoC)", color="black")
ax2.tick_params(axis="y", labelcolor="black")

# 범례 설정
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

plt.title("Result: T_eng, T_bsg and SoC vs Time")
plt.grid(True)
plt.show()