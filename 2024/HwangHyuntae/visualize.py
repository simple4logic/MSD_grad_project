import json
import matplotlib.pyplot as plt

# JSON 파일 경로
json_path = "episode_info.json"

# JSON 파일 읽기
with open(json_path, "r") as f:
    data = json.load(f)

# 각 키에 해당하는 값을 리스트로 추출합니다.
times = [entry["time"] for entry in data]
soc_values = [entry["SoC"] for entry in data]
fuel_dot_values = [entry["fuel_dot"] for entry in data]
speed = [entry["current_speed"] for entry in data]

# 플롯 생성: 두 개의 y축 값을 한 그래프에 그리기 (예: 두 곡선을 한 그림에)
plt.figure(figsize=(10, 5))

# plt.plot(times, soc_values, label="SoC", color="blue")
plt.plot(times, speed, label="Speed", color="green")
plt.plot(times, fuel_dot_values, label="Fuel Dot", color="red")

plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Time vs SoC and Fuel Dot")
plt.legend()
plt.grid(True)
plt.show()
