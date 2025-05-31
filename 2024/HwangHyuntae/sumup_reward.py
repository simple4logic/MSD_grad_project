import os
import json
import matplotlib.pyplot as plt
import pandas as pd

def analyze_total_rewards(directory: str, test_name: str = "wltp"):
    """
    지정된 디렉터리(directory)에서 'results_{test_name}_{episode}.json' 파일을 찾고,
    각 파일에 기록된 total_reward 값을 합산하여 에피소드별로 계산한 뒤,
    콘솔에 표를 출력하고 그래프로 시각화합니다.
    """
    episodes = []
    
    # 디렉터리 내 파일 검색
    for fname in os.listdir(directory):
        if fname.startswith(f"results_{test_name}_") and fname.endswith(".json"):
            try:
                # 파일명에서 에피소드 번호 추출 (예: results_wltp_5000.json -> 5000)
                parts = fname.rstrip(".json").split("_")
                ep = int(parts[-1])
                episodes.append((ep, os.path.join(directory, fname)))
            except ValueError:
                # 파일명이 예상 형식이 아니라면 무시
                continue
    
    # 에피소드 번호 순으로 정렬
    episodes.sort(key=lambda x: x[0])
    
    episode_nums = []
    total_rewards = []
    
    # 각 JSON 파일을 읽고 total_reward 합산
    for ep, filepath in episodes:
        with open(filepath, 'r') as f:
            data = json.load(f)
        # 이 에피소드 전체 스텝의 total_reward 값을 합산
        total_reward_sum = sum(entry.get("total_reward", 0) or 0 for entry in data)
        episode_nums.append(ep)
        total_rewards.append(total_reward_sum)
    
    # 에피소드별 Total Reward 데이터를 DataFrame으로 정리
    df = pd.DataFrame({
        "Episode": episode_nums,
        "Total Reward": total_rewards
    })
    
    # 1) 콘솔에 표 형태로 출력
    print("=== Episode별 Total Reward ===")
    print(df.to_string(index=False))
    
    # 2) 꺾은선 그래프로 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(df["Episode"], df["Total Reward"], marker='o', linewidth=2, color='teal')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"Total Reward vs Episode ({test_name.upper()})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def analyze_soc_and_fuel_rewards(directory: str, test_name: str = "wltp"):
    """
    지정된 디렉터리(directory)에서 'results_{test_name}_{episode}.json' 파일을 찾고,
    각 파일에 기록된 soc_reward, fuel_reward 값을 합산하여 에피소드별로 계산한 뒤,
    콘솔에 표를 출력하고 하나의 그래프에 함께 시각화합니다.
    """
    episodes = []
    
    # 디렉터리 내 파일 검색
    for fname in os.listdir(directory):
        if fname.startswith(f"results_{test_name}_") and fname.endswith(".json"):
            try:
                # 파일명에서 에피소드 번호 추출 (예: results_wltp_5000.json -> 5000)
                parts = fname.rstrip(".json").split("_")
                ep = int(parts[-1])
                episodes.append((ep, os.path.join(directory, fname)))
            except ValueError:
                # 예상치 못한 파일명 형식이면 무시
                continue
    
    if not episodes:
        print(f"[경고] '{directory}' 경로에 'results_{test_name}_*.json' 파일이 없습니다.")
        return
    
    # 에피소드 번호 순으로 정렬
    episodes.sort(key=lambda x: x[0])
    
    episode_nums = []
    soc_rewards = []
    fuel_rewards = []
    
    # 각 JSON 파일을 읽고 SOC Reward, Fuel Reward 합산
    for ep, filepath in episodes:
        with open(filepath, 'r') as f:
            data = json.load(f)
        soc_reward_sum = sum(entry.get("soc_reward", 0) or 0 for entry in data)
        fuel_reward_sum = sum(entry.get("fuel_reward", 0) or 0 for entry in data)
        
        episode_nums.append(ep)
        soc_rewards.append(soc_reward_sum)
        fuel_rewards.append(fuel_reward_sum)
    
    # 에피소드별 SOC/Fuel Reward 데이터를 DataFrame으로 정리
    df = pd.DataFrame({
        "Episode": episode_nums,
        "SOC Reward": soc_rewards,
        "Fuel Reward": fuel_rewards
    })
    
    # 1) 콘솔에 표 형태로 출력
    print("=== Episode별 SOC Reward / Fuel Reward ===")
    print(df.to_string(index=False))
    
    # 2) 하나의 그래프로 함께 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(df["Episode"], df["SOC Reward"],   marker='s', linewidth=2, color='orange', label="SOC Reward")
    plt.plot(df["Episode"], df["Fuel Reward"],  marker='^', linewidth=2, color='purple', label="Fuel Reward")
    
    plt.xlabel("Episode")
    plt.ylabel("Reward Value")
    plt.title(f"SOC Reward & Fuel Reward vs Episode ({test_name.upper()})")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_multiple_soc(directory: str, test_name: str = "wltp", interval: int = 5000):
    """
    지정된 디렉터리(directory)에서 'results_{test_name}_{episode}.json' 파일 중,
    episode가 interval 단위로 떨어지는 파일을 찾아 각 에피소드의 SoC(time-series)를
    하나의 그래프에 겹쳐서 표시합니다.
    """
    soc_series = []
    labels = []

    # 디렉터리에서 파일 검색
    for fname in os.listdir(directory):
        if fname.startswith(f"results_{test_name}_") and fname.endswith(".json"):
            parts = fname.rstrip(".json").split("_")
            try:
                ep = int(parts[-1])
            except ValueError:
                continue
            if ep % interval == 0:
                filepath = os.path.join(directory, fname)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                # time과 SoC 값을 추출
                times = [entry.get("time", 0) for entry in data]
                soc_values = [entry.get("SoC", 0) for entry in data]
                soc_series.append((times, soc_values))
                labels.append(f"Episode {ep}")

    if not soc_series:
        print(f"[경고] '{directory}' 경로에 interval={interval} 단위로 떨어지는 'results_{test_name}_*.json' 파일이 없습니다.")
        return

    # 여러 SoC 궤적을 하나의 그래프에 겹쳐서 그리기
    plt.figure(figsize=(10, 6))
    for (times, soc_values), label in zip(soc_series, labels):
        plt.plot(times, soc_values, linewidth=2, label=label)

    plt.xlabel("Time")
    plt.ylabel("State of Charge (SoC)")
    plt.title(f"SoC Time-Series Overlay (Every {interval} Episodes [{test_name.upper()}])")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # analyze_total_rewards(directory="json_data", test_name="udds")
    # analyze_soc_and_fuel_rewards(directory="json_data", test_name="wltp")

    plot_multiple_soc(directory="json_data", test_name="udds", interval=5000)
