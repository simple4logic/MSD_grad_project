import argparse
import glob
import json
import os
from stable_baselines3 import SAC
from visualize_train_result import plot_results

from env import HEV

def parse_args():
    parser = argparse.ArgumentParser(
        description="SAC으로 HEV 학습을 시작하거나 재개합니다."
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="do visualization if set"
    )
    parser.add_argument(
        "--test-name", "--test",
        type=str,
        default="wltp",
        choices=["udds", "wltp", "hwfet"],
        help="choose test cycle name"
    )
    parser.add_argument(
        "--target-eps", "--eps",
        type=int,
        default="0",
        help="choose which episode to test"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    chkpt_dir = os.path.join(".", "checkpoints", "model")
    if args.target_eps != 0:
        latest_eps = args.target_eps
    else:
        pattern = os.path.join(chkpt_dir, "sac_hev_*_eps.zip")
        all_files = glob.glob(pattern)
        if not all_files:
            raise FileNotFoundError(f"No checkpoint found in {chkpt_dir}")
        def eps_from_path(p):
            name = os.path.basename(p)
            return int(name.split("_")[-2])
        latest_file = max(all_files, key=eps_from_path)
        latest_eps  = eps_from_path(latest_file)
    
    last_trained_file = os.path.join(chkpt_dir, f"sac_hev_{latest_eps}_eps.zip")
    print(f"Start verifying target episode {latest_eps}")

    # 1. 학습된 모델 불러오기
    model = SAC.load(last_trained_file)

    # 2. Urban 사이클(또는 다른 cycle)을 사용하는 환경 생성
    # profile_filename 인자를 'urban.csv'로 전달합니다.
    test_cycle = os.path.join("test_cycle", args.test_name + ".csv")  # Urban 사이클 파일 경로
    env = HEV(start_time=0, step_size=1, config=None, profile_name=test_cycle)

    # 결과 저장을 위한 빈 리스트 생성
    results = []

    # 3. 검증 루프 실행
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        # 에이전트의 행동 예측 (deterministic=True로 선택)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # 각 스텝의 결과를 딕셔너리로 저장
        results.append({
            "time"                  : info.get("time", None),
            "reward"                : reward,
            "ratio"                 : info.get("ratio", None).item(), # 혼자 nparray라서
            "SoC"                   : info.get("SoC", None),
            "T_req"                 : info.get("T_req", None),
            "T_eng"                 : info.get("T_eng", None),
            "T_bsg"                 : info.get("T_bsg", None),
            "T_eng_max"             : info.get("T_eng_max", None),
            "prev_w_eng"            : info.get("prev_w_eng", None),
            "soc_reward"            : info.get("soc_reward", None),
            "fuel_reward"           : info.get("fuel_reward", None),
            "total_reward"          : info.get("total_reward", None),
        })
        # print("Current Step : ", info.get("time", None))

    print("Total Reward:", total_reward)

    # JSON 파일로 저장 (indent=4로 가독성 있게 저장)
    output_file = os.path.join("json_data", "results_" + args.test_name + ".json") 
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print("Results saved to", output_file)

    if args.visualize:
        plot_results(args.test_name)