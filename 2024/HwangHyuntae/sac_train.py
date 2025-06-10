# train_sac_hev.py
import glob
import os
import argparse

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from callback import TqdmProgressBarCallback, SaveInfoCallback

from env import HEV


class EpisodeCheckpointCallback(BaseCallback):
    """
    매 N 에피소드마다 모델(.zip)과 replay buffer(.pkl)를 저장합니다.
    파일명: {name_prefix}_{episode}_eps(.zip/.pkl)
    """

    def __init__(
        self,
        save_freq_episodes: int,
        save_path: str,
        name_prefix: str,
        save_replay_buffer: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq_eps = save_freq_episodes
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self._episode_count = 0

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is not None:
            for info in infos:
                # Gymnasium RecordEpisodeStatistics wrapper 가
                # 에피소드 종료 시 info["episode"] 를 포함합니다.
                if "episode" in info:
                    self._episode_count += 1
                    if self._episode_count % self.save_freq_eps == 0:
                        model_path = os.path.join(
                            self.save_path,
                            "model",
                            # .zip model 저장
                            f"{self.name_prefix}_{self._episode_count}_eps"
                        )
                        self.model.save(model_path)
                        if self.save_replay_buffer:
                            buf_path = os.path.join(
                                self.save_path,
                                # .pkl buffer 저장
                                f"{self.name_prefix}_replay_buffer_{self._episode_count}_eps"
                            )
                            self.model.save_replay_buffer(buf_path)
        return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="SAC으로 HEV 학습을 시작하거나 재개합니다."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="이전 체크포인트에서 학습을 이어갈지 여부"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=".\checkpoints",
        help="체크포인트를 저장/로드할 폴더"
    )
    parser.add_argument(
        "--checkpoint-freq", "--freq",
        type=int,
        default=500,
        help="몇 에피소드마다 체크포인트를 저장할지"
    )
    parser.add_argument(
        "--episodes", "--eps",
        type=int,
        default=5000,
        help="처음부터 학습할 총 에피소드 수"
    )
    parser.add_argument(
        "--extra-episodes", "--extra",
        default=500,
        type=int,
        help="재개 시 추가로 학습할 에피소드 수 (지정하지 않으면 --episodes 사용)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    chkpt_dir = args.checkpoint_path
    os.makedirs(chkpt_dir, exist_ok=True)

    # 하이퍼파라미터 (depend on the test cycle)
    start_time = 0
    step_size = 1
    stop_time = 1800

    if args.resume:
        episodes_to_train = args.extra_episodes or args.episodes
    else:
        episodes_to_train = args.episodes

    total_timesteps = stop_time * episodes_to_train

    # env setup
    train_cycle = os.path.join("test_cycle", "wltp.csv")
    env = HEV(
        start_time=start_time,
        step_size=step_size,
        config=None,
        profile_name=train_cycle
    )

    # callback configuration
    progress_cb = TqdmProgressBarCallback(total_timesteps=total_timesteps)
    info_cb = SaveInfoCallback(log_path="json_data/final_episode_info.json")
    eps_chkpt_cb = EpisodeCheckpointCallback(
        save_freq_episodes=args.checkpoint_freq,
        save_path=chkpt_dir,
        name_prefix="sac_hev",
        save_replay_buffer=True
    )

    # 모델 생성 또는 로드
    if args.resume:
        pattern = os.path.join(chkpt_dir, "sac_hev_*_eps.zip")
        all_files = glob.glob(pattern)
        if not all_files:
            raise FileNotFoundError(f"No checkpoint found in {chkpt_dir}")

        def eps_from_path(p):
            name = os.path.basename(p)
            return int(name.split("_")[-2])
        latest_file = max(all_files, key=eps_from_path)
        latest_eps = eps_from_path(latest_file)
        eps_chkpt_cb._episode_count = latest_eps  # 콜백에 에피소드 수 업데이트

        model_file = latest_file
        buffer_file = os.path.join(
            chkpt_dir,
            f"sac_hev_replay_buffer_{latest_eps}_eps"
        )

        print(f"Resuming from episode {latest_eps}:")
        print(f"  - model: {model_file}")
        print(f"  - buffer: {buffer_file}")

        model = SAC.load(model_file, env=env)
        model.load_replay_buffer(buffer_file)
    else:
        model = SAC("MlpPolicy", env=env, verbose=0)

    # set callbacks
    callbacks = CallbackList([progress_cb, info_cb, eps_chkpt_cb])

    # 학습 시작/재개
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=not args.resume
    )

    # 최종 모델 저장
    final_name = "final_model_resume" if args.resume else "final_model"
    model.save(os.path.join(chkpt_dir, final_name))
