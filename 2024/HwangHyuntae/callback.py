import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import json
from tqdm import tqdm

class TqdmProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps)

    def _on_step(self) -> bool:
        self.pbar.update(1)
        return True

    def _on_training_end(self):
        self.pbar.close()


def convert_to_serializable(obj):
    # numpy 배열인 경우 list로 변환, numpy 스칼라인 경우 float로 변환
    if isinstance(obj, np.ndarray):
        # 만약 0차원 배열이면 float로 변환, 아니면 list로 변환
        if obj.ndim == 0:
            return float(obj)
        return obj.tolist()
    return obj


class SaveInfoCallback(BaseCallback):
    def __init__(self, log_path: str, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.log_file = open(self.log_path, "w")
        self.log_file.write("[\n")
        self.first_entry = True

    def _on_step(self) -> bool:
        # step() 함수의 info는 self.locals["infos"]에 담겨 있음
        infos = self.locals.get("infos", None)
        if infos is not None:
            for info in infos:
                # 환경의 info 딕셔너리가 그대로 저장되도록 함
                serializable_info = {k: convert_to_serializable(v) for k, v in info.items()}
                if not self.first_entry:
                    self.log_file.write(",\n")
                else:
                    self.first_entry = False
                self.log_file.write(json.dumps(serializable_info))
        return True

    def _on_training_end(self):
        self.log_file.write("\n]")
        self.log_file.close()
