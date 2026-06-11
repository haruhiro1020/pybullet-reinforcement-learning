# Pybullet (Pythonでの3次元物理シミュレータ) による2軸ロボットアームの強化学習 (DQN)


# 標準ライブラリの読み込み
import numpy as np
import glob
from tkinter import messagebox
import csv
from natsort import natsorted


# 自作モジュールの読み込み
from pybullet_main import MainPyBulletRobot
from constant import *



CONST_SEED = 1      # シード値 (常に同じ結果としたいから)
HAND_FLG   = True   # ハンドの装着有無

# ロボットの関節数 ↓
N_JOINTS   = DIMENTION_2D               # 2軸ロボットアーム
# ロボットの関節数 ↑

# カメラ数 ↓
# N_CAMERAS  = CAMERANUM.SINGLE.value     # 1台のカメラ
N_CAMERAS  = CAMERANUM.MULTI.value      # 複数台のカメラ
# カメラ数 ↑

# PyBulletの環境 ↓
ENV_URDF   = ENVURDF.TWODOF.value       # 2軸ロボットアーム
# PyBulletの環境 ↑

# 強化学習のエージェント ↓
# AGENT_TYPE = AGENT.RANDOM.value         # ランダム法
# AGENT_TYPE = AGENT.MONTECARLO.value     # モンテカルロ法
# AGENT_TYPE = AGENT.SARSA.value          # SARSA
# AGENT_TYPE = AGENT.QLEARNING.value      # Q学習
AGENT_TYPE = AGENT.DQN.value            # DQN
# 強化学習のエージェント ↑

# 強化学習の環境 ↓
ENV_TYPE   = ENV.TWODOF.value           # 2軸ロボットアーム
# 強化学習の環境 ↑

# 強化学習の学習フェーズ ↓
# REINFORCE_LEARN = REIN_PHASE.LEARN      # 学習フェーズ
# REINFORCE_LEARN = REIN_PHASE.IMITATION  # 学習の再現フェーズ
REINFORCE_LEARN = REIN_PHASE.INFERENCE  # 学習後の推論フェーズ
# 強化学習の学習フェーズ ↑

# 強化学習の再現フェーズの定数 ↓
REINFORCE_IMITATION_IDX = 0             # 強化学習の再現フェーズ時のファイル番号
# 強化学習の再現フェーズの定数 ↑



def __get_all_action_files():
    """
    フォルダ内に保存されている行動ファイルを全部取得

    戻り値
        list[str]: 全部の行動ファイル
    """
    all_action_files = glob.glob(f"{RESULT_FOLDER_NAME}/*.csv")
    all_action_files = natsorted(all_action_files)

    return all_action_files


def __print_episode_from_action_file_name(action_file):
    """
    行動ファイル名のエピソード番号を画面に表示

    パラメータ
        action_file (str): 行動ファイル
    """
    under_score_split = action_file.split('_')[1]
    extension_split   = under_score_split.split('.')
    episode = int(extension_split[0])

    return episode


def __get_action_file_from_list(idx):
    """
    リストから行動ファイルを取得

    パラメータ
        idx (int): リストの要素番号

    戻り値
        str: 行動ファイル
    """
    all_action_files = __get_all_action_files()

    if (idx < 0 or idx >= len(all_action_files)):
        raise ValueError(f"idx is abnormal. idx is {idx}")

    selected_action_file = all_action_files[idx]
    __print_episode_from_action_file_name(selected_action_file)

    return selected_action_file


def main():
    """
    メイン処理
    """
    # 把持対象物が保存されている URDF ファイル名
    grasp_urdf = f"{URDF_FOLDER_NAME}/grasp_object.urdf"

    # 探索空間を指定
    interpolation = INTERPOLATION.JOINT.value       # 関節空間

    # Pybulletを使用するインスタンス作成
    my_robot = MainPyBulletRobot(interpolation, N_JOINTS, ENV_URDF, grasp_urdf, N_CAMERAS, hand=HAND_FLG, learn=REINFORCE_LEARN)

    # シード値の設定
    np.random.seed(CONST_SEED)

    # 強化学習の再現フェーズ時のファイルを取得
    file_name = ""
    if REINFORCE_LEARN == REIN_PHASE.IMITATION:
        file_name = __get_action_file_from_list(REINFORCE_IMITATION_IDX)

    # 強化学習の実行
    my_robot.run(AGENT_TYPE, ENV_TYPE, imitation_file_name=file_name)

    if REINFORCE_LEARN == REIN_PHASE.LEARN:
        messagebox.showinfo(
            "Fin ReinForcement",
            "強化学習の実行が終了しました"
        )


if __name__ == "__main__":
    main()
