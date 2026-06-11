# Pybullet (Pythonでの3次元物理シミュレータ) による2軸ロボットアームの強化学習 (Q学習)


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
AGENT_TYPE = AGENT.QLEARNING.value      # Q学習
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



def __get_all_action_files() -> list[str]:
    """
    フォルダ内に保存されている行動ファイルを全部取得

    戻り値
        list[str]: 全部の行動ファイル
    """
    # フォルダ内のCSVで保存されている行動ファイル名を取得
    all_action_files = glob.glob(f"{RESULT_FOLDER_NAME}/*.csv")
    # ファイル名を昇順に並び替える
    all_action_files = natsorted(all_action_files)

    return all_action_files


def __print_episode_from_action_file_name(action_file: str) -> int:
    """
    行動ファイル名のエピソード番号を画面に表示

    パラメータ
        action_file (str): 行動ファイル

    戻り値
        int: エピソード番号
    """
    # ファイル名から，エピソード番号の取得
    # ファイル名は episode_{エピソード番号}.csv として，設計している
    # "_" 以降の文字列を取得
    under_score_split = action_file.split('_')[1]
    # "." で文字列を分割
    extension_split   = under_score_split.split('.')
    # エピソード番号を取得
    episode = int(extension_split[0])

    return episode


def __get_action_file_from_list(idx: int) -> str:
    """
    リストから行動ファイルを取得

    パラメータ
        idx (int): リストの要素番号

    戻り値
        str: 行動ファイル
    """
    # フォルダ内に保存されている行動ファイルを全部取得
    all_action_files = __get_all_action_files()

    # パラメータ idx の範囲確認
    if (idx < 0 or idx >= len(all_action_files)):
        # 範囲外
        raise ValueError(f"idx is abnormal. idx is {idx}")

    # 要素番号に応じた行動ファイル
    selected_action_file = all_action_files[idx]

    # 行動ファイル内のエピソード番号を表示
    __print_episode_from_action_file_name(selected_action_file)

    return selected_action_file


def main() -> None:
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
        # 学習時はDIRECTモードであり，学習が完了したことを知らせたいから，メッセージボックスを採用
        messagebox.showinfo(
            "Fin ReinForcement",
            "強化学習の実行が終了しました"
        )


if __name__ == "__main__":
    # 本ファイルがメインで呼ばれた時の処理
    main()
