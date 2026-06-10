# Pybullet (Pythonでの3次元物理シミュレータ) による2軸ロボットアームの強化学習 (ランダムエージェント)


# 標準ライブラリの読み込み
import numpy as np


# 自作モジュールの読み込み
from pybullet_main import MainPyBulletRobot
from constant import *



CONST_SEED = 1      # シード値 (常に同じ結果としたいから)
HAND_FLG   = True   # ハンドの装着有無

# ロボットの関節数 ↓
N_JOINTS   = DIMENTION_2D               # 2軸ロボットアーム
# ロボットの関節数 ↑

# PyBulletの環境 ↓
ENV_URDF   = ENVURDF.TWODOF.value       # 2軸ロボットアーム
# PyBulletの環境 ↑

# 強化学習のエージェント ↓
AGENT_TYPE = AGENT.RANDOM               # ランダム法
# 強化学習のエージェント ↑

# 強化学習の環境 ↓
ENV_TYPE   = ENV.TWODOF                 # 2軸ロボットアーム
# 強化学習の環境 ↑



def main():
    """
    メイン処理
    """
    # 把持対象物が保存されている URDF ファイル名
    grasp_urdf = f"{URDF_FOLDER_NAME}/grasp_object.urdf"

    # 探索空間を指定 (強化学習では，常に探索空間を関節空間とする．直交空間には対応していないから)
    interpolation = INTERPOLATION.JOINT.value       # 関節空間

    # カメラ数を指定
    n_cameras = CAMERANUM.MULTI.value               # 複数台のカメラ

    # Pybulletを使用するインスタンス作成
    my_robot = MainPyBulletRobot(interpolation, DIMENTION_2D, ENV_URDF, grasp_urdf, n_cameras, hand=HAND_FLG)

    # シード値の設定
    np.random.seed(CONST_SEED)

    # 強化学習の実行
    my_robot.run(AGENT_TYPE, ENV_TYPE)


if __name__ == "__main__":
    # 本ファイルがメインで呼ばれた時の処理
    main()
