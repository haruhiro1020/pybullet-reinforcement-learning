# PyBulletの干渉判定に関するクラス


# ライブラリの読み込み
import pybullet as p    # PyBullet
import numpy as np      # 数値計算
import time             # 時間
from typing import Protocol


# 自作モジュールの読み込み
from constant import *  # 定数
from pybullet_robot import PyBulletRobotController


class _EnvironmentLike(Protocol):
    """
    干渉判定に使用する環境オブジェクトのプロトコル (Protocol：型チェック用の仮想インターフェース)
    """
    @property
    def environment_id(self) -> int:
        ...


class PyBulletInteference:
    """
    PyBulletの干渉判定クラス

    プロパティ
        __robot(): ロボットクラス
        __environment(): 環境クラス


    メソッド
        public
            is_line_interference(): 2点間の干渉判定
    """
    # 定数の定義
    __SIMULATION_SLEEP_TIME = 0.05   # シミュレーションの待機時間 [sec]
    __INTERFERENCE_MARGIN   = 0.15   # 干渉判定のマージン [m]


    def __init__(self, robot: PyBulletRobotController, environment: _EnvironmentLike) -> None:
        """
        コンストラクタ

        パラメータ
            robot(PyBulletRobotController): PyBulletでのロボットクラス
            environment(_EnvironmentLike): environment_id プロパティを持つ環境オブジェクト
        """
        # プロパティの初期化
        self.__robot = robot
        self.__environment = environment


    def is_line_interference(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        """
        2点間の干渉判定

        パラメータ
            pos1(numpy.ndarray): 位置1
            pos2(numpy.ndarray): 位置2

        戻り値
            bool: True/False = 干渉あり/干渉なし
        """
        # 戻り値
        is_interference = True

        # 2点の干渉判定
        if self.is_interference_pos(pos2):
            return is_interference
        if self.is_interference_pos(pos1):
            return is_interference

        # pos1からpos2へ移動
        theta = self.__robot.convert_pos_to_theta(pos2)
        self.__robot.set_joint(theta)
        # グリッパーの実行
        self.__robot.run_gripper(open=True)

        # 待機時間
        time.sleep(self.__SIMULATION_SLEEP_TIME)

        # ロボットと干渉物との干渉判定
        close_points = p.getClosestPoints(self.__robot.robot_id, self.__environment.environment_id, self.__INTERFERENCE_MARGIN)
        if len(close_points) == 0:  # 干渉なし
            is_interference = False

        return is_interference


    def is_interference_pos(self, pos: np.ndarray) -> bool:
        """
        位置にジャンプして干渉判定

        パラメータ
            pos(numpy.ndarray): 位置/関節

        戻り値
            bool: True/False = 干渉あり/干渉なし
        """
        # 戻り値
        is_interference = True

        # 位置から関節角度に変換
        theta = self.__robot.convert_pos_to_theta(pos)
        # 位置にジャンプ
        self.__robot.set_jump_joint(theta)
        # グリッパーの実行
        self.__robot.run_gripper(open=True)

        # 待機時間
        time.sleep(self.__SIMULATION_SLEEP_TIME)

        # ロボットと干渉物との干渉判定
        close_points = p.getClosestPoints(self.__robot.robot_id, self.__environment.environment_id, self.__INTERFERENCE_MARGIN)
        if len(close_points) == 0:  # 干渉なし
            is_interference = False

        return is_interference
