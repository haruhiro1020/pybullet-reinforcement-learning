# 環境制御クラスの作成


# 外部ライブラリの読み込み
import numpy as np


# 自作モジュールの読み込み
from constant import ENV
from Environment.DiscretizeEnvironment import TwoDOFEnv     # 2軸ロボットアーム環境
from pybullet_robot import PyBulletRobotController



class PyBulletEnvironmentController:
    """
    環境制御クラス
        環境タイプに応じた環境クラスを生成し，操作を委譲する
    """

    def __init__(self, env: ENV, robot: PyBulletRobotController, state_goal: np.ndarray):
        """
        コンストラクタ

        パラメータ
            env(ENV): 環境
            robot(PyBulletRobotController): ロボット制御クラス
            state_goal(numpy.ndarray): 目標状態
        """
        self.__env = self.__make_environment_cls(env, robot, state_goal)


    def __make_environment_cls(self, env: ENV, robot: PyBulletRobotController, state_goal: np.ndarray):
        """
        環境に応じたクラスの作成

        パラメータ
            env(ENV): 環境
            robot(PyBulletRobotController): ロボット制御クラス
            state_goal(numpy.ndarray): 目標状態

        戻り値
            環境に応じたクラス
        """
        envs = {
            ENV.TWODOF: TwoDOFEnv,
        }

        env_cls = envs.get(env)
        if env_cls is None:
            raise ValueError(f"env is abnormal. env is {env}")

        return env_cls(robot, state_goal)


    @property
    def n_action(self):
        """
        行動数の取得
        """
        return self.__env.n_action

    @property
    def state_goal(self):
        """
        目標位置[m]の取得
        """
        return self.__env.state_goal

    @property
    def state_start(self):
        """
        初期位置[m]の取得
        """
        return self.__env.state_start

    @property
    def state_dimention(self):
        """
        状態の次元数の取得
        """
        return self.__env.state_dimention


    def reset(self):
        """
        初期化

        戻り値
            tuple: 初期関節状態
        """
        return self.__env.reset()


    def step(self, action):
        """
        1ステップ実行

        パラメータ
            action(int): 行動

        戻り値
            tuple: 次状態
            float: 報酬
            bool: 完了フラグ
        """
        return self.__env.step(action)
