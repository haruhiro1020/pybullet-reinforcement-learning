# 環境制御クラスの作成


from constant import *          # 定数
from Environment.BaseEnvironment import BaseEnvironment
from Environment.DiscretizeEnvironment import TwoDOFEnv # 2軸ロボットアーム環境
from pybullet_robot import PyBulletRobotController



class PyBulletEnvironmentController:
    """
    環境制御クラス

    プロパティ

    メソッド
        public
        protected
        private
    """
    # 定数の定義
    __SIM_SLEEP_TIME_STEP_DIRECT = 0.0001   # 1ステップ終了後のシミュレーションの待機時間 [sec] (PyBulletのDirectモード)
    __SIM_SLEEP_TIME_STEP_GUI    = 0.1      # 1ステップ終了後のシミュレーションの待機時間 [sec] (PyBulletのGUIモード)


    def __init__(self, env: ENV, robot: PyBulletRobotController, state_goal, phase:REIN_PHASE):
        """
        コンストラクタ

        パラメータ
            env(ENV): 環境
            robot(PyBulletRobotController): ロボット制御クラス
            state_goal(numpy.ndarray): 目標状態
            phase(REIN_PHASE): 強化学習のフェーズ
        """
        # 環境クラスの取得
        env_cls = self.__get_environment_class(env)
        
        # １ステップあたりのシミュレーション待機時間をフェーズに応じて選択
        if phase == REIN_PHASE.LEARN:
            # 学習フェーズ
            sim_sleep_time_step = self.__SIM_SLEEP_TIME_STEP_DIRECT
        else:
            # 学習フェーズ以外
            sim_sleep_time_step = self.__SIM_SLEEP_TIME_STEP_GUI

        # 環境クラスのインスタンス作成
        self.__env = env_cls(robot, state_goal, sim_sleep_time_step)


    def __get_environment_class(self, env: ENV):
        """
        環境クラスの取得

        パラメータ
            env(ENV): 環境

        戻り値
            環境に応じたクラス
        """
        envs = {
            ENV.TWODOF.value: TwoDOFEnv,
        }

        env_cls = envs.get(env)
        if env_cls is None:
            raise ValueError(f"env is abnormal. env is {env}")

        return env_cls


    # 定数のゲッター ↓
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
    # 定数のゲッター ↑


    # public ↓
    def reset(self):
        """
        初期化

        戻り値
            numpy.ndarray: 初期関節状態
        """
        return self.__env.reset()


    def step(self, action):
        """
        1ステップ実行

        パラメータ
            action(int): 行動

        戻り値
            numpy.ndarray: 次状態
            float: 報酬
            bool: 完了フラグ
        """
        return self.__env.step(action)
    # public ↑

