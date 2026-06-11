# 環境制御クラスの作成


# 外部ライブラリの読み込み
import pybullet as p


# 自作モジュールの読み込み
from constant import ENV, REIN_PHASE    # 環境・学習フェーズ
from Environment.BaseEnvironment import BaseEnvironment
from Environment.DiscretizeEnvironment import TwoDOFEnv     # 2軸ロボットアーム環境
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
    __PLANE_URDF = "plane.urdf"     # 地面に関する urdf ファイル

    # 各ロボットアームに応じた環境のベース位置
    __ENVIRONMENT_POS_2DOF = [0   ,  0,  0  ]   # 2軸ロボットアームの環境のベース位置
    __ENVIRONMENT_POS_3DOF = [1.5 ,  0, -0.5]   # 3軸ロボットアームの環境のベース位置
    __ENVIRONMENT_POS_6DOF = [1.75,  0,  0  ]   # 6軸ロボットアームの環境のベース位置

    # 学習フェーズに応じたシミュレーション待機時間 [sec]
    __SIM_SLEEP_TIME_LEARN = 0.0001     # 学習時 (DIRECT モード: 高速化優先)
    __SIM_SLEEP_TIME_GUI   = 0.001      # 推論・再現時 (GUI モード: 可視化優先)


    def __init__(self, env: ENV, robot: PyBulletRobotController, state_goal, learn: REIN_PHASE):
        """
        コンストラクタ

        パラメータ
            env(ENV): 環境
            robot(PyBulletRobotController): ロボット制御クラス
            state_goal(numpy.ndarray): 目標状態
            learn(REIN_PHASE): 強化学習のフェーズ
        """
        # 環境に応じたクラスの取得
        env_cls = self.__get_environment_cls(env)

        # 学習フェーズに応じたシミュレーション待機時間を設定
        sim_sleep_time = (
            self.__SIM_SLEEP_TIME_LEARN
            if learn == REIN_PHASE.LEARN
            else self.__SIM_SLEEP_TIME_GUI
        )

        # 環境クラスのインスタンス作成
        self.__env: BaseEnvironment = env_cls(robot, state_goal, sim_sleep_time)


    def __get_environment_cls(self, env: ENV):
        """
        環境に応じたクラスの取得

        パラメータ
            env(ENV): 環境

        戻り値
            環境に応じたクラス
        """
        # 環境番号と環境クラスを保存する辞書型データの定義
        envs = {
            ENV.TWODOF.value: TwoDOFEnv
        }

        # 環境クラスとURDF定義
        env_cls = None

        for key, value in envs.items():
            if env == key:
                # 環境番号が一致
                env_cls = value
                break

        if env_cls is None:
            # 環境番号が一致しなかった
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
            tuple: 初期関節状態
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
