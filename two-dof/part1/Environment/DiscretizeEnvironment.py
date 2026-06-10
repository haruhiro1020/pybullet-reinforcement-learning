# 強化学習の全環境のベースを定義


# 外部ライブラリの読み込み
import numpy as np
from enum import Enum
from enum import auto
from abc import ABC, abstractmethod
import pybullet as p    # PyBullet
import time


# 自作モジュールの読み込み
from constant import *
from pybullet_robot import PyBulletRobotController



class _DiscretizeBaseEnvironment(ABC):
    """
    状態が離散値のベースクラス (抽象クラス)

    プロパティ

    メソッド
        public
            reset(): 初期化
            step(): 1ステップ実行
    """
    # 定数の定義

    # シミュレーションに関する定数 ↓
    _SIMULATION_SLEEP_TIME = 0.0001   # シミュレーションの待機時間 [sec]
    # シミュレーションに関する定数 ↑

    # ロボットに関する定数 ↓
    _ROBOT_MOVE_VAL  =  round(np.deg2rad(5), 2)    # 関節角度の移動量 [rad]
    _ROBOT_JOINT_MAX =  np.pi      # 関節の最大値 [rad]
    _ROBOT_JOINT_MIN = -np.pi      # 関節の最小値 [rad]
    # ロボットに関する定数 ↑

    # オーバーライド必須の定数 ↓
    _DIMENTION = DIMENTION_NONE         # 次元数

    # 行動に関する定数 ↓
    _N_ACTION = 0                       # 行動数
    class _ACTION(Enum):
        """
        行動に関する定数
        """
        pass
    # 行動に関する定数 ↑

    # 状態に関する定数 ↓
    _STATE_NUM  = 73                    # 状態数 (73は5度刻みになる)
    _STATE_GRID = np.linspace(_ROBOT_JOINT_MIN, _ROBOT_JOINT_MAX, _STATE_NUM)
    # オーバーライド必須の定数 ↑

    _THRESHOLD_GOAL = 0.1               # 手先位置と目標位置との閾値 [m]


    def __init__(self, robot: PyBulletRobotController, state_goal):
        """
        コンストラクタ

        パラメータ
            robot(PyBulletRobotController): ロボット制御クラス
            state_goal(numpy.ndarray): 目標状態(関節角度 [rad])
        """
        # 定数のオーバーライド確認
        self.__chk_override_constant()

        # 引数の次元数確認
        if len(state_goal) != self._DIMENTION:
            # 引数が異常
            raise ValueError(f"len(state_goal) is abnormal. len(state_goal) is {len(state_goal)}")

        # 目標状態の更新
        self._state_goal  = state_goal
        # 初期状態の更新 (サブクラスで _STATE_START が定義された後に確定させる)
        self._state_start = self._STATE_START
        # ロボットの更新
        self._robot = robot

        # 状態の初期化
        self.reset()


    def __chk_override_constant(self):
        """
        定数のオーバーライド確認
        """
        # 値が変わっていないものは未オーバーライドとして異常扱い
        match_errors = {
            "self._DIMENTION":   (self._DIMENTION,   DIMENTION_NONE),
            "self._N_ACTION":    (self._N_ACTION,     0),
        }
        # 値が一致していないものは未オーバーライドとして異常扱い
        mismatch_errors = {
            "len(self._ACTION)": (len(self._ACTION), self._N_ACTION),
        }

        for tag, (source, target) in match_errors.items():
            if source == target:
                raise NotImplementedError(f"{tag} is abnormal. {tag} is {source}.")

        for tag, (source, target) in mismatch_errors.items():
            if source != target:
                raise NotImplementedError(f"{tag} is abnormal. {tag} is {source}.")


    # プロパティのゲッター ↓
    @property
    def n_action(self):
        """
        行動数を取得
        """
        return self._N_ACTION


    @property
    def state_goal(self):
        """
        目標状態(関節角度[rad])を取得
        """
        return self._state_goal


    @property
    def state_start(self):
        """
        初期状態(関節角度[rad])を取得
        """
        return self._state_start


    @property
    def state_dimention(self):
        """
        状態の次元数を取得
        """
        return self._DIMENTION
    # プロパティのゲッター ↑


    # public ↓
    @abstractmethod
    def reset(self):
        """
        初期化

        戻り値
            numpy.ndarray: 初期状態
        """
        pass


    @abstractmethod
    def step(self, action):
        """
        1ステップ実行

        パラメータ
            action(numpy.ndarray): 行動

        戻り値
            numpy.ndarray: 次の状態
            float: 報酬
            bool: 完了フラグ
        """
        pass



class TwoDOFEnv(_DiscretizeBaseEnvironment):
    """
    ２軸ロボットアームの環境クラス

    プロパティ

    メソッド
        public
        protected
        private
    """
    # 定数の定義
    # オーバーライド必須の定数 ↓
    # シミュレーションに関する定数 ↓
    _SIMULATION_SLEEP_TIME = 0.001      # シミュレーションの待機時間 [sec]
    # シミュレーションに関する定数 ↑

    _DIMENTION = DIMENTION_2D           # 次元数

    # 行動に関する定数 ↓
    _N_ACTION = 9          # 行動数
    class _ACTION(Enum):
        """
        行動に関する定数
        """
        # 定数名は "関節１の移動角度_関節２の移動角度" で定義

        # 関節１の角度をプラス ↓
        PLUS_PLUS  = 0
        PLUS_NONE  = auto()
        PLUS_MINUS = auto()
        # 関節１の角度をプラス ↑

        # 関節１の角度は変わらない ↓
        NONE_PLUS  = auto()
        NONE_NONE  = auto()
        NONE_MINUS = auto()
        # 関節１の角度は変わらない ↑

        # 関節１の角度をマイナス ↓
        MINUS_PLUS  = auto()
        MINUS_NONE  = auto()
        MINUS_MINUS = auto()
        # 関節１の角度をマイナス ↑
    # 行動に関する定数 ↑
    # オーバーライド必須の定数 ↑

    # 状態に関する定数 ↓
    # 初期関節状態: グリッドの中央インデックス = 関節角度 0 [rad]
    _STATE_START = (int(_DiscretizeBaseEnvironment._STATE_NUM // 2), int(_DiscretizeBaseEnvironment._STATE_NUM // 2))
    # 状態に関する定数 ↑


    def __init__(self, robot: PyBulletRobotController, state_goal):
        """
        コンストラクタ

        パラメータ
            robot(PyBulletRobotController): ロボット制御クラス
            state_goal(numpy.ndarray): 目標状態(関節角度 [rad])
        """
        # 親クラスのコンストラクタ
        super().__init__(robot, state_goal)

        # 初期関節角度から，初期状態を取得
        self._state_start = self.__cnvrt_state_to_joints(self._STATE_START)


    def reset(self):
        """
        初期化

        戻り値
            tuple: 初期状態
        """
        # 状態から関節角度を計算してロボットに設定
        self.__set_thetas_from_state(self._STATE_START)

        # エージェント位置の初期化
        # 初期関節角度を初期状態へ変換
        self._agent_state = self.__discretize_state(
            self.__cnvrt_state_to_joints(self._STATE_START)
        )

        return self._agent_state


    def __cnvrt_state_to_joints(self, state):
        """
        状態(インデックス)から関節角度に変換

        パラメータ
            state(tuple): 状態 (グリッドインデックス)

        戻り値
            numpy.ndarray: 関節角度 [rad]
        """
        # 戻り値を定義
        joints = np.zeros(self._DIMENTION)

        for i in range(len(state)):
            # 状態から関節角度へ変換
            joints[i] = self._STATE_GRID[state[i]]

        return joints


    def __set_thetas_from_state(self, state):
        """
        状態から関節角度を計算してロボットに設定

        パラメータ
            state(tuple): 状態
        """
        # 状態から関節角度に変換
        joints = self.__cnvrt_state_to_joints(state)

        # ロボットに次状態を渡し，動かす
        self._robot.set_joint(joints)
        # グリッパーの実行
        self._robot.run_gripper(open=True)
        # 実行
        p.stepSimulation()

        # シミュレーション実行の待機時間
        time.sleep(self._SIMULATION_SLEEP_TIME)


    def step(self, action):
        """
        1ステップ実行

        パラメータ
            action(int): 行動

        戻り値
            tuple: 次の状態
            float: 報酬
            bool: 完了フラグ
        """
        # 行動後の次状態を取得
        next_state = self.__next_state(action)

        # 状態から関節角度を取得して，ロボットを動かす
        self.__set_thetas_from_state(next_state)

        # 手先位置と目標位置との距離を計算
        dist = self.__calc_dist()

        # 報酬の計算
        reward = self.__calc_reward(dist)

        # 距離が近傍であるかの確認
        done = self.__is_closed(dist)

        if done:
            # 近傍なら，報酬を更新
            reward += 100

        # エージェント位置の更新
        self._agent_state = next_state

        return next_state, reward, done


    def __next_state(self, action):
        """
        次状態の取得

        パラメータ
            action(int): 行動

        戻り値
            tuple: 次状態
        """
        # 関節角度の差分を計算
        delta_thetas = self.__get_delta_thetas(action)

        # 現在の状態から関節角度を取得
        thetas = self.__cnvrt_state_to_joints(self._agent_state)

        # 次状態の関節角度を計算
        next_thetas = thetas + delta_thetas
        # 次状態を離散化
        next_state = self.__discretize_state(next_thetas)

        return next_state


    def __get_delta_thetas(self, action):
        """
        関節角度の差分を計算

        パラメータ
            action(int): 行動

        戻り値
            numpy.ndarray: 関節角度の差分
        """
        # 行動をキー，行動内容をバリューとして，辞書型データにまとめる
        actions = {
            self._ACTION.PLUS_PLUS.value:   np.array([ self._ROBOT_MOVE_VAL,  self._ROBOT_MOVE_VAL]),
            self._ACTION.PLUS_NONE.value:   np.array([ self._ROBOT_MOVE_VAL,  0.0                 ]),
            self._ACTION.PLUS_MINUS.value:  np.array([ self._ROBOT_MOVE_VAL, -self._ROBOT_MOVE_VAL]),
            self._ACTION.NONE_PLUS.value:   np.array([ 0.0                 ,  self._ROBOT_MOVE_VAL]),
            self._ACTION.NONE_NONE.value:   np.array([ 0.0                 ,  0.0                 ]),
            self._ACTION.NONE_MINUS.value:  np.array([ 0.0                 , -self._ROBOT_MOVE_VAL]),
            self._ACTION.MINUS_PLUS.value:  np.array([-self._ROBOT_MOVE_VAL,  self._ROBOT_MOVE_VAL]),
            self._ACTION.MINUS_NONE.value:  np.array([-self._ROBOT_MOVE_VAL,  0.0                 ]),
            self._ACTION.MINUS_MINUS.value: np.array([-self._ROBOT_MOVE_VAL, -self._ROBOT_MOVE_VAL])
        }

        delta_thetas = actions.get(action)
        if delta_thetas is None:
            raise ValueError(f"action is abnormal. action is {action}")

        return delta_thetas


    def __discretize_state(self, state):
        """
        状態の離散化

        パラメータ
            state(numpy.ndarray): 関節角度 [rad]

        戻り値
            tuple: 離散化した状態 (グリッドインデックス)
        """
        # 関節分のデータを保存する領域を確保
        joints_idx = [0 for _ in range(self._DIMENTION)]

        # 各関節に離散化した状態を保存
        for i in range(self._DIMENTION):
            joints_idx[i] = int(np.argmin(np.abs(self._STATE_GRID - state[i])))

        return tuple(joints_idx)


    def __calc_dist(self):
        """
        手先位置と目標位置との距離[m]を計算

        戻り値
            float: 手先位置と目標位置との距離 [m]
        """
        # 手先位置(3次元)を取得
        ee_pos, ee_ori = self._robot.get_ee_pos()

        # state_goal と次元を合わせるため，手先位置を _DIMENTION 次元に切り出す
        # (2軸アームは XY 平面で動作するため，Z成分は不要)
        ee_pos_nd = ee_pos[:self._DIMENTION]

        # 手先位置と目標位置との差分を計算
        difference = self._state_goal - ee_pos_nd

        # 手先位置と目標位置との距離を計算
        dist = np.linalg.norm(difference)

        return dist


    def __calc_reward(self, dist):
        """
        報酬の取得

        パラメータ
            dist(float): 手先位置と目標位置との距離

        戻り値
            float: 報酬(手先位置と目標位置との距離のマイナス値)
        """
        # 報酬は手先位置と目標位置との距離のマイナス値とする
        reward = -dist

        return reward


    def __is_closed(self, dist):
        """
        手先位置と目標位置との距離が近傍であるかの判定

        パラメータ
            dist(float): 手先位置と目標位置との距離

        戻り値
            bool: True / False = 近傍である / 近傍ではない
        """
        return dist <= self._THRESHOLD_GOAL
