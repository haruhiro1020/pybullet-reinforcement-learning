# 強化学習の全環境のベースを定義


# 外部ライブラリの読み込み
import numpy as np
from enum import Enum
from enum import auto
from abc import ABC, abstractmethod
import pybullet as p    # PyBullet (Python製3次元物理シミュレータ)
import time


# 自作モジュールの読み込み
from constant import *
from pybullet_robot import PyBulletRobotController



class _DiscretizeBaseEnvironment(ABC):
    """
    状態が離散値 (discretize：連続値をグリッドに丸めた値) のベースクラス (抽象クラス)

    プロパティ

    メソッド
        public
            reset(): 初期化
            step(): 1ステップ実行
    """
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
    # オーバーライド必須の定数 ↑

    # 状態に関する定数 ↓
    # _STATE_NUM = 73                    # 状態数 (73は5度刻みになる)
    _STATE_NUM   = 37                   # 状態数 (37は10度刻みになる)
    _STATE_GRID  = np.linspace(_ROBOT_JOINT_MIN, _ROBOT_JOINT_MAX, _STATE_NUM)
    # 状態に関する定数 ↑

    _THRESHOLD_GOAL = 0.1               # 手先位置と目標位置との閾値 [m]
    _THRESHOLD_INIT = 0.001             # 現在関節角度と初期関節角度との閾値 [rad]

    _STATE_START = (int(_STATE_NUM/2), int(_STATE_NUM/2))       # 初期関節状態


    def __init__(self, robot: PyBulletRobotController, state_goal, sim_sleep_time_step):
        """
        コンストラクタ

        パラメータ
            robot(PyBulletRobotController): ロボット制御クラス
            state_goal(numpy.ndarray): 目標状態(関節角度 [rad])
            sim_sleep_time_step(float): 1ステップ終了後のシミュレーションの待機時間 [sec]
        """
        self.__chk_override_constant()

        if len(state_goal) != self._DIMENTION:
            raise ValueError(f"len(state_goal) is abnormal. len(state_goal) is {len(state_goal)}")

        self._state_goal          = state_goal
        self._state_start         = self._STATE_START
        self._robot               = robot
        self._sim_sleep_time_step = sim_sleep_time_step

        self.reset()


    def __chk_override_constant(self):
        """
        サブクラスで必須定数がオーバーライドされているかの確認
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
    # オーバーライド必須の定数 ↓
    _DIMENTION = DIMENTION_2D           # 次元数

    # 行動に関する定数 ↓
    _N_ACTION = 9          # 行動数
    class _ACTION(Enum):
        """
        行動に関する定数
        定数名は "関節１の移動角度_関節２の移動角度" で定義
        """
        PLUS_PLUS   = 0
        PLUS_NONE   = auto()
        PLUS_MINUS  = auto()
        NONE_PLUS   = auto()
        NONE_NONE   = auto()
        NONE_MINUS  = auto()
        MINUS_PLUS  = auto()
        MINUS_NONE  = auto()
        MINUS_MINUS = auto()
    # 行動に関する定数 ↑
    # オーバーライド必須の定数 ↑

    _REWARD_SUCCESS = 100           # 成功時の報酬


    def __init__(self, robot: PyBulletRobotController, state_goal, sim_sleep_time_step):
        """
        コンストラクタ

        パラメータ
            robot(PyBulletRobotController): ロボット制御クラス
            state_goal(numpy.ndarray): 目標状態(関節角度 [rad])
            sim_sleep_time_step(float): 1ステップ終了後のシミュレーションの待機時間 [sec]
        """
        super().__init__(robot, state_goal, sim_sleep_time_step)
        self._state_start = self.__cnvrt_state_to_joints(self._STATE_START)


    def reset(self):
        """
        初期化

        戻り値
            tuple: 初期状態
        """
        near_flg   = False
        init_thetas = self.__cnvrt_state_to_joints(self._STATE_START)

        while not near_flg:
            self.__set_thetas_from_state(self._STATE_START)
            current_pos, _ = self._robot.get_ee_pos()
            current_thetas  = self._robot.convert_pos_to_theta(current_pos, True)

            if np.linalg.norm(init_thetas - current_thetas) <= self._THRESHOLD_INIT:
                near_flg = True

        self._agent_state = self._STATE_START
        print(f"self._agent_state = {self._agent_state}")

        return self._agent_state


    def __cnvrt_state_to_joints(self, state):
        """
        状態から関節角度に変換

        パラメータ
            state(tuple): 状態

        戻り値
            numpy.ndarray: 関節角度
        """
        joints = np.zeros(self._DIMENTION)
        for i in range(len(state)):
            joints[i] = self._STATE_GRID[state[i]]
        return joints


    def __set_thetas_from_state(self, state):
        """
        状態から関節角度を計算してロボットに設定

        パラメータ
            state(tuple): 状態
        """
        joints = self.__cnvrt_state_to_joints(state)
        self._robot.set_joint(joints)
        self._robot.run_gripper(open=True)
        p.stepSimulation()
        time.sleep(self._sim_sleep_time_step)


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
        next_state = self.__next_state(action)
        self.__set_thetas_from_state(next_state)

        dist   = self.__calc_dist()
        reward = self.__calc_reward(dist)
        done   = self.__is_closed(dist)

        if done:
            reward += self._REWARD_SUCCESS

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
        delta_thetas = self.__get_delta_thetas(action)
        thetas       = self.__cnvrt_state_to_joints(self._agent_state)
        next_thetas  = thetas + delta_thetas
        return self.__discretize_state(next_thetas)


    def __get_delta_thetas(self, action):
        """
        関節角度の差分を計算

        パラメータ
            action(int): 行動

        戻り値
            numpy.ndarray: 関節角度の差分
        """
        mv = self._ROBOT_MOVE_VAL
        actions = {
            self._ACTION.PLUS_PLUS.value:   np.array([ mv,  mv]),
            self._ACTION.PLUS_NONE.value:   np.array([ mv,  0.0]),
            self._ACTION.PLUS_MINUS.value:  np.array([ mv, -mv]),
            self._ACTION.NONE_PLUS.value:   np.array([ 0.0,  mv]),
            self._ACTION.NONE_NONE.value:   np.array([ 0.0,  0.0]),
            self._ACTION.NONE_MINUS.value:  np.array([ 0.0, -mv]),
            self._ACTION.MINUS_PLUS.value:  np.array([-mv,  mv]),
            self._ACTION.MINUS_NONE.value:  np.array([-mv,  0.0]),
            self._ACTION.MINUS_MINUS.value: np.array([-mv, -mv]),
        }

        delta_thetas = actions.get(action)
        if delta_thetas is None:
            raise ValueError(f"action is abnormal. action is {action}")

        return delta_thetas


    def __discretize_state(self, state):
        """
        状態の離散化 (discretize：連続値を最近傍グリッドに丸める)

        パラメータ
            state(numpy.ndarray): 状態

        戻り値
            tuple: 離散化した状態
        """
        joints_idx = [
            int(np.argmin(np.abs(self._STATE_GRID - state[i])))
            for i in range(self._DIMENTION)
        ]
        return tuple(joints_idx)


    def __calc_dist(self):
        """
        手先位置と目標位置との距離[m]を計算

        戻り値
            float: 距離 [m]
        """
        ee_pos, _ = self._robot.get_ee_pos()
        return float(np.linalg.norm(self._state_goal - ee_pos))


    def __calc_reward(self, dist):
        """
        報酬の計算

        パラメータ
            dist(float): 手先位置と目標位置との距離

        戻り値
            float: 報酬 (距離のマイナス値)
        """
        return -dist


    def __is_closed(self, dist):
        """
        手先位置と目標位置との距離が近傍かの判定

        パラメータ
            dist(float): 手先位置と目標位置との距離

        戻り値
            bool: True = 近傍, False = 非近傍
        """
        return dist <= self._THRESHOLD_GOAL
