# ２軸ロボットアームに関する環境を定義


# 外部ライブラリの読み込み
import numpy as np
from enum import Enum
from enum import auto


# 自作モジュールの読み込み
from .BaseEnvironment import BaseEnvironment


class TwoDOFEnv(BaseEnvironment):
    """
    ２軸ロボットアームの環境クラス

    プロパティ
        __agent_state(numpy.ndarray): エージェント位置 [m]
        __state_start_pos(numpy.ndarray): 初期位置 [m]

    メソッド
        public
            定数のゲッター
                state_start: 初期位置[m]の取得
                state_goal:  目標位置[m]の取得
            reset(): 初期化
            step(): 1ステップ実行

        private
            __next_state(): 次状態の取得
            __fk(): 順運動学(関節角度から手先位置への変換)
            __calc_dist(): 手先位置と目標位置との距離[m]を計算
            __is_closed(): 手先位置と目標位置との距離が近傍であるかの判定
            __calc_reward(): 報酬の取得
    """
    # 定数の定義
    __DIMENTION      = 2        # 次元数
    __THRESHOLD_GOAL = 0.01     # 手先位置と目標位置との閾値 [m]

    # 行動に関する定数 ↓
    __N_ACTION = 9          # 行動数
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

    # ロボットに関する定数 ↓
    __ROBOT_LINKS_LEN =  np.array([1.0, 1.0])       # ロボットのリンク長
    __ROBOT_MOVE_VAL  =  round(np.deg2rad(5), 2)    # 関節角度の移動量 [rad]
    __ROBOT_JOINT_MAX =  np.pi      # 関節の最大値 [rad]
    __ROBOT_JOINT_MIN = -np.pi      # 関節の最小値 [rad]
    # ロボットに関する定数 ↑

    # 状態に関する定数 ↓
    __STATE_START = np.array([ 0.0, 0.0])   # 初期関節状態
    __STATE_GOAL  = np.array([-1.0, 1.0])   # 目標位置状態
    __STATE_NUM   = 73                      # 状態数 (73は5度刻みになる)
    __STATE_GRID  = np.linspace(__ROBOT_JOINT_MIN, __ROBOT_JOINT_MAX, __STATE_NUM)
    # 状態に関する定数 ↑


    def __init__(self, robot_id):
        """
        コンストラクタ
        """
        # エージェント位置
        self.__agent_state = self.__STATE_START.copy()
        # 順運動学により，初期関節角度から初期位置へ変換
        self.__state_start_pos = self.__fk(self.__STATE_START)


    # 定数のゲッター ↓
    @property
    def n_action(self):
        """
        行動数の取得
        """
        return self.__N_ACTION

    @property
    def state_goal(self):
        """
        目標位置[m]の取得
        """
        return self.__STATE_GOAL

    @property
    def state_start(self):
        """
        初期位置[m]の取得
        """
        return self.__state_start_pos

    @property
    def state_dimention(self):
        """
        状態の次元数の取得
        """
        return self.__DIMENTION
    # 定数のゲッター ↑


    # public ↓
    def reset(self):
        """
        初期化

        戻り値
            numpy.ndarray: 初期関節状態
        """
        # エージェント位置の初期化
        self.__agent_state = self.__STATE_START.copy()

        return self.__STATE_START


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
        # 行動後の次状態を取得
        next_state = self.__next_state(action)

        # 手先位置と目標位置との距離を計算
        dist = self.__calc_dist(next_state)

        # 報酬の計算
        reward = self.__calc_reward(dist)

        # 距離が近傍であるかの確認
        done = self.__is_closed(dist)

        if done:
            reward += 100

        # エージェント位置の更新
        self.__agent_state = next_state.copy()

        return next_state, reward, done
    # public ↑


    # private ↓
    def __next_state(self, action):
        """
        次状態の取得

        パラメータ
            action(int): 行動

        戻り値
            numpy.ndaray: 次状態
        """
        # 関節角度の差分を計算
        delta_thetas = self.__get_delta_thetas(action)

        # 次状態の計算
        next_state = self.__agent_state + delta_thetas
        # 次状態を離散化
        next_state = self.__discretize_state(next_state)

        return next_state


    def __discretize_state(self, state):
        """
        状態の離散化

        パラメータ
            state(numpy.ndarray): 状態

        戻り値
            numpy.ndarray: 離散化した状態
        """
        # 離散化した関節１の値
        joint1_idx = np.argmin(np.abs(self.__STATE_GRID - state[0]))
        joint1 = self.__STATE_GRID[joint1_idx]

        # 離散化した関節２の値
        joint2_idx = np.argmin(np.abs(self.__STATE_GRID - state[1]))
        joint2 = self.__STATE_GRID[joint2_idx]

        return np.array([joint1, joint2])


    def __get_delta_thetas(self, action):
        """
        関節角度の差分を計算

        パラメータ
            action(int): 行動

        戻り値
            numpy.ndaray: 関節角度の差分
        """
        # 行動をキー，行動内容をバリューとして，辞書型データにまとめる
        actions = {
            self._ACTION.PLUS_PLUS.value:   np.array([ self.__ROBOT_MOVE_VAL,  self.__ROBOT_MOVE_VAL]),
            self._ACTION.PLUS_NONE.value:   np.array([ self.__ROBOT_MOVE_VAL,  0.0                  ]),
            self._ACTION.PLUS_MINUS.value:  np.array([ self.__ROBOT_MOVE_VAL, -self.__ROBOT_MOVE_VAL]),
            self._ACTION.NONE_PLUS.value:   np.array([ 0.0                  ,  self.__ROBOT_MOVE_VAL]),
            self._ACTION.NONE_NONE.value:   np.array([ 0.0                  ,  0.0                  ]),
            self._ACTION.NONE_MINUS.value:  np.array([ 0.0                  , -self.__ROBOT_MOVE_VAL]),
            self._ACTION.MINUS_PLUS.value:  np.array([-self.__ROBOT_MOVE_VAL,  self.__ROBOT_MOVE_VAL]),
            self._ACTION.MINUS_NONE.value:  np.array([-self.__ROBOT_MOVE_VAL,  0.0                  ]),
            self._ACTION.MINUS_MINUS.value: np.array([-self.__ROBOT_MOVE_VAL, -self.__ROBOT_MOVE_VAL])
        }

        delta_thetas = actions.get(action)
        if delta_thetas is None:
            raise ValueError(f"action is abnormal. action is {action}")

        return delta_thetas


    def __fk(self, joint_valus):
        """
        順運動学(関節角度から手先位置への変換)

        パラメータ
            joint_valus(numpy.ndarray): 関節角度

        戻り値
            numpy.ndarray: 手先位置
        """
        # 関節数の正常確認
        if len(joint_valus) != self.__DIMENTION:
            # 異常
            raise ValueError(f"joint_valus's length is abnormal. joint_valus's length is {len(joint_valus)}")

        # ロボットのリンク長をローカルに保存
        l1, l2 = self.__ROBOT_LINKS_LEN
        # ロボットの関節角度をローカルに保存
        theta1, theta2 = joint_valus

        # 順運動学により，手先位置の計算
        x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
        y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)

        # 手先位置を一つにまとめる
        pos = np.array([x, y])

        return pos


    def __calc_dist(self, joint_valus):
        """
        手先位置と目標位置との距離[m]を計算

        パラメータ
            joint_valus(numpy.ndarray): 関節角度 [rad]

        戻り値
            float: 手先位置と目標位置との距離 [m]
        """
        # 関節角度から，手先位置への順運動学
        pos = self.__fk(joint_valus)

        # 手先位置と目標位置との差分を計算
        difference = self.__STATE_GOAL - pos

        # 手先位置と目標位置との距離を計算
        dist = np.linalg.norm(difference)

        return dist


    def __is_closed(self, dist):
        """
        手先位置と目標位置との距離が近傍であるかの判定

        パラメータ
            dist(float): 手先位置と目標位置との距離

        戻り値
            bool: True / False = 近傍である / 近傍ではない
        """
        return dist <= self.__THRESHOLD_GOAL


    def __calc_reward(self, dist):
        """
        報酬の取得

        パラメータ
            dist(float): 手先位置と目標位置との距離

        戻り値
            float: 報酬(手先位置と目標位置との距離のマイナス値)
        """
        reward = -dist
        return reward
    # private ↑





