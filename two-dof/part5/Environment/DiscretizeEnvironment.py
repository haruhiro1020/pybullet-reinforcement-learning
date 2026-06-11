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
    # _STATE_NUM = 73                   # 状態数 (73は5度刻みになる)
    _STATE_NUM  = 37                    # 状態数 (37は10度刻みになる)
    _STATE_GRID = np.linspace(_ROBOT_JOINT_MIN, _ROBOT_JOINT_MAX, _STATE_NUM)
    # 状態に関する定数 ↑

    _THRESHOLD_GOAL = 0.1               # 手先位置と目標位置との閾値 [m]
    _THRESHOLD_INIT = 0.001             # 現在関節角度と初期関節角度との閾値 [rad]

    # 状態に関する定数 ↓
    # 初期関節状態: グリッドの中央インデックス = 関節角度 0 [rad]
    _STATE_START = (int(_STATE_NUM // 2), int(_STATE_NUM // 2))
    # 状態に関する定数 ↑


    def __init__(self, robot: PyBulletRobotController, state_goal, sim_sleep_time_step):
        """
        コンストラクタ

        パラメータ
            robot(PyBulletRobotController): ロボット制御クラス
            state_goal(numpy.ndarray): 目標状態(関節角度 [rad])
            sim_sleep_time_step(float): 1ステップ終了後のシミュレーションの待機時間 [sec]
        """
        # 定数のオーバーライド確認
        self.__chk_override_constant()

        # 引数の次元数確認
        if len(state_goal) != self._DIMENTION:
            # 引数が異常
            raise ValueError(f"len(state_goal) is abnormal. len(state_goal) is {len(state_goal)}")

        # 目標状態の更新
        self._state_goal  = state_goal
        # 初期状態の更新
        self._state_start = self._STATE_START
        # ロボットの更新
        self._robot = robot
        # シミュレーションの待機時間を更新
        self._sim_sleep_time_step = sim_sleep_time_step

        # 状態の初期化
        self.reset()


    def __chk_override_constant(self):
        """
        定数のオーバーライド確認
        """
        # 比較元と比較先が一致していたら，異常とする辞書型データの作成
        compare_match_datas = {
            "self._DIMENTION":   [self._DIMENTION, DIMENTION_NONE],
            "self._N_ACTION":    [self._N_ACTION, 0]
        }

        # 比較元と比較先が不一致なら，異常とする辞書型データの作成
        compare_not_match_datas = {
            "len(self._ACTION)": [len(self._ACTION), self._N_ACTION]
        }

        # オーバーライドが正しいかの確認
        for tag, datas in compare_match_datas.items():
            # データから，比較元と比較先を取得
            sorce, target = datas
            if sorce == target:
                # オーバーライドしていないから，エラー発報
                raise NotImplementedError(f"{tag} is abnormal. {tag} is {sorce}.")

        # オーバーライドが正しいかの確認
        for tag, datas in compare_not_match_datas.items():
            # データから，比較元と比較先を取得
            sorce, target = datas
            if sorce != target:
                # オーバーライドしていないから，エラー発報
                raise NotImplementedError(f"{tag} is abnormal. {tag} is {sorce}.")


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

    _REWARD_SUCCESS = 100           # 成功時の報酬


    def __init__(self, robot: PyBulletRobotController, state_goal, sim_sleep_time_step):
        """
        コンストラクタ

        パラメータ
            robot(PyBulletRobotController): ロボット制御クラス
            state_goal(numpy.ndarray): 目標状態(関節角度 [rad])
            sim_sleep_time_step(float): 1ステップ終了後のシミュレーションの待機時間 [sec]
        """
        # 親クラスのコンストラクタ
        super().__init__(robot, state_goal, sim_sleep_time_step)

        # 初期関節角度から，初期状態を取得
        self._state_start = self.__cnvrt_state_to_joints(self._STATE_START)


    def reset(self):
        """
        初期化

        戻り値
            tuple: 初期状態
        """
        # 近傍フラグの初期化
        near_flg = False
        # 初期状態の位置を計算
        init_thetas = self.__cnvrt_state_to_joints(self._STATE_START)

        while not near_flg:
            # 初期値へ移動
            self.__set_thetas_from_state(self._STATE_START)
            # 現在位置と現在姿勢の取得
            current_pos, current_ori = self._robot.get_ee_pos()
            # 位置から角度へ変換
            current_thetas = self._robot.convert_pos_to_theta(current_pos, True)

            # 現在角度と初期角度が近傍になれば，ループから抜ける
            if (np.linalg.norm(init_thetas - current_thetas) <= self._THRESHOLD_INIT):
                near_flg = True

        # エージェント位置の初期化
        self._agent_state = self._STATE_START
        print(f"self._agent_state = {self._agent_state}")

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
            reward += self._REWARD_SUCCESS

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
        next_state  = self.__discretize_state(next_thetas)

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

        # 行動内容を保存
        delta_thetas = None

        # 辞書型データから，行動に応じた行動内容を取得
        for key, value in actions.items():
            if action == key:
                # 行動が辞書型データに保存されている
                delta_thetas = value
                break

        if delta_thetas is None:
            # 行動が辞書型データに保存されていないため，異常
            raise ValueError(f"action is abnorma. action is {action}")

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
        # 手先位置を取得
        ee_pos, ee_ori = self._robot.get_ee_pos()

        # 手先位置と目標位置との差分を計算
        difference = self._state_goal - ee_pos

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
