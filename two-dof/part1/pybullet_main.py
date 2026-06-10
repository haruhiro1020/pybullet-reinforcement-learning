# PyBulletのメイン処理を記載


# ライブラリの読み込み
import pybullet as p    # PyBullet
import pybullet_data    # PyBulletで使用するデータ
import numpy as np      # 数値計算ライブラリ


# サードパーティの読み込み


# 自作モジュールの読み込み
from constant import *
from pybullet_grasp import PyBulletGraspObject                  # 把持物体に関して
from pybullet_robot import PyBulletRobotController              # ロボットに関して
from pybullet_camera import PyBulletCameraContoller             # カメラに関して
from pybullet_agent_controller import PyBulletAgentController   # 強化学習のエージェントに関して
from pybullet_environment_controller import PyBulletEnvironmentController   # 強化学習の環境に関して



class MainPyBulletRobot:
    # 定数の定義
    __MAX_EPISODES = 1              # 強化学習のエピソード数


    def __init__(self, interpolation, n_robot_joint, environment_urdf, grasp_urdf, n_cameras, hand=False):
        """
        コンストラクタ

        パラメータ
            interpolation(str): 補間方法 (関節空間/位置空間)
            n_robot_joint(int): ロボットアームの関節数(2, 3, 6だけ)
            environment_urdf(str): 環境のファイル名 (urdf)
            grasp_urdf(str): 把持物体のファイル名 (urdf)
            n_cameras(CAMERANUM): カメラの数
            hand(bool): ハンドの装着有無 True/False = 装着/未装着
        """
        # PyBulletの初期化
        p.connect(p.GUI)
        # パスの追加
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # シミュレーションの初期化
        p.resetSimulation()
        # 重力の設定 (下(-z軸)方向の加速度)
        p.setGravity(0, 0, -GRAVITY_VALUE)

        # 地面を読み込む
        p.loadURDF("plane.urdf")
        # 環境を読み込む
        p.loadURDF(environment_urdf, basePosition=[0, 0, 0], useFixedBase=True)

        # ロボットの初期化
        self.__robot = PyBulletRobotController(n_robot_joint, interpolation, hand)

        # 把持物体の初期化
        self.__grasp_obj = PyBulletGraspObject(grasp_urdf, self.__robot.n_robot_joint)

        # カメラの初期化
        self.__camera = PyBulletCameraContoller(n_cameras)


    # メイン処理 ↓
    def run(self, agent_type: AGENT, env_type: ENV):
        """
        実行
            強化学習によりロボットを動かす

        パラメータ
            agent_type(AGENT): エージェントタイプ
            env_type(ENV): 環境タイプ

        戻り値
            result(bool): True/False = 経路生成に成功/失敗
        """
        # 把持物体の位置をカメラから取得
        end_pos = self.__get_grasp_pos_from_camera()

        # 強化学習の環境の設定
        self.__env = PyBulletEnvironmentController(env_type, self.__robot, end_pos)
        # 強化学習のエージェントの設定
        self.__agent = PyBulletAgentController(agent_type, self.__env.n_action, self.__env.state_dimention)

        # 強化学習の学習フェーズ
        self.__learning()

        return True


    def __learning(self):
        """
        強化学習の学習フェーズ
        """
        # 全エピソードの報酬
        rewards_history = []

        # エピソード分ループ
        for episode in range(self.__MAX_EPISODES):
            # 強化学習用の環境とエージェントを初期化
            state = self.__env.reset()
            self.__agent.reset()

            # 完了フラグ
            done = False
            # １エピソード分の報酬の合計
            total_reward = 0
            # ループ数の保存
            count = 0

            print(f"episode = {episode} is start.")

            # 完了するまでループ
            while not done:
                # 状態から行動を取得
                action = self.__agent.get_action(state)
                # １ステップ実行
                next_state, reward, done = self.__env.step(action)
                # データの追加
                self.__agent.add(state, action, reward)

                # 報酬の更新
                total_reward += reward

                # 状態の更新
                state = next_state

                # カウンタの更新
                count += 1

            # 1エピソード分の情報を使って，データの更新
            self.__agent.update()

            # 報酬を履歴に追加
            rewards_history.append(total_reward)

            print(f"count = {count}")
            print(f"total_reward = {total_reward}")
            print(f"episode = {episode} is fin.")
            print()


    def __get_grasp_pos_from_camera(self):
        """
        カメラから把持物体の位置を取得

        戻り値
            numpy.ndarray: 把持物体の2次元カルテシアン位置 [m]
        """
        # カメラから物体位置(3次元)の取得
        pos = self.__camera.get_pos(self.__grasp_obj.grasp_obj_id)
        # 今回は2軸ロボットアームだけに対応するため，物体位置を2次元に変換する
        # 報酬計算で手先位置(カルテシアン座標)と比較するため，関節角度には変換しない
        pos = pos[:DIMENTION_2D]

        return pos
    # メイン処理 ↑


    # 運動学関連 ↓
    def convert_pos_to_theta(self, pos):
        """
        位置から関節角度に変換

        パラメータ
            pos(numpy.ndarray): 位置 / 関節角度

        戻り値
            thetas(numpy.ndarray): 関節角度
        """
        thetas = self.__robot.convert_pos_to_theta(pos, force=True)

        return thetas
    # 運動学関連 ↑
