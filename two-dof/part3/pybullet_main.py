# PyBulletのメイン処理を記載 (SARSA)


# ライブラリの読み込み
import pybullet as p    # PyBullet
import pybullet_data    # PyBulletで使用するデータ
import numpy as np      # 数値計算ライブラリ
import os               # OSライブラリ (ディレクトリ作成に使用する)
import shutil           # ディレクトリの削除に使用
import csv              # CSVファイルの読み書き


# サードパーティの読み込み


# 自作モジュールの読み込み
from constant import *
from pybullet_grasp import PyBulletGraspObject                  # 把持物体に関して
from pybullet_robot import PyBulletRobotController              # ロボットに関して
from pybullet_camera import PyBulletCameraContoller             # カメラに関して
from pybullet_agent_controller import PyBulletAgentController   # 強化学習のエージェントに関して
from pybullet_environment_controller import PyBulletEnvironmentController   # 強化学習の環境に関して



class MainPyBulletRobot:
    """
    PyBulletのメインクラス


    プロパティ
        __robot(PyBulletRobotController): ロボットアームの制御クラス
        __interpolation(str): 探索空間 (直交空間/関節空間)
        __environment(PyBulletEnvironment): 環境クラス
        __camera(PyBulletCamera): カメラクラス
        __env(PyBulletEnvironmentController): 強化学習の環境制御クラス
        __agent(PyBulletAgentController): 強化学習のエージェント制御クラス
        __learn(REIN_PHASE): 強化学習のフェーズ


    メソッド
        public

            メイン処理関連
                run(): 実行 (始点から終点まで，干渉しない経路を生成)

            運動学関連
                convert_pos_to_theta(): 位置から関節角度に変換 (クラス外で使う用)


        private

            メイン処理関連
                __learning(): 強化学習の学習フェーズ
                __imitation(): 強化学習の再現フェーズ
                __inference(): 強化学習の推論フェーズ
                __get_grasp_pos_from_camera(): カメラから把持物体の位置を取得
    """
    # 定数の定義
    __MAX_EPISODES    = 12000       # 強化学習のエピソード数
    __N_SAVE_EPISODES =   100       # 保存するエピソード数


    def __init__(self, interpolation: str, n_robot_joint: int, environment_urdf: str, grasp_urdf: str, n_cameras: int, hand: bool = False, learn: REIN_PHASE = REIN_PHASE.LEARN) -> None:
        """
        コンストラクタ

        パラメータ
            interpolation(str): 補間方法 (関節空間/位置空間)
            n_robot_joint(int): ロボットアームの関節数(2, 3, 6だけ)
            environment_urdf(str): 環境のファイル名 (urdf)
            grasp_urdf(str): 把持物体のファイル名 (urdf)
            n_cameras(int): カメラの数
            hand(bool): ハンドの装着有無 True/False = 装着/未装着
            learn(REIN_PHASE): 強化学習のフェーズ
        """
        # PyBulletの初期化
        if learn == REIN_PHASE.LEARN:
            # 学習時は，後ろで動かし続ける
            p.connect(p.DIRECT)
            # ファイル保存用のフォルダ削除
            self.__remove_folder()
        elif (learn == REIN_PHASE.IMITATION) or (learn == REIN_PHASE.INFERENCE):
            # 推論または再現では，GUI上のロボットを動かす
            p.connect(p.GUI)
        else:
            # 異常な値
            raise ValueError(f"learn is abnormal. learn is {learn}")

        # ファイル保存用のフォルダ作成
        self.__make_folder()

        # プロパティの初期化
        self.__learn = learn

        # パスの追加
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # シミュレーションの初期化
        p.resetSimulation()
        # 重力の設定 (下(-z軸)方向の加速度)
        p.setGravity(0, 0, -GRABITY_VALUE)

        # ロボットの初期化
        self.__robot = PyBulletRobotController(n_robot_joint, interpolation, hand)

        # 把持物体の初期化
        self.__grasp_obj = PyBulletGraspObject(grasp_urdf, self.__robot.n_robot_joint)

        # カメラの初期化
        self.__camera = PyBulletCameraContoller(n_cameras)


    def __make_folder(self) -> None:
        """
        ファイル保存用のフォルダ作成
        """
        # フォルダが存在していても，エラー発報させない(exist_ok=True)
        os.makedirs(RESULT_FOLDER_NAME, exist_ok=True)


    def __remove_folder(self) -> None:
        """
        ファイル保存用のフォルダを削除
        """
        # フォルダの存在確認
        if os.path.isdir(RESULT_FOLDER_NAME):
            # 存在するため，フォルダを削除
            shutil.rmtree(RESULT_FOLDER_NAME)


    # メイン処理 ↓
    def run(self, agent_type: int, env_type: int, imitation_file_name: str = '') -> None:
        """
        実行
            強化学習によりロボットを動かす

        パラメータ
            agent_type(int): エージェントタイプ (AGENT列挙型の .value)
            env_type(int): 環境タイプ (ENV列挙型の .value)
            imitation_file_name(str): 再現フェーズ用ファイル名
        """
        # 把持物体の位置をカメラから取得
        end_pos = self.__get_grasp_pos_from_camera()

        # 強化学習の環境の設定
        self.__env = PyBulletEnvironmentController(env_type, self.__robot, end_pos, self.__learn)
        # 強化学習のエージェントの設定
        self.__agent = PyBulletAgentController(agent_type, self.__env.n_action, self.__env.state_dimention)

        if self.__learn == REIN_PHASE.LEARN:
            # 強化学習の学習フェーズ
            self.__learning()
        elif self.__learn == REIN_PHASE.IMITATION:
            # 強化学習の再現フェーズ
            self.__imitation(imitation_file_name)
        elif self.__learn == REIN_PHASE.INFERENCE:
            # 強化学習の推論フェーズ
            self.__inference()
        else:
            # 異常な引数
            raise ValueError(f"self.__learn is abnormal. self.__learn is {self.__learn}")


    def __learning(self) -> None:
        """
        強化学習の学習フェーズ (SARSA)

        SARSAはon-policy TD学習のため，1ステップごとにQ値を更新する．
        ループ前に最初の行動を選択し，ループ内では (s, a, r, s', a') のセットで更新する．
        """
        # 全エピソードの報酬
        rewards_history: list[float] = []

        # ファイル保存するエピソードのインターバル
        save_episode_interval = int(self.__MAX_EPISODES / self.__N_SAVE_EPISODES)
        # インターバルが「0」以下での対処方法
        save_episode_interval = max(save_episode_interval, 1)

        # エピソード分ループ
        for episode in range(self.__MAX_EPISODES):
            # 強化学習用の環境とエージェントを初期化
            state = self.__env.reset()
            self.__agent.reset()

            # 完了フラグ
            done = False
            # １エピソード分の報酬の合計
            total_reward = 0.0
            # ループ数の保存
            count = 0
            # １エピソード分の行動を保存
            action_per_episode: list[int] = []

            # SARSAは最初の行動をループ前に選択する (s, a, r, s', a' の a に相当)
            action = self.__agent.get_action(state)

            print(f"episode = {episode} is start.")

            # 完了するまでループ
            while not done:
                # １ステップ実行: (s, a) → (s', r, done)
                next_state, reward, done = self.__env.step(action)

                # 次の行動を選択 (on-policy: 同じε-greedy方策で選択)
                next_action = self.__agent.get_action(next_state)

                # Q値を即時更新: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
                self.__agent.add(state, action, reward, next_state, next_action)

                # 報酬の更新
                total_reward += reward

                # 状態と行動の更新 (s ← s', a ← a')
                state  = next_state
                action = next_action

                # カウンタの更新
                count += 1

                # 行動の保存
                action_per_episode.append(action)

            # エピソード終了: 探索確率 ε の更新 (Q値更新はadd()内で完了済み)
            self.__agent.update()

            # 報酬を履歴に追加
            rewards_history.append(total_reward)

            # 行動をファイルに保存
            if episode % save_episode_interval == 0:
                self.__save_action(episode, action_per_episode)

            print(f"count = {count}")
            print(f"total_reward = {total_reward}")
            print(f"episode = {episode} is fin.")
            print()

        # パラメータをファイルに書き込む
        self.__agent.save()


    def __save_action(self, episode: int, actions: list[int]) -> None:
        """
        １エピソード分の行動をファイルに保存

        パラメータ
            episode (int): エピソード番号
            actions (list[int]): １エピソード分の行動
        """
        # ファイル名の作成
        file_name = f"{RESULT_FOLDER_NAME}/episode_{episode}.csv"

        # CSVファイルに行動を保存
        with open(file_name, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(actions)


    def __imitation(self, file_name: str) -> None:
        """
        強化学習の再現フェーズ

        パラメータ
            file_name(str): 行動が保存されているファイル
        """
        # 環境の初期化
        self.__env.reset()

        # １エピソード分の報酬の合計
        total_reward = 0.0

        with open(file_name, "r") as f:
            # 行動をファイルから2次元配列の形で取得
            actions = csv.reader(f)
            # 全行動の実行
            for list_action in actions:
                for str_action in list_action:
                    # 文字列から，数値型に型変換
                    int_action = int(str_action)
                    # １ステップ実行
                    next_state, reward, done = self.__env.step(int_action)

                    print(f"next_state = {next_state}")
                    print(f"action = {int_action}")
                    print()

                    # 報酬の更新
                    total_reward += reward

                print(f"len(list_action) = {len(list_action)}")
        print(f"total_reward = {total_reward}")
        print()


    def __inference(self) -> None:
        """
        強化学習の推論フェーズ
        """
        # パラメータをファイルから読み込む
        self.__agent.load()

        # 探索確率を 0 として，活用だけとする
        self.__agent.set_epsilon(0.0)

        # 強化学習用の環境とエージェントを初期化
        state = self.__env.reset()
        self.__agent.reset()

        # 完了フラグ
        done = False
        # １エピソード分の報酬の合計
        total_reward = 0.0
        # ループ数の保存
        count = 0

        print("inference is start.")

        # 完了するまでループ
        while not done:
            # 状態から行動を取得
            action = self.__agent.get_action(state)
            # １ステップ実行
            next_state, reward, done = self.__env.step(action)

            # 報酬の更新
            total_reward += reward

            # 状態の更新
            state = next_state

            # カウンタの更新
            count += 1

        print(f"count = {count}")
        print(f"total_reward = {total_reward}")
        print("inference is fin.")
        print()


    def __get_grasp_pos_from_camera(self) -> np.ndarray:
        """
        カメラから把持物体の位置を取得

        戻り値
            numpy.ndarray: 把持物体の位置
        """
        # カメラから物体位置(3次元)の取得
        pos = self.__camera.get_pos(self.__grasp_obj.grasp_obj_id)
        # 今回は2軸ロボットアームだけに対応するため，物体位置を2次元に変換する
        pos = pos[:DIMENTION_2D]
        # 探索方法に応じた終点に変換
        if self.__robot.interpolation == INTERPOLATION.JOINT.value:
            # 逆運動学により，関節角度を取得
            pos = self.__robot.convert_pos_to_theta(pos)

        return pos
    # メイン処理 ↑


    # 運動学関連 ↓
    def convert_pos_to_theta(self, pos: np.ndarray) -> np.ndarray:
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
