# モンテカルロエージェントの作成


# 外部ライブラリの読み込み
import numpy as np
from collections import defaultdict
import json
import ast


# 自作モジュールの読み込み
from .BaseAgent import BaseAgent



class MonteCarloAgent(BaseAgent):
    """
    モンテカルロ法 (Monte Carlo：エピソード終了後に収益を遡って更新する手法) エージェントのクラス

    プロパティ
    メソッド
        public
            get_action(): 行動の取得
            reset(): 初期化
            update(): データの更新
            add(): データの追加
            save(): 学習データの保存
            load(): 学習データの読み込み
    """
    # 定数の定義
    _GAMMA = 0.9        # 割引率 (discount factor：将来の報酬をどれだけ割り引くか)
    _ALPHA = 0.1        # 更新量 (learning rate：1ステップあたりのパラメータ更新量)

    # 探索確率 ε (epsilon) に関する定数
    _EPSILON_MIN   = 0.1        # 探索の最小確率
    _EPSILON_MAX   = 0.9        # 探索の最大確率
    _EPSILON_DECAY = 0.9999     # 探索確率の減衰率

    __INITIAL_Q_VALUE = 0       # 行動価値関数 (Q値) の初期値

    __PARAMETER_FILE = "montecarlo_param.json"      # パラメータを保存する JSON ファイル


    def __init__(self, n_action):
        """
        コンストラクタ

        パラメータ
            n_action(int): 行動数
        """
        self.__n_action = n_action
        self.__epsilon  = self._EPSILON_MAX
        self.__memory   = []
        self.__Q        = defaultdict(lambda: self.__INITIAL_Q_VALUE)


    def __update_epsilon(self):
        """
        探索確率の更新
        """
        if self.__epsilon > self._EPSILON_MIN:
            self.__epsilon *= self._EPSILON_DECAY


    def set_epsilon(self, value):
        """
        探索確率 ε の設定

        パラメータ
            value(float): 設定値
        """
        if 0.0 <= value <= 1.0:
            self.__epsilon = value


    def get_action(self, state):
        """
        行動の取得

        パラメータ
            state(tuple): 状態

        戻り値
            int: 行動
        """
        if np.random.rand() < self.__epsilon:
            # ε-greedy (イプシロン・グリーディ) 探索：確率 ε でランダム行動
            action = np.random.choice(self.__n_action)
        else:
            # 活用：行動価値関数が最大となる行動を選択
            Qs = [self.__Q[state, a] for a in range(self.__n_action)]
            action = np.argmax(Qs)

        return int(action)


    def reset(self):
        """
        初期化
        """
        self.__memory.clear()


    def update(self):
        """
        データの更新
        """
        G = 0.0

        # 終了状態から逆順に収益 G (return：割引報酬の総和) を計算
        for data in reversed(self.__memory):
            state, action, reward = data
            G = reward + self._GAMMA * G
            key = (state, action)
            self.__Q[key] = self.__Q[key] + self._ALPHA * (G - self.__Q[key])

        self.__update_epsilon()


    def add(self, state, action, reward):
        """
        データの追加

        パラメータ
            state(numpy.ndarray): 状態
            action(int): 行動
            reward(float): 報酬
        """
        self.__memory.append((state, action, reward))


    def save(self):
        """
        学習データの保存
        """
        save_data = {
            "Q": {str(k): v for k, v in self.__Q.items()}
        }
        with open(self.__PARAMETER_FILE, "w") as f:
            json.dump(save_data, f)


    def load(self):
        """
        学習データの読み込み
        """
        with open(self.__PARAMETER_FILE, "r") as f:
            data = json.load(f)

        # ast.literal_eval (安全な式評価) でタプルキーを復元
        self.__Q = {ast.literal_eval(k): v for k, v in data["Q"].items()}
