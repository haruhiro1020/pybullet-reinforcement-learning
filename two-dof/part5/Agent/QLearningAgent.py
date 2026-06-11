# Q学習エージェントの作成

# 外部ライブラリの読み込み
import numpy as np
from collections import defaultdict
import json

# 自作モジュールの読み込み
from .BaseAgent import BaseAgent


class QLearningAgent(BaseAgent):
    """
    Q学習エージェントのクラス

    プロパティ
    メソッド
        public
            get_action(): 行動の取得
            reset(): 初期化
            update(): データの更新 (1エピソード終了ごとに呼ぶ: εの更新のみ)
            add(): データの追加 (1ステップごとに呼ぶ: Q値の更新)
    """
    # 定数の定義
    _GAMMA = 0.9        # 割引率
    _ALPHA = 0.1        # 更新量

    # 探索確率に関する定数
    _EPSILON_MIN   = 0.1        # 探索の最小確率
    _EPSILON_MAX   = 0.9        # 探索の最大確率
    _EPSILON_DECAY = 0.9999     # 探索確率の倍率

    __INITIAL_Q_VALUE = 0       # 行動価値関数の初期値

    __PARAMATER_FILE = "qlearning_param.json"       # パラメータを保存する JSON ファイル


    def __init__(self, n_action):
        """
        コンストラクタ

        パラメータ
            n_action(int): 行動数
        """
        # 行動数
        self.__n_action = n_action
        # 探索確率
        self.__epsilon  = self._EPSILON_MAX

        # 全状態・行動の行動価値関数を辞書型で定義
        self.__Q = defaultdict(lambda: self.__INITIAL_Q_VALUE)


    def __update_epsilon(self):
        """
        探索確率の更新
        """
        if self.__epsilon > self._EPSILON_MIN:
            # 探索確率の修正
            self.__epsilon *= self._EPSILON_DECAY

        print(f"self.__epsilon = {self.__epsilon}")


    def set_epsilon(self, value):
        """
        探索確率である ε の設定

        パラメータ
            value(float): 設定値
        """
        if value >= 0.0 and value <= 1.0:
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
            # ランダム探索
            action = np.random.choice(self.__n_action)
        else:
            # 行動価値関数が最大となる行動を活用
            Qs = [self.__Q[state, a] for a in range(self.__n_action)]
            action = np.argmax(Qs)

        return int(action)


    def reset(self):
        """
        初期化 (エピソード開始時に呼ぶ)
        """
        pass


    def add(self, state, action, reward, next_state, done=False):
        """
        データの追加 + Q値の即時更新 (1ステップごとに呼ぶ)

        Q学習の更新式:
            Q(s, a) ← Q(s, a) + α * [r + γ * max_a'(Q(s', a')) - Q(s, a)]

        パラメータ
            state(tuple)      : 現在の状態
            action(int)       : 現在の行動
            reward(float)     : 報酬
            next_state(tuple) : 次の状態
            done(bool)        : 終了フラグ (Q学習では未使用。DQNとのAPI統一のため受け取る)
        """
        key = (state, action)

        # 次状態での全行動のQ値を取得して，最大値を選択 (off-policy: greedy方策)
        next_Qs = [self.__Q[next_state, a] for a in range(self.__n_action)]
        max_next_Q = max(next_Qs)

        # TD誤差の計算: r + γ * max_a'(Q(s', a')) - Q(s, a)
        td_error = reward + self._GAMMA * max_next_Q - self.__Q[key]

        # 行動価値関数の更新
        self.__Q[key] = self.__Q[key] + self._ALPHA * td_error


    def update(self):
        """
        データの更新 (1エピソード終了ごとに呼ぶ)

        Q値の更新は add() 内でステップごとに完了しているため，
        ここでは探索確率 ε の更新のみ行う．
        """
        self.__update_epsilon()


    def save(self):
        """
        学習データの保存
        """
        save_datas = {
            "Q": {str(k): v for k, v in self.__Q.items()}
        }

        with open(self.__PARAMATER_FILE, "w") as f:
            json.dump(save_datas, f)


    def load(self):
        """
        学習データの読み込み
        """
        with open(self.__PARAMATER_FILE, "r") as f:
            datas = json.load(f)

        Q = datas["Q"]
        self.__Q = {eval(k): v for k, v in Q.items()}
