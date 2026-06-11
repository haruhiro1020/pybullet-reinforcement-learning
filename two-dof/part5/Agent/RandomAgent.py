# ランダムエージェントの作成


# 外部ライブラリの読み込み
import numpy as np


# 自作モジュールの読み込み
from .BaseAgent import BaseAgent



class RandomAgent(BaseAgent):
    """
    ランダムエージェントのクラス

    プロパティ
    メソッド
        public
            get_action(): 行動の取得
            reset(): 初期化
            update(): データの更新
            add(): データの追加
    """
    # 定数の定義


    def __init__(self, n_action):
        """
        コンストラクタ

        パラメータ
            n_action(int): 行動数
        """
        self.__n_action = n_action


    def set_epsilon(self, value):
        """
        探索確率である ε の設定

        パラメータ
            value(float): 設定値
        """
        # 何もしない
        pass


    def get_action(self, state):
        """
        行動の取得

        パラメータ
            state(tuple): 状態

        戻り値
            int: 行動
        """
        # 行動をランダムに選択
        action = np.random.choice(range(self.__n_action))
        return int(action)


    def reset(self):
        """
        初期化
        """
        # 何もしない
        pass


    def update(self):
        """
        データの更新
        """
        # 何もしない
        pass


    def add(self, state, action, reward):
        """
        データの追加

        パラメータ
            state(numpy.ndarray): 状態
            action(int): 行動
            reward(float): 報酬
        """
        # 何もしない
        pass


    def save(self):
        """
        学習データの保存
        """
        # 何もしない
        pass


    def load(self):
        """
        学習データの読み込み
        """
        # 何もしない
        pass
