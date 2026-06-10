# 強化学習の全エージェントのベースを定義

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    全エージェントのベースクラス (抽象クラス)

    メソッド
        public
            get_action(): 行動の取得
            reset(): 初期化
            update(): 更新
            add(): データの追加
            set_epsilon(): 探索確率の設定
            save(): 学習データの保存
            load(): 学習データの読み込み
    """

    @property
    def Q(self):
        """
        行動価値関数 (Q-function：状態と行動のペアに対する期待収益) の取得
        """
        raise NotImplementedError("Q() is necessary override.")

    @property
    def V(self):
        """
        状態価値関数 (V-function：状態に対する期待収益) の取得
        """
        raise NotImplementedError("V() is necessary override.")

    @abstractmethod
    def set_epsilon(self, value):
        """
        探索確率 ε (epsilon：ε-greedy法で使うランダム行動の確率) の設定

        パラメータ
            value(float): 設定値
        """
        pass

    @abstractmethod
    def get_action(self, state):
        """
        行動の取得

        パラメータ
            state(numpy.ndarray): 状態

        戻り値
            int: 行動
        """
        pass

    @abstractmethod
    def reset(self):
        """
        初期化
        """
        pass

    @abstractmethod
    def update(self):
        """
        データの更新
        """
        pass

    @abstractmethod
    def add(self, state, action, reward):
        """
        データの追加

        パラメータ
            state(numpy.ndarray): 状態
            action(int): 行動
            reward(float): 報酬
        """
        pass

    @abstractmethod
    def save(self):
        """
        学習データの保存
        """
        pass

    @abstractmethod
    def load(self):
        """
        学習データの読み込み
        """
        pass
