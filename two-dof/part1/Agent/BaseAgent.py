# 強化学習の全エージェントのベースを定義


# 外部ライブラリの読み込み
from abc import ABC, abstractmethod



class BaseAgent(ABC):
    """
    全エージェントのベースクラス (抽象クラス)

    プロパティ

    メソッド
        public
            get_action(): 行動の取得
            reset(): 初期化
            update(): 更新
            add(): データの追加
            set_epsilon(): 探索確率の設定
    """
    @abstractmethod
    def set_epsilon(self, value):
        """
        探索確率である ε の設定

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
    def eval(self, state, action, reward, done, next_state):
        """
        1ステップ分のデータで更新 (オンライン学習：Q学習・SARSAなど1ステップごとに更新する手法で使用)

        パラメータ
            state(numpy.ndarray): 状態
            action(int): 行動
            reward(float): 報酬
            done(bool): 完了フラグ
            next_state(numpy.ndarray): 次状態
        """
        pass
