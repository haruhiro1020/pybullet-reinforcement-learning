# 強化学習の全環境のベースを定義

# 外部ライブラリの読み込み
from abc import ABC, abstractmethod



class BaseEnvironment(ABC):
    """
    全環境のベースクラス (抽象クラス)

    プロパティ

    メソッド
        public
            reset(): 初期化
            step(): 1ステップ実行
    """
    # 定数の定義


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
