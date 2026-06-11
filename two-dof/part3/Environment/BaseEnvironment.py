# 強化学習の全環境のベースを定義

from abc import ABC, abstractmethod

import numpy as np


class BaseEnvironment(ABC):
    """
    全環境のベースクラス (抽象クラス)

    メソッド
        public
            reset(): 初期化
            step(): 1ステップ実行
    """

    @abstractmethod
    def reset(self) -> tuple:
        """
        初期化

        戻り値
            tuple: 初期状態
        """
        pass

    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """
        1ステップ実行

        パラメータ
            action(int): 行動

        戻り値
            tuple: 次の状態
            float: 報酬
            bool: 完了フラグ
        """
        pass
