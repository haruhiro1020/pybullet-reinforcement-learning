# 強化学習の全エージェントのベースを定義


class BaseAgent:
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
    # 定数の定義


    # プロパティのゲッター ↓
    @property
    def Q(self):
        """
        行動価値関数の取得
        """
        raise NotImplementedError("Q() is necessary override.")


    @property
    def V(self):
        """
        状態価値関数の取得
        """
        raise NotImplementedError("V() is necessary override.")
    # プロパティのゲッター ↑


    def set_epsilon(self, value):
        """
        探索確率である ε の設定

        パラメータ
            value(float): 設定値
        """
        raise NotImplementedError("set_epsilon() is necessary override.")


    def get_action(self, state):
        """
        行動の取得

        パラメータ
            state(numpy.ndarray): 状態

        戻り値
            int: 行動
        """
        raise NotImplementedError("get_action() is necessary override.")


    def reset(self):
        """
        初期化
        """
        raise NotImplementedError("reset() is necessary override.")


    def update(self):
        """
        データの更新
        """
        raise NotImplementedError("update() is necessary override.")


    def add(self, state, action, reward):
        """
        データの追加

        パラメータ
            state(numpy.ndarray): 状態
            action(int): 行動
            reward(float): 報酬
        """
        raise NotImplementedError("add() is necessary override.")


    def save(self):
        """
        学習データの保存
        """
        raise NotImplementedError("save() is necessary override.")


    def load(self):
        """
        学習データの読み込み
        """
        raise NotImplementedError("load() is necessary override.")
