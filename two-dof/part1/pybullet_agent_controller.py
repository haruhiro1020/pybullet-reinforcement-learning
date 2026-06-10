# エージェント制御クラスの作成


# 自作モジュールの読み込み
from constant import AGENT                  # エージェント定数
from Agent.BaseAgent import BaseAgent       # ベースエージェントクラス (抽象クラス)
from Agent.RandomAgent import RandomAgent   # ランダムエージェントクラス



class PyBulletAgentController:
    """
    エージェント制御クラス
        エージェントタイプに応じたエージェントクラスを生成し，操作を委譲する
    """

    def __init__(self, agent_type: AGENT, n_action, state_dimention=None):
        """
        コンストラクタ

        パラメータ
            agent_type(AGENT): エージェント
            n_action(int): 行動数
            state_dimention(int): 状態の次元数 (将来の拡張用)
        """
        # エージェントクラスのインスタンス取得
        self.__agent: BaseAgent = self.__make_agent_class(agent_type, n_action)


    def __make_agent_class(self, agent_type: AGENT, n_action: int) -> BaseAgent:
        """
        エージェントクラスの作成

        パラメータ
            agent_type(AGENT): エージェントタイプ
            n_action(int): 行動数

        戻り値
            エージェントに応じたクラス
        """
        agents = {
            AGENT.RANDOM: RandomAgent,
        }

        agent_cls = agents.get(agent_type)
        if agent_cls is None:
            raise ValueError(f"agent is abnormal. agent is {agent_type}")

        return agent_cls(n_action)


    def set_epsilon(self, value):
        """
        探索確率である ε の設定

        パラメータ
            value(float): 設定値
        """
        self.__agent.set_epsilon(value)


    def get_action(self, state):
        """
        行動の取得

        パラメータ
            state(tuple): 状態

        戻り値
            int: 行動
        """
        return self.__agent.get_action(state)


    def reset(self):
        """
        初期化
        """
        self.__agent.reset()


    def update(self):
        """
        1エピソード分のデータで更新
        """
        self.__agent.update()


    def add(self, state, action, reward):
        """
        データの追加

        パラメータ
            state(numpy.ndarray): 状態
            action(int): 行動
            reward(float): 報酬
        """
        self.__agent.add(state, action, reward)


    def eval(self, state, action, reward, done, next_state):
        """
        1ステップ分のデータで更新 (オンライン学習用)

        パラメータ
            state(tuple): 状態
            action(int): 行動
            reward(float): 報酬
            done(bool): 完了フラグ
            next_state(tuple): 次状態
        """
        self.__agent.eval(state, action, reward, done, next_state)
