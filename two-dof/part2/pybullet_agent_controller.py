# エージェント制御クラスの作成


from constant import AGENT                  # エージェント定数
from Agent.BaseAgent import BaseAgent       # ベースエージェントクラス (抽象クラス)
from Agent.RandomAgent import RandomAgent   # ランダムエージェントクラス
from Agent.MontecarloAgent import MonteCarloAgent   # モンテカルロエージェントクラス



class PyBulletAgentController:
    """
    エージェント制御クラス

    プロパティ

    メソッド
        public
        protected
        private
    """
    # 定数の定義


    def __init__(self, agent_type: AGENT, n_action, state_dim):
        """
        コンストラクタ

        パラメータ
            agent_type(AGENT): エージェント
            n_action(int): 行動数
            state_dim(int): 状態の次元数
        """
        agent_cls = self.__get_agent_class(agent_type)
        self.__agent: BaseAgent = agent_cls(n_action)


    def __get_agent_class(self, agent_type: AGENT):
        """
        エージェントクラスの取得

        パラメータ
            agent_type(AGENT): エージェントタイプ

        戻り値
            エージェントに応じたクラス
        """
        agents = {
            AGENT.RANDOM.value:     RandomAgent,
            AGENT.MONTECARLO.value: MonteCarloAgent,
        }

        agent_cls = agents.get(agent_type)
        if agent_cls is None:
            raise ValueError(f"agent is abnormal. agent is {agent_type}")

        return agent_cls


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


    def save(self):
        """
        学習データの保存
        """
        self.__agent.save()


    def load(self):
        """
        学習データの読み込み
        """
        self.__agent.load()



