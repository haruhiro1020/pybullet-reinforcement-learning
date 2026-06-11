# エージェント制御クラスの作成


from constant import AGENT                              # エージェント定数
from Agent.BaseAgent import BaseAgent                   # ベースエージェントクラス (抽象クラス)
from Agent.RandomAgent import RandomAgent               # ランダムエージェントクラス
from Agent.MonteCarloAgent import MonteCarloAgent       # モンテカルロエージェントクラス
from Agent.SarsaAgent import SarsaAgent                 # SARSAエージェントクラス
from Agent.QLearningAgent import QLearningAgent         # Q学習エージェントクラス
from Agent.DQNAgent import DQNAgent                     # DQNエージェントクラス



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
        # エージェントクラスの取得
        agent_cls = self.__get_agent_class(agent_type)

        # DQN は state_dim も必要なため，agent_type に応じてインスタンス作成
        if agent_type == AGENT.DQN.value:
            self.__agent: BaseAgent = agent_cls(n_action, state_dim)
        else:
            self.__agent: BaseAgent = agent_cls(n_action)

        # エージェントタイプをプロパティに保存
        self.__agent_type = agent_type


    def __get_agent_class(self, agent_type: AGENT):
        """
        エージェントクラスの取得

        パラメータ
            agent_type(AGENT): エージェントタイプ

        戻り値
            エージェントに応じたクラス
        """
        # エージェント番号とエージェントクラスを保存する辞書型データの定義
        agents = {
            AGENT.RANDOM.value:     RandomAgent,
            AGENT.MONTECARLO.value: MonteCarloAgent,
            AGENT.SARSA.value:      SarsaAgent,
            AGENT.QLEARNING.value:  QLearningAgent,
            AGENT.DQN.value:        DQNAgent
        }

        # エージェントクラスの定義
        agent_cls = None

        for key, value in agents.items():
            if agent_type == key:
                # エージェント番号が一致
                agent_cls = value
                break

        if agent_cls is None:
            # エージェント番号が一致しなかった
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


    def add(self, state, action, reward, next_state=None, next_action=None, done=False):
        """
        データの追加

        引数の組み合わせに応じてエージェントの add() を呼び分ける

        パラメータ
            state(tuple)    : 状態
            action(int)     : 行動
            reward(float)   : 報酬
            next_state(tuple): 次の状態 (SARSA・Q学習・DQNで使用)
            next_action(int) : 次の行動 (SARSAで使用)
            done(bool)       : 終了フラグ (DQN・Q学習で使用)
        """
        if next_action is not None:
            # SARSA: (s, a, r, s', a') の5要素で更新
            self.__agent.add(state, action, reward, next_state, next_action)
        elif next_state is not None:
            # Q学習・DQN: (s, a, r, s', done) の5要素で更新
            self.__agent.add(state, action, reward, next_state, done)
        else:
            # モンテカルロ法: (s, a, r) の3要素でメモリに追加
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
