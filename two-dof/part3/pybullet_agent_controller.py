# エージェント制御クラスの作成

from typing import Optional

from constant import AGENT                              # エージェント定数
from Agent.BaseAgent import BaseAgent                   # ベースエージェントクラス (抽象クラス)
from Agent.RandomAgent import RandomAgent               # ランダムエージェントクラス
from Agent.MonteCarloAgent import MonteCarloAgent       # モンテカルロエージェントクラス
from Agent.SarsaAgent import SarsaAgent                 # SARSAエージェントクラス



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


    def __init__(self, agent_type: int, n_action: int, state_dim: int) -> None:
        """
        コンストラクタ

        パラメータ
            agent_type(int): エージェント (AGENT列挙型の .value)
            n_action(int): 行動数
            state_dim(int): 状態の次元数
        """
        # エージェントクラスの取得
        agent_cls = self.__get_agent_class(agent_type)

        # エージェントクラスのインスタンス作成
        self.__agent: BaseAgent = agent_cls(n_action)

        # エージェントタイプをプロパティに保存
        self.__agent_type = agent_type


    def __get_agent_class(self, agent_type: int) -> type[BaseAgent]:
        """
        エージェントクラスの取得

        パラメータ
            agent_type(int): エージェントタイプ (AGENT列挙型の .value)

        戻り値
            type[BaseAgent]: エージェントに応じたクラス
        """
        # エージェント番号とエージェントクラスを保存する辞書型データの定義
        agents: dict[int, type[BaseAgent]] = {
            AGENT.RANDOM.value:     RandomAgent,
            AGENT.MONTECARLO.value: MonteCarloAgent,
            AGENT.SARSA.value:      SarsaAgent
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


    def set_epsilon(self, value: float) -> None:
        """
        探索確率である ε の設定

        パラメータ
            value(float): 設定値
        """
        self.__agent.set_epsilon(value)


    def get_action(self, state: tuple) -> int:
        """
        行動の取得

        パラメータ
            state(tuple): 状態

        戻り値
            int: 行動
        """
        return self.__agent.get_action(state)


    def reset(self) -> None:
        """
        初期化
        """
        self.__agent.reset()


    def update(self) -> None:
        """
        1エピソード分のデータで更新
        """
        self.__agent.update()


    def add(self, state: tuple, action: int, reward: float, next_state: Optional[tuple] = None, next_action: Optional[int] = None) -> None:
        """
        データの追加

        パラメータ
            state(tuple)             : 状態
            action(int)              : 行動
            reward(float)            : 報酬
            next_state(tuple | None) : 次の状態 (SARSAで使用)
            next_action(int | None)  : 次の行動 (SARSAで使用)
        """
        if next_state is not None and next_action is not None:
            self.__agent.add(state, action, reward, next_state, next_action)
        else:
            self.__agent.add(state, action, reward)


    def save(self) -> None:
        """
        学習データの保存
        """
        self.__agent.save()


    def load(self) -> None:
        """
        学習データの読み込み
        """
        self.__agent.load()
