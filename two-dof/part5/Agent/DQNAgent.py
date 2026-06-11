# DQNエージェントの作成

# 外部ライブラリの読み込み
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# 自作モジュールの読み込み
from .BaseAgent import BaseAgent


class _QNetwork(nn.Module):
    """
    Q関数を近似するニューラルネットワーク (2層MLP)

    入力: 状態 (正規化済み)
    出力: 各行動のQ値
    """

    def __init__(self, state_dim, n_action, hidden_dim):
        """
        コンストラクタ

        パラメータ
            state_dim(int) : 状態の次元数
            n_action(int)  : 行動数
            hidden_dim(int): 隠れ層のユニット数
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_action)
        )

    def forward(self, x):
        return self.layers(x)


class DQNAgent(BaseAgent):
    """
    DQNエージェントのクラス

    Deep Q-Network (DQN) は，Q関数をニューラルネットワークで近似する．
    経験再生 (Experience Replay) とターゲットネットワーク (Target Network) により
    学習を安定化させる．

    プロパティ
    メソッド
        public
            get_action(): 行動の取得
            reset(): 初期化
            update(): データの更新 (1エピソード終了ごとに呼ぶ: εの更新のみ)
            add(): 経験の追加 + ミニバッチ学習 (1ステップごとに呼ぶ)
    """
    # 強化学習パラメータ
    _GAMMA = 0.9                # 割引率

    # 探索確率に関する定数
    _EPSILON_MIN   = 0.1        # 探索の最小確率
    _EPSILON_MAX   = 0.9        # 探索の最大確率
    _EPSILON_DECAY = 0.9999     # 探索確率の倍率

    # ニューラルネットワークに関する定数
    _LR         = 0.001         # Adam の学習率
    _HIDDEN_DIM = 64            # 隠れ層のユニット数

    # 経験再生に関する定数
    _BUFFER_SIZE     = 10000    # 経験再生バッファのサイズ
    _BATCH_SIZE      = 64       # ミニバッチサイズ
    _MIN_REPLAY_SIZE = 500      # 学習開始に必要な最小経験数

    # ターゲットネットワークに関する定数
    _TARGET_UPDATE_INTERVAL = 100   # ターゲットネットワーク更新間隔 [ステップ]

    # 状態に関する定数 (DiscretizeEnvironment._STATE_NUM と一致させる)
    _STATE_NUM = 37             # 1軸あたりの状態数

    __PARAMETER_FILE = "dqn_param.pth"  # パラメータを保存するファイル


    def __init__(self, n_action, state_dim=2):
        """
        コンストラクタ

        パラメータ
            n_action(int) : 行動数
            state_dim(int): 状態の次元数 (デフォルト: 2軸)
        """
        self.__n_action   = n_action
        self.__state_dim  = state_dim
        self.__epsilon    = self._EPSILON_MAX
        self.__step_count = 0

        # メインQ ネットワーク (毎ステップ更新)
        self.__q_network = _QNetwork(state_dim, n_action, self._HIDDEN_DIM)
        # ターゲットQ ネットワーク (_TARGET_UPDATE_INTERVAL ステップごとに更新)
        self.__target_network = _QNetwork(state_dim, n_action, self._HIDDEN_DIM)
        self.__target_network.load_state_dict(self.__q_network.state_dict())
        self.__target_network.eval()

        # Adam オプティマイザ
        self.__optimizer = optim.Adam(self.__q_network.parameters(), lr=self._LR)

        # 損失関数 (平均二乗誤差)
        self.__loss_fn = nn.MSELoss()

        # 経験再生バッファ
        self.__replay_buffer = deque(maxlen=self._BUFFER_SIZE)


    def __state_to_tensor(self, state):
        """
        状態 (グリッドインデックスのタプル) をテンソルに変換

        インデックスを [0, 1] に正規化してネットワークの入力とする
        (インデックス / (STATE_NUM - 1))

        パラメータ
            state(tuple): 状態 (グリッドインデックス)

        戻り値
            torch.FloatTensor: 正規化済み状態テンソル
        """
        normalized = [s / (self._STATE_NUM - 1) for s in state]
        return torch.FloatTensor(normalized)


    def __update_epsilon(self):
        """
        探索確率の更新
        """
        if self.__epsilon > self._EPSILON_MIN:
            self.__epsilon *= self._EPSILON_DECAY

        print(f"self.__epsilon = {self.__epsilon}")


    def set_epsilon(self, value):
        """
        探索確率である ε の設定

        パラメータ
            value(float): 設定値
        """
        if 0.0 <= value <= 1.0:
            self.__epsilon = value


    def get_action(self, state):
        """
        行動の取得

        パラメータ
            state(tuple): 状態

        戻り値
            int: 行動
        """
        if np.random.rand() < self.__epsilon:
            # ランダム探索
            return int(np.random.choice(self.__n_action))

        # ニューラルネットワークで Q値を計算し，最大の行動を選択 (活用)
        state_tensor = self.__state_to_tensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.__q_network(state_tensor)

        return int(q_values.argmax().item())


    def reset(self):
        """
        初期化 (エピソード開始時に呼ぶ)
        """
        # DQN はステップごとに学習するためメモリ不要
        pass


    def add(self, state, action, reward, next_state, done=False):
        """
        経験の追加 + ミニバッチ学習 (1ステップごとに呼ぶ)

        DQNの損失関数:
            L(θ) = E[(r + γ * max_a'(Q(s',a';θ⁻)) * (1-done) - Q(s,a;θ))²]

        パラメータ
            state(tuple)      : 現在の状態
            action(int)       : 現在の行動
            reward(float)     : 報酬
            next_state(tuple) : 次の状態
            done(bool)        : 終了フラグ (Trueのとき次状態の価値を0とする)
        """
        # 経験再生バッファに追加
        self.__replay_buffer.append((state, action, reward, next_state, done))

        # 最小経験数に達したらネットワークを学習
        if len(self.__replay_buffer) >= self._MIN_REPLAY_SIZE:
            self.__train()

        self.__step_count += 1

        # ターゲットネットワークの定期更新
        if self.__step_count % self._TARGET_UPDATE_INTERVAL == 0:
            self.__update_target_network()


    def __train(self):
        """
        ミニバッチサンプリングによるネットワーク更新
        """
        # バッファからランダムサンプリング
        batch = random.sample(self.__replay_buffer, self._BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        # テンソルに変換
        state_tensors      = torch.stack([self.__state_to_tensor(s) for s in states])
        action_tensors     = torch.LongTensor(actions)
        reward_tensors     = torch.FloatTensor(rewards)
        next_state_tensors = torch.stack([self.__state_to_tensor(s) for s in next_states])
        done_tensors       = torch.FloatTensor(dones)

        # 現在のQ値: Q(s, a; θ)
        current_q = self.__q_network(state_tensors).gather(1, action_tensors.unsqueeze(1)).squeeze(1)

        # ターゲット値: r + γ * max_a'(Q(s', a'; θ⁻)) * (1 - done)
        # done=True のとき (1-done)=0 となり，次状態の価値を使わない
        with torch.no_grad():
            max_next_q = self.__target_network(next_state_tensors).max(1)[0]
            target_q   = reward_tensors + self._GAMMA * max_next_q * (1 - done_tensors)

        # 損失計算 + 逆伝播 + パラメータ更新
        loss = self.__loss_fn(current_q, target_q)
        self.__optimizer.zero_grad()
        loss.backward()
        self.__optimizer.step()


    def __update_target_network(self):
        """
        ターゲットネットワークをメインネットワークの重みで更新 (θ⁻ ← θ)
        """
        self.__target_network.load_state_dict(self.__q_network.state_dict())


    def update(self):
        """
        データの更新 (1エピソード終了ごとに呼ぶ)

        ネットワークの更新は add() 内でステップごとに完了しているため，
        ここでは探索確率 ε の更新のみ行う．
        """
        self.__update_epsilon()


    def save(self):
        """
        学習データの保存 (メインネットワークの重みを保存)
        """
        torch.save(self.__q_network.state_dict(), self.__PARAMETER_FILE)


    def load(self):
        """
        学習データの読み込み
        """
        self.__q_network.load_state_dict(
            torch.load(self.__PARAMETER_FILE, weights_only=True)
        )
        self.__target_network.load_state_dict(self.__q_network.state_dict())
