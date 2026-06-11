# ソースコード構成

PyBullet シミュレータ上のロボットアームに対して、強化学習アルゴリズムを段階的に適用するソースコード群です。  
`two-dof`（2軸） シリーズで構成され、各シリーズは Part 1〜5 まで **同一のアルゴリズム順序** で進みます。

---

## アルゴリズム一覧

各 Part でそのパート固有のアルゴリズムを新たに導入します。  
前パートのエージェントはすべて引き継がれるため、Part が進むほどエージェントの種類が増えていきます。

| Part | 新規アルゴリズム | 分類 | 概要 |
|------|----------------|------|------|
| Part 1 | ランダム | ベースライン | 行動空間（Action Space：エージェントがとれる全行動の集合）からランダムに行動を選択 |
| Part 2 | モンテカルロ法 | モデルフリー / オンポリシー | エピソード（1試行の始まりから終わりまでの一連の行動列）完了後に累積報酬で価値関数を更新 |
| Part 3 | SARSA | TD 法 / オンポリシー | SARSA（State-Action-Reward-State-Action：現在と次ステップの行動価値を TD 誤差で逐次更新） |
| Part 4 | Q 学習 | TD 法 / オフポリシー | Q-Learning（Q学習：最大行動価値を使って貪欲に更新。SARSA のオフポリシー版） |
| Part 5 | DQN | 深層 RL / 離散行動 | DQN（Deep Q-Network：ニューラルネットで Q 値を近似。Experience Replay・Target Network を使用） |

---

## シリーズ構成

```
src/
└── two-dof/        # 2軸ロボットアームシリーズ (part1〜12)
```

各シリーズの各 Part は以下の共通ディレクトリ構成を持ちます。

```
partN/
├── main.py                            # エントリポイント
├── constant.py                        # 定数定義
├── pybullet_main.py                   # PyBullet メイン処理
├── Agent/
│   ├── BaseAgent.py                   # エージェント抽象基底クラス
│   ├── RandomAgent.py                 # ランダムエージェント
│   └── <アルゴリズム名>Agent.py       # 各 Part の新規エージェント
├── Environment/
│   ├── BaseEnvironment.py             # 環境抽象基底クラス
│   └── DiscretizeEnvironment.py       # 離散値環境
└── URDF/                              # ロボット・環境の 3D モデル定義ファイル
```
